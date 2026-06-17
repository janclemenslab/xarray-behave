"""Audio loader

should return:
    audio_data: np.array[time, samples]
    non_audio_data: np.array[time, samples]
    samplerate: Optional[float]
"""

# [x] daq.h5
# [x] wav, ....
# [x] npz, npy
# [x] generic audio (pysoundfile)
# [ ] npy_dir
# [ ] mmap (Bartul)

import h5py
import os
import numpy as np
import logging
from .. import io
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


class SoundFileArray:
    """Small NumPy-like wrapper for random-access reads from an audio file."""

    ndim = 2

    def __init__(self, filename: str, dtype: str = "float32"):
        import soundfile as sf

        self.filename = str(filename)
        self.dtype = np.dtype(dtype)
        with sf.SoundFile(self.filename) as file:
            self.shape = (len(file), file.channels)
            self.sampling_rate = file.samplerate

    def __getitem__(self, key):
        if key is Ellipsis:
            key = (slice(None), slice(None))
        elif not isinstance(key, tuple):
            key = (key, slice(None))
        elif any(item is Ellipsis for item in key):
            key = tuple(slice(None) if item is Ellipsis else item for item in key)

        if len(key) == 1:
            key = (key[0], slice(None))
        if len(key) != 2:
            raise IndexError("Audio data must be indexed as samples[, channels].")

        sample_key, channel_key = key
        sample_scalar = np.isscalar(sample_key)
        if sample_scalar:
            sample_index = int(sample_key)
            if sample_index < 0:
                sample_index += self.shape[0]
            if sample_index < 0 or sample_index >= self.shape[0]:
                raise IndexError("sample index out of range")
            start = sample_index
            stop = sample_index + 1
            step = 1
        elif isinstance(sample_key, slice):
            start, stop, step = sample_key.indices(self.shape[0])
            if step < 0:
                raise ValueError("negative sample strides are not supported for lazy audio")
        else:
            sample_indices = np.asarray(sample_key)
            if sample_indices.dtype == bool:
                sample_indices = np.flatnonzero(sample_indices)
            sample_indices = sample_indices.astype(int, copy=False)
            sample_indices = np.where(sample_indices < 0, sample_indices + self.shape[0], sample_indices)
            if sample_indices.size == 0:
                data = np.empty((0, self.shape[1]), dtype=self.dtype)
                return data[:, channel_key]
            start = int(sample_indices.min())
            stop = int(sample_indices.max()) + 1
            step = None

        import soundfile as sf

        data, _ = sf.read(
            self.filename,
            start=start,
            stop=stop,
            dtype=self.dtype.name,
            always_2d=True,
        )
        if isinstance(sample_key, slice) and step != 1:
            data = data[::step]
        elif not sample_scalar and not isinstance(sample_key, slice):
            data = data[sample_indices - start]

        data = data[:, channel_key]
        if sample_scalar:
            data = data[0]
        return data


def split_song_and_nonsong(data, song_channels=None, return_nonsong_channels=False):
    song = data
    nonsong = None
    if song_channels is not None:
        song = song[:, song_channels]
        if return_nonsong_channels:
            nonsong = np.delete(data, song_channels, axis=-1)
    return song, nonsong


@io.register_provider
class Ethodrome(io.BaseProvider):
    KIND = "audio"
    NAME = "ethodrome h5"
    SUFFIXES = ["_daq.h5"]

    def load(
        self,
        filename: Optional[str],
        song_channels: Optional[Sequence[int]] = None,
        return_nonsong_channels: bool = False,
        lazy: bool = False,
        **kwargs,
    ):
        """[summary]

        Args:
            filename ([type]): [description]
            song_channels (List[int], optional): Sequence of integers as indices into 'samples' dataset.
                                                 Taken from 'song_channels' dataset of the h5 file if it exists.
                                                 Defaults to [0,..., 15].
            return_nonsong_channels (bool, optional): will return the data not in song_channels as separate array. Defaults to False
            lazy (bool, optional): If True, will load song as dask.array, which allows lazy indexing.
                                Otherwise, will load the full recording from disk (slow). Defaults to False.

        Returns:
            [type]: [description]
        """
        if filename is None:
            filename = self.path

        if song_channels is None:
            with h5py.File(filename, mode="r") as f:
                if "song_channels" in f:
                    song_channels = np.asarray(f["song_channels"][:])
            if song_channels is None:  # the first 16 channels in the data are the mic recordings
                song_channels = np.arange(16)

        non_song = None
        samplerate = None
        if lazy:
            f = h5py.File(filename, mode="r", rdcc_w0=0, rdcc_nbytes=100 * (1024**2), rdcc_nslots=50000)
            # convert to dask array since this allows lazily evaluated indexing...
            import dask.array as daskarray

            da = daskarray.from_array(f["samples"], chunks=(10000, 1))

            # FIXME: code in "if" and in "else" is identical - refactor to function
            nb_channels = f["samples"].shape[1]
            song_channels = song_channels[song_channels < nb_channels]
            song = da[:, song_channels]
            if return_nonsong_channels:
                non_song_channels = list(set(range(nb_channels)) - set(song_channels))
                non_song = da[:, non_song_channels]

            if "rate" in f.attrs:
                samplerate = f.attrs["rate"]
            elif "rate" in f["samples"].attrs:
                samplerate = f["samples"].attrs["rate"]
            else:
                logger.info("   No sampling rate information in daq.h5 file - setting samplerate to default 10_000Hz.")
                samplerate = 10_000
        else:
            with h5py.File(filename, "r") as f:
                da = f["samples"]
                nb_channels = f["samples"].shape[1]
                song_channels = song_channels[song_channels < nb_channels]
                song = da[:, song_channels]
                if return_nonsong_channels:
                    non_song_channels = list(set(range(nb_channels)) - set(song_channels))
                    non_song = da[:, non_song_channels]

                if "rate" in f.attrs:
                    samplerate = f.attrs["rate"]
                elif "rate" in f["samples"].attrs:
                    samplerate = f["samples"].attrs["rate"]
                else:
                    logger.info("   No sampling rate information in daq.h5 file - setting samplerate to default 10_000Hz.")
                    samplerate = 10_000

        return song, non_song, samplerate


@io.register_provider
class Npz(io.BaseProvider):
    KIND = "audio"
    NAME = "npz"
    SUFFIXES = [".npz"]

    def load(
        self,
        filename: Optional[str],
        song_channels: Optional[Sequence[int]] = None,
        return_nonsong_channels: bool = False,
        lazy: bool = False,
        audio_dataset: Optional[str] = None,
        **kwargs,
    ):
        if filename is None:
            filename = self.path
        if audio_dataset is None:
            audio_dataset = "data"

        with np.load(filename) as file:
            try:
                sampling_rate = float(file["samplerate"])
            except KeyError:
                try:
                    sampling_rate = float(file["samplerate_Hz"])
                except KeyError:
                    sampling_rate = None
            data = file[audio_dataset]

        data = data[:, np.newaxis] if data.ndim == 1 else data  # adds singleton dim for single-channel wavs

        if song_channels is None:  # the first 16 channels in the data are the mic recordings
            song_channels = np.arange(np.min((16, data.shape[1])))

        # split song and non-song channels
        song, non_song = split_song_and_nonsong(data, song_channels, return_nonsong_channels)

        return data, non_song, sampling_rate


@io.register_provider
class Npy(io.BaseProvider):
    KIND = "audio"
    NAME = "npy"
    SUFFIXES = [".npy"]

    def load(
        self,
        filename: Optional[str],
        song_channels: Optional[Sequence[int]] = None,
        return_nonsong_channels: bool = False,
        lazy: bool = False,
        **kwargs,
    ):
        if filename is None:
            filename = self.path

        data = np.load(filename)
        sampling_rate = None

        song, non_song = split_song_and_nonsong(data, song_channels, return_nonsong_channels)
        return song, non_song, sampling_rate


@io.register_provider
class AudioFile(io.BaseProvider):
    KIND = "audio"
    NAME = "generic audio file"
    SUFFIXES = [".wav", ".aif", ".mp3", ".flac"]

    def load(
        self,
        filename: Optional[str],
        song_channels: Optional[Sequence[int]] = None,
        return_nonsong_channels: bool = False,
        lazy: bool = False,
        **kwargs,
    ):
        if filename is None:
            filename = self.path

        if lazy:
            import dask.array as daskarray

            data = SoundFileArray(filename)
            sampling_rate = data.sampling_rate
            data = daskarray.from_array(data, chunks=(100_000, data.shape[1]), asarray=False)
            song, non_song = split_song_and_nonsong(data, song_channels, return_nonsong_channels)
            return song, non_song, sampling_rate

        import librosa

        data, sampling_rate = librosa.load(filename, sr=None, mono=False)
        data = data.T
        data = data[:, np.newaxis] if data.ndim == 1 else data  # adds singleton dim for single-channel wavs

        song, non_song = split_song_and_nonsong(data, song_channels, return_nonsong_channels)
        return song, non_song, sampling_rate


@io.register_provider
class H5file(io.BaseProvider):
    KIND = "audio"
    NAME = "h5"
    SUFFIXES = [".h5", ".hdf5", ".hdfs"]

    def load(
        self,
        filename: Optional[str],
        song_channels: Optional[Sequence[int]] = None,
        return_nonsong_channels: bool = False,
        lazy: bool = False,
        audio_dataset: Optional[str] = None,
        **kwargs,
    ):
        if filename is None:
            filename = self.path
        if audio_dataset is None:
            audio_dataset = "data"

        import h5py

        sampling_rate = None
        with h5py.File(filename, mode="r") as file:
            data = file[audio_dataset][:]
            try:
                sampling_rate = file.attrs["samplerate"]
            except:
                pass
            try:
                sampling_rate = file["samplerate"][0]
            except:
                pass

        data = data[:, np.newaxis] if data.ndim == 1 else data  # adds singleton dim for single-channel wavs
        song, non_song = split_song_and_nonsong(data, song_channels, return_nonsong_channels)
        return song, non_song, sampling_rate


@io.register_provider
class MMAPfile(io.BaseProvider):
    KIND = "audio"
    NAME = "mmap"
    SUFFIXES = [".mmap"]

    def load(
        self,
        filename: Optional[str],
        song_channels: Optional[Sequence[int]] = None,
        return_nonsong_channels: bool = False,
        lazy: bool = True,
        audio_dataset: Optional[str] = None,
        **kwargs,
    ):
        if filename is None:
            filename = self.path
        if audio_dataset is None:
            audio_dataset = "data"

        # parse filename parse, expected format: SOME_RANDOMN-NAME_{sampling_rate_Hz}_{nb_samples}_{nb_channels}_{dtype}.mmap
        trunk = os.path.splitext(os.path.basename(filename))[0]
        tokens = trunk.split("_")
        sampling_rate, nb_samples, nb_channels, dtype = float(tokens[-4]), int(tokens[-3]), int(tokens[-2]), tokens[-1]
        logger.info(f"{filename} with {nb_samples} samples, {nb_channels} channels, at {sampling_rate} Hz, type {dtype}.")

        song = np.memmap(filename, mode="r", dtype=dtype, shape=(nb_samples, nb_channels))
        non_song = None
        return song, non_song, sampling_rate
