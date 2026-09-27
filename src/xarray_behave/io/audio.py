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
