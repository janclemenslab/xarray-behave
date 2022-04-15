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


import h5py
import numpy as np
from .. import io
from typing import Optional, Sequence


def split_song_and_nonsong(data, song_channels = None, return_nonsong_channels = False):
        song = data
        nonsong = None
        if song_channels is not None:
            song = song[:, song_channels]
            if  return_nonsong_channels:
                nonsong = np.delete(data, song_channels, axis=-1)
        return song, nonsong


@io.register_provider
class Ethodrome(io.BaseProvider):

    KIND = 'audio'
    NAME = 'ethodrome h5'
    SUFFIXES = ['_daq.h5']

    def load(self, filename: Optional[str], song_channels: Optional[Sequence[int]] = None,
             return_nonsong_channels: bool = False, lazy: bool = False,
             **kwargs):
        """[summary]

        Args:
            filename ([type]): [description]
            song_channels (List[int], optional): Sequence of integers as indices into 'samples' datasaet.
                                                Defaults to [0,..., 15].
            return_nonsong_channels (bool, optional): will return the data not in song_channels as separate array. Defaults to False
            lazy (bool, optional): If True, will load song as dask.array, which allows lazy indexing.
                                Otherwise, will load the full recording from disk (slow). Defaults to False.

        Returns:
            [type]: [description]
        """
        if filename is None:
            filename = self.path

        if song_channels is None:  # the first 16 channels in the data are the mic recordings
            song_channels = np.arange(16)

        non_song = None
        samplerate = None
        if lazy:
            f = h5py.File(filename, mode='r', rdcc_w0=0, rdcc_nbytes=100 * (1024 ** 2), rdcc_nslots=50000)
            # convert to dask array since this allows lazily evaluated indexing...
            import dask.array as daskarray
            da = daskarray.from_array(f['samples'], chunks=(10000, 1))
            nb_channels = f['samples'].shape[1]
            song_channels = song_channels[song_channels < nb_channels]
            song = da[:, song_channels]
            if return_nonsong_channels:
                non_song_channels = list(set(list(range(nb_channels))) - set(song_channels))
                non_song = da[:, non_song_channels]
        else:
            with h5py.File(filename, 'r') as f:
                nb_channels = f['samples'].shape[1]
                song_channels = song_channels[song_channels < nb_channels]
                song = f['samples'][:, song_channels]
                if return_nonsong_channels:
                    non_song_channels = list(set(list(range(nb_channels))) - set(song_channels))
                    non_song = f['samples'][:, non_song_channels]

        return song, non_song, samplerate


@io.register_provider
class Npz(io.BaseProvider):

    KIND = 'audio'
    NAME = 'npz'
    SUFFIXES = ['.npz']

    def load(self, filename: Optional[str], song_channels: Optional[Sequence[int]] = None,
             return_nonsong_channels: bool = False, lazy: bool = False,
             audio_dataset: Optional[str] = None,
             **kwargs):

        if filename is None:
            filename = self.path
        if audio_dataset is None:
            audio_dataset = 'data'

        with np.load(filename) as file:
            try:
                sampling_rate = file['samplerate']
            except KeyError:
                try:
                    sampling_rate = file['samplerate_Hz']
                except KeyError:
                    sampling_rate = None
            data = file[audio_dataset]

        data = data[:, np.newaxis] if data.ndim==1 else data  # adds singleton dim for single-channel wavs

        if song_channels is None:  # the first 16 channels in the data are the mic recordings
            song_channels = np.arange(np.min((16, data.shape[1])))

        # split song and non-song channels
        song, non_song = split_song_and_nonsong(data, song_channels, return_nonsong_channels)

        return data, non_song, sampling_rate


@io.register_provider
class Npy(io.BaseProvider):

    KIND = 'audio'
    NAME = 'npy'
    SUFFIXES = ['.npy']

    def load(self, filename: Optional[str], song_channels: Optional[Sequence[int]] = None,
             return_nonsong_channels: bool = False, lazy: bool = False,
             **kwargs):

        if filename is None:
            filename = self.path

        data = np.load(filename)
        sampling_rate = None

        song, non_song = split_song_and_nonsong(data, song_channels, return_nonsong_channels)
        return song, non_song, sampling_rate

@io.register_provider
class AudioFile(io.BaseProvider):

    KIND = 'audio'
    NAME = 'generic audio file'
    SUFFIXES = ['.wav', '.aif', '.mp3', '.flac']

    def load(self, filename: Optional[str], song_channels: Optional[Sequence[int]] = None,
             return_nonsong_channels: bool = False, lazy: bool = False,
             **kwargs):

        if filename is None:
            filename = self.path

        # import soundfile
        # data, sampling_rate = soundfile.read(filename)
        import librosa
        data, sampling_rate = librosa.load(filename, sr=None, mono=False)
        data = data.T
        data = data[:, np.newaxis] if data.ndim==1 else data  # adds singleton dim for single-channel wavs

        song, non_song = split_song_and_nonsong(data, song_channels, return_nonsong_channels)
        return song, non_song, sampling_rate


@io.register_provider
class Wav(io.BaseProvider):

    KIND = 'audio'
    NAME = 'wav'
    SUFFIXES = ['.wav']

    def load(self, filename: Optional[str], song_channels: Optional[Sequence[int]] = None,
             return_nonsong_channels: bool = False, lazy: bool = False,
             audio_dataset: Optional[str] = None):

        if filename is None:
            filename = self.path

        import scipy.io.wavfile
        sampling_rate, data = scipy.io.wavfile.read(filename)
        data = data[:, np.newaxis] if data.ndim==1 else data  # adds singleton dim for single-channel wavs

        song, non_song = split_song_and_nonsong(data, song_channels, return_nonsong_channels)
        return song, non_song, sampling_rate


@io.register_provider
class H5file(io.BaseProvider):

    KIND = 'audio'
    NAME = 'h5'
    SUFFIXES = ['.h5', '.hdf5', '.hdfs']

    def load(self, filename: Optional[str], song_channels: Optional[Sequence[int]] = None,
             return_nonsong_channels: bool = False, lazy: bool = False,
             audio_dataset: Optional[str] = None,
             **kwargs):

        if filename is None:
            filename = self.path
        if audio_dataset is None:
            audio_dataset = 'data'

        import h5py
        sampling_rate = None
        with h5py.File(filename, mode='r') as file:
            data = file[audio_dataset][:]
            try:
                sampling_rate = file.attrs['samplerate']
            except:
                pass
            try:
                sampling_rate = file['samplerate'][0]
            except:
                pass

        data = data[:, np.newaxis] if data.ndim==1 else data  # adds singleton dim for single-channel wavs
        song, non_song = split_song_and_nonsong(data, song_channels, return_nonsong_channels)
        return song, non_song, sampling_rate
