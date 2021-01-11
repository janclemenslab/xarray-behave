"""Audio loader

should return:
    audio_data: np.array[time, samples]
    non_audio_data: np.array[time, samples]
    samplerate: Optional[float]
"""
# daq.h5
# wav, ....
# npz, npy
# npy_dir


"""Tracks loaders

should return:
    x: np.array[frames, flies, body_parts, x/y]
    body_parts: List[str]
    first_tracked_frame: int
    last_tracked_frame: int
    background: np.array[width, height, pixels?]
"""

import h5py
import flammkuchen
import numpy as np
import pandas as pd
from .. import io
from typing import Optional, Sequence


@io.register_provider
class Ethodrome(io.BaseProvider):

    KIND = 'audio'
    NAME = 'ethodrome h5'
    SUFFIXES = ['_daq.h5']

    def load(self, filename: Optional[str], song_channels: Optional[Sequence[int]] = None,
             return_nonsong_channels: bool = False, lazy: bool = False):
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
