"""Tracks loaders

should return:
    x: np.array[frames, tracks(flies), body_parts, y/x]
    track_names: List
    body_parts: List[str]
    frame_numbers List[np.intp]
"""

import h5py
import flammkuchen
import numpy as np
import pandas as pd
from .. import io
from typing import Optional
import logging
import xarray as xr


class Tracks():
    def make(self, filename: Optional[str] = None):
        if filename is None:
            filename = self.path

        body_pos, track_names, body_parts, frame_numbers = self.load(filename)

        positions = xr.DataArray(data=body_pos,
                                 dims=['frame_number', 'flies', 'bodyparts', 'coords'],
                                 coords={'frame_number': frame_numbers,
                                         'bodyparts': body_parts,
                                         'flies': track_names,
                                         'coords': ['y', 'x']},
                                 attrs={'description': 'coords are "allocentric" - rel. to the full frame',
                                        'type': 'tracks',
                                        'spatial_units': 'pixels',
                                        'loader': self.NAME,
                                        'kind': self.KIND,
                                        'path': filename,})
        return positions


@io.register_provider
class Ethotracker(Tracks, io.BaseProvider):

    KIND = 'tracks'
    NAME = 'ethotracker'
    SUFFIXES = ['_tracks.h5', '_tracks_fixed.h5']

    def load(self, filename: Optional[str] = None):
        """Load tracker data"""
        if filename is None:
            filename = self.path

        with h5py.File(filename, 'r') as f:
            if 'data' in f.keys():  # in old-style or unfixed tracks, everything is in the 'data' group
                data = flammkuchen.load(filename)
                chbb = data['chambers_bounding_box'][:]
                heads = data['lines'][:, 0, :, 0, :]   # nframe, fly id, coordinates
                tails = data['lines'][:, 0, :, 1, :]   # nframe, fly id, coordinates
                box_centers = data['centers'][:, 0, :, :]   # nframe, fly id, coordinates
                background = data['background'][:]
                first_tracked_frame = data['start_frame']
                last_tracked_frame = data['frame_count']
            else:
                chbb = f['chambers_bounding_box'][:]
                heads = f['lines'][:, 0, :, 0, :]   # nframe, fly id, coordinates
                tails = f['lines'][:, 0, :, 1, :]   # nframe, fly id, coordinates
                box_centers = f['centers'][:, 0, :, :]   # nframe, fly id, coordinates
                background = f['background'][:]
                first_tracked_frame = f.attrs['start_frame']
                last_tracked_frame = f.attrs['frame_count']
        # everything to frame coords
        heads = heads[..., ::-1]
        tails = tails[..., ::-1]
        heads = heads + chbb[1][0][:]
        tails = tails + chbb[1][0][:]
        box_centers = box_centers + chbb[1][0][:]
        body_parts = ['head', 'center', 'tail']

        x = np.stack((heads, box_centers, tails), axis=2)
        x = x[first_tracked_frame:last_tracked_frame, ...]
        frame_numbers = np.arange(first_tracked_frame, last_tracked_frame, dtype=np.intp)
        track_names = np.arange(x.shape[1])
        return x, track_names, body_parts, frame_numbers


@io.register_provider
class CSV_tracks(Tracks, io.BaseProvider):

    KIND = 'tracks'
    NAME = 'generic csv'
    SUFFIXES = ['_tracks.csv']

    def load(self, filename: Optional[str] = None):
        """Load tracker data from CSV file.

        Head of the CSV file should look like this:
        track	track1	track1	track1	track1	track1	track1
        part	a	    a	    b	    b  	    c	    c
        coord	x	    y	    x	    y	    x	    y
        0       0.09    0.09    0.09    0.09    0.09    0.09
        1       0.09    0.09    0.09    0.09    0.09    0.09
        2       0.09    0.09    0.09    0.09    0.09    0.09
        ...

        First column contains the framenumber (does not need to start at 0),
        remaining columns contain coordinate date for different tracks/trackparts.

        Args:
            filename (Optional[str], optional): Path to the CSV file. Defaults to None.

        Returns:
            x: np.array([frames, tracks, parts, coords (y/x)]), track_names: List[], track_parts: List[], frame_numbers: np.array[np.intp]
        """

        if filename is None:
            filename = self.path
        logging.warning('Loading tracks from CSV.')
        df = pd.read_csv(filename, header=[0, 1, 2], index_col=0)
        track_names = df.columns.levels[df.columns.names.index('track')].to_list()
        track_parts = df.columns.levels[df.columns.names.index('part')].to_list()
        track_coord = df.columns.levels[df.columns.names.index('coord')].to_list()
        coord_order = [track_coord.index('y'), track_coord.index('x')]
        x = np.reshape(df.values, (-1, len(track_names), len(track_parts), len(track_coord)))
        x = x[..., coord_order]
        frame_numbers = df.index.to_numpy().astype(np.intp)
        return x, track_names, track_parts, frame_numbers
