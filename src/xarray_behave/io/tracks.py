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
from typing import Optional


@io.register_provider
class Ethotracker(io.BaseProvider):

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
        # first_tracked_frame, last_tracked_frame = data['start_frame'], data['frame_count']
        x = np.stack((heads, box_centers, tails), axis=2)
        x = x[first_tracked_frame:last_tracked_frame, ...]
        return x, body_parts, first_tracked_frame, last_tracked_frame, background


@io.register_provider
class CSV_tracks(io.BaseProvider):

    KIND = 'tracks'
    NAME = 'generic csv'
    SUFFIXES = ['_tracks.csv']

    def load(self, filename: Optional[str] = None):
        """Load tracker data"""
        if filename is None:
            filename = self.path
        # #df with cols: framenumber, tracknumber, head, center, tail
        # # in frame coords
        # df = pd.read_csv(filename)

        # # reshape df to [frames, flies, parts, x/y]
        # return x, body_parts, first_tracked_frame, last_tracked_frame
        logging.warning('Loading generic tracks from CSV not implemented yet.')

