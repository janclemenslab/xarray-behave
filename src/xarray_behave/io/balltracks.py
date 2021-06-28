"""Load FicTrac ball tracking data

should return:
    x: pd.DataFrame[frames, (variables)]
"""
import numpy as np
import pandas as pd
from .. import io
from typing import Optional
import logging
import xarray as xr

class BallTracks():

    def make(self, filename: Optional[str] = None):
        if filename is None:
            filename = self.path

        balltracks, first_balltracked_frame, last_balltracked_frame = self.load(filename)

        xr_balltracks = xr.DataArray(data=balltracks.values,
                                    dims=['frame_number_ball', 'data_ball'],
                                    coords={'frame_number_ball': balltracks['frame counter (starts at 1)']-1,
                                            'data_ball': list(balltracks.columns)},
                                    attrs={'loader': self.NAME,
                                           'kind': self.KIND,
                                           'path': filename,})
        return xr_balltracks


@io.register_provider
class FicTrac_balltracks(io.BaseProvider, BallTracks):

    KIND = 'balltracks'
    NAME = 'fictrac csv'
    SUFFIXES = ['_fictrac.csv']

    def load(self, filename: Optional[str] = None):
        """Load tracker data"""
        if filename is None:
            filename = self.path

        balltracks = pd.read_csv(filename, index_col=0)
        first_balltracked_frame = balltracks['frame counter (starts at 1)'].values[0]
        last_balltracked_frame = balltracks['frame counter (starts at 1)'].values[-1]
        return balltracks, first_balltracked_frame, last_balltracked_frame
