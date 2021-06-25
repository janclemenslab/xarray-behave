"""Load parameters for DLP movies

"""
import numpy as np
import pandas as pd
from .. import io
from typing import Optional
import logging
import xarray as xr

class MovieParams():

    def make(self, filename: Optional[str] = None):
        if filename is None:
            filename = self.path

        movieparams, first_movie_frame, last_movie_frame = self.load(filename)

        xr_balltracks = xr.DataArray(data=movieparams.values,
                                    dims=['frame_number_movie', 'params_movie'],
                                    coords={'frame_number_movie': np.arange(first_movie_frame, last_movie_frame, dtype=np.intp),
                                            'params_movie': list(movieparams.columns)},
                                    attrs={'loader': self.NAME,
                                           'kind': self.KIND,
                                           'path': filename,})
        return xr_balltracks


@io.register_provider
class DLP_params(io.BaseProvider, MovieParams):

    KIND = 'movieparams'
    NAME = 'npz'
    SUFFIXES = ['_movieparams.npz']

    def load(self, filename: Optional[str] = None):
        """Load tracker data"""
        if filename is None:
            filename = self.path

        movieparams = np.load(filename)
        movieparams = pd.DataFrame(dict(movieparams))
        first_movie_frame = 0
        last_movie_frame = np.median([len(movieparams[k]) for k in movieparams.keys()])
        return movieparams, first_movie_frame, last_movie_frame
