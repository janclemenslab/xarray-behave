"""Load parameters for DLP movies

"""
import numpy as np
import pandas as pd
import h5py
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
class DLP_h5log(io.BaseProvider, MovieParams):

    KIND = 'movieparams'
    NAME = 'h5'
    SUFFIXES = ['_dlp.h5']

    def load(self, filename: Optional[str] = None):
        """Load tracker data.

        Data are stored as nested dict:
        - first level is the name of the DLP_runner, second level is the logged parameter
        - and one single-level key (systemtime) with the system time for each frame

        Args:
            filename (Optional[str], optional): Defaults to None.

        Returns:
            dict: dict with keys {DLPRUNNER_PARAM: values}
            int, int: first_movie_frame, last_movie_frame
        """
        if filename is None:
            filename = self.path

        movieparams = dict()
        with h5py.File(filename, mode='r') as f:
            keys = list(f.keys())
            keys.remove('systemtime')

            movieparams['systemtime'] = f['systemtime'][:]
            for key in keys:
                val = f[key]
                for k, v  in val.items():
                    movieparams[f'{key}_{k}'] = v[:]

        first_movie_frame = int(0)
        last_movie_frame = int(np.median([len(movieparams[k]) for k in movieparams.keys()]))
        movieparams = pd.DataFrame(movieparams)
        return movieparams, first_movie_frame, last_movie_frame


@io.register_provider
class DLP_params(io.BaseProvider, MovieParams):

    KIND = 'movieparams'
    NAME = 'npz'
    SUFFIXES = ['_movieparams.npz']

    def load(self, filename: Optional[str] = None):
        """Load DLP movieparams from stimulus file.

        Args:
            filename (Optional[str], optional): Defaults to None.

        Returns:
            dict: dict with keys {DLPRUNNER_PARAM: values}
            int, int: first_movie_frame, last_movie_frame
        """
        if filename is None:
            filename = self.path

        movieparams = np.load(filename)
        movieparams = pd.DataFrame(dict(movieparams))
        first_movie_frame = int(0)
        last_movie_frame = int(np.median([len(movieparams[k]) for k in movieparams.keys()]))
        return movieparams, first_movie_frame, last_movie_frame
