"""xarray tools for behavioral data."""

__version__ = "0.37.4"

import os

os.environ["QT_API"] = "pyside6"

from .xarray_behave import assemble_metrics, load, save
from .api import discover, assemble, resample

__all__ = ["assemble", "assemble_metrics", "discover", "load", "resample", "save"]
