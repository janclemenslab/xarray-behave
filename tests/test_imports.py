import pytest

import numpy as np
import xarray as xr
import pandas as pd


def test_imports():
    import xarray_behave as xb
    from xarray_behave import event_utils
    import xarray_behave.annot
    import xarray_behave.gui
    import xarray_behave.io
