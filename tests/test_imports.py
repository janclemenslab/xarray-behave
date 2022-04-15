import pytest


def test_imports():
    import xarray_behave as xb
    import xarray_behave.event_utils
    import xarray_behave.metrics
    import xarray_behave.loaders
    import xarray_behave.annot

    from xarray_behave.io import (annotations, annotations_manual, audio,
                                  balltracks, movieparams, poses, tracks)
    from xarray_behave.gui import (app, das, formbuilder, table,
                                   utils, views)
