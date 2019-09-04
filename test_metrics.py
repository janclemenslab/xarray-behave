import xarray_behave as xb
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
datename = 'localhost-20190703_164840'
root = '/Volumes/ukme04/#Common/backlight'
# datename = 'localhost-20190712_160816'
datename = 'localhost-20190708_160012'
root = '/Volumes/ukme04/#Common/chainingmic'
dataset = xb.assemble(datename, root)
%time dsm_old = xb.load('test_metrics_old.zarr', lazy=False)
%time dsm = xb.assemble_metrics(dataset)

%time print(np.nansum(np.abs(dsm.abs_features.values - dsm_old.abs_features.values)))
%time print(np.nansum(np.abs(dsm.rel_features.values - dsm_old.rel_features.values)))
