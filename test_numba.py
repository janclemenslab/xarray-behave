import xarray_behave as xb
import xarray_behave.metrics as mt
import numpy as np
import time

datename = 'localhost-20181120_144618'
root = ''
ds = xb.assemble(datename, root, include_song=False)
t0 = time.time()
dsm = xb.assemble_metrics(ds)
print(time.time() - t0)
