import xarray_behave as xb
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
datename = 'localhost-20190703_164840'
root = '/Volumes/ukme04/#Common/backlight'
# datename = 'localhost-20190712_160816'
datename = 'localhost-20190708_160012'
root = '/Volumes/ukme04/#Common/chainingmic'
dataset = xb.assemble('localhost-20180716_130902', root='')
# xb.load(, dataset)