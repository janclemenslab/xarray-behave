# xarray-behave

## Installation
First, manually install dependencies:
```shell
conda install numpy scipy pandas xarray zarr h5py
pip install git+http://github.com/janclemenslab/samplestamps
```
Install with GUI (requires extra dependencies):
```shell
conda install pyqtgraph scikit-image opencv
pip install git+http://github.com/postpop/videoreader
pip install git+http://github.com/janclemenslab/xarray-behave#egg=xarray-behave[gui]
```
Install without GUI (e.g. on the cluster):
```shell
pip install git+http://github.com/janclemenslab/xarray-behave
```


## Usage
See `demo.ipynb`.

Use `python -m xarray_behave.ui datename` for starting the UI. `python -m xarray_behave.ui --help` for usage/arguments/keys.