# xarray-behave

## Installation
First, manually install dependencies:
```shell
conda install pyside2 numpy scipy pandas xarray zarr h5py dask netCDF4 bottleneck toolz pytables
conda install pysoundfile -c conda-forge
pip install git+http://github.com/janclemenslab/samplestamps
pip install flammkuchen
```
Install with GUI (requires extra dependencies):
```shell
conda install pyqtgraph scikit-image opencv
pip install defopt
pip install pyvideoreader --no-deps
pip install git+http://github.com/janclemenslab/xarray-behave#egg=xarray-behave[gui]
```
Install without GUI (e.g. on the cluster):
```shell
pip install git+http://github.com/janclemenslab/xarray-behave
```

For playback of sound you need a working installation of [pysimpleaudio](https://simpleaudio.readthedocs.io). See the [docs](https://simpleaudio.readthedocs.io/en/stable/installation.html) if you have trouble installing.

## Usage
See `demo.ipynb`.

Use `python -m xarray_behave.ui datename` for starting the UI. `python -m xarray_behave.ui --help` for usage/arguments/keys.
