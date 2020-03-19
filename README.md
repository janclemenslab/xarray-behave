# xarray-behave

## Installation
Install a working conda installation with python 3.7 (see [here](https://docs.conda.io/en/latest/miniconda.html)).

Then in a terminal run:
```shell
conda create -n xb python=3.7 -y
conda activate xb
conda install pyqtgraph python-sounddevice -c conda-forge -y
pip install git+http://github.com/janclemenslab/xarray-behave#egg=xarray-behave[gui]
```

## Old:
First, manually install dependencies:
```shell
conda install numpy scipy pandas xarray zarr h5py dask netCDF4 bottleneck toolz pytables
conda install pysoundfile -c conda-forge
pip install simpleaudio
pip install flammkuchen
pip install samplestamps
```

Install with GUI (requires extra dependencies):
```shell
conda install pyqtgraph scikit-image opencv
conda install pysoundfile pyside2 -c conda-forge
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

Use `xb datename` for starting the UI. `xb --help` for usage/arguments/keys.
