# xarray-behave

## Installation
Install a working conda installation with python 3.7 (see [here](https://docs.conda.io/en/latest/miniconda.html)).

### Non-GUI
If you do not want to use the GUI, e.g. if just want to create, load, save datasets:
```shell
conda create -n xb python=3.7 -y
conda activate xb
conda install zarr
python -m pip install xarray-behave
```
See `demo.ipynb` for usage examples.

### GUI
For using the GUI
```shell
conda create -n xb_gui python=3.7 -y
conda activate xb_gui
conda install pyside2 pyqtgraph=0.11.0rc0 zarr python-sounddevice zarr -c conda-forge -y
python -m pip install xarray-behave[gui]
```
The gui can be started by typing `xb` in a terminal. See `xb --help` for usage/arguments/keys.
