# xarray-behave

## Installation
Install a working conda installation with python 3.7 (see [here](https://docs.conda.io/en/latest/miniconda.html)). If you have conda already installed, make sure you have conda v4.8.4+, You can update conda from an older version with `conda update conda`.

If you want to use xb with DAS, follow the [installation instructions for DAS](https://janclemenslab.org/das/install.html).
The [DAS docs](https://janclemenslab.org/das) also contain instructions on annotating audio recordings using the GUI.

### GUI
For using with the GUI:
```shell
conda create -c ncb -c conda-forge -n xb python=3.8 xarray-behave
```
The GUI can be started by activating the new environment `conda activate xb` and then typing `xb` in a terminal. See `xb --help` for usage/arguments/keys.
GUI usage is documented [here](https://janclemenslab.org/das/tutorials_gui/tutorials_gui.html).

### Non-GUI
If you do not want to use the GUI, e.g. if just want to create, load, save datasets:
```shell
conda create -c ncb -c conda-forge -n xb python=3.8 xarray-behave-nogui
```

See `demo.ipynb` for usage examples.
