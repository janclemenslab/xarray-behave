# xarray-behave

## Installation
Install a working conda installation with python 3.7 (see [here](https://docs.conda.io/en/latest/miniconda.html)). If you have conda already installed, make sure you have conda v4.8.4+. If not, update from an older version with `conda update conda`.

If you want to use xb with DeepSS, following the [installation instructions for DeepSS](https://janclemenslab.org/deepss/install.html).
The [DeepSS docs](https://janclemenslab.org/deepss) also contain instructions on annotating audio recordings using the GUI.

### GUI
For using with the GUI:
```shell
conda env create -f https://raw.githubusercontent.com/janclemenslab/xarray-behave/master/env/xb_gui.yml -n xb
```
The GUI can be started by activating the new environment `conda activate xb` and then typing `xb` in a terminal. See `xb --help` for usage/arguments/keys.
GUI usage is documented [here](https://janclemenslab.org/deepss/tutorials_gui/tutorials_gui.html).

### Non-GUI
If you do not want to use the GUI, e.g. if just want to create, load, save datasets:
```shell
conda env create -f https://raw.githubusercontent.com/janclemenslab/xarray-behave/master/env/xb.yml -n xb
```

See `demo.ipynb` for usage examples.
