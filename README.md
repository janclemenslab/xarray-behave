# xarray-behave

Install a working conda installation (see [here](https://docs.conda.io/en/latest/miniconda.html)).

Install PySide6/Qt and FFmpeg together from conda-forge so they use compatible ICU libraries. The Python dependency is `PySide6>=6.10,<7`, which is satisfied by Conda's package as well as the PyPI distribution.

```shell
conda env create -n xb -y -f https://raw.githubusercontent.com/janclemenslab/xarray-behave/refs/heads/master/env/xb.yml
```

See `demo.ipynb` for usage examples.
