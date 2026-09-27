# xarray-behave

Install a working conda installation (see [here](https://docs.conda.io/en/latest/miniconda.html)).

```shell
conda env create -n xb -y -f https://raw.githubusercontent.com/janclemenslab/xarray-behave/refs/heads/master/env/xb.yml
```

See `demo.ipynb` for usage examples.

## GUI configuration

The GUI saves reusable state to `~/.das.yaml` when it closes cleanly. This
includes the window and panel layout, display and annotation options, audio and
threshold settings, event-type presets, and the last reusable load-dialog
choices. Source-specific values such as file paths, HDF5 dataset names, inferred
sample rates, and pixel calibration are deliberately not saved.

Configuration is applied in this order, with later values taking precedence:

1. Built-in defaults.
2. `~/.das.yaml`.
3. `.das.yaml` in a directory source, or beside a file or `.zarr` source.
4. A file supplied with `--config`.
5. Explicit command-line options and edits made in the load dialog.

For example:

```shell
python -m xarray_behave.gui.app recording.wav --config lab-profile.yaml
```

Use **File > Save Configuration As...** to create a local or shared profile.
Configuration files are versioned YAML mappings. A minimal profile is:

```yaml
version: 1
window:
  panels:
    sidebar: true
    timeline: true
    event_table: true
viewer:
  spectrogram:
    colormap: magma
    compression: 3
  audio:
    waveform_all: true
    playback_all: false
event_types:
  - name: pulse
    fixed_duration: true
    duration_seconds: 0.01
    duration_editable: false
    color_hex: "#ffd166"
    visible: true
    editable: true
```
