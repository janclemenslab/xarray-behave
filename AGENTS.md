# AGENTS.md

Guide for coding agents working in this repository. This file is meant to save
future agents from re-parsing the whole codebase before small changes.

## Local Workflow

- Prefer Python.
- Use the `das-conformer` conda environment for Python commands:
  `conda run -n das-conformer python -m pytest -q ...`
- If packages are needed, prefer `uv pip install ...` inside the conda env when
  possible.
- Check `git status --short` before editing. This repo may have user-owned
  uncommitted changes; do not revert or reformat unrelated files.
- Keep changes surgical. Match the existing style, even where it is older or
  inconsistent.

## Project Snapshot

- Package: `xarray-behave`
- Source layout: `src/xarray_behave`
- Public API: `xarray_behave.assemble`, `assemble_metrics`, `load`, `save`
- GUI entry points: `xarray_behave.gui.app.main`, `main_das`, and CLI via
  `defopt.run(main)`
- Internal GUI/data boundary: `src/xarray_behave/_dataset_service.py` contains
  non-Qt dataset orchestration used by the GUI.
- GUI runtime uses full `PySide6>=6.10`. Qt Multimedia is imported directly from
  `PySide6.QtMultimedia`; do not switch transport playback back to `qtpy`
  multimedia wrappers or split `PySide6-Essentials/Addons` pins without testing
  real audio output.
- Build backend: `flit_core` in `pyproject.toml`.
- Version lives in `src/xarray_behave/__init__.py`.
- `__init__.py` also sets `QT_API=pyside6`.

## What The Package Does

`xarray-behave` assembles behavioral recordings into self-describing
`xarray.Dataset` objects. Inputs can include audio/DAQ data, camera timestamps,
video tracking, pose tracking, ball tracking, DLP movie parameters, and automatic
or manual annotations. The main assembly path aligns everything onto either a
target sample grid or original frame times, then stores data arrays with time
coordinates and metadata.

The main model is:

- `time`: seconds on the analysis grid.
- `sampletime`: seconds on the raw audio/sample grid.
- `nearest_frame`: video frame nearest to each analysis time.
- `song_raw`: raw audio, dims `sampletime, channels`.
- `non_song_raw`: non-song channels, dims `sampletime, no_song_channels`.
- `song_events`: binary event traces, dims `time, event_types`, with legacy
  `event_categories` normalized to `"event"`.
- `event_names` and `event_times`: table-like annotation representation.
- `body_positions`: tracked body positions, dims
  `time, flies, bodyparts, coords`.
- `pose_positions`: egocentric pose positions, dims
  `time, flies, poseparts, coords`.
- `pose_positions_allo`: allocentric pose positions in frame coordinates.
- `balltracks` and `movieparams`: optional aligned auxiliary traces.

Spatial coordinates use `coords=["y", "x"]`. Many loaders ingest x/y formats
and swap into y/x before returning arrays.

## Core Modules

- `src/xarray_behave/xarray_behave.py`
  - `assemble(...)` is the central pipeline. It infers paths from
    `root/dat_path/res_path/datename`, loads timestamps, audio, tracks, poses,
    annotations, and optional auxiliary data, aligns arrays to the target grid,
    builds an `xarray.Dataset`, converts positions to millimeters when
    `pixel_size_mm` is available, and applies fly identity swaps when requested.
  - `target_sampling_rate=0` or `None` disables resampling to a uniform target
    grid and uses frame times.
  - `align_time(...)` interpolates frame-indexed arrays onto target samples.
  - `assemble_metrics(...)` computes absolute and relative behavioral features
    from poses or body tracking.
  - `convert_spatial_units(...)` mutates dataset arrays in place.
  - `save(...)` writes a zipped zarr store using `zarr.ZipStore`; it removes
    dict-valued `song_events.attrs["event_times"]` before saving because xarray
    cannot serialize that attr.
  - `load(...)` opens zipped zarr stores, optionally copying to a temp store and
    normalizing byte-string coordinates.

- `src/xarray_behave/annot.py`
  - `Events` is the annotation container used across assembly and GUI code.
  - Internally each event name maps to an `N x 3` array:
    `[start_seconds, stop_seconds, channel]`.
  - Everything is an event. Instantaneous events have `start == stop`; events
    with duration use different start/stop values.
  - `categories` is legacy compatibility metadata. Inputs may still contain
    `"segment"`, but code should normalize all values to `"event"` and must not
    use categories to decide behavior.
  - Empty names are preserved through DataFrame/list conversion with NaN
    conventions: empty event names use `[nan, nan]`.
  - Common methods: `from_df`, `from_lists`, `from_dataset`, `to_df`,
    `to_dataset`, `add_name`, `add_time`, `delete_time`, `change_name`,
    `filter_range`, `delete_range`, `find_next`, `find_prev`.

- `src/xarray_behave/event_utils.py`
  - Converts between binary `song_events` traces and event-time dictionaries.
  - Connected runs in binary traces become event start/stop rows. Single-sample
    runs become point events with `start == stop`.
  - Category inference helpers are legacy shims and return `"event"` for every
    event type.
  - `eventtimes_to_traces(...)` updates existing event names only.
  - `update_traces(...)` rebuilds traces and can add new events.

- `src/xarray_behave/metrics.py`
  - NumPy/SciPy helpers for smoothing, derivatives, distance, orientation,
    velocity, acceleration, angular velocity/acceleration, vector projection,
    and wing/internal angles.

- `src/xarray_behave/loaders.py`
  - Timestamp loading helpers, channel merging, fly swap utilities, and legacy
    key normalization.
  - `swap_flies(...)` mutates selected dataset arrays from each swap time onward.

- `src/xarray_behave/_dataset_service.py`
  - Thin internal service layer between `gui.app` and dataset logic.
  - Wraps GUI-facing assembly from media/project recordings and files/directories,
    zarr loading, song filtering, legacy event-category normalization, event-time
    extraction, display unit conversion, and save preparation.
  - Must stay free of Qt imports. Keep helpers small and avoid extra validation;
    this module exists to separate concerns, not to change behavior.

- `src/xarray_behave/gui/project.py`
  - Stores project recordings, their media paths, embedded annotations, and
    project-specific GUI settings in `.xbp.yaml` files.
  - Annotation types and presets belong to the project settings, not the global
    GUI config. Project saves must not rewrite annotation sidecar files.

## IO Provider System

`src/xarray_behave/io/__init__.py` defines `BaseProvider`,
`register_provider`, and `get_loader(kind, basename, ...)`.

Provider modules register themselves at import time. Keep the final import line
in `io/__init__.py` updated when adding a new provider module, otherwise
`get_loader` will not know about it.

To add a loader, follow the existing pattern:

1. Subclass `io.BaseProvider`.
2. Set `KIND`, `NAME`, and `SUFFIXES`.
3. Decorate with `@io.register_provider`.
4. Implement `load(...)`; implement `make(...)` when the provider returns an
   `xarray.DataArray`.
5. Return arrays in the expected dims and y/x coordinate order.

Registered provider families:

- Audio:
  - `Ethodrome`: `_daq.h5`
  - `Npz`: `.npz`
  - `Npy`: `.npy`
  - `AudioFile`: `.wav`, `.aif`, `.mp3`, `.flac`
  - `H5file`: `.h5`, `.hdf5`, `.hdfs`
  - `MMAPfile`: `.mmap`
- Timestamps:
  - `CamStamps`: `_timestamps.h5`
  - `DaqStamps`: `.h5`
  - `CsvStamps`: `_timestamps.csv`
- Tracks:
  - `Ethotracker`: `_tracks.h5`, `_tracks_fixed.h5`
  - `CSV_tracks`: `_tracks.csv`
- Poses:
  - `Leap`: `_poses.h5`, `_poses_leap.h5`
  - `DeepPoseKit`: `_poses_dpk.zarr`
  - `Sleap`: `_poses_sleap.h5`, `_sleap.h5`, `_sleap.h5.slp`
- Annotations:
  - `DAS`: `_song.h5`, `_vibration.h5`, `_pulse.h5`, `_sine.h5`,
    `_dss.h5`, `_das.h5`
  - `FlySongSegmenter`: `_song.mat`
  - manual CSV/zarr/mat loaders in `annotations_manual.py`
  - definitions loader: `_definitions.csv`
  - Legacy annotation files may contain event/segment labels. Loaders should
    preserve start/stop timing but normalize all categories to `"event"`.
- Auxiliary:
  - FicTrac ball tracks: `_fictrac.csv`
  - DLP movie params: `_dlp.h5`, `_movieparams.npz`

## Sample/Time Conversion

`src/xarray_behave/io/samplestamps` contains conversion utilities:

- `SampStamp` builds interpolators between samples, frames, and timestamps.
- `SimpleStamp` is used for audio-only data with a regular sampling rate.
- `utils.monotonize(...)` truncates at the first monotonicity violation; tests
  cover strict and non-strict increasing/decreasing behavior.

Assembly depends heavily on these converters, so changes here can affect
alignment throughout the package.

## GUI Architecture

- `src/xarray_behave/gui/app.py`
  - `MainWindow` handles initial menus, dialogs, DAS integration, saving UI,
    annotation editing, and project recording management.
  - Dataset assembly/loading/filtering/event extraction/save prep is delegated to
    `xarray_behave._dataset_service`; keep new non-Qt dataset orchestration
    there instead of adding it directly to GUI methods.
  - `PSV` is the main viewer/controller for synchronized waveform, spectrogram,
    event timeline/table widgets, transport controls, and optional movie view.
  - In the audio-focused layout the center stack is waveform, spectrogram,
    event timeline, then event table, with initial splitter weights `1:4:1:2`.
    Event presets live in the left preset panel. The current audio channel
    selector overlays the waveform or spectrogram; there is intentionally no top
    event/channel selector row.
  - Switching a project recording keeps the window, project list, and preset
    panel alive. Refresh only recording-dependent views and rebuild the preset
    panel only when annotation types actually differ.
  - Event table/timeline edits update the shared `annot.Events` instance.
    Keep table-row selection, timeline selection, and audio view synchronization
    in `PSV` rather than duplicating annotation mutation in widget classes.
  - Transport playback is QMediaPlayer-backed when an audio source path is
    available. Playback should run continuously; when the visible window flips,
    update the displayed range without calling `setPosition()`/`play()` again.
  - DAS helpers provide current-audio slices and prediction callbacks.
    Focused tests live in `tests/test_gui_das.py`.
- `src/xarray_behave/gui/event_widgets.py`
  - Qt/PyQtGraph widgets for the event table, waveform pane, event timeline, and
    preset sidebar. `EventsTableWidget` owns table UI only; `WaveformPane` owns
    the audio waveform display/playhead/annotation overlay; `EventTimelineWidget`
    owns event-bar display and can optionally include an embedded waveform for
    standalone use. They emit signals and do not own dataset save/load behavior.
  - Table cells allow event-name dropdown edits and start/stop second edits.
    Selection can be linked to the audio view; changing the audio range selects
    overlapping rows.
  - Timeline bars support selecting, creating, moving, and edge-resizing events.
  - `EventTypePreset`, `EventPresetPanel`, and `EventTypePresetDialog` provide
    the audio-only preset sidebar. Presets are GUI metadata over event names:
    name, fixed-duration mode, default duration, editability after creation, and
    color. `ChannelSelectorPanel` hosts the compact channel selector above the
    preset panel. Presets must not replace the `Events` start/stop/channel
    storage model.
  - `WaveformPane` draws continuous traces for normal/default zooms and switches
    to min/max overview pairs only for windows of at least 4 seconds. Keep this
    threshold behavior in mind when changing waveform performance or appearance.
- `src/xarray_behave/gui/views.py`
  - PyQtGraph view/items for traces, spectrograms, annotations, draggable body
    and pose points, and movie display.
- `src/xarray_behave/gui/modern_video.py`
  - PyAV-backed reader used by the GUI movie path. It exposes the small reader
    protocol expected by `views.MovieView` (`read`, `__getitem__`, frame shape,
    frame count, and rate metadata).
- `src/xarray_behave/gui/media_dialog.py`
  - Collects named audio/video sources and optional timestamp, offset, rate, and
    dataset overrides before `dataset_service.assemble_from_media(...)` loads them.
- `src/xarray_behave/gui/formbuilder.py`
  - YAML-driven Qt form builder. Forms live under `src/xarray_behave/gui/forms`.
- `src/xarray_behave/gui/utils.py`
  - Color palettes, fast plotting, image widget,
    nearest-index helpers, worker/thread helpers, and checkable combo box.
- `src/xarray_behave/gui/style_profile.py`
  - Shared dark Qt stylesheet and timeline colors adapted from `xb_gui`.
    Prefer using these constants for new GUI widgets instead of introducing a
    separate look.
GUI tests are written to avoid opening real windows where possible by using
`__new__`, monkeypatching imported modules, and faking datasets.

## Tests And Verification

Use the requested conda env:

```shell
conda run -n das-conformer python -m pytest -q
```

Useful narrower checks:

```shell
conda run -n das-conformer python -m pytest -q tests/test_annot.py tests/test_sampstamps.py
conda run -n das-conformer python -m pytest -q tests/test_event_utils.py tests/test_gui_event_widgets.py
conda run -n das-conformer python -m pytest -q tests/test_dataset_service.py tests/test_gui_das.py tests/test_imports.py
conda run -n das-conformer python -m pytest -q tests/test_gui_utils.py tests/test_gui_das.py
conda run -n das-conformer python -m pytest -q tests/test_imports.py tests/test_io.py
```

Assembly tests in `tests/test_assemble.py` and
`tests/test_assemble_metrics.py` use fixture data paths under `tests/data`.
They are closer to smoke tests than strict structure assertions.

The GitHub Actions workflow only runs import/IO/samplestamp smoke tests after
creating an env from `env/xb.yml`.

## Common Change Patterns

- Adding a new file format: implement a provider in `src/xarray_behave/io`,
  register it, import the module from `io/__init__.py`, and add a focused test
  for suffix matching and return shape.
- Changing annotations: update `annot.Events` and `event_utils` tests first.
  Preserve the `N x 3` internal layout and NaN empty-name conventions unless the
  task explicitly asks for a migration. Do not reintroduce event-vs-segment
  behavior; duration belongs in start/stop seconds.
- Changing assembly behavior: add or update an assembly fixture test where
  possible, but keep it narrow because full assembly can be slow.
- Changing GUI dataset creation/loading/saving behavior: update
  `_dataset_service.py` first and add/adjust a focused test in
  `tests/test_dataset_service.py`. Keep `gui.app` responsible for Qt dialogs and
  viewer construction.
- Changing project persistence or recording switching: update `gui/project.py`,
  the focused `tests/test_gui_project.py` / `tests/test_gui_event_widgets.py`,
  and keep project-wide panels intact during a recording change.
- Changing event table/timeline behavior: update `src/xarray_behave/gui/event_widgets.py`
  and `tests/test_gui_event_widgets.py`. Keep widgets signal-driven and keep
  dataset mutation in `PSV`.
- Changing waveform, transport, side-panel, or audio-channel UI behavior usually
  touches `gui.app`, `gui.event_widgets`, `gui.style_profile`, and
  `tests/test_gui_event_widgets.py` together. Smoke launch the audio GUI after
  such changes:
  `conda run -n das-conformer python -m xarray_behave.gui.app scratch/dat/Dmel_male.wav --skip-dialog`.
- Changing GUI DAS behavior: update `tests/test_gui_das.py`; the tests
  intentionally monkeypatch the external `das` module.
- Changing color maps or GUI helper behavior: update `tests/test_gui_utils.py`.
- Changing saved dataset structure: check `save`, `load`, GUI save logic, and
  manual annotation loaders together.

## Pitfalls

- Many functions mutate arrays or datasets in place (`Events`, `swap_flies`,
  `convert_spatial_units`, GUI annotation edits). Avoid assuming pure returns.
- `get_loader(...)` returns either one loader or a list depending on
  `stop_after_match`.
- `eventtimes_to_traces(...)` does not add new event names; use
  `update_traces(...)` when new names must appear.
- Category values are legacy. Treat any `"segment"` strings from old files or
  tests as input compatibility only, and normalize them to `"event"` before GUI
  use or saving.
- Event rows are identified in the GUI table/timeline by event name plus row
  index. After deleting or moving rows, refresh widgets from `self.event_times`
  before reusing old row ids.
- Fixed-duration presets are enforced in `PSV`, not in `Events`. For locked
  fixed-duration event types, table edits and timeline edge drags should move
  the whole event while preserving the preset duration.
- The active event type is stored in `PSV._current_event_name` and mirrored by
  the preset side panel. Do not reintroduce a hidden or visible top combo box as
  the source of truth.
- The current audio channel selector lives in `ChannelSelectorPanel.channel_combo`;
  `PSV.cb2` is a compatibility alias used by existing channel-selection methods.
- Project recordings can have different audio data and annotations, but share
  project presets and settings. Do not recreate the whole viewer just to switch
  recordings.
- The linked table/audio behavior is optional. Respect
  `EventsTableWidget.sync_enabled` before changing the audio range in response
  to table selection.
- The waveform panel and event timeline both use seconds. `WaveformPane` is the
  top audio panel; do not replace it with the legacy `views.TraceView` unless
  restoring all playhead, annotation overlay, click, and threshold behavior.
- Some loader code catches broad exceptions and logs instead of failing. When
  debugging assembly, inspect logs and the final dataset variables.
- `pixel_size_mm` can be `nan`; conversion should be skipped when there is no
  valid scale.
- `song_raw` may be a NumPy array, dask array, h5-backed array, or memmap-like
  object depending on loader and lazy settings.
- GUI code uses both sample indices and seconds. Check whether a variable is an
  index (`time0`, `time1`) or seconds (`t0`, `t1`, `mouseT`) before changing it.
- The event table and timeline use seconds. Convert to sample indices only at
  the `PSV` boundary when changing the audio view.
- `_dataset_service.py` intentionally preserves GUI-era behavior, including small
  quirks in form-data mapping. Treat it as orchestration glue unless explicitly
  asked to change behavior.
- Keep generated or temporary files out of the repo. Some tests/examples write
  `test.txt`; do not treat it as source.


## Inspiration for extensions
- https://www.pamguard.org/olhelp/detectors/whistleMoanHelp/docs/whistleMoan_Overview.html
- https://www.ravensoundsoftware.com/article-categories/raven-workbench/
