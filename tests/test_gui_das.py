import sys
import wave
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xarray_behave import annot
from xarray_behave.gui.app import MainWindow, _num_flies


class _ComputedSlice:
    def __init__(self, values):
        self.values = values
        self.computed = False

    def compute(self):
        self.computed = True
        return self.values


class _LazyAudio:
    def __init__(self, values):
        self.values = values
        self.last_slice = None

    def __getitem__(self, item):
        self.last_slice = _ComputedSlice(self.values[item])
        return self.last_slice


class _Dataset(SimpleNamespace):
    def __contains__(self, key):
        return hasattr(self, key)


def test_num_flies_uses_dimension_size_not_coordinate_values():
    assert _num_flies(xr.Dataset(coords={"flies": ["chamber0", "chamber1"]})) == 2
    assert _num_flies(xr.Dataset(coords={"flies": [2, 5]})) == 2
    assert _num_flies(xr.Dataset(coords={"time": [0.0]})) == 1


def test_das_gui_starts_with_single_file_actions(qtbot):
    window = MainWindow(is_das=True)
    qtbot.addWidget(window)
    labels = [action.text() for action in window.file_menu.actions()]
    assert "Open audio file" in labels
    assert "Open etho folder" in labels
    assert "Import folder as project" not in labels
    assert "Open project" not in labels
    assert "Make dataset for training" in [action.text() for action in window.das_menu.actions()]


def test_das_empty_startup_opens_annotation_workspace(qtbot, monkeypatch, tmp_path):
    from xarray_behave.gui import app as gui_app, gui_config

    manager = gui_config.GuiConfigManager(home=tmp_path)
    monkeypatch.setattr(gui_config, "GuiConfigManager", lambda *_: manager)
    monkeypatch.setattr(gui_app.QtWidgets.QApplication, "exec_", lambda self: 0)
    opened = []
    monkeypatch.setattr(gui_app.MainWindow, "from_file", classmethod(lambda cls, qt_keycode=None, **kw: opened.append(("file", kw))))
    monkeypatch.setattr(gui_app.MainWindow, "from_dir", classmethod(lambda cls, qt_keycode=None, **kw: opened.append(("folder", kw))))

    gui_app.main_das()
    window = gui_app.QtWidgets.QApplication.instance()._xarray_behave_mainwin
    qtbot.addWidget(window)

    assert isinstance(window, gui_app.PSV)
    assert window.isVisible() and not opened
    assert not window.ds.data_vars and window.tmax == 0
    assert window.slice_view.isVisible() and window.spec_view.isVisible()
    assert window.events_table.table.model().rowCount() == 0
    assert not window.transport_panel.isEnabled()
    assert not window._viewer_actions["save_annotations"].isEnabled()
    assert not window._viewer_actions["threshold_mode"].isEnabled()
    assert gui_app.QtWidgets.QApplication.activeModalWidget() is None
    actions = {action.text(): action for action in window.file_menu.actions()}
    for label in ("Open audio file", "Open etho folder"):
        assert actions[label].isEnabled()
        actions[label].trigger()
    assert opened == [("file", {"is_das": True, "window": window}), ("folder", {"is_das": True, "window": window})]


@pytest.mark.parametrize("source_kind", ["file", "folder", "zarr"])
def test_empty_workspace_loads_into_same_window(qtbot, monkeypatch, tmp_path, source_kind):
    from xarray_behave.gui import app as gui_app, gui_config

    audio = tmp_path / "clip.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 8_000, 0, "NONE", "not compressed"))
        stream.writeframes(np.zeros(8_000, dtype="<i2").tobytes())
    manager = gui_config.GuiConfigManager(home=tmp_path / "home")
    gui_app._set_config_manager(manager)
    window = gui_app.PSV(xr.Dataset(), is_das=True, config_manager=manager)
    qtbot.addWidget(window)
    window.resize(1100, 750)
    window_id = window.winId()
    geometry = window.geometry()
    menu_bar = window.menuBar()
    menus = menu_bar.findChildren(gui_app.QtWidgets.QMenu, options=gui_app.QtCore.Qt.FindDirectChildrenOnly)
    menu_labels = [action.text() for action in window.bar.actions()]
    monkeypatch.setattr(gui_app.QtWidgets.QFileDialog, "getOpenFileName", lambda *_, **__: (str(audio), ""))
    monkeypatch.setattr(gui_app.QtWidgets.QFileDialog, "getExistingDirectory", lambda **_: str(tmp_path))
    monkeypatch.setattr(gui_app.YamlDialog, "exec_", lambda self: gui_app.QtWidgets.QDialog.Accepted)

    class VideoReader:
        frame_rate = 25
        frame_width = 5
        frame_height = 4

        def __getitem__(self, index):
            return np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

    if source_kind != "file":
        ds = xr.Dataset(
            {"song_raw": (("sampletime", "channels"), np.zeros((8_000, 1)))},
            coords={"sampletime": np.arange(8_000) / 8_000},
            attrs={"target_sampling_rate_Hz": 1_000},
        )
        ds.song_raw.attrs["sampling_rate_Hz"] = 8_000
        monkeypatch.setattr(gui_app.dataset_service, "assemble_from_dir", lambda *_, **__: ds)
        monkeypatch.setattr(gui_app.dataset_service, "load_from_zarr", lambda *_, **__: ds)
        monkeypatch.setattr(gui_app.modern_video, "PyAVVideoReader", lambda _: VideoReader())
    monkeypatch.setattr(gui_app, "PSV", lambda *_, **__: pytest.fail("Opening data must reuse the empty window"))
    label = {"file": "Open audio file", "folder": "Open etho folder", "zarr": "Load dataset"}[source_kind]
    next(action for action in window.file_menu.actions() if action.text() == label).trigger()

    assert window.isVisible() and window.tmax == 8_000
    assert window.geometry() == geometry
    # Adding the OpenGL movie view can recreate Qt's native surface.
    if source_kind != "folder":
        assert window.winId() == window_id
    assert window._has_current_das_audio()
    assert window.outer_splitter.isEnabled() and window.transport_panel.isEnabled()
    assert window._viewer_actions["save_annotations"].isEnabled()
    assert window._viewer_actions["threshold_mode"].isEnabled()
    gui_app.QtCore.QCoreApplication.sendPostedEvents(None, gui_app.QtCore.QEvent.DeferredDelete)
    assert window.menuBar() is menu_bar
    assert menu_bar.findChildren(gui_app.QtWidgets.QMenu, options=gui_app.QtCore.Qt.FindDirectChildrenOnly) == menus
    assert [action.text() for action in window.bar.actions()] == menu_labels
    toolbars = window.findChildren(gui_app.QtWidgets.QToolBar)
    assert [toolbar for toolbar in toolbars if toolbar.isVisible()] == [window.annotation_toolbar]
    if source_kind == "folder":
        assert window.movie_view is not None and window.movie_panel.isVisible()


@pytest.mark.parametrize("source_kind", ["file", "folder", "zarr"])
@pytest.mark.parametrize("cancel_picker", [True, False])
def test_cancel_loading_keeps_empty_workspace(qtbot, monkeypatch, tmp_path, source_kind, cancel_picker):
    from xarray_behave.gui import app as gui_app, gui_config

    manager = gui_config.GuiConfigManager(home=tmp_path / "home")
    gui_app._set_config_manager(manager)
    window = gui_app.PSV(xr.Dataset(), is_das=True, config_manager=manager)
    qtbot.addWidget(window)
    original_panel = window.centralWidget()
    audio = tmp_path / "clip.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 8_000, 0, "NONE", "not compressed"))
        stream.writeframes(np.zeros(8_000, dtype="<i2").tobytes())
    path = "" if cancel_picker else str(tmp_path if source_kind == "folder" else audio)
    monkeypatch.setattr(gui_app.QtWidgets.QFileDialog, "getOpenFileName", lambda *_, **__: (path, ""))
    monkeypatch.setattr(gui_app.QtWidgets.QFileDialog, "getExistingDirectory", lambda **_: path)
    monkeypatch.setattr(gui_app.YamlDialog, "exec_", lambda self: gui_app.QtWidgets.QDialog.Rejected)
    label = {"file": "Open audio file", "folder": "Open etho folder", "zarr": "Load dataset"}[source_kind]

    next(action for action in window.file_menu.actions() if action.text() == label).trigger()

    assert window.isVisible() and not window.ds.data_vars
    assert window.centralWidget() is original_panel


def test_make_dataset_action_runs_builder_without_blocking_gui(qtbot, monkeypatch, tmp_path):
    from xarray_behave.gui import app as gui_app

    window = MainWindow(is_das=True)
    qtbot.addWidget(window)
    source = tmp_path / "recordings"
    source.mkdir()
    destination = tmp_path / "dataset"
    notices = []
    message_threads = []

    def accept_dialog(dialog):
        assert dialog.form.fields["store_folder"].text() == str(source) + ".npy"
        labels = dialog.form.findChildren(gui_app.QtWidgets.QLabel)
        assert any(label.text() == "Create a DAS training dataset from audio files and their annotation CSV files." for label in labels)
        assert "text" not in dialog.form.get_form_data()
        assert dialog.form.fields["data_folder"].isEnabled()
        assert dialog.form.fields["store_folder"].isEnabled()
        next(button for button in dialog.form.findChildren(gui_app.QtWidgets.QPushButton)
             if button.text() == "Select Dataset folder").click()
        return gui_app.QtWidgets.QDialog.Accepted

    builder = ModuleType("das.data.dataset_builder")
    builder.make_training_dataset = lambda *args, **kwargs: notices.append((args, kwargs)) or args[1]
    monkeypatch.setitem(sys.modules, "das.data.dataset_builder", builder)
    monkeypatch.setattr(gui_app.YamlDialog, "exec_", accept_dialog)
    folder_choices = iter((str(source) + "/", str(destination)))
    monkeypatch.setattr(gui_app.QtWidgets.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: next(folder_choices))
    app = gui_app.QtWidgets.QApplication.instance()
    monkeypatch.setattr(
        gui_app.QtWidgets.QMessageBox,
        "information",
        lambda *args: message_threads.append(gui_app.QtCore.QThread.currentThread() == app.thread()),
    )

    window.das_make()
    qtbot.waitUntil(lambda: not app._das_dataset_jobs)

    assert notices == [((str(source) + "/", str(destination)),
                        {"split_by": "samples", "validation_fraction": 0.2,
                         "test_fraction": 0.2, "seed": None})]
    assert message_threads == [True]


def test_add_das_prediction_rows_preserves_known_categories():
    window = MainWindow.__new__(MainWindow)
    window.event_times = annot.Events(categories={"pulse": "event", "sine": "segment"})

    rows = pd.DataFrame(
        [
            {"name": "pulse", "start_seconds": 0.1, "stop_seconds": 0.101},
            {"name": "sine", "start_seconds": 0.2, "stop_seconds": 0.3},
        ]
    )

    added = window._add_das_prediction_rows(rows, suffix="_proposals", time_offset_seconds=0.5)

    assert added == 2
    assert window.event_times.categories["pulse_proposals"] == "event"
    assert window.event_times.categories["sine_proposals"] == "event"
    np.testing.assert_allclose(window.event_times["pulse_proposals"][0, :2], [0.6, 0.601])
    np.testing.assert_allclose(window.event_times["sine_proposals"][0, :2], [0.7, 0.8])


def test_das_current_audio_slices_loaded_audio():
    window = MainWindow.__new__(MainWindow)
    window.fs_song = 1_000.2
    audio = np.arange(20).reshape(10, 2)
    lazy_audio = _LazyAudio(audio)
    window.ds = _Dataset(
        sampletime=SimpleNamespace(values=np.arange(10) / 1_000),
        song_raw=SimpleNamespace(data=lazy_audio),
    )

    sliced_audio, samplerate, offset = window._das_current_audio(0.002, 0.006)

    np.testing.assert_array_equal(sliced_audio, audio[2:6])
    assert lazy_audio.last_slice.computed
    assert samplerate == 1_000
    assert offset == 0.002


def test_das_current_audio_duration_uses_sample_times():
    window = MainWindow.__new__(MainWindow)
    window.ds = _Dataset(
        sampletime=SimpleNamespace(values=np.arange(10) / 1_000),
        song_raw=SimpleNamespace(data=np.zeros(10)),
    )

    assert window._das_current_audio_duration() == pytest.approx(0.01)


def test_das_annotated_regions_uses_current_event_times():
    window = MainWindow.__new__(MainWindow)
    window.event_times = annot.Events(
        {
            "pulse": np.array([[0.5, 0.5]]),
            "song": np.array([[0.2, 0.4], [0.8, 0.7]]),
        },
        categories={"pulse": "event", "song": "segment"},
    )

    assert window._das_annotated_regions() == [(0.5, 0.5), (0.2, 0.4), (0.7, 0.8)]


def test_handle_das_predictions_adds_proposals_and_refreshes_state():
    window = MainWindow.__new__(MainWindow)
    window.event_times = annot.Events(categories={"song": "segment"})
    refreshed = {"selector": False, "xy": False}
    window.update_eventtype_selector = lambda: refreshed.__setitem__("selector", True)
    window.update_xy = lambda: refreshed.__setitem__("xy", True)

    annotations = pd.DataFrame([{"name": "song", "start_seconds": 0.1, "stop_seconds": 0.2}])
    window._handle_das_predictions(annotations, time_offset_seconds=1.0)

    assert "song_proposals" in window.event_times
    np.testing.assert_allclose(window.event_times["song_proposals"][0, :2], [1.1, 1.2])
    assert refreshed == {"selector": True, "xy": True}


@pytest.mark.parametrize("bounds,fixed", [([(0.1, 0.2), (0.4, 0.7)], False), ([(0.1, 0.1)], True)])
def test_new_proposal_presets_follow_predicted_durations(qtbot, bounds, fixed):
    from xarray_behave.gui.app import PSV
    from test_gui_event_widgets import _audio_dataset

    window = PSV(_audio_dataset())
    qtbot.addWidget(window)
    annotations = pd.DataFrame([{"name": "vocal", "start_seconds": start, "stop_seconds": stop} for start, stop in bounds])

    window._handle_das_predictions(annotations, time_offset_seconds=0.0)

    preset = window.event_presets["vocal_proposals"]
    assert preset.fixed_duration == fixed
    assert preset.duration_editable == (not fixed)
    np.testing.assert_allclose(window.event_times["vocal_proposals"][:, :2], bounds)
    if not fixed:
        np.testing.assert_allclose(window._bounds_for_event_edit("vocal_proposals", 0.1, 0.2, 0.1, 0.35, "stop"), [0.1, 0.35])


def test_open_das_train_window_uses_current_audio_and_duration(monkeypatch):
    calls = {}

    class _Destroyed:
        def connect(self, callback):
            calls["destroyed_callback"] = callback

    class FakeDASConformerWindow:
        destroyed = _Destroyed()

        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def setAttribute(self, value):
            calls["attribute"] = value

        def show(self):
            calls["shown"] = True

    fake_module = ModuleType("das.gui_app")
    fake_module.DASConformerWindow = FakeDASConformerWindow
    monkeypatch.setitem(sys.modules, "das.gui_app", fake_module)

    window = MainWindow.__new__(MainWindow)
    window._has_current_das_audio = lambda: True
    window._das_current_audio = lambda start, stop: (np.zeros(10), 1_000, start)
    window._das_current_audio_duration = lambda: 0.01
    window._das_annotated_regions = lambda: [(0.2, 0.4)]
    window._handle_das_predictions = lambda annotations, time_offset_seconds: None
    window.event_times = annot.Events({"pulse": np.array([[0.2, 0.4, -1]])})

    das_window = window._open_das_window("train", use_current_audio=True)

    assert isinstance(das_window, FakeDASConformerWindow)
    assert calls["kwargs"]["initial_tab"] == "train"
    assert calls["kwargs"]["current_audio_provider"] is window._das_current_audio
    assert calls["kwargs"]["current_duration_provider"] is window._das_current_audio_duration
    assert calls["kwargs"]["annotated_region_provider"] is window._das_annotated_regions
    annotations_provider = calls["kwargs"]["current_annotations_provider"]
    assert annotations_provider()["name"].tolist() == ["pulse"]
    window.event_times.add_name("sine", times=np.array([[0.5, 0.6, -1]]))
    assert annotations_provider()["name"].tolist() == ["pulse", "sine"]
    assert calls["kwargs"]["on_predictions"] is window._handle_das_predictions
