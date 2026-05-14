import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd

from xarray_behave import annot
from xarray_behave.gui.app import MainWindow


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
    assert window.event_times.categories["sine_proposals"] == "segment"
    assert window.event_times["pulse_proposals"][0, 0] == window.event_times["pulse_proposals"][0, 1]
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


def test_open_daws_window_uses_whisper_gui_and_current_audio(monkeypatch):
    calls = {}

    class _Destroyed:
        def connect(self, callback):
            calls["destroyed_callback"] = callback

    class FakeDASWhisperWindow:
        destroyed = _Destroyed()

        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def setAttribute(self, value):
            calls["attribute"] = value

        def show(self):
            calls["shown"] = True

    fake_module = ModuleType("das_whisper.gui_app")
    fake_module.DASWhisperWindow = FakeDASWhisperWindow
    monkeypatch.setitem(sys.modules, "das_whisper.gui_app", fake_module)

    window = MainWindow.__new__(MainWindow)
    window._has_current_das_audio = lambda: True
    window._das_current_audio = lambda start, stop: (np.zeros(10), 1_000, start)
    window._handle_das_predictions = lambda annotations, time_offset_seconds: None

    daws_window = window._open_daws_window("predict", use_current_audio=True)

    assert isinstance(daws_window, FakeDASWhisperWindow)
    assert calls["kwargs"]["initial_tab"] == "predict"
    assert calls["kwargs"]["current_audio_provider"] is window._das_current_audio
    assert calls["kwargs"]["on_predictions"] is window._handle_das_predictions
    assert calls["shown"] is True
    assert window._daws_windows == [daws_window]
