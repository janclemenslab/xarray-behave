import numpy as np

import xarray_behave  # noqa: F401 - sets QT_API before qtpy imports
from qtpy import QtCore, QtWidgets
from xarray_behave.annot import Events
from xarray_behave.gui.app import PSV
from xarray_behave.gui.event_widgets import (
    EventBarsView,
    EventPresetPanel,
    EventRecord,
    EventTimelineWidget,
    EventTypePreset,
    EventsTableWidget,
    records_from_events,
)
from xarray_behave.gui.table import Table


def _app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_records_from_events_preserves_intervals_and_channels():
    events = Events(
        {
            "pulse": np.array([[0.1, 0.1, 2]]),
            "song": np.array([[0.2, 0.4, -1]]),
        }
    )

    records = records_from_events(events)

    assert [record.name for record in records] == ["pulse", "song"]
    assert records[0].duration_seconds == 0
    assert records[0].channel == 2
    assert records[1].duration_seconds == 0.2


def test_events_table_selects_overlapping_visible_range():
    _app()
    events = Events(
        {
            "early": np.array([[0.1, 0.2, -1]]),
            "visible": np.array([[1.0, 1.4, -1]]),
            "late": np.array([[3.0, 3.1, -1]]),
        }
    )
    widget = EventsTableWidget()
    widget.set_events(events)

    widget.select_overlapping_range(0.9, 1.1)

    selected = widget.selected_records()
    assert [record.name for record in selected] == ["visible"]


def test_events_table_can_select_multiple_overlapping_rows():
    _app()
    events = Events(
        {
            "a": np.array([[1.0, 2.0, -1]]),
            "b": np.array([[1.5, 1.6, -1]]),
            "c": np.array([[4.0, 4.1, -1]]),
        }
    )
    widget = EventsTableWidget()
    widget.set_events(events)

    widget.select_overlapping_range(1.25, 1.75)

    assert sorted(record.name for record in widget.selected_records()) == ["a", "b"]


def test_events_table_defaults_to_start_time_sort():
    _app()
    events = Events(
        {
            "late": np.array([[3.0, 3.1, -1]]),
            "early": np.array([[0.1, 0.2, -1]]),
        }
    )
    widget = EventsTableWidget()
    widget.set_events(events)

    assert widget.table.item(0, widget._COL_START).text() == "0.100000"


def test_annotation_name_editor_is_name_only():
    _app()
    dialog = Table([["pulse"], ["sine"]])

    assert dialog.table.columnCount() == 1
    assert dialog.get_table_data()[0][0][0] == "pulse"


def test_event_bars_drag_mode_resizes_interval_edges():
    _app()
    widget = EventBarsView()
    widget.setXRange(0, 10, padding=0)
    record = EventRecord("song\x1f0", "song", 0, 1.0, 3.0, -1)

    assert widget._drag_mode(record, 1.01) == "resize_start"
    assert widget._drag_mode(record, 2.0) == "move"
    assert widget._drag_mode(record, 2.99) == "resize_stop"


def test_event_bars_drag_mode_moves_locked_duration_events():
    _app()
    widget = EventBarsView()
    widget.setXRange(0, 10, padding=0)
    record = EventRecord("song\x1f0", "song", 0, 1.0, 3.0, -1)
    widget._locked_duration_ids = {record.id}

    assert widget._drag_mode(record, 1.01) == "move"
    assert widget._drag_mode(record, 2.99) == "move"


def test_event_timeline_can_hide_embedded_waveform():
    _app()
    widget = EventTimelineWidget(show_waveform=False)

    widget.set_waveform(np.array([1.0, 2.0]), np.array([0.0, 1.0]))

    assert widget.waveform is None
    assert widget.splitter is None
    assert np.allclose(widget.events.viewRange()[0], [1.0, 2.0])


def test_preset_panel_emits_selection_and_formats_fixed_duration():
    _app()
    panel = EventPresetPanel()
    selected = []
    panel.selection_changed.connect(selected.append)
    panel.set_presets([EventTypePreset("pulse", fixed_duration=True, duration_seconds=0.0)], selected_name="pulse")

    assert panel.current_name() == "pulse"
    assert "fixed 0s" in panel.list_widget.item(0).text()


def test_fixed_duration_creation_uses_click_as_onset():
    window = PSV.__new__(PSV)
    window.tmax = 10_000
    window.fs_song = 1_000
    window.event_presets = {
        "pulse": EventTypePreset("pulse", fixed_duration=True, duration_seconds=0.25, duration_editable=False)
    }

    assert window._bounds_for_event_creation("pulse", 1.0, 4.0) == (1.0, 1.25)


def test_playhead_starts_at_boundary_and_scrolls_to_last_sample():
    class App:
        def processEvents(self):
            pass

    window = PSV.__new__(PSV)
    window.tmin = 0
    window.tmax = 1_000
    window.fs_song = 1_000
    window.fs_other = 1_000
    window._span = 200
    window._t0 = 0
    window.vr = None
    window.app = App()
    window.update_xy = lambda: None
    window.update_frame = lambda: None

    assert window.time0 == 0
    assert window.time1 == 200

    window.t0 = window.tmax

    assert window.t0 == 999
    assert window.time0 == 800
    assert window.time1 == 1_000


def test_start_playback_starts_at_visible_window_beginning():
    class Timer:
        def __init__(self):
            self.started = False

        def start(self):
            self.started = True

    class Clock:
        def __init__(self):
            self.restarted = False

        def restart(self):
            self.restarted = True

    class AudioPlayer:
        def __init__(self):
            self.positions = []
            self.played = False

        def setPosition(self, value):
            self.positions.append(value)

        def play(self):
            self.played = True

    class App:
        def processEvents(self):
            pass

    window = PSV.__new__(PSV)
    window.tmin = 0
    window.tmax = 1_000
    window._t0 = 500
    window.fs_song = 1_000
    window.fs_other = 1_000
    window._span = 200
    window.STOP = True
    window._is_playing = False
    window._playback_timer = Timer()
    window._playback_clock = Clock()
    window._audio_player = AudioPlayer()
    window._playback_window_start = None
    window._playback_window_stop = None
    window.vr = None
    window.app = App()
    window.update_xy = lambda: None
    window.update_frame = lambda: None
    states = []
    window._set_play_button_state = lambda *, playing: states.append(playing)

    window._start_playback()

    assert window.t0 == 400
    assert window.time0 == 400
    assert window.time1 == 600
    assert window._playback_window_start == 400
    assert window._playback_window_stop == 600
    assert window._is_playing is True
    assert window.STOP is False
    assert window._playback_timer.started is True
    assert window._playback_clock.restarted is True
    assert window._audio_player.positions == [400]
    assert window._audio_player.played is True
    assert states == [True]


def test_playback_tick_flips_to_next_window_at_visible_end():
    class App:
        def processEvents(self):
            pass

    class AudioPlayer:
        def __init__(self):
            self._position = 601
            self.positions = []
            self.play_count = 0

        def position(self):
            return self._position

        def setPosition(self, value):
            self.positions.append(value)
            self._position = value

        def play(self):
            self.play_count += 1

    window = PSV.__new__(PSV)
    window.tmin = 0
    window.tmax = 1_000
    window._t0 = 500
    window.fs_song = 1_000
    window.fs_other = 1_000
    window._span = 200
    window.STOP = False
    window._is_playing = True
    window._playback_anchor_sample = 400
    window._playback_window_start = 400
    window._playback_window_stop = 600
    window._audio_player = AudioPlayer()
    window._playback_clock = type("Clock", (), {"restart": lambda self: None})()
    window._playback_timer = type("Timer", (), {"start": lambda self: None})()
    window.vr = None
    window.app = App()
    window.update_xy = lambda: None
    window.update_frame = lambda: None
    window._set_play_button_state = lambda *, playing: None

    window._on_playback_tick()

    assert window._playback_window_start == 600
    assert window._playback_window_stop == 800
    assert window.t0 == 601
    assert window.time0 == 600
    assert window.time1 == 800
    assert window._audio_player.positions == []
    assert window._audio_player.play_count == 0


def test_navigation_seek_clears_playback_window_and_recenters():
    class App:
        def processEvents(self):
            pass

    window = PSV.__new__(PSV)
    window.tmin = 0
    window.tmax = 1_000
    window._t0 = 400
    window.fs_song = 1_000
    window.fs_other = 1_000
    window._span = 200
    window._playback_window_start = 400
    window._playback_window_stop = 600
    window._is_playing = False
    window._playback_clock = type("Clock", (), {"restart": lambda self: None})()
    window._audio_player = None
    window.vr = None
    window.app = App()
    window.update_xy = lambda: None
    window.update_frame = lambda: None

    window._seek_playhead(700)

    assert window._playback_window_start is None
    assert window._playback_window_stop is None
    assert window.t0 == 700
    assert window.time0 == 600
    assert window.time1 == 800


def test_locked_fixed_duration_table_start_edit_moves_whole_event():
    window = PSV.__new__(PSV)
    window.tmax = 10_000
    window.fs_song = 1_000
    window.event_presets = {
        "pulse": EventTypePreset("pulse", fixed_duration=True, duration_seconds=0.25, duration_editable=False)
    }

    assert window._bounds_for_event_edit("pulse", 1.0, 1.25, 2.0, 1.25, changed_edge="start") == (2.0, 2.25)
    assert window._bounds_for_event_edit("pulse", 1.0, 1.25, 1.0, 2.0, changed_edge="stop") == (1.75, 2.0)


def test_legacy_point_event_drag_accepts_qpointf_and_persists():
    class Combo:
        def currentIndex(self):
            return 1

    class Position:
        event_index = 0
        position = 1.0

        def pos(self):
            return QtCore.QPointF(2.0, 0.0)

    window = PSV.__new__(PSV)
    window.edit_only_current_events = False
    window.event_times = Events({"pulse": np.array([[1.0, 1.0, -1]])})
    window.event_presets = {
        "pulse": EventTypePreset("pulse", fixed_duration=True, duration_seconds=0.0, duration_editable=False)
    }
    window.eventList = [(0, "pulse")]
    window.cb = Combo()
    window.tmax = 10_000
    window.fs_song = 1_000
    window.annot_view = type("AnnotView", (), {"mousePoint": None})()
    window.update_xy = lambda: None

    window.on_position_change_finished(Position())

    np.testing.assert_allclose(window.event_times["pulse"][0, :2], [2.0, 2.0])
