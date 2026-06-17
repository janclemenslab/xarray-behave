import numpy as np

import xarray_behave  # noqa: F401 - sets QT_API before qtpy imports
from qtpy import QtCore, QtWidgets
from xarray_behave.annot import Events
from xarray_behave.gui import app as gui_app
from xarray_behave.gui.app import PSV
from xarray_behave.gui.event_widgets import (
    AudioChannelSettings,
    AudioSettingsDialog,
    ChannelSelectorPanel,
    EventBarsView,
    EventPresetPanel,
    EventRecord,
    EventTimelineWidget,
    EventTypePreset,
    EventsTableWidget,
    WaveformPane,
    records_from_events,
)


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


def test_records_from_events_time_range_preserves_original_indices():
    events = Events({"pulse": np.array([[0.0, 0.0, -1], [1.0, 1.1, -1], [2.0, 2.1, -1]])})

    records = records_from_events(events, start_seconds=0.9, stop_seconds=1.2)

    assert [record.id for record in records] == ["pulse\x1f1"]
    assert records[0].index == 1


def test_records_from_events_channel_filter_preserves_original_indices():
    events = Events({"pulse": np.array([[0.0, 0.0, 0], [1.0, 1.1, 1], [2.0, 2.1, -1]])})

    records = records_from_events(events, channel_filter=1)

    assert [record.id for record in records] == ["pulse\x1f1"]
    assert records[0].index == 1
    assert records[0].channel == 1


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


def test_events_table_channel_filter_uses_exact_channel():
    _app()
    widget = EventsTableWidget()
    events = Events({"pulse": np.array([[0.1, 0.1, -1], [0.2, 0.2, 0], [0.3, 0.3, 1]])})

    widget.set_events(events, channel_filter=0)

    assert widget.table.rowCount() == 1
    assert widget.table.item(0, widget._COL_CHANNEL).text() == "0"
    assert widget._record_id_for_row(0) == "pulse\x1f1"


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
    assert panel.list_widget.item(0).text() == ""
    row = panel.list_widget.itemWidget(panel.list_widget.item(0))
    assert row.name_label.text() == "pulse"
    assert "fixed 0s" in row.detail_label.text()
    assert panel.title_label.text() == "Annotations"


def test_channel_selector_panel_hosts_channel_selector_and_settings_button():
    _app()
    panel = ChannelSelectorPanel()
    requested = []
    panel.settings_requested.connect(lambda: requested.append(True))

    panel.set_channels(["Merged channels", "Channel 0"])
    panel.settings_button.click()

    assert panel.title_label.text() == "Audio"
    assert panel.channel_combo.count() == 2
    assert panel.channel_combo.currentText() == "Merged channels"
    assert panel.channel_combo.isEnabled()
    assert requested == [True]


def test_audio_settings_dialog_defaults_and_accepts_changes():
    _app()
    dialog = AudioSettingsDialog(AudioChannelSettings())
    changes = []
    dialog.settings_changed.connect(changes.append)

    assert dialog.settings() == AudioChannelSettings(waveform_all=True, events_all=True, playback_all=False)

    dialog.waveform_current_radio.setChecked(True)
    dialog.scale_y_current_radio.setChecked(True)
    dialog.events_current_radio.setChecked(True)
    dialog.playback_all_radio.setChecked(True)

    assert dialog.settings() == AudioChannelSettings(
        waveform_all=False,
        events_all=False,
        playback_all=True,
        scale_y_all=False,
    )
    assert changes[-1] == AudioChannelSettings(
        waveform_all=False,
        events_all=False,
        playback_all=True,
        scale_y_all=False,
    )


def test_preset_panel_emits_row_and_global_layer_toggles():
    _app()
    panel = EventPresetPanel()
    visibility_changes = []
    editability_changes = []
    all_visibility_changes = []
    all_editability_changes = []
    panel.visibility_changed.connect(lambda name, visible: visibility_changes.append((name, visible)))
    panel.editability_changed.connect(lambda name, editable: editability_changes.append((name, editable)))
    panel.visibility_all_changed.connect(all_visibility_changes.append)
    panel.editability_all_changed.connect(all_editability_changes.append)

    panel.set_presets(
        [
            EventTypePreset("pulse", visible=True, editable=True),
            EventTypePreset("song", visible=False, editable=False),
        ],
        selected_name="pulse",
    )
    row = panel.list_widget.itemWidget(panel.list_widget.item(0))

    row.visibility_button.click()
    row.editability_button.click()
    panel.visibility_all_button.click()
    panel.editability_all_button.click()

    assert visibility_changes == [("pulse", False)]
    assert editability_changes == [("pulse", False)]
    assert all_visibility_changes == [True]
    assert all_editability_changes == [True]


def test_preset_panel_selects_rows_by_name_after_refresh():
    _app()
    panel = EventPresetPanel()
    selected = []
    panel.selection_changed.connect(selected.append)
    panel.set_presets([EventTypePreset("pulse"), EventTypePreset("song")], selected_name="song")
    stale_row = panel.list_widget.itemWidget(panel.list_widget.item(0))

    panel.set_presets([EventTypePreset("pulse", color_hex="#d7263d"), EventTypePreset("song")], selected_name="song")
    stale_row.selection_requested.emit("pulse")

    assert panel.current_name() == "pulse"
    assert selected[-1] == "pulse"


def test_events_table_locks_rows_by_event_name():
    _app()
    widget = EventsTableWidget()
    widget.set_events(Events({"pulse": np.array([[0.1, 0.1, -1]])}), locked_event_names=["pulse"])

    assert not widget.table.cellWidget(0, widget._COL_TYPE).isEnabled()
    assert not (widget.table.item(0, widget._COL_START).flags() & QtCore.Qt.ItemIsEditable)


def test_waveform_pane_updates_playhead():
    _app()
    widget = WaveformPane()
    widget.set_waveform(np.array([1.0, 2.0]), np.array([0.0, 1.0]))

    widget.set_playhead(1.5)

    assert widget._playhead.value() == 1.5


def test_waveform_pane_overlays_other_channels_behind_primary():
    _app()
    widget = WaveformPane()
    yranges = []
    widget.setYRange = lambda ymin, ymax, padding=0: yranges.append((ymin, ymax, padding))

    widget.set_waveform(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    selected_range = yranges[-1]

    widget.set_waveform(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 0.0]),
        y_other=np.array([[1.0, 2.0], [0.5, 1.5], [0.0, 1.0]]),
    )
    all_range = yranges[-1]
    widget.set_waveform(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 0.0]),
        y_other=np.array([[1.0, 2.0], [0.5, 1.5], [0.0, 1.0]]),
        scale_y_all=False,
    )
    selected_with_overlay_range = yranges[-1]

    assert len(widget._other_curves) == 2
    assert all(curve.zValue() < widget._curve.zValue() for curve in widget._other_curves)
    assert selected_range[1] < 1.1
    assert all_range[1] > 2.0
    assert selected_with_overlay_range[1] < 1.1


def test_waveform_pane_uses_continuous_trace_below_four_seconds():
    _app()
    widget = WaveformPane()
    calls = []
    widget._curve.setData = lambda *args, **kwargs: calls.append((args, kwargs))

    widget.set_waveform(np.linspace(0.0, 1.0, 10_000), np.sin(np.linspace(0.0, 100.0, 10_000)))

    assert len(calls[-1][0][0]) == 10_000
    assert calls[-1][1]["connect"] == "finite"


def test_waveform_pane_uses_overview_pairs_at_four_seconds():
    _app()
    widget = WaveformPane()
    calls = []
    widget._curve.setData = lambda *args, **kwargs: calls.append((args, kwargs))

    widget.set_waveform(
        np.linspace(0.0, 4.0, 10_000),
        np.sin(np.linspace(0.0, 100.0, 10_000)),
        max_points=1000,
    )

    assert len(calls[-1][0][0]) == 2000
    assert calls[-1][1]["connect"] == "pairs"


def test_preset_selection_sets_current_event_without_top_combo():
    window = PSV.__new__(PSV)
    window.event_times = Events({"pulse": np.array([[0.1, 0.1, -1]]), "song": np.array([[0.2, 0.3, -1]])})
    window._current_event_name = "pulse"
    updates = []
    window.update_xy = lambda: updates.append(True)

    window._on_preset_selected("song")

    assert window.current_event_name == "song"
    assert updates == [True]


def test_no_events_warning_add_button_opens_preset_dialog(monkeypatch):
    _app()
    window = PSV.__new__(PSV)
    window.event_times = Events({})
    window._current_event_name = None
    calls = []
    warning_instances = []

    class Warning:
        def __init__(self, parent=None):
            self.parent = parent
            self.button = object()
            warning_instances.append(self)

        def exec(self):
            return None

        def clickedButton(self):
            return self.button

    monkeypatch.setattr(gui_app, "NoEventsRegisteredWarning", Warning)
    window._create_preset_from_panel = lambda: calls.append("create")

    window.on_trace_clicked(0.5, QtCore.Qt.MouseButton.LeftButton)

    assert warning_instances[0].parent is window
    assert calls == ["create"]


def test_numeric_event_shortcut_sets_current_event_without_top_combo():
    window = PSV.__new__(PSV)
    window.event_times = Events({"pulse": np.array([[0.1, 0.1, -1]]), "song": np.array([[0.2, 0.3, -1]])})
    window.eventList = [(0, "pulse"), (1, "song")]
    window._current_event_name = "pulse"
    refreshed = []
    updates = []
    window._refresh_preset_panel = lambda selected_name=None: refreshed.append(selected_name)
    window.update_xy = lambda: updates.append(True)

    window.change_event_type("2")

    assert window.current_event_name == "song"
    assert refreshed[-1] == "song"
    assert updates[-1] is True


def test_visible_event_times_filters_hidden_presets():
    window = PSV.__new__(PSV)
    window.event_times = Events({"pulse": np.array([[0.1, 0.1, -1]]), "song": np.array([[0.2, 0.3, -1]])})
    window.event_presets = {
        "pulse": EventTypePreset("pulse", visible=False),
        "song": EventTypePreset("song", visible=True),
    }

    visible = window._visible_event_times()

    assert visible.names == ["song"]


def test_selected_channel_event_filter_is_exact():
    rows = np.array([[0.1, 0.1, -1], [0.2, 0.2, 0], [0.3, 0.3, 1]])
    window = PSV.__new__(PSV)
    window.audio_channel_settings = AudioChannelSettings(events_all=False)
    window.cb2 = type("ChannelCombo", (), {"currentText": lambda self: "Channel 0"})()

    filtered = window._filter_event_rows_for_audio_channel(rows)

    np.testing.assert_array_equal(filtered[:, 2], np.array([0]))

    window.cb2 = type("ChannelCombo", (), {"currentText": lambda self: "Merged channels"})()
    filtered = window._filter_event_rows_for_audio_channel(rows)

    np.testing.assert_array_equal(filtered[:, 2], np.array([-1]))


def test_audio_settings_dialog_live_update_reverts_on_cancel(monkeypatch):
    class Signal:
        def __init__(self):
            self._callback = None

        def connect(self, callback):
            self._callback = callback

        def emit(self, settings):
            self._callback(settings)

    live_settings = AudioChannelSettings(waveform_all=False, events_all=False, playback_all=True)
    seen_live = []

    class Dialog:
        def __init__(self, settings, parent):
            self.settings_changed = Signal()
            self._parent = parent
            assert settings == AudioChannelSettings()

        def exec_(self):
            self.settings_changed.emit(live_settings)
            seen_live.append(self._parent.audio_channel_settings)
            return QtWidgets.QDialog.Rejected

    monkeypatch.setattr(gui_app.event_widgets, "AudioSettingsDialog", Dialog)
    window = PSV.__new__(PSV)
    window.audio_channel_settings = AudioChannelSettings()
    window.show_all_channels = True
    window._is_playing = False
    window.STOP = False

    window._edit_audio_settings()

    assert seen_live == [live_settings]
    assert window.audio_channel_settings == AudioChannelSettings()
    assert window.show_all_channels is True


def test_locked_preset_blocks_timeline_creation():
    window = PSV.__new__(PSV)
    window.event_times = Events({"pulse": np.zeros((0, 3))})
    window.event_presets = {"pulse": EventTypePreset("pulse", visible=True, editable=False)}

    window._on_timeline_event_created("pulse", 0.1, 0.2)

    assert len(window.event_times["pulse"]) == 0


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


def test_transport_loop_button_uses_audio_window_shortcut():
    _app()
    played = []
    window = PSV.__new__(PSV)
    window.tmin = 0
    window.tmax = 1_000
    window._t0 = 500
    window.fs_song = 1_000
    window.fs_other = 1_000
    window._span = 200
    window.vr = None
    window.play_audio = lambda qt_keycode: played.append(qt_keycode)

    panel = window._build_transport()

    loop_button = panel.findChild(QtWidgets.QToolButton, "transportLoopButton")
    assert loop_button is not None
    assert loop_button.text() == "Loop"
    loop_button.click()
    assert played == ["E"]


def test_play_audio_starts_window_playhead_range():
    class SongRaw:
        data = np.arange(1_000)[:, None]

    class Dataset:
        song_raw = SongRaw()

        def __contains__(self, key):
            return key == "song_raw"

    started = []
    window = PSV.__new__(PSV)
    window.ds = Dataset()
    window.tmin = 0
    window.tmax = 1_000
    window._t0 = 500
    window.fs_song = 1_000
    window.fs_other = 1_000
    window._span = 200
    window._playback_window_start = None
    window._playback_window_stop = None
    window.cb2 = type("ChannelCombo", (), {"currentText": lambda self: "Channel 0"})()
    window._start_window_audio_playhead = lambda window_start, window_stop, *, all_channels: started.append(
        (window_start, window_stop, all_channels)
    )

    window.play_audio("E")

    assert started == [(400, 600, False)]


def test_play_audio_all_channels_uses_qt_window_playhead():
    class SongRaw:
        data = np.arange(2_000).reshape(1_000, 2)

    class Dataset:
        song_raw = SongRaw()

        def __contains__(self, key):
            return key == "song_raw"

    started = []
    window = PSV.__new__(PSV)
    window.ds = Dataset()
    window.tmin = 0
    window.tmax = 1_000
    window._t0 = 500
    window.fs_song = 1_000
    window.fs_other = 1_000
    window._span = 200
    window._playback_window_start = None
    window._playback_window_stop = None
    window.audio_channel_settings = AudioChannelSettings(playback_all=True)
    window.cb2 = type("ChannelCombo", (), {"currentText": lambda self: "Channel 0"})()
    window._start_window_audio_playhead = lambda window_start, window_stop, *, all_channels: started.append(
        (window_start, window_stop, all_channels)
    )

    window.play_audio("E")

    assert started == [(400, 600, True)]


def test_window_audio_playhead_sweeps_without_advancing_window():
    class Timer:
        def __init__(self):
            self.started = False
            self.stopped = False

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

    class Sink:
        def __init__(self):
            self.elapsed = 0
            self.stopped = False

        def processedUSecs(self):
            return self.elapsed

        def stop(self):
            self.stopped = True

    class ChannelCombo:
        def __init__(self):
            self.enabled = []

        def count(self):
            return 2

        def setEnabled(self, enabled):
            self.enabled.append(enabled)

    window = PSV.__new__(PSV)
    window.tmin = 0
    window.tmax = 1_000
    window._t0 = 500
    window.fs_song = 1_000
    window.fs_other = 1_000
    window._span = 200
    window._is_playing = False
    window._playback_window_start = None
    window._playback_window_stop = None
    window._window_audio_timer = Timer()
    window._array_audio_sink = Sink()
    window._array_audio_buffer = None
    window._array_audio_bytes = None
    window.x = np.arange(400, 600) / 1_000
    window.vr = None
    window.update_xy = lambda: None
    window.update_frame = lambda: None
    window.cb2 = ChannelCombo()
    window._start_qt_array_audio_window = lambda window_start, window_stop, *, all_channels: True

    window._start_window_audio_playhead(400, 600, all_channels=False)

    assert window.t0 == 400
    assert window.time0 == 400
    assert window.time1 == 600
    assert window._window_audio_timer.started is True
    assert window.cb2.enabled[-1] is False

    window._array_audio_sink.elapsed = 50_000
    window._on_window_audio_tick()

    assert window.t0 == 450
    assert window.time0 == 400
    assert window.time1 == 600

    window._array_audio_sink.elapsed = 250_000
    window._on_window_audio_tick()

    assert window.t0 == 599
    assert window.time0 == 400
    assert window.time1 == 600
    assert window._window_audio_timer.stopped is True
    assert window.cb2.enabled[-1] is True


def test_channel_shortcuts_ignore_playback_locked_selector():
    class ChannelCombo:
        def __init__(self):
            self.index = 0

        def currentIndex(self):
            return self.index

        def count(self):
            return 2

        def setCurrentIndex(self, index):
            self.index = index

    window = PSV.__new__(PSV)
    window.cb2 = ChannelCombo()
    window.select_loudest_channel = True
    window._window_audio_start_sample = None
    window._is_playing = True

    window.set_next_channel(None)

    assert window.cb2.index == 0

    window._is_playing = False
    window.set_next_channel(None)

    assert window.cb2.index == 1


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

    class QMediaPlayer:
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

    class ChannelCombo:
        def __init__(self):
            self.enabled = []

        def currentText(self):
            return "Channel 0"

        def count(self):
            return 2

        def setEnabled(self, enabled):
            self.enabled.append(enabled)

    class SongRaw:
        data = np.arange(1_000)[:, None]

    class Dataset:
        song_raw = SongRaw()

        def __contains__(self, key):
            return key == "song_raw"

    window = PSV.__new__(PSV)
    window.ds = Dataset()
    window.tmin = 0
    window.tmax = 1_000
    window._t0 = 500.0000000001
    window.fs_song = 1_000
    window.fs_other = 1_000
    window._span = 200.0000000001
    window.STOP = True
    window._is_playing = False
    window._playback_timer = Timer()
    window._playback_clock = Clock()
    window._audio_player = QMediaPlayer()
    window.cb2 = ChannelCombo()
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
    assert window.cb2.enabled[-1] is False


def test_start_playback_qt_array_audio_queues_before_transport_clock():
    class Timer:
        def __init__(self):
            self.started = False

        def start(self):
            self.started = True
            events.append("timer_start")

    class Clock:
        def restart(self):
            events.append("clock_restart")

    class App:
        def processEvents(self):
            pass

    class SongRaw:
        data = np.arange(1_000)[:, None]

    class Dataset:
        song_raw = SongRaw()

        def __contains__(self, key):
            return key == "song_raw"

    events = []
    window = PSV.__new__(PSV)
    window.ds = Dataset()
    window.tmin = 0
    window.tmax = 1_000
    window._t0 = 500.0000000001
    window.fs_song = 1_000
    window.fs_other = 1_000
    window._span = 200.0000000001
    window.STOP = True
    window._is_playing = False
    window._playback_timer = Timer()
    window._playback_clock = Clock()
    window._audio_player = None
    window.cb2 = type("ChannelCombo", (), {"currentText": lambda self: "Channel 0"})()
    window._playback_window_start = None
    window._playback_window_stop = None
    window.vr = None
    window.app = App()
    window.update_xy = lambda: None
    window.update_frame = lambda: None
    window._set_play_button_state = lambda *, playing: None
    window._start_qt_array_audio_window = lambda start, stop, *, all_channels: events.append(
        ("audio_play", start, stop, all_channels)
    ) or True

    window._start_playback()

    assert window._playback_window_start == 400
    assert window._playback_window_stop == 600
    assert window._playback_audio_stop_sample == 1_000
    assert events == [("audio_play", 400, 1_000, False), "clock_restart", "timer_start"]


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
    window.audio_channel_settings = AudioChannelSettings(playback_all=True)
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


def test_playback_tick_does_not_replay_continuous_array_audio_at_visible_end():
    class App:
        def processEvents(self):
            pass

    class Clock:
        def nsecsElapsed(self):
            return 201_000_000

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
    window._playback_audio_stop_sample = 1_000
    window._audio_player = None
    window._playback_clock = Clock()
    window.vr = None
    window.app = App()
    window.update_xy = lambda: None
    window.update_frame = lambda: None
    window._set_play_button_state = lambda *, playing: None
    calls = []
    window._start_transport_array_audio = lambda: calls.append(True)

    window._on_playback_tick()

    assert window._playback_window_start == 600
    assert window._playback_window_stop == 800
    assert window.t0 == 601
    assert calls == []


def test_start_transport_array_audio_prefers_qt_array_audio():
    class SongRaw:
        data = np.arange(1_000)[:, None]

    class Dataset:
        song_raw = SongRaw()

        def __contains__(self, key):
            return key == "song_raw"

    window = PSV.__new__(PSV)
    window.ds = Dataset()
    window.tmax = 1_000
    window.fs_song = 1_000
    window._playback_window_start = 400
    window._playback_window_stop = 600
    window._playback_audio_stop_sample = None
    window.cb2 = type("ChannelCombo", (), {"currentText": lambda self: "Channel 0"})()
    window._audio_player = None
    window._start_qt_array_audio_window = lambda start, stop, *, all_channels: calls.append(
        (start, stop, all_channels)
    ) or True

    calls = []

    assert window._start_transport_array_audio() is True
    assert calls == [(400, 1_000, False)]
    assert window._playback_audio_stop_sample == 1_000


def test_playback_tick_uses_qt_array_audio_processed_clock():
    class App:
        def processEvents(self):
            pass

    class Sink:
        def processedUSecs(self):
            return 201_000

    class Clock:
        def nsecsElapsed(self):
            return 999_000_000

    window = PSV.__new__(PSV)
    window.tmin = 0
    window.tmax = 1_000
    window._t0 = 400
    window.fs_song = 1_000
    window.fs_other = 1_000
    window._span = 200
    window.STOP = False
    window._is_playing = True
    window._playback_anchor_sample = 400
    window._playback_window_start = 400
    window._playback_window_stop = 600
    window._playback_audio_stop_sample = 1_000
    window._audio_player = None
    window._array_audio_sink = Sink()
    window._array_audio_start_sample = 400
    window._playback_clock = Clock()
    window.vr = None
    window.app = App()
    window.update_xy = lambda: None
    window.update_frame = lambda: None
    window._set_play_button_state = lambda *, playing: None
    window._start_transport_array_audio = lambda: False

    window._on_playback_tick()

    assert window.t0 == 601
    assert window._playback_window_start == 600
    assert window._playback_window_stop == 800


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


def test_point_event_drag_accepts_qpointf_and_persists():
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
    window._current_event_name = "pulse"
    window.tmax = 10_000
    window.fs_song = 1_000
    window.annot_view = type("AnnotView", (), {"mousePoint": None})()
    window.update_xy = lambda: None

    window.on_position_change_finished(Position())

    np.testing.assert_allclose(window.event_times["pulse"][0, :2], [2.0, 2.0])
