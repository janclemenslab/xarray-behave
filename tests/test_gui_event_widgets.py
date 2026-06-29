import numpy as np
import xarray as xr

import xarray_behave  # noqa: F401 - sets QT_API before qtpy imports
from qtpy import QtCore, QtWidgets
from xarray_behave.annot import Events
from xarray_behave.gui import app as gui_app, gui_config, view_dialog, views
from xarray_behave.gui.app import PSV
from xarray_behave.gui.event_widgets import (
    AudioChannelSettings,
    EventBarsView,
    EventPresetPanel,
    EventRecord,
    EventTimelineWidget,
    EventTypePreset,
    EventsTableWidget,
    ThresholdingPanel,
    WaveformPane,
    WaveformSettingsDialog,
    records_from_events,
)


def _app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _audio_dataset(event_times=None):
    sampletime = np.arange(1_000) / 1_000
    ds = xr.Dataset(
        {"song_raw": (("sampletime", "channels"), np.zeros((len(sampletime), 2)))},
        coords={"sampletime": sampletime},
        attrs={"target_sampling_rate_Hz": 1_000},
    )
    ds.song_raw.attrs["sampling_rate_Hz"] = 1_000
    if event_times is not None:
        ds.attrs["event_times"] = event_times
    return ds


def test_psv_restores_and_captures_persistent_gui_state(tmp_path):
    _app()
    manager = gui_config.GuiConfigManager(home=tmp_path)
    manager.config = {
        "version": 1,
        "window": {
            "panels": {"timeline": False, "event_table": True, "sidebar": False},
            "splitter_sizes": {"waveform": 150, "spectrogram": 350},
        },
        "viewer": {
            "waveform": {"color": "#ff6a74", "y_limits": [-2.0, 3.0]},
            "spectrogram": {"compression": 3, "resolution": 128, "colormap": "magma"},
            "audio": {"waveform_all": False, "events_all": False, "playback_all": True, "scale_y_all": False},
            "annotations": {"table_audio_link": False, "table_audio_filter": True, "show_labels": False},
            "thresholding": {"enabled": True, "value": 0.4, "min_distance": 0.05},
        },
        "selection": {"event_type": "pulse", "audio_channel": "Channel 1"},
        "event_types": [
            {
                "name": "pulse",
                "fixed_duration": True,
                "duration_seconds": 0.01,
                "duration_editable": False,
                "color_hex": "#ffd166",
                "visible": True,
                "editable": False,
            }
        ],
    }
    window = PSV(
        _audio_dataset(Events({"song": np.array([[0.1, 0.2, 0]])})),
        config_manager=manager,
    )

    assert window.event_times.names == ["pulse", "song"]
    assert window.current_event_name == "song"
    assert window._event_preset("pulse").duration_seconds == 0.01
    assert window._event_preset("pulse").editable is False
    assert window.current_channel_name == "Channel 0"
    assert window.slice_view.waveform_color == "#ff6a74"
    assert window.slice_view.waveform_y_limits == (-2.0, 3.0)
    assert window.spec_compression_ratio == 3
    assert window.spec_win == 128
    assert window.spec_colormap == "magma"
    assert window.events_table.sync_enabled is False
    assert window.events_table.window_filter_enabled is True
    assert window.threshold_mode is True
    assert window.thres_value == 0.4
    assert window.show_timeline is False
    assert window.show_sidebar is False
    assert window.transport_panel.isVisible() is True
    assert not hasattr(window, "annot_view")

    view_menu = next(menu for menu in window.bar.findChildren(QtWidgets.QMenu) if menu.title() == "View")
    view_labels = {action.text() for action in view_menu.actions()}
    assert "Show transport" not in view_labels
    assert "Show ethogram" not in view_labels
    assert "Video, waveform, and spectrogram display parameters" not in view_labels

    snapshot = window._config_snapshot()
    assert "selection" not in snapshot
    assert snapshot["window"]["panels"]["timeline"] is False
    assert "transport" not in snapshot["window"]["panels"]
    assert "ethogram" not in snapshot["window"]["panels"]
    assert snapshot["viewer"]["audio"]["playback_all"] is True
    assert snapshot["viewer"]["annotations"]["table_audio_filter"] is True
    assert [item["name"] for item in snapshot["event_types"]] == ["pulse", "song"]

    window.close()
    assert "selection" not in gui_config.read_config(tmp_path / ".das.yaml")


def _toolbar_action(window, tooltip: str):
    for action in window.annotation_toolbar.actions():
        if action.toolTip() == tooltip:
            return action
    raise AssertionError(f"Missing toolbar action {tooltip!r}")


def test_psv_annotation_toolbar_exposes_and_toggles_view_actions():
    _app()
    window = PSV(_audio_dataset(Events({"song": np.array([[0.1, 0.2, 0]])})))

    tooltips = {action.toolTip() for action in window.annotation_toolbar.actions() if action.toolTip()}
    assert {
        "Open audio/annotations",
        "Import annotations",
        "Save annotations",
        "Show waveform",
        "Show spectrogram",
        "Show event timeline",
        "Show annotation table",
        "Show annotation type table",
        "Thresholding mode",
    }.issubset(tooltips)
    for action in window.annotation_toolbar.actions():
        if action.toolTip():
            assert not action.icon().isNull()

    waveform_action = _toolbar_action(window, "Show waveform")
    waveform_action.trigger()
    assert window.show_trace is False
    assert waveform_action.isChecked() is False
    assert window.slice_view.isVisible() is False
    assert window.cb2.parent() is window.spec_view
    assert window.cb2.isVisible()
    assert window.cb2.x() < window.spec_view.settings_button.x()

    spectrogram_action = _toolbar_action(window, "Show spectrogram")
    spectrogram_action.trigger()
    assert window.show_spec is False
    assert spectrogram_action.isChecked() is False
    assert window.spec_view.isVisible() is False
    assert window.cb2.isHidden()

    timeline_action = _toolbar_action(window, "Show event timeline")
    timeline_action.trigger()
    assert window.show_timeline is False
    assert timeline_action.isChecked() is False
    assert window.event_timeline.isVisible() is False

    table_action = _toolbar_action(window, "Show annotation table")
    table_action.trigger()
    assert window.show_event_table is False
    assert table_action.isChecked() is False
    assert window.events_table.isVisible() is False

    type_action = _toolbar_action(window, "Show annotation type table")
    type_action.trigger()
    assert window.show_sidebar is False
    assert type_action.isChecked() is False
    assert window.left_sidebar.isVisible() is False

    threshold_action = _toolbar_action(window, "Thresholding mode")
    threshold_action.trigger()
    assert window.threshold_mode is True
    assert threshold_action.isChecked() is True
    assert window.show_sidebar is True
    assert type_action.isChecked() is True
    assert window.threshold_panel.isVisible() is True

    window.close()


def test_psv_annotation_toolbar_includes_movie_action_when_video_is_loaded():
    _app()

    class FakeVideoReader:
        frame_rate = 1_000
        frame_width = 5
        frame_height = 4

        def __getitem__(self, index):
            return np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

    window = PSV(_audio_dataset(Events({"song": np.array([[0.1, 0.2, 0]])})), vr=FakeVideoReader())

    movie_action = _toolbar_action(window, "Show movie")
    movie_action.trigger()
    assert window.show_movie is False
    assert movie_action.isChecked() is False
    assert window.movie_view.isVisible() is False

    window.close()


def test_import_annotations_merge_refreshes_viewer(monkeypatch):
    _app()
    window = PSV(_audio_dataset(Events({"pulse": np.array([[0.1, 0.1, 0], [0.2, 0.2, 0]])})))

    imported = Events(
        {
            "pulse": np.array([[0.2, 0.2, 0], [0.3, 0.3, 0]]),
            "chirp": np.array([[0.4, 0.5, -1]]),
        }
    )
    monkeypatch.setattr(gui_app.dataset_service, "load_annotation_file", lambda filename: imported)

    window.import_annotations(filename="/tmp/import_annotations.csv", mode="merge")

    assert window.event_times.names == ["pulse", "chirp"]
    assert sorted(map(tuple, window.event_times["pulse"])) == [
        (0.1, 0.1, 0.0),
        (0.2, 0.2, 0.0),
        (0.3, 0.3, 0.0),
    ]
    assert "chirp" in {record.name for record in window.events_table._records_by_id.values()}
    assert "chirp" in {
        window.preset_panel.list_widget.item(row).data(QtCore.Qt.UserRole)
        for row in range(window.preset_panel.list_widget.count())
    }

    window.close()


def test_import_annotations_replace_drops_old_presets(monkeypatch):
    _app()
    window = PSV(_audio_dataset(Events({"old": np.array([[0.1, 0.1, -1]])})))

    imported = Events({"new": np.array([[0.2, 0.3, -1]])})
    monkeypatch.setattr(gui_app.dataset_service, "load_annotation_file", lambda filename: imported)

    window.import_annotations(filename="/tmp/import_annotations.csv", mode="replace")

    assert window.event_times.names == ["new"]
    assert list(window.event_presets) == ["new"]
    assert window.current_event_name == "new"
    assert {record.name for record in window.events_table._records_by_id.values()} == {"new"}

    window.close()


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


def test_link_table_audio_view_scrolls_without_filtering_table():
    _app()
    events = Events(
        {
            "early": np.array([[0.1, 0.2, -1]]),
            "visible": np.array([[1.0, 1.4, -1]]),
            "late": np.array([[3.0, 3.1, -1]]),
        }
    )
    window = PSV(_audio_dataset(events))
    window.x = np.array([0.9, 1.1])
    window.events_table.link_checkbox.setChecked(True)
    window.events_table.window_filter_checkbox.setChecked(False)

    window._refresh_event_widgets(sync_table_to_view=True)

    assert window.events_table.table.rowCount() == 3
    assert [record.name for record in window.events_table.selected_records()] == ["visible"]

    window.events_table.window_filter_checkbox.setChecked(True)
    assert window.events_table.table.rowCount() == 1
    assert [record.name for record in window.events_table.selected_records()] == ["visible"]

    window.close()


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


def test_thresholding_panel_syncs_values_and_emits_changes():
    _app()
    panel = ThresholdingPanel()
    threshold_changes = []
    envelope_changes = []
    min_distance_changes = []
    duration_filter_changes = []
    duration_range_changes = []
    bandpass_filter_changes = []
    bandpass_range_changes = []
    generated = []
    panel.threshold_changed.connect(threshold_changes.append)
    panel.envelope_std_changed.connect(envelope_changes.append)
    panel.min_distance_changed.connect(min_distance_changes.append)
    panel.duration_filter_changed.connect(duration_filter_changes.append)
    panel.duration_range_changed.connect(duration_range_changes.append)
    panel.bandpass_filter_changed.connect(bandpass_filter_changes.append)
    panel.bandpass_range_changed.connect(bandpass_range_changes.append)
    panel.generate_requested.connect(lambda: generated.append(True))

    panel.set_limits(duration_max=2.0, frequency_max=500.0)
    panel.set_values(
        threshold=0.25,
        envelope_std=0.003,
        min_distance=0.04,
        duration_enabled=True,
        duration_range=(0.05, 0.25),
        bandpass_enabled=True,
        bandpass_range=(100.0, 300.0),
    )
    panel.threshold_spin.setValue(0.5)
    panel.envelope_std_spin.setValue(0.006)
    panel.min_distance_spin.setValue(0.08)
    panel.duration_checkbox.setChecked(False)
    panel.duration_range.setValue([0.1, 0.4])
    panel.bandpass_checkbox.setChecked(False)
    panel.bandpass_range.setValue([150.0, 250.0])
    panel.generate_button.click()

    assert panel.threshold_spin.value() == 0.5
    assert panel.envelope_std_spin.value() == 0.006
    assert panel.min_distance_spin.value() == 0.08
    assert panel.duration_range.value() == [0.1, 0.4]
    assert panel.bandpass_range.value() == [150.0, 250.0]
    assert threshold_changes[-1] == 0.5
    assert envelope_changes[-1] == 0.006
    assert min_distance_changes[-1] == 0.08
    assert duration_filter_changes[-1] is False
    assert duration_range_changes[-1] == [0.1, 0.4]
    assert bandpass_filter_changes[-1] is False
    assert bandpass_range_changes[-1] == [150.0, 250.0]
    assert generated == [True]


def test_waveform_pane_draws_threshold_overlay_only_when_enabled():
    _app()
    widget = WaveformPane()
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x)
    envelope = np.abs(y)

    widget.set_waveform(x, y)
    widget.set_threshold_data(x, envelope, enabled=True, threshold=0.4)
    env_x, env_y = widget._threshold_curve.getData()

    assert widget.threshold == 0.4
    assert widget.threshold_line in widget.getPlotItem().items
    assert widget._threshold_curve in widget.getPlotItem().items
    assert np.allclose(env_x, x)
    assert np.allclose(env_y, envelope)

    widget.set_threshold_data(None, None, enabled=False)

    assert widget.threshold_line not in widget.getPlotItem().items
    assert widget._threshold_curve not in widget.getPlotItem().items


def test_threshold_duration_mode_adds_interval_events():
    window = PSV.__new__(PSV)
    window.STOP = True
    window.event_times = Events({"song": np.zeros((0, 3))})
    window.event_presets = {"song": EventTypePreset("song", fixed_duration=False)}
    window._current_event_name = "song"
    window.fs_song = 100
    window.tmax = 1_000
    window.x = np.arange(8) / window.fs_song
    window.envelope = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    window.thres_value = 0.5
    window.thres_min_dist = 0.011
    window.thres_duration_enabled = True
    window.thres_duration_min = 0.05
    window.thres_duration_max = 0.1
    window.slice_view = type("SliceView", (), {"threshold": 0.5})()
    window.update_xy = lambda: None

    window.threshold(None)

    np.testing.assert_allclose(window.event_times["song"], [[0.01, 0.07, -1.0]])


def test_threshold_bandpass_filters_signal_before_envelope():
    window = PSV.__new__(PSV)
    window.fs_song = 1_000
    seconds = np.arange(0.0, 2.0, 1 / window.fs_song)
    low = np.sin(2 * np.pi * 20 * seconds)
    high = np.sin(2 * np.pi * 200 * seconds)
    window.y = low + high
    window.thres_bandpass_enabled = True
    window.thres_bandpass_low = 150.0
    window.thres_bandpass_high = 250.0

    filtered = window._threshold_signal()
    core = slice(200, -200)

    assert np.corrcoef(filtered[core], high[core])[0, 1] > 0.9
    assert abs(np.corrcoef(filtered[core], low[core])[0, 1]) < 0.2


def test_waveform_pane_overlays_channel_selector_for_multichannel_audio_only():
    _app()
    widget = WaveformPane()

    widget.set_channels(["Merged channels", "Channel 0"], show_selector=False)

    assert widget.channel_combo.isHidden()

    widget.set_channels(["Merged channels", "Channel 0", "Channel 1"], show_selector=True)

    assert not widget.channel_combo.isHidden()
    assert widget.channel_combo.count() == 3
    assert widget.channel_combo.currentText() == "Merged channels"
    assert widget.channel_combo.isEnabled()
    assert widget.channel_combo.x() < widget.settings_button.x()
    widget.close()


def test_waveform_settings_dialog_merges_display_and_audio_settings_live():
    _app()
    widget = WaveformPane()
    widget.set_audio_settings(AudioChannelSettings())
    audio_changes = []
    widget.audio_settings_changed.connect(audio_changes.append)
    yranges = []
    widget.setYRange = lambda ymin, ymax, padding=0: yranges.append((ymin, ymax, padding))
    widget.set_waveform(
        np.array([0.0, 1.0]),
        np.array([-0.25, 0.5]),
        y_other=np.array([[10.0], [20.0]]),
    )
    dialog = WaveformSettingsDialog(widget)

    assert widget.settings_button.objectName() == "waveformSettingsButton"
    assert widget.settings_button.size() == QtCore.QSize(22, 22)
    assert dialog.auto_limits_source_group.isEnabled()
    assert not dialog.fixed_limits_group.isEnabled()
    color_index = dialog.color_combo.findData("#ff6a74")
    dialog.color_combo.setCurrentIndex(color_index)
    dialog.auto_limits_checkbox.setChecked(False)
    assert not dialog.auto_limits_source_group.isEnabled()
    assert dialog.fixed_limits_group.isEnabled()
    dialog.lower_spin.setValue(-2.0)
    dialog.upper_spin.setValue(3.0)

    assert widget.waveform_color == "#ff6a74"
    assert widget.waveform_y_limits == (-2.0, 3.0)
    assert yranges[-1] == (-2.0, 3.0, 0)

    dialog.auto_limits_checkbox.setChecked(True)

    assert widget.waveform_y_limits is None

    assert dialog.audio_settings() == AudioChannelSettings(waveform_all=True, events_all=True, playback_all=False)

    dialog.waveform_current_radio.setChecked(True)
    dialog.scale_y_current_radio.setChecked(True)
    dialog.events_current_radio.setChecked(True)
    dialog.playback_all_radio.setChecked(True)

    assert dialog.audio_settings() == AudioChannelSettings(
        waveform_all=False,
        events_all=False,
        playback_all=True,
        scale_y_all=False,
    )
    assert widget.audio_settings == AudioChannelSettings(
        waveform_all=False,
        events_all=False,
        playback_all=True,
        scale_y_all=False,
    )
    assert audio_changes[-1] == AudioChannelSettings(
        waveform_all=False,
        events_all=False,
        playback_all=True,
        scale_y_all=False,
    )
    assert yranges[-1][1] < 1.0
    dialog.close()
    widget.close()


def test_waveform_settings_dialog_is_singleton():
    app = _app()
    widget = WaveformPane()

    widget._open_settings_dialog()
    dialog = widget._settings_dialog
    widget._open_settings_dialog()

    assert widget._settings_dialog is dialog

    dialog.reject()
    app.processEvents()

    assert widget._settings_dialog is None
    widget.close()


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


def test_events_table_type_combo_changes_selection_from_start_of_edit():
    _app()
    widget = EventsTableWidget()
    widget.set_events(
        Events(
            {
                "pulse": np.array([[0.1, 0.1, -1], [0.2, 0.2, -1]]),
                "song": np.zeros((0, 3)),
            }
        )
    )
    changed = []
    widget.type_changed.connect(lambda records, name: changed.append(([record.id for record in records], name)))
    selected_ids = ["pulse\x1f0", "pulse\x1f1"]
    widget.select_ids(selected_ids)
    combo = widget.table.cellWidget(0, widget._COL_TYPE)

    widget._remember_type_combo_selection("pulse\x1f0")
    widget.select_ids(["pulse\x1f0"])
    combo.setCurrentText("song")
    combo.activated.emit(combo.currentIndex())

    assert changed == [(selected_ids, "song")]


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


def test_locked_preset_blocks_timeline_creation():
    window = PSV.__new__(PSV)
    window.event_times = Events({"pulse": np.zeros((0, 3))})
    window.event_presets = {"pulse": EventTypePreset("pulse", visible=True, editable=False)}

    window._on_timeline_event_created("pulse", 0.1, 0.2)

    assert len(window.event_times["pulse"]) == 0


def test_fixed_duration_creation_uses_click_as_center():
    window = PSV.__new__(PSV)
    window.tmax = 10_000
    window.fs_song = 1_000
    window.event_presets = {
        "pulse": EventTypePreset("pulse", fixed_duration=True, duration_seconds=0.25, duration_editable=False)
    }

    assert window._bounds_for_event_creation("pulse", 1.0) == (0.875, 1.125)


def test_non_fixed_trace_creation_uses_two_clicks():
    _app()
    window = PSV.__new__(PSV)
    window.event_times = Events({"song": np.zeros((0, 3))})
    window.event_presets = {"song": EventTypePreset("song", fixed_duration=False)}
    window._current_event_name = "song"
    window.sinet0 = None
    window.sinet0_event_name = None
    window.cb2 = type("ChannelCombo", (), {"currentText": lambda self: "Channel 0"})()
    updates = []
    window.update_xy = lambda: updates.append(True)

    window.on_trace_clicked(0.75, QtCore.Qt.MouseButton.LeftButton)

    assert len(window.event_times["song"]) == 0
    assert window.sinet0 == 0.75
    assert window.sinet0_event_name == "song"
    assert updates == [True]

    window.on_trace_clicked(1.25, QtCore.Qt.MouseButton.LeftButton)

    np.testing.assert_allclose(window.event_times["song"], [[0.75, 1.25, 0]])
    assert window.sinet0 is None
    assert window.sinet0_event_name is None
    assert updates == [True, True]


def test_non_fixed_timeline_click_creation_uses_two_clicks():
    window = PSV.__new__(PSV)
    window.event_times = Events({"song": np.zeros((0, 3))})
    window.event_presets = {"song": EventTypePreset("song", fixed_duration=False)}
    window._current_event_name = "song"
    window.sinet0 = None
    window.sinet0_event_name = None
    window.cb2 = type("ChannelCombo", (), {"currentText": lambda self: "Merged channels"})()
    updates = []
    window.update_xy = lambda: updates.append("xy")
    window._after_event_edit = lambda: updates.append("after")

    window._on_timeline_event_created("song", 0.25, 0.25)

    assert len(window.event_times["song"]) == 0
    assert window.sinet0 == 0.25
    assert window.sinet0_event_name == "song"
    assert updates == ["xy"]

    window._on_timeline_event_created("song", 0.75, 0.75)

    np.testing.assert_allclose(window.event_times["song"], [[0.25, 0.75, -1]])
    assert window.sinet0 is None
    assert window.sinet0_event_name is None
    assert updates == ["xy", "after"]


def test_pending_non_fixed_boundary_is_drawn():
    window = PSV.__new__(PSV)
    window.event_times = Events({"song": np.zeros((0, 3))})
    window.event_presets = {"song": EventTypePreset("song", fixed_duration=False)}
    window._current_event_name = "song"
    window.sinet0 = 0.75
    window.sinet0_event_name = "song"
    window.eventtype_colors = np.array([[10, 20, 30]])
    window.show_event_text = False
    window.show_trace = True
    window.show_tracks = False
    window.show_spec = False
    calls = []
    window.slice_view = type(
        "SliceView",
        (),
        {"add_event": lambda self, xx, event_index, pen, movable=False, text=None: calls.append((xx, event_index, movable, text))},
    )()

    window._plot_pending_event_boundary(np.array([0.0, 1.0]))

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], [0.75])
    assert calls[0][1:] == (0, False, None)


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


def test_qmedia_start_position_signal_preserves_playback_window():
    class Timer:
        def __init__(self):
            self.active = False

        def start(self):
            self.active = True

        def isActive(self):
            return self.active

    class Clock:
        def restart(self):
            pass

    class QMediaPlayer:
        def __init__(self):
            self.callback = None

        def setPosition(self, value):
            self.callback(value)

        def play(self):
            pass

    class App:
        def processEvents(self):
            pass

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
    window._t0 = 500
    window.fs_song = 1_000
    window.fs_other = 1_000
    window._span = 200
    window.STOP = True
    window._is_playing = False
    window._playback_timer = Timer()
    window._playback_clock = Clock()
    window._audio_player = QMediaPlayer()
    window._audio_player.callback = window._on_audio_position_changed
    window.cb2 = type("ChannelCombo", (), {"currentText": lambda self: "Channel 0"})()
    window._playback_window_start = None
    window._playback_window_stop = None
    window.vr = None
    window.app = App()
    window.update_xy = lambda: None
    window.update_frame = lambda: None
    window._set_play_button_state = lambda *, playing: None

    window._start_playback()

    assert window._playback_window_start == 400
    assert window._playback_window_stop == 600
    assert window.t0 == 400
    assert window.time0 == 400
    assert window.time1 == 600


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
    window.update_xy = lambda: None

    window.on_position_change_finished(Position())

    np.testing.assert_allclose(window.event_times["pulse"][0, :2], [2.0, 2.0])


def test_spectrogram_view_opens_settings_dialog(monkeypatch):
    _app()

    model = object()
    widget = views.SpecView(model=model, callback=lambda *args: None)
    calls = []

    class Signal:
        def __init__(self):
            self._callback = None

        def connect(self, callback):
            self._callback = callback

        def emit(self, *args):
            self._callback(*args)

    class Dialog:
        def __init__(self, parent, model):
            self.finished = Signal()
            self._visible = True
            calls.append(("init", parent, model))

        def isVisible(self):
            return self._visible

        def show(self):
            calls.append(("show",))

        def raise_(self):
            calls.append(("raise",))

        def activateWindow(self):
            calls.append(("activate",))

        def deleteLater(self):
            calls.append(("delete",))

    monkeypatch.setattr(view_dialog, "SpectrogramSettingsDialog", Dialog)

    widget.resize(320, 180)
    widget._position_settings_button()
    widget._open_settings_dialog()
    dialog = widget._settings_dialog
    widget._open_settings_dialog()

    assert widget.settings_button.objectName() == "spectrogramSettingsButton"
    assert widget.settings_button.x() == widget.width() - widget.settings_button.width() - 8
    assert widget.settings_button.size() == QtCore.QSize(22, 22)
    assert widget.settings_button.iconSize() == QtCore.QSize(18, 18)
    assert calls[0][0] == "init"
    assert calls[0][2] is model
    assert widget._settings_dialog is dialog
    assert calls[1:] == [("show",), ("raise",), ("activate",), ("show",), ("raise",), ("activate",)]
    dialog.finished.emit(QtWidgets.QDialog.Rejected)
    assert widget._settings_dialog is None
    assert calls[-1] == ("delete",)
    widget.close()


def test_spectrogram_invalid_frequency_bounds_keep_nonempty_display():
    _app()

    x = np.linspace(0.0, 0.25, 1024)
    y = np.sin(2 * np.pi * 120 * x)

    class Model:
        fs_song = 4_096
        spec_mel = False
        spec_win = 128
        spec_compression_ratio = 0.0
        fmin = 10_000.0
        fmax = -20.0
        spec_denoise = False
        spec_levels = [None, None]
        t0 = 0

    model = Model()
    model.x = x
    widget = views.SpecView(model=model, callback=lambda *args: None)

    widget.update_spec(x, y)

    assert widget.S.size > 0
    assert widget.S.shape[0] >= 1
    widget.close()


def test_spectrogram_settings_dialog_updates_model_live():
    _app()

    class SpecView:
        S = np.array([[0.0, 1.0], [2.0, 3.0]])

    class Model:
        fs_song = 1_000
        fmin = None
        fmax = None
        spec_levels = [None, None]
        spec_compression_ratio = 0.0
        spec_colormap = "turbo"
        spec_view = SpecView()
        resolution_calls = []

        def inc_freq_res(self, key):
            self.resolution_calls.append(("inc", key))

        def dec_freq_res(self, key):
            self.resolution_calls.append(("dec", key))

    model = Model()
    dialog = view_dialog.SpectrogramSettingsDialog(model=model)

    dialog.colormap_combo.setCurrentText("magma")
    dialog.frequency_slider.sld.setValue((100.0, 200.0))
    dialog.level_slider.checkbox.setChecked(False)
    dialog.level_slider.sld.setValue((0.5, 1.5))
    dialog.compression_slider.sld.setValue(4.0)
    dialog.resolution_slider.slider.setValue(-1)
    dialog.resolution_slider.slider.setValue(1)

    assert model.spec_colormap == "magma"
    np.testing.assert_allclose([model.fmin, model.fmax], [100.0, 200.0])
    np.testing.assert_allclose(model.spec_levels, [0.5, 1.5])
    assert model.spec_compression_ratio == 4.0
    assert model.resolution_calls == [("inc", None), ("dec", None), ("dec", None)]
    dialog.close()
