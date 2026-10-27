import numpy as np
import pandas as pd
import pytest

import xarray_behave  # noqa: F401 - sets QT_API before qtpy imports
from qtpy import QtWidgets

from xarray_behave import _dataset_service as dataset_service
from xarray_behave.annot import Events
from xarray_behave.gui import app as gui_app
from xarray_behave.gui.annotation_import import AnnotationImportDialog
from xarray_behave.gui.annotation_import import add_annotation_import_button, finish_annotation_import
from test_gui_event_widgets import _audio_dataset


def test_preview_and_custom_column_mapping_import_all_rows(qtbot, tmp_path):
    filename = tmp_path / "labels.csv"
    pd.DataFrame({"end": np.arange(12) + 100, "label": ["001"] * 12, "begin": np.arange(12)}).to_csv(filename, index=False)
    dialog = AnnotationImportDialog(filename, 1000)
    qtbot.addWidget(dialog)

    assert dialog.preview.rowCount() == 10
    assert dialog.preview.columnCount() == 3
    assert dialog.preview.item(0, 1).text() == "001"
    assert dialog.preview.editTriggers() == QtWidgets.QAbstractItemView.NoEditTriggers
    for key, column in {"name": "label", "start_seconds": "begin", "stop_seconds": "end"}.items():
        dialog.column_selectors[key].setCurrentText(column)
    dialog.units.setCurrentText("Samples")
    dialog._import()

    assert dialog.result() == QtWidgets.QDialog.Accepted
    assert dialog.events.names == ["001"]
    assert len(dialog.events["001"]) == 12
    np.testing.assert_allclose(dialog.events["001"][0], [0, 0.1, -1])


@pytest.mark.parametrize("separator", [",", "\t", ";"])
def test_standard_columns_default_to_seconds_and_preserve_empty_types(qtbot, tmp_path, separator):
    filename = tmp_path / "labels.csv"
    pd.DataFrame({"name": ["NA", "empty"], "start_seconds": [0.1, np.nan], "stop_seconds": [0.2, np.nan], "channel": [1, -1]}).to_csv(
        filename, sep=separator, index=False
    )
    dialog = AnnotationImportDialog(filename, 1000)
    qtbot.addWidget(dialog)
    dialog._import()

    assert dialog.result() == QtWidgets.QDialog.Accepted
    np.testing.assert_allclose(dialog.events["NA"], [[0.1, 0.2, 1]])
    assert len(dialog.events["empty"]) == 0


def test_tab_delimited_raven_columns_are_preselected(qtbot, tmp_path):
    filename = tmp_path / "bird.selections.txt"
    filename.write_text("Selection\tChannel\tBegin Time (s)\tEnd Time (s)\tAnnotation\n1\t2\t0.1\t0.2\tcall\n")
    dialog = AnnotationImportDialog(filename, 1000)
    qtbot.addWidget(dialog)
    dialog._import()

    assert dialog.result() == QtWidgets.QDialog.Accepted
    np.testing.assert_allclose(dialog.events["call"], [[0.1, 0.2, 1]])


def test_utf8_bom_in_first_column_does_not_break_mapping(qtbot, tmp_path):
    filename = tmp_path / "frog.selections.csv"
    filename.write_text("\ufeffSelection,Begin Time (s),End Time (s)\n1,0.541088,0.557072\n")
    dialog = AnnotationImportDialog(filename, 44100)
    qtbot.addWidget(dialog)
    assert dialog.frame.columns[0] == "Selection"
    dialog._import()
    assert dialog.result() == QtWidgets.QDialog.Accepted
    np.testing.assert_allclose(dialog.events["1"], [[0.541088, 0.557072, -1]])


@pytest.mark.parametrize("start,stop", [("bad", 1), (-1, 1), (2, 1), (0, np.inf), (np.nan, 1)])
def test_invalid_times_leave_dialog_open(qtbot, tmp_path, start, stop):
    filename = tmp_path / "labels.csv"
    pd.DataFrame({"name": ["call"], "start_seconds": [start], "stop_seconds": [stop]}).to_csv(filename, index=False)
    dialog = AnnotationImportDialog(filename, 1000)
    qtbot.addWidget(dialog)
    dialog._import()

    assert dialog.result() != QtWidgets.QDialog.Accepted
    assert dialog.events is None
    assert dialog.error.text()


def test_sample_import_requires_positive_rate():
    frame = pd.DataFrame({"name": ["call"], "start_seconds": [10], "stop_seconds": [20]})
    with pytest.raises(ValueError, match="positive"):
        dataset_service.annotation_table_to_events(frame, {column: column for column in frame}, "samples", 0)


@pytest.mark.parametrize("column,value", [("channel", 0.5), ("channel", "bad"), ("Channel", 0), ("Channel", "")])
def test_invalid_channels_are_rejected(column, value):
    frame = pd.DataFrame({"name": ["call"], "start_seconds": [0.1], "stop_seconds": [0.2], column: [value]})
    with pytest.raises(ValueError, match="Channels"):
        dataset_service.annotation_table_to_events(
            frame, {key: key for key in ("name", "start_seconds", "stop_seconds")}, "seconds", 1000
        )


def test_mapping_error_can_be_corrected_before_import(qtbot, tmp_path):
    filename = tmp_path / "labels.csv"
    filename.write_text("name,start_seconds,stop_seconds\ncall,0.1,0.2\n")
    dialog = AnnotationImportDialog(filename, 1000)
    qtbot.addWidget(dialog)
    dialog.column_selectors["name"].setCurrentText("start_seconds")
    dialog._import()
    assert "separate name column" in dialog.error.text()
    assert dialog.events is None

    dialog.column_selectors["name"].setCurrentText("name")
    dialog._import()
    assert dialog.result() == QtWidgets.QDialog.Accepted


@pytest.mark.parametrize("mode", ["merge", "replace", "cancel_wizard", "cancel_mode"])
def test_viewer_import_opens_wizard_and_preserves_cancelled_annotations(qtbot, tmp_path, monkeypatch, mode):
    window = gui_app.PSV(_audio_dataset(Events({"old": [[0.1, 0.2, -1]]})))
    qtbot.addWidget(window)
    filename = tmp_path / "labels.csv"
    filename.write_text("label,begin,end\nnew,200,300\n")
    monkeypatch.setattr(gui_app.QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (str(filename), ""))
    monkeypatch.setattr(window, "_choose_annotation_import_mode", lambda filename: "cancel" if mode == "cancel_mode" else mode)
    dialogs = []

    def run_wizard(dialog):
        dialogs.append(dialog)
        assert dialog.sampling_rate == 1000
        if mode == "cancel_wizard":
            return QtWidgets.QDialog.Rejected
        dialog.units.setCurrentText("Samples")
        dialog._import()
        return dialog.result()

    monkeypatch.setattr(AnnotationImportDialog, "exec_", run_wizard)
    window._viewer_actions["import_annotations"].trigger()

    assert len(dialogs) == 1
    annotation_menu = next(menu for menu in window.bar.findChildren(QtWidgets.QMenu) if menu.title() == "Annotations")
    assert window._viewer_actions["import_annotations"] in annotation_menu.actions()
    expected_names = {"old", "new"} if mode == "merge" else {"new"} if mode == "replace" else {"old"}
    assert set(window.event_times.names) == expected_names
    if "new" in expected_names:
        np.testing.assert_allclose(window.event_times["new"], [[0.2, 0.3, -1]])
        assert not window.event_presets["new"].fixed_duration
        assert window.event_presets["new"].duration_editable
        assert "new" in {record.name for record in window.events_table._records_by_id.values()}


def test_import_infers_duration_presets_and_preserves_existing_settings(qtbot, tmp_path, monkeypatch):
    window = gui_app.PSV(_audio_dataset(Events({"existing": [[0.1, 0.1, -1]]})))
    qtbot.addWidget(window)
    existing_preset = window.event_presets["existing"]
    filename = tmp_path / "frog.selections.csv"
    filename.write_text(
        "\ufeffSelection,View,Channel,Begin Time (s),End Time (s),Delta Time (s),Notes\n"
        "1,Waveform 1,1,0.541088,0.557072,0.0160,\n"
        "2,Waveform 1,1,0.598617,0.626943,0.0283,\n"
    )

    def run_wizard(dialog):
        dialog.column_selectors["name"].setCurrentText("View")
        dialog._import()
        return dialog.result()

    monkeypatch.setattr(AnnotationImportDialog, "exec_", run_wizard)
    window.import_annotations(filename=str(filename), mode="merge")

    np.testing.assert_allclose(window.event_times["Waveform 1"], [[0.541088, 0.557072, 0], [0.598617, 0.626943, 0]])
    preset = window.event_presets["Waveform 1"]
    assert not preset.fixed_duration
    assert preset.duration_editable
    assert window.event_presets["existing"] == existing_preset
    np.testing.assert_allclose(window._bounds_for_event_edit("Waveform 1", 0.541088, 0.557072, 0.54, 0.57, "stop"), [0.54, 0.57])

    window._set_imported_event_times(Events({"point": [[0.5, 0.5, -1]]}))
    assert window.event_presets["point"].fixed_duration
    assert window.event_presets["point"].duration_seconds == 0


@pytest.mark.parametrize("kind", ["from_file", "from_dir"])
def test_load_dialog_stages_import_without_writing_source(qtbot, tmp_path, monkeypatch, kind):
    filename = tmp_path / "labels.csv"
    original = "label,begin,end\nnew,200,300\n"
    filename.write_text(original)
    dialog = gui_app.YamlDialog(f"{gui_app.package_dir}/gui/forms/{kind}.yaml")
    qtbot.addWidget(dialog)
    dialog.form["annotation_path"] = str(filename)
    if kind == "from_file":
        dialog.form["samplerate"] = 1000

    def run_wizard(wizard):
        if kind == "from_dir":
            wizard.sample_rate.setValue(1000)
        wizard.units.setCurrentText("Samples")
        wizard._import()
        return wizard.result()

    monkeypatch.setattr(AnnotationImportDialog, "exec_", run_wizard)
    add_annotation_import_button(dialog)
    next(button for button in dialog.findChildren(QtWidgets.QPushButton) if button.text() == "Import annotations...").click()
    ds = _audio_dataset(Events({"old": [[0.1, 0.2, -1]]}))
    ds.attrs["annotation_load_errors"] = [{"filename": str(filename), "error": "wrong columns"}]
    result = finish_annotation_import(ds, dialog)

    assert dataset_service.event_times_from_dataset(result).names == ["new"]
    np.testing.assert_allclose(result.event_times.values, [[0.2, 0.3, -1]])
    assert "annotation_load_errors" not in result.attrs
    assert filename.read_text() == original


@pytest.mark.parametrize("choice", ["wizard", "continue", "cancel", "cancel_wizard"])
def test_open_audio_offers_wizard_for_failed_annotations(qtbot, tmp_path, monkeypatch, choice):
    import soundfile as sf

    audio = tmp_path / "clip.wav"
    sf.write(audio, np.zeros(1000), 1000)
    annotations = tmp_path / "clip_annotations.csv"
    annotations.write_text("label,begin,end\nnew,200,300\n")
    shown = []

    def run_error(message):
        shown.append(message)
        assert message.icon() == QtWidgets.QMessageBox.Critical
        assert str(annotations) in message.text()
        prefix = "Open import" if choice in {"wizard", "cancel_wizard"} else "Continue" if choice == "continue" else "Cancel"
        message.chosen = next(button for button in message.buttons() if button.text().startswith(prefix))
        return 0

    def run_wizard(wizard):
        assert wizard.sampling_rate == 1000
        if choice == "cancel_wizard":
            return QtWidgets.QDialog.Rejected
        wizard.units.setCurrentText("Samples")
        wizard._import()
        return wizard.result()

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", run_error)
    monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton", lambda message: message.chosen)
    monkeypatch.setattr(AnnotationImportDialog, "exec_", run_wizard)
    captured = []
    monkeypatch.setattr(gui_app, "PSV", lambda ds, **kwargs: captured.append(ds) or ds)
    result = gui_app.MainWindow.from_file(str(audio), skip_dialog=True)

    assert len(shown) == 1
    if choice in {"cancel", "cancel_wizard"}:
        assert result is None
        assert captured == []
    elif choice == "wizard":
        np.testing.assert_allclose(result.event_times.values, [[0.2, 0.3, -1]])
    else:
        assert not dataset_service.event_times_from_dataset(result).names


def test_folder_import_sample_rate_can_be_corrected(qtbot, tmp_path):
    filename = tmp_path / "labels.csv"
    filename.write_text("name,start_seconds,stop_seconds\ncall,100,200\n")
    wizard = AnnotationImportDialog(filename, None)
    qtbot.addWidget(wizard)
    wizard.units.setCurrentText("Samples")
    wizard._import()
    assert wizard.events is None
    wizard.sample_rate.setValue(1000)
    wizard._import()
    np.testing.assert_allclose(wizard.events["call"], [[0.1, 0.2, -1]])


@pytest.mark.parametrize("separator,suffix", [(",", ".csv"), ("\t", ".csv"), ("\t", ".tsv"), ("\t", ".selections.txt")])
def test_raven_tables_load_without_wizard(tmp_path, separator, suffix):
    filename = tmp_path / ("bird" + suffix)
    pd.DataFrame({"Annotation": ["001"], "Begin Time (s)": [0.1], "End Time (s)": [0.2], "Channel": [2]}).to_csv(
        filename, sep=separator, index=False, encoding="utf-8-sig"
    )
    events = dataset_service.load_annotation_file(str(filename))
    assert events.names == ["001"]
    np.testing.assert_allclose(events["001"], [[0.1, 0.2, 1]])
