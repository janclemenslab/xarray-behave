"""Preview and map columns in text annotation tables."""

import logging
from pathlib import Path

import pandas as pd
from qtpy import QtWidgets

from .. import _dataset_service as dataset_service

logger = logging.getLogger(__name__)


class AnnotationImportDialog(QtWidgets.QDialog):
    def __init__(self, filename, sampling_rate, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import annotations")
        self.resize(800, 480)
        self.sampling_rate = sampling_rate
        self.events = None
        self.frame = pd.read_csv(filename, sep=None, engine="python", encoding="utf-8-sig", dtype=str, keep_default_na=False)

        layout = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(f"{Path(filename).name} — first {min(10, len(self.frame))} of {len(self.frame)} rows", self)
        layout.addWidget(info)
        self.preview = QtWidgets.QTableWidget(min(10, len(self.frame)), len(self.frame.columns), self)
        self.preview.setHorizontalHeaderLabels(list(self.frame.columns))
        self.preview.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for row, values in enumerate(self.frame.head(10).itertuples(index=False, name=None)):
            for column, value in enumerate(values):
                self.preview.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
        self.preview.resizeColumnsToContents()
        layout.addWidget(self.preview, 1)

        form = QtWidgets.QFormLayout()
        self.column_selectors = {}
        for index, (key, label, defaults) in enumerate([
            ("name", "Name column", ("name", "Annotation")),
            ("start_seconds", "Start column", ("start_seconds", "start", "Begin Time (s)")),
            ("stop_seconds", "Stop column", ("stop_seconds", "stop", "End Time (s)")),
        ]):
            selector = QtWidgets.QComboBox(self)
            selector.addItems(list(self.frame.columns))
            selector.setCurrentIndex(min(index, len(self.frame.columns) - 1))
            for default in defaults:
                if default in self.frame.columns:
                    selector.setCurrentText(default)
                    break
            self.column_selectors[key] = selector
            form.addRow(label, selector)
        self.units = QtWidgets.QComboBox(self)
        self.units.addItem("Seconds", "seconds")
        self.units.addItem("Samples", "samples")
        form.addRow("Start / stop units", self.units)
        if sampling_rate is None:
            self.sample_rate = QtWidgets.QDoubleSpinBox(self)
            self.sample_rate.setRange(0, 10_000_000)
            self.sample_rate.setSuffix(" Hz")
            form.addRow("Audio sample rate (required for samples)", self.sample_rate)
        else:
            form.addRow("Sample rate", QtWidgets.QLabel(f"{sampling_rate:g} Hz (active audio; samples start at 0)", self))
        layout.addLayout(form)

        self.error = QtWidgets.QLabel(self)
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=self)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Import")
        buttons.accepted.connect(self._import)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _import(self):
        if hasattr(self, "sample_rate"):
            self.sampling_rate = self.sample_rate.value()
        columns = {key: selector.currentText() for key, selector in self.column_selectors.items()}
        try:
            self.events = dataset_service.annotation_table_to_events(
                self.frame, columns, self.units.currentData(), self.sampling_rate
            )
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self.accept()


def open_annotation_import(filename, sampling_rate, parent=None):
    try:
        if Path(filename).suffix.lower() in {".csv", ".tsv", ".txt"}:
            wizard = AnnotationImportDialog(filename, sampling_rate, parent)
            return wizard.events if wizard.exec_() == QtWidgets.QDialog.Accepted else None
        return dataset_service.load_annotation_file(filename)
    except Exception as exc:
        logger.exception("Could not import annotations from %s", filename)
        QtWidgets.QMessageBox.warning(parent, "Import annotations", str(exc))
        return None


def add_annotation_import_button(dialog):
    dialog.annotation_import = None
    button = QtWidgets.QPushButton("Import annotations...", dialog)
    layout = dialog.form.form_layout
    row, _ = layout.getWidgetPosition(dialog.form.fields["annotation_path"])
    layout.insertRow(row + 2, "", button)

    def import_annotations():
        filename = dialog.form["annotation_path"].strip()
        if not filename or not Path(filename).is_file():
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                dialog, "Import annotations", filename, "Annotation files (*.csv *.tsv *.txt *.zarr *.mat);;All files (*)"
            )
        if not filename:
            return
        rate = dialog.form.get_form_data().get("samplerate")
        events = open_annotation_import(filename, rate, dialog)
        if events is not None:
            dialog.form["annotation_path"] = filename
            dialog.annotation_import = (filename, rate, events)
            button.setText(f"Import annotations... ({len(events)} types ready)")

    button.clicked.connect(import_annotations)


def finish_annotation_import(ds, dialog):
    staged = getattr(dialog, "annotation_import", None)
    errors = ds.attrs.pop("annotation_load_errors", [])
    if staged is not None and staged[:2] == (
        dialog.form["annotation_path"], dialog.form.get_form_data().get("samplerate")
    ):
        ds = dataset_service.replace_event_times(ds, staged[2])
        ds.attrs["event_times"] = staged[2]
        errors = [error for error in errors if error["filename"] != staged[0]]
    for error in errors:
        message = QtWidgets.QMessageBox(dialog)
        message.setIcon(QtWidgets.QMessageBox.Critical)
        message.setWindowTitle("Could not load annotations")
        message.setText(f"Could not load annotations from {error['filename']}.")
        message.setInformativeText(error["error"])
        wizard_button = message.addButton("Open import wizard...", QtWidgets.QMessageBox.ActionRole)
        continue_button = message.addButton("Continue without these annotations", QtWidgets.QMessageBox.AcceptRole)
        message.addButton(QtWidgets.QMessageBox.Cancel)
        message.setDefaultButton(wizard_button)
        message.exec_()
        if message.clickedButton() is wizard_button:
            rate = ds.song_raw.attrs["sampling_rate_Hz"] if "song_raw" in ds else None
            events = open_annotation_import(error["filename"], rate, dialog)
            if events is None:
                return None
            events = dataset_service.merge_event_times(dataset_service.event_times_from_dataset(ds), events)
            ds = dataset_service.replace_event_times(ds, events)
            ds.attrs["event_times"] = events
        elif message.clickedButton() is not continue_button:
            return None
    return ds
