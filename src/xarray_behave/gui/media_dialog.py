"""Dialog for assembling a dataset from multiple media files."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from qtpy import QtCore, QtWidgets


AUDIO_COLUMNS = ("Name", "File", "Timestamp file", "Offset (s)", "Sample rate (Hz)", "Dataset/key")
VIDEO_COLUMNS = ("Name", "File", "Timestamp file", "Offset (s)", "FPS")


class MediaFilesDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, files=None):
        super().__init__(parent)
        self.setWindowTitle("New dataset from media files")
        self.resize(1050, 560)

        layout = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel("The first audio is the reference clock. Positive offsets move a source later.", self)
        layout.addWidget(info)

        self.audio_table = self._make_table(AUDIO_COLUMNS)
        self.video_table = self._make_table(VIDEO_COLUMNS)
        layout.addWidget(self._media_group("Audio", self.audio_table, "audio"), 1)
        layout.addWidget(self._media_group("Video", self.video_table, "video"), 1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if files:
            self.populate_from_manifest(files)

    @staticmethod
    def _make_table(columns):
        table = QtWidgets.QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        for index in (1, 2):
            table.horizontalHeader().setSectionResizeMode(index, QtWidgets.QHeaderView.Stretch)
        return table

    def _media_group(self, title, table, kind):
        group = QtWidgets.QGroupBox(title, self)
        layout = QtWidgets.QVBoxLayout(group)
        layout.addWidget(table)

        row = QtWidgets.QHBoxLayout()
        add_button = QtWidgets.QPushButton(f"Add {title.lower()} files...", group)
        add_button.clicked.connect(lambda: self._choose_files(kind))
        remove_button = QtWidgets.QPushButton("Remove selected", group)
        remove_button.clicked.connect(lambda: self._remove_selected(table))
        timestamp_button = QtWidgets.QPushButton("Set timestamp file...", group)
        timestamp_button.clicked.connect(lambda: self._choose_timestamp(table))
        row.addWidget(add_button)
        row.addWidget(remove_button)
        row.addWidget(timestamp_button)
        row.addStretch(1)
        layout.addLayout(row)
        return group

    def _choose_files(self, kind):
        if kind == "audio":
            file_filter = (
                "Audio and array files (*.wav *.aif *.mp3 *.flac *.h5 *.hdf5 *.hdfs *.npy *.npz *.mmap);;" "All files (*)"
            )
        else:
            file_filter = "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)"
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, f"Add {kind} files", "", file_filter)
        if kind == "audio":
            self.add_audio_files(paths)
        else:
            self.add_video_files(paths)

    def _choose_timestamp(self, table):
        row = table.currentRow()
        if row < 0:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select timestamp file",
            "",
            "Timestamp files (*.csv *.h5 *.hdf5 *.hdfs);;All files (*)",
        )
        if path:
            table.item(row, 2).setText(path)

    @staticmethod
    def _remove_selected(table):
        rows = sorted({index.row() for index in table.selectionModel().selectedRows()}, reverse=True)
        for row in rows:
            table.removeRow(row)

    def add_audio_files(self, paths):
        for path in paths:
            self._add_path(self.audio_table, path, self._infer_audio_dataset(path))

    def add_video_files(self, paths):
        for path in paths:
            self._add_path(self.video_table, path)

    def populate_from_manifest(self, files):
        self.audio_table.setRowCount(0)
        self.video_table.setRowCount(0)
        timestamps = files.get("timestamps", {})
        for name, entry in files.get("audio", {}).items():
            self._add_manifest_rows(self.audio_table, name, entry, timestamps)
        for name, entry in files.get("video", {}).items():
            self._add_manifest_rows(self.video_table, name, entry, timestamps)

    def _add_manifest_rows(self, table, name, entry, timestamps):
        entry_data = entry if isinstance(entry, Mapping) else {}
        timestamp = entry_data.get("timestamp_path") or self._entry_path(timestamps.get(name, {}))
        rate_key = "sampling_rate_Hz" if table is self.audio_table else "frame_rate_Hz"
        for index, path in enumerate(self._entry_paths(entry)):
            dataset = entry_data.get("audio_dataset", "") if table is self.audio_table else ""
            if table is self.audio_table and not dataset:
                dataset = self._infer_audio_dataset(path)
            self._add_path(
                table,
                path,
                dataset,
                name=name if index == 0 else None,
                timestamp=timestamp,
                offset=entry_data.get("offset_seconds", "0"),
                rate=entry_data.get(rate_key, ""),
            )

    def _add_path(self, table, path, dataset="", name=None, timestamp=None, offset="0", rate=""):
        path = str(Path(path).expanduser())
        if any(table.item(row, 1).text() == path for row in range(table.rowCount())):
            return
        name = self._unique_name(name or Path(path).stem)
        values = [name, path, self._timestamp_sidecar(path) if timestamp is None else timestamp, offset, rate]
        if table is self.audio_table:
            values.append(dataset)

        row = table.rowCount()
        table.insertRow(row)
        for column, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(str(value))
            if column == 1:
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            table.setItem(row, column, item)

    @staticmethod
    def _entry_path(entry):
        if isinstance(entry, str):
            return entry
        if not isinstance(entry, Mapping):
            return ""
        if "path" in entry:
            return entry["path"]
        paths = entry.get("paths", [])
        return paths[0] if paths else ""

    @staticmethod
    def _entry_paths(entry):
        if isinstance(entry, str):
            return [entry]
        if not isinstance(entry, Mapping):
            return []
        if "paths" in entry:
            return entry["paths"]
        if "path" in entry:
            return [entry["path"]]
        return []

    def _unique_name(self, stem):
        name = re.sub(r"\W+", "_", stem).strip("_").lower() or "media"
        if name[0].isdigit():
            name = f"media_{name}"
        existing = {
            table.item(row, 0).text() for table in (self.audio_table, self.video_table) for row in range(table.rowCount())
        }
        base = name
        suffix = 2
        while name in existing:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    @staticmethod
    def _timestamp_sidecar(path):
        path = Path(path)
        if path.name.lower().endswith("_daq.h5"):
            return ""
        for suffix in ("_timestamps.h5", "_timeStamps.h5", "_timestamps.csv"):
            candidate = path.with_name(path.stem + suffix)
            if candidate.exists():
                return str(candidate)
        return ""

    @staticmethod
    def _infer_audio_dataset(path):
        path = Path(path)
        if path.name.lower().endswith("_daq.h5"):
            return "samples"
        try:
            if path.suffix.lower() == ".npz":
                with np.load(path) as file:
                    candidates = [
                        name for name in file.files if name not in {"samplerate", "samplerate_Hz"} and file[name].ndim in (1, 2)
                    ]
            elif path.suffix.lower() in {".h5", ".hdf5", ".hdfs"}:
                import h5py

                with h5py.File(path, "r") as file:
                    candidates = []

                    def visit(name, value):
                        if isinstance(value, h5py.Dataset) and value.ndim in (1, 2):
                            candidates.append(name)

                    file.visititems(visit)
            else:
                return ""
        except (OSError, ValueError):
            return ""
        if "data" in candidates:
            return "data"
        return candidates[0] if len(candidates) == 1 else ""

    def media_data(self):
        if self.audio_table.rowCount() == 0:
            raise ValueError("Add at least one audio file.")
        audio = [self._row_data(self.audio_table, row, "audio") for row in range(self.audio_table.rowCount())]
        video = [self._row_data(self.video_table, row, "video") for row in range(self.video_table.rowCount())]
        names = [item["name"] for item in audio + video]
        if len(names) != len(set(names)):
            raise ValueError("Media names must be unique.")
        return {"audio": audio, "video": video}

    @staticmethod
    def _row_data(table, row, kind):
        name = table.item(row, 0).text().strip()
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise ValueError(f"Invalid media name {name!r}; use letters, numbers, and underscores.")
        path = Path(table.item(row, 1).text()).expanduser()
        if not path.is_file():
            raise ValueError(f"Media file does not exist: {path}")
        timestamp = table.item(row, 2).text().strip()
        if timestamp and not Path(timestamp).expanduser().is_file():
            raise ValueError(f"Timestamp file does not exist: {timestamp}")
        try:
            offset = float(table.item(row, 3).text())
        except ValueError as exc:
            raise ValueError(f"Offset for {name!r} must be a number.") from exc
        if not np.isfinite(offset):
            raise ValueError(f"Offset for {name!r} must be finite.")

        rate_text = table.item(row, 4).text().strip()
        rate = None
        if rate_text:
            try:
                rate = float(rate_text)
            except ValueError as exc:
                raise ValueError(f"Rate for {name!r} must be a number.") from exc
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError(f"Rate for {name!r} must be positive.")

        data = {"name": name, "path": str(path), "offset_seconds": offset}
        if timestamp:
            data["timestamp_path"] = str(Path(timestamp).expanduser())
        if rate is not None:
            data["sampling_rate_Hz" if kind == "audio" else "frame_rate_Hz"] = rate
        if kind == "audio":
            dataset = table.item(row, 5).text().strip()
            if dataset:
                data["audio_dataset"] = dataset
        return data

    def accept(self):
        try:
            self.media_data()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid media selection", str(exc))
            return
        super().accept()
