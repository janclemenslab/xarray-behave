"""PLOT SONG AND PLAY VIDEO IN SYNC

`python -m xarray_behave.ui datename root`
"""

import os
import sys
import logging
import inspect
from pathlib import Path
from functools import partial

import defopt
import h5py
import functools

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.signal
import scipy.signal.windows
from typing import Callable, Optional, List

from qtpy import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

import xarray_behave
from .. import _dataset_service as dataset_service, api as v2_api, xarray_behave as xb, loaders as ld, annot
from .formbuilder import YamlDialog
from .media_dialog import MediaFilesDialog
from .widgets import ChkBxFileDialog, ZarrOverwriteWarning, NoEventsRegisteredWarning
from . import utils, views, event_widgets, gui_config, modern_video, project as project_model
from .style_profile import TEXT_PRIMARY, WINDOW_STYLESHEET

logger = logging.getLogger(__name__)

try:
    from PySide6.QtCore import QUrl
    from PySide6.QtMultimedia import QAudioFormat, QAudioOutput, QAudioSink, QMediaDevices, QMediaPlayer
except Exception:  # pragma: no cover - optional Qt runtime module
    QAudioFormat = None
    QAudioOutput = None
    QAudioSink = None
    QMediaDevices = None
    QMediaPlayer = None
    QUrl = None

try:
    import numba

    pg.setConfigOption("useNumba", True)
except ImportError:
    pass

sys.setrecursionlimit(10**6)  # increase recursion limit to avoid errors when keeping key pressed for a long time
package_dir: str = xarray_behave.__path__[0]


def _get_config_manager() -> gui_config.GuiConfigManager:
    app = QtWidgets.QApplication.instance()
    if app is not None and hasattr(app, "_xarray_behave_config_manager"):
        return app._xarray_behave_config_manager
    manager = gui_config.GuiConfigManager()
    if app is not None:
        app._xarray_behave_config_manager = manager
    return manager


def _set_config_manager(manager: gui_config.GuiConfigManager) -> None:
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app._xarray_behave_config_manager = manager


def _apply_dialog_config(form, manager: gui_config.GuiConfigManager, kind: str) -> None:
    form.set_form_data(manager.dialog_values(kind))


class DataSource:
    def __init__(self, type: str, name: str):
        self.type = type
        self.name = name


class ProjectPanel(QtWidgets.QGroupBox):
    recording_activated = QtCore.Signal(str)
    add_requested = QtCore.Signal()
    edit_requested = QtCore.Signal(str)
    remove_requested = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__("Project", parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.recordings = QtWidgets.QListWidget(self)
        self.recordings.setUniformItemSizes(True)
        self.recordings.itemClicked.connect(self._activate)
        layout.addWidget(self.recordings, 1)

        buttons = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton("Add", self)
        self.edit_button = QtWidgets.QPushButton("Edit/Relink", self)
        self.remove_button = QtWidgets.QPushButton("Remove", self)
        self.add_button.clicked.connect(self.add_requested)
        self.edit_button.clicked.connect(lambda: self._emit_selected(self.edit_requested))
        self.remove_button.clicked.connect(lambda: self._emit_selected(self.remove_requested))
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.edit_button)
        buttons.addWidget(self.remove_button)
        layout.addLayout(buttons)

    def set_project(self, document, current_name: str | None = None) -> None:
        self.setTitle(_project_title(document))
        self.recordings.blockSignals(True)
        self.recordings.clear()
        for recording in document.recordings:
            suffix = " [missing]" if not recording.available else ""
            if recording.annotations_changed:
                suffix += " *"
            item = QtWidgets.QListWidgetItem(recording.name + suffix)
            item.setData(QtCore.Qt.UserRole, recording.name)
            item.setToolTip(str(recording.audio_path))
            if not recording.available:
                item.setForeground(QtGui.QColor("#888888"))
            self.recordings.addItem(item)
            if recording.name == current_name:
                self.recordings.setCurrentItem(item)
        self.recordings.blockSignals(False)

    def set_current_recording(self, name: str) -> None:
        for index in range(self.recordings.count()):
            if self.recordings.item(index).data(QtCore.Qt.UserRole) == name:
                self.recordings.setCurrentRow(index)
                return

    def _selected_name(self) -> str | None:
        item = self.recordings.currentItem()
        return None if item is None else item.data(QtCore.Qt.UserRole)

    def _activate(self, item) -> None:
        self.recording_activated.emit(item.data(QtCore.Qt.UserRole))

    def _emit_selected(self, signal) -> None:
        name = self._selected_name()
        if name is not None:
            signal.emit(name)


def _num_flies(ds):
    return int(ds.sizes["flies"]) if "flies" in ds.sizes else 1


def _apply_cli_bandpass_filter(form, spec_freq_min, spec_freq_max, skip_dialog: bool):
    if not skip_dialog or (spec_freq_min is None and spec_freq_max is None):
        return

    form_data = {"filter_song": "yes"}
    if spec_freq_min is not None:
        form_data["f_low"] = spec_freq_min
    if spec_freq_max is not None:
        form_data["f_high"] = spec_freq_max
    form.set_form_data(form_data)


def _project_title(document: project_model.Project) -> str:
    return document.path.name[: -len(project_model.PROJECT_SUFFIX)] if document.path is not None else "Untitled Project"


def _add_project_annotation_types(document: project_model.Project) -> None:
    configured = document.settings.setdefault("event_types", [])
    known = {item["name"] for item in configured}
    added = False
    for recording in document.recordings:
        for name in recording.annotations.names:
            if name in known:
                continue
            configured.append({"name": name})
            known.add(name)
            added = True
    document.settings = gui_config.sanitize_config(document.settings)
    if added and document.is_saved:
        document.document_changed = True


def _apply_project_cli_settings(manager, events_string="", spec_freq_min=None, spec_freq_max=None) -> None:
    config = gui_config.deep_merge({}, manager.config)
    spectrogram = config.setdefault("viewer", {}).setdefault("spectrogram", {})
    if spec_freq_min is not None:
        spectrogram["fmin"] = spec_freq_min
    if spec_freq_max is not None:
        spectrogram["fmax"] = spec_freq_max
    configured = config.setdefault("event_types", [])
    known = {item["name"] for item in configured}
    for value in events_string.split(";"):
        name = value.strip().split(",", 1)[0]
        if name and name not in known:
            configured.append({"name": name})
            known.add(name)
    manager.config = gui_config.sanitize_config(config)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        parent=None,
        title="Deep Audio Segmenter",
        media_manifest: Optional[str] = None,
        is_das: bool = False,
        project_document: project_model.Project | None = None,
    ):
        super().__init__(parent)

        self.parent = parent
        self.is_das = bool(is_das)
        self.project = project_document
        self.current_recording_name = None
        self.project_panel = None
        self._switching_project = False
        self.config_manager = _get_config_manager()
        media_callback = partial(self.from_media, manifest=media_manifest) if media_manifest else self.from_media

        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtGui.QApplication([])
        self.app.setWindowIcon(QtGui.QIcon(package_dir + "/gui/icon.png"))
        # self.app.setFont(self.font)  # sets app-wide font

        self.resize(400, 200)
        self.setWindowTitle(title)
        self.setWindowIcon(QtGui.QIcon(package_dir + "/gui/icon.png"))
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # build menu
        self.bar = self.menuBar()

        self.file_menu = self.bar.addMenu("File")
        if self.is_das:
            self._add_keyed_menuitem(self.file_menu, "Open audio file", self.new_project_from_file)
            self._add_keyed_menuitem(self.file_menu, "Import folder as project", self.new_project_from_folder)
            self._add_keyed_menuitem(self.file_menu, "Open project", self.open_project)
            if self.project is not None:
                self._add_keyed_menuitem(self.file_menu, "Add recordings", self._add_project_recordings)
                self.file_menu.addSeparator()
                self._add_keyed_menuitem(self.file_menu, "Save Project", self.save_project)
                self._add_keyed_menuitem(self.file_menu, "Save Project As...", self.save_project_as)
        else:
            self._add_keyed_menuitem(self.file_menu, "New from media files", media_callback)
            self._add_keyed_menuitem(self.file_menu, "New from file", self.from_file)
            self._add_keyed_menuitem(self.file_menu, "New from ethodrome folder", self.from_dir)
        self.file_menu.addSeparator()
        self._add_keyed_menuitem(self.file_menu, "Load dataset", self.from_zarr)
        self.file_menu.addSeparator()
        self.file_menu.addAction("Save Configuration As...", self.save_gui_config_as)
        self.file_menu.addAction("Exit", self.close)

        self.das_menu = self.bar.addMenu("DAS")
        self._add_keyed_menuitem(self.das_menu, "Train", self.das_train, None)
        self._add_keyed_menuitem(self.das_menu, "Predict", self.das_predict, None)

        # add initial buttons
        self.hb = QtWidgets.QVBoxLayout()
        if self.project is not None:
            self.project_panel = self._make_project_panel()
            self.hb.addWidget(self.project_panel, 1)
        elif self.is_das:
            self.hb.addWidget(self.add_button("Open audio file", self.new_project_from_file))
            self.hb.addWidget(self.add_button("Import folder as project", self.new_project_from_folder))
            self.hb.addWidget(self.add_button("Open project", self.open_project))
        else:
            self.hb.addWidget(self.add_button("Create dataset from media files", media_callback))
            self.hb.addWidget(self.add_button("Load audio from file", self.from_file))
            self.hb.addWidget(self.add_button("Create dataset from ethodrome folder", self.from_dir))
            self.hb.addWidget(self.add_button("Load dataset (zarr)", self.from_zarr))

        self.cb = pg.GraphicsLayoutWidget()
        self.cb.setLayout(self.hb)
        self.setCentralWidget(self.cb)
        self._restore_window_geometry()

    def closeEvent(self, event):
        if self.project is not None and not self._switching_project:
            self._capture_project_state()
            if self.project.is_dirty:
                choice = QtWidgets.QMessageBox.warning(
                    self,
                    "Unsaved project changes",
                    "Save changes before closing?",
                    QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
                    QtWidgets.QMessageBox.Save,
                )
                if choice == QtWidgets.QMessageBox.Cancel:
                    event.ignore()
                    return
                if choice == QtWidgets.QMessageBox.Save and not self.save_project():
                    event.ignore()
                    return
            self._finish_close(event, save_global=False)
            return
        self._finish_close(event, save_global=True)

    def _finish_close(self, event, *, save_global: bool) -> None:
        if save_global:
            try:
                self.config_manager.save_global(self._config_snapshot())
            except Exception:
                logger.exception("Could not save global GUI configuration to %s", self.config_manager.global_path)
        stuff_to_delete = list(self.__dict__.keys())
        for stuff in stuff_to_delete:
            try:
                del self.__dict__[stuff]
            except KeyError:
                pass
            except Exception as e:
                print(e)
        import gc

        gc.collect()
        event.accept()

    def _config_snapshot(self):
        config = gui_config.deep_merge({}, self.config_manager.config)
        config.pop("selection", None)
        config.pop("event_types", None)
        geometry = self.geometry()
        config["version"] = gui_config.CONFIG_VERSION
        config.setdefault("window", {})["geometry"] = {
            "x": int(geometry.x()),
            "y": int(geometry.y()),
            "width": int(geometry.width()),
            "height": int(geometry.height()),
            "maximized": bool(self.isMaximized()),
        }
        return gui_config.sanitize_config(config)

    def save_gui_config_as(self, qt_keycode=None):
        del qt_keycode
        default_path = (
            gui_config.local_config_path(self.config_manager.source)
            if self.config_manager.source
            else Path.cwd() / gui_config.CONFIG_FILENAME
        )
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save GUI configuration",
            str(default_path),
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if not filename:
            return
        try:
            saved = self.config_manager.save_as(filename, self._config_snapshot())
            logger.info("Saved GUI configuration to %s", saved)
        except Exception:
            logger.exception("Could not save GUI configuration to %s", filename)

    def _restore_window_geometry(self):
        geometry = self.config_manager.config.get("window", {}).get("geometry", {})
        try:
            x = int(geometry["x"])
            y = int(geometry["y"])
            width = max(200, int(geometry["width"]))
            height = max(150, int(geometry["height"]))
        except (KeyError, TypeError, ValueError):
            return
        target = QtCore.QRect(x, y, width, height)
        screens = QtGui.QGuiApplication.screens()
        if screens and not any(screen.availableGeometry().intersects(target) for screen in screens):
            return
        self.setGeometry(target)
        if bool(geometry.get("maximized", False)):
            self.showMaximized()

    def add_button(self, text: str, callback: Callable) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(self)
        button.setText(text)
        button.clicked.connect(callback)
        return button

    @classmethod
    def new_project_from_file(
        cls,
        qt_keycode=None,
        filename: str | None = None,
        events_string: str = "",
        spec_freq_min=None,
        spec_freq_max=None,
    ):
        del qt_keycode
        if not filename:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                None,
                "Open audio file",
                "",
                "Audio files (*.wav *.aif *.aiff *.mp3 *.flac *.ogg *.m4a *.h5 *.hdf5 *.hdfs *.npy *.npz *.mmap);;All files (*)",
            )
        if not filename:
            return None
        manager = _get_config_manager()
        manager.load_for_source(filename)
        _apply_project_cli_settings(manager, events_string, spec_freq_min, spec_freq_max)
        document = project_model.project_from_audio_files([filename], manager.config)
        _add_project_annotation_types(document)
        return MainWindow.from_project(document, config_manager=manager)

    @classmethod
    def new_project_from_folder(
        cls,
        qt_keycode=None,
        dirname: str | None = None,
        events_string: str = "",
        spec_freq_min=None,
        spec_freq_max=None,
    ):
        del qt_keycode
        if not dirname:
            dirname = QtWidgets.QFileDialog.getExistingDirectory(None, "Import folder as project")
        if not dirname:
            return None
        paths = project_model.audio_files_in_folder(dirname)
        if not paths:
            QtWidgets.QMessageBox.warning(None, "Import folder", "No supported audio files were found.")
            return None
        manager = _get_config_manager()
        manager.load_for_source(dirname)
        _apply_project_cli_settings(manager, events_string, spec_freq_min, spec_freq_max)
        document = project_model.project_from_audio_files(paths, manager.config, document_changed=True)
        _add_project_annotation_types(document)
        return MainWindow.from_project(document, config_manager=manager)

    @classmethod
    def open_project(cls, qt_keycode=None, filename: str | None = None):
        del qt_keycode
        if not filename:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                None, "Open project", "", "xarray-behave projects (*.xbp.yaml);;All files (*)"
            )
        if not filename:
            return None
        try:
            document = project_model.read_project(filename)
            manager = _get_config_manager()
            document.settings = manager.load_for_project(filename, document.settings)
            _add_project_annotation_types(document)
            return MainWindow.from_project(document, config_manager=manager)
        except Exception as exc:
            logger.exception("Could not open project %s", filename)
            QtWidgets.QMessageBox.warning(None, "Could not open project", str(exc))
            return None

    @classmethod
    def from_project(
        cls,
        document: project_model.Project,
        recording_name: str | None = None,
        config_manager: gui_config.GuiConfigManager | None = None,
    ):
        manager = config_manager or _get_config_manager()
        manager.config = gui_config.sanitize_config(document.settings)
        recording = None
        if recording_name is not None:
            candidate = document.recording(recording_name)
            if candidate.available:
                recording = candidate
        if recording is None:
            recording = next((item for item in document.recordings if item.available), None)
        if recording is None:
            window = MainWindow(
                title=_project_title(document),
                is_das=True,
                project_document=document,
            )
            window.config_manager = manager
            QtWidgets.QMessageBox.warning(
                window, "Missing media", "No project recording is currently available. Relink one to continue."
            )
            window.show()
            return window
        try:
            ds = dataset_service.assemble_project_recording(recording)
        except Exception as exc:
            logger.exception("Could not open project recording %s", recording.name)
            QtWidgets.QMessageBox.warning(None, "Could not open recording", f"{recording.audio_path}\n\n{exc}")
            return None
        return PSV(
            ds,
            title=f"{_project_title(document)} - {recording.name}",
            data_source=DataSource("project", str(recording.audio_path)),
            config_manager=manager,
            project_document=document,
            recording_name=recording.name,
        )

    def _make_project_panel(self) -> ProjectPanel:
        panel = ProjectPanel(self)
        panel.recording_activated.connect(self._open_project_recording)
        panel.add_requested.connect(self._add_project_recordings)
        panel.edit_requested.connect(self._edit_project_recording)
        panel.remove_requested.connect(self._remove_project_recording)
        panel.set_project(self.project, self.current_recording_name)
        return panel

    def _refresh_project_panel(self) -> None:
        if self.project_panel is not None:
            self.project_panel.set_project(self.project, self.current_recording_name)

    def _capture_project_state(self, *, refresh_project_panel: bool = True) -> None:
        if self.project is None:
            return
        if self.current_recording_name is not None and hasattr(self, "event_times"):
            self.project.recording(self.current_recording_name).set_annotations(self.event_times)
        config = self._config_snapshot()
        config["event_types"] = (
            self._event_type_settings()
            if hasattr(self, "_event_type_settings")
            else self.project.settings.get("event_types", [])
        )
        if self.project.is_saved or len(self.project.recordings) > 1 or self.project.document_changed:
            self.project.set_settings(config)
        else:
            self.project.settings = config
        if refresh_project_panel:
            self._refresh_project_panel()

    def _add_project_recordings(self, qt_keycode=None) -> None:
        del qt_keycode
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Add recordings",
            "",
            "Audio files (*.wav *.aif *.aiff *.mp3 *.flac *.ogg *.m4a *.h5 *.hdf5 *.hdfs *.npy *.npz *.mmap);;All files (*)",
        )
        if not paths:
            return
        self._capture_project_state()
        added = self.project.add_recordings(paths)
        _add_project_annotation_types(self.project)
        self._refresh_project_panel()
        if self.current_recording_name is None and added:
            self._open_project_recording(added[0].name)

    def _edit_project_recording(self, name: str) -> None:
        recording = self.project.recording(name)
        files = {
            "audio": {recording.name: dict(recording.audio)},
            "video": {video.get("name", f"camera_{index + 1}"): dict(video) for index, video in enumerate(recording.videos)},
        }
        dialog = MediaFilesDialog(self, files=files)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        try:
            media = dialog.media_data()
            if len(media["audio"]) != 1:
                raise ValueError("A project recording requires exactly one primary audio file.")
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Edit recording", str(exc))
            return
        audio = dict(media["audio"][0])
        audio.pop("name", None)
        recording.audio = audio
        recording.videos = media["video"]
        self.project.document_changed = True
        self._refresh_project_panel()

    def _remove_project_recording(self, name: str) -> None:
        confirmed = QtWidgets.QMessageBox.question(
            self,
            "Remove recording",
            f"Remove '{name}' from the project? Media files will not be deleted.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if confirmed != QtWidgets.QMessageBox.Yes:
            return
        self._capture_project_state()
        was_current = name == self.current_recording_name
        self.project.remove_recording(name)
        self._refresh_project_panel()
        if was_current:
            replacement = MainWindow.from_project(self.project, config_manager=self.config_manager)
            if replacement is not None:
                self.app._xarray_behave_mainwin = replacement
                self._switching_project = True
                self.close()

    def _open_project_recording(self, name: str) -> None:
        if name == self.current_recording_name:
            return
        recording = self.project.recording(name)
        if not recording.available:
            QtWidgets.QMessageBox.warning(self, "Missing media", f"Relink the missing audio file for '{name}' first.")
            return
        self._capture_project_state(refresh_project_panel=False)
        try:
            ds = dataset_service.assemble_project_recording(recording)
        except Exception as exc:
            logger.exception("Could not open project recording %s", recording.name)
            QtWidgets.QMessageBox.warning(self, "Could not open recording", f"{recording.audio_path}\n\n{exc}")
            return
        self._load_project_recording(ds, recording)

    def save_project(self, qt_keycode=None) -> bool:
        del qt_keycode
        if self.project is None:
            return False
        if self.project.path is None:
            return self.save_project_as()
        self._capture_project_state()
        try:
            saved = project_model.write_project(self.project.path, self.project)
            logger.info("Saved project to %s", saved)
            if self.current_recording_name is not None:
                self.setWindowTitle(f"{_project_title(self.project)} - {self.current_recording_name}")
            self._refresh_project_panel()
            return True
        except Exception as exc:
            logger.exception("Could not save project")
            QtWidgets.QMessageBox.warning(self, "Could not save project", str(exc))
            return False

    def save_project_as(self, qt_keycode=None) -> bool:
        del qt_keycode
        if self.project is None:
            return False
        if self.project.path is not None:
            default_path = self.project.path
        elif self.project.recordings:
            audio = self.project.recordings[0].audio_path
            default_path = audio.with_name(audio.stem + project_model.PROJECT_SUFFIX)
        else:
            default_path = Path.cwd() / ("project" + project_model.PROJECT_SUFFIX)
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save project", str(default_path), "xarray-behave projects (*.xbp.yaml)"
        )
        if not filename:
            return False
        self._capture_project_state()
        try:
            saved = project_model.write_project(filename, self.project)
            self.config_manager.source = str(saved)
            logger.info("Saved project to %s", saved)
            if self.current_recording_name is not None:
                self.setWindowTitle(f"{_project_title(self.project)} - {self.current_recording_name}")
            self._refresh_project_panel()
            return True
        except Exception as exc:
            logger.exception("Could not save project")
            QtWidgets.QMessageBox.warning(self, "Could not save project", str(exc))
            return False

    def _add_keyed_menuitem(
        self,
        parent,
        label: str,
        callback,
        qt_keycode=None,
        checkable=False,
        checked=True,
    ):
        """Add new action to menu and register key press."""
        menuitem = parent.addAction(label)
        menuitem.setCheckable(checkable)
        menuitem.setChecked(checked)
        if qt_keycode is not None:
            menuitem.setShortcut(qt_keycode)
        menuitem.triggered.connect(lambda: callback(qt_keycode))
        return menuitem

    def _get_filename_from_ds(self, suffix: str):
        try:
            if "filebase" in self.ds.attrs:
                savefilename = Path(f"{self.ds.attrs['filebase']}{suffix}")
            else:
                savefilename = Path(
                    self.ds.attrs["root"],
                    self.ds.attrs["res_path"],
                    self.ds.attrs["datename"],
                    f"{self.ds.attrs['datename']}{suffix}",
                )
        except KeyError:
            savefilename = Path("")
        return str(savefilename)

    def save_swaps(self, qt_keycode=None):
        savefilename = self._get_filename_from_ds(suffix="_idswaps.txt")
        savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save swaps to",
            str(savefilename),
            filter="txt files (*.txt);;all files (*)",
        )
        if len(savefilename):
            logger.info(f"   Saving list of swap indices to {savefilename}.")
            os.makedirs(os.path.dirname(savefilename), exist_ok=True)
            np.savetxt(savefilename, self.swap_events, fmt="%f %d %d", header="index fly1 fly2")
            logger.info("Done.")

    def save_definitions(self, qt_keycode=None):
        savefilename = self._get_filename_from_ds(suffix="_definitions.csv")
        savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            caption="Save definitions to",
            dir=str(savefilename),
            filter="CSV files (*_definitions.csv);;all files (*)",
        )
        if len(savefilename):
            # get defs from annot and save them to csv
            logger.info(f"   Saving definitions to {savefilename}.")
            defs = [[key, val] for key, val in annot.Events(self.event_times).categories.items()]
            os.makedirs(os.path.dirname(savefilename), exist_ok=True)
            np.savetxt(savefilename, defs, delimiter=",", fmt="%s")
            logger.info("Done.")

    def save_annotations(self, qt_keycode=None):
        """Save annotations to csv.
        Each annotation is a row with name, start_seconds, stop_seconds.
        start_seconds = stop_seconds for events like pulses."""
        savefilename = self._get_filename_from_ds(suffix="_annotations.csv")

        # TODO: Add explanatory text to ChkBxFileDialog
        dialog = ChkBxFileDialog(
            caption="Save annotations to",
            checkbox_titles=[
                "Save definitions to separate file",
                "Preserve empty",
                "Save channel information",
            ],
            directory=savefilename,
        )
        dialog.set_checked("Save channel information", True)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return False
        savefilename = dialog.selectedUrls()[0].toLocalFile()
        if not savefilename:
            return False

        logger.info(f"   Saving annotations to {savefilename}.")
        self.export_to_csv(
            savefilename,
            preserve_empty=dialog.checked("Preserve empty"),
            with_channels=dialog.checked("Save channel information"),
        )
        self._saved_annotations = self._annotation_snapshot()
        logger.info("Done.")

        if dialog.checked("Save definitions to separate file"):
            self.save_definitions()
        return True

    def export_to_csv(
        self,
        savefilename: str = None,
        start_seconds: float = 0,
        end_seconds: float = np.inf,
        which_events: Optional[List[str]] = None,
        match_to_samples: bool = False,
        preserve_empty: bool = True,
        with_channels: bool = True,
        qt_keycode=None,
    ):
        """[summary]

        Args:
            savefilename (str, optional): [description]. Defaults to None.
            start_seconds (int, optional): [description]. Defaults to 0.
            end_seconds (float, optional): [description]. Defaults to None.
            which_events (float, optional): [description]. Defaults to None.
            match_to_samples (bool, optiona): Will adjust seconds so that seconds * samplerate yields the correct index.
                                              Otherwise, seconds will correspond to the correct time stamp of the event sample.
                                              Only relevant for xb.datasets with timestamp info.
                                              Defaults to False.
            preserve_empty (bool, optional): Preserve event names without annotations. Defaults to True.
            with_channels (bool, optional): Add channels column to file. Defaults to True.
            qt_keycode ([type], optional): [description]. Defaults to None.
        """
        if which_events is None:
            which_events = self.event_times.names
        samplerate = self.fs_song
        event_times = annot.Events(self.event_times)
        for name in event_times.names:
            if name not in which_events:
                event_times.delete_name(name)
            else:
                if match_to_samples:
                    idx = utils.find_nearest_idx(self.ds.sampletime, event_times[name])
                    expected_time = idx / samplerate
                    error = expected_time - event_times[name]
                    times_correct = self.ds.sampletime.data[idx] + error
                    event_times[name] = times_correct
                event_times[name] = event_times.filter_range(name, start_seconds, end_seconds) - start_seconds

        df = event_times.to_df(preserve_empty=preserve_empty, with_channels=with_channels)
        df = df.sort_values(by="start_seconds", ascending=True, ignore_index=True)
        os.makedirs(os.path.dirname(savefilename), exist_ok=True)
        df.to_csv(savefilename, index=False)

    def _add_das_prediction_rows(self, rows, suffix: str = "", time_offset_seconds: float = 0.0) -> int:
        if rows is None or len(rows) == 0:
            return 0

        prediction_rows = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        added = 0
        for row in prediction_rows.itertuples(index=False):
            name = str(row.name)
            start_seconds = float(row.start_seconds) + time_offset_seconds
            stop_seconds = float(row.stop_seconds) + time_offset_seconds
            if not np.isfinite(start_seconds) or not np.isfinite(stop_seconds):
                continue

            category = "event"

            self.event_times.add_time(name + suffix, start_seconds, stop_seconds, category=category)
            added += 1
        return added

    def _refresh_annotations_after_das(self):
        self.nb_eventtypes = len(self.event_times)
        self.eventtype_colors = utils.make_colors(self.nb_eventtypes)
        self.update_eventtype_selector()
        refresh = getattr(self, "_update_xy_with_event_table_refresh", None) or getattr(self, "update_xy", None)
        if refresh is not None:
            refresh()

    def _has_current_das_audio(self) -> bool:
        if not hasattr(self, "ds"):
            return False
        if hasattr(self, "_audio_dataarray_for_source"):
            return self._audio_dataarray_for_source() is not None
        return hasattr(self.ds, "song_raw")

    def _das_audio_and_times(self):
        if hasattr(self, "_audio_dataarray_for_source"):
            return self._audio_dataarray_for_source().data, self._audio_time_values()
        return self.ds.song_raw.data, np.asarray(self.ds.sampletime.values)

    def _das_current_audio(self, start_seconds: float, stop_seconds: float | None):
        if not self._has_current_das_audio():
            raise ValueError("No current audio is loaded.")

        audio_data, sampletime = self._das_audio_and_times()
        start_index = utils.find_nearest_idx(sampletime, start_seconds)
        end_index = None if stop_seconds is None else utils.find_nearest_idx(sampletime, stop_seconds)
        audio = audio_data[start_index:end_index]
        try:
            audio = audio.compute()
        except AttributeError:
            pass

        return (
            np.asarray(audio),
            int(round(self.fs_song)),
            float(sampletime[start_index]),
        )

    def _das_current_audio_duration(self) -> float:
        if not self._has_current_das_audio():
            raise ValueError("No current audio is loaded.")

        _, sampletime = self._das_audio_and_times()
        if sampletime.size == 0:
            raise ValueError("Current audio has no samples.")
        if sampletime.size == 1:
            return float(sampletime[0])

        step = float(np.median(np.diff(sampletime)))
        return float(sampletime[-1] + step)

    def _das_annotated_regions(self) -> list[tuple[float, float]]:
        if not hasattr(self, "event_times"):
            raise ValueError("No annotations are loaded.")

        rows = annot.Events(self.event_times).to_df(preserve_empty=False, with_channels=False)
        regions: list[tuple[float, float]] = []
        for row in rows.itertuples(index=False):
            start_seconds = float(row.start_seconds)
            stop_seconds = float(row.stop_seconds)
            if not np.isfinite(start_seconds) or not np.isfinite(stop_seconds):
                continue
            start_seconds, stop_seconds = sorted((start_seconds, stop_seconds))
            regions.append((start_seconds, stop_seconds))

        if not regions:
            raise ValueError("No annotated regions are available.")
        return regions

    def _handle_das_predictions(self, annotations, time_offset_seconds: float):
        added = self._add_das_prediction_rows(
            annotations,
            suffix="_proposals",
            time_offset_seconds=time_offset_seconds,
        )
        if added == 0:
            logger.warning("Found no song.")
            return
        logger.info(f"   Added {added} predicted annotations.")
        self._refresh_annotations_after_das()

    def _open_das_window(self, initial_tab: str, *, use_current_audio: bool = False):
        try:
            from das.gui_app import DASConformerWindow
        except ImportError as e:
            logger.exception(e)
            logger.info("   Failed to import das-conformer. Install it in the current environment to use DAS.")
            return None

        current_audio_provider = self._das_current_audio if use_current_audio and self._has_current_das_audio() else None
        on_predictions = self._handle_das_predictions if current_audio_provider is not None else None
        window_kwargs = {
            "initial_tab": initial_tab,
            "current_audio_provider": current_audio_provider,
            "on_predictions": on_predictions,
            "parent": self,
        }
        das_signature = inspect.signature(DASConformerWindow)
        accepts_extra_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in das_signature.parameters.values()
        )
        if current_audio_provider is not None and (
            accepts_extra_kwargs or "current_duration_provider" in das_signature.parameters
        ):
            window_kwargs["current_duration_provider"] = self._das_current_audio_duration
        if current_audio_provider is not None and (
            accepts_extra_kwargs or "annotated_region_provider" in das_signature.parameters
        ):
            window_kwargs["annotated_region_provider"] = self._das_annotated_regions
        window = DASConformerWindow(**window_kwargs)
        window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        if not hasattr(self, "_das_windows"):
            self._das_windows = []
        self._das_windows.append(window)

        def forget_window(*_args, das_window=window):
            try:
                windows = getattr(self, "_das_windows", None)
            except RuntimeError:
                return
            if windows is not None and das_window in windows:
                windows.remove(das_window)

        window.destroyed.connect(forget_window)
        window.show()
        return window

    def das_train(self, qt_keycode=None):
        del qt_keycode
        self._open_das_window("train", use_current_audio=True)

    def das_predict(self, qt_keycode=None):
        del qt_keycode
        self._open_das_window("predict", use_current_audio=True)

    @classmethod
    def from_media(
        cls,
        qt_keycode=None,
        *,
        manifest: Optional[str] = None,
        datename: str = "",
        root: str = "",
        dat_path: str = "dat",
        res_path: str = "res",
    ):
        del qt_keycode
        files = None
        if manifest is not None:
            try:
                files = v2_api.discover(datename, root=root, dat_path=dat_path, res_path=res_path, manifest=manifest)
            except Exception as exc:
                logger.exception("Could not read media manifest")
                QtWidgets.QMessageBox.warning(None, "Could not read manifest", str(exc))
                return None
        dialog = MediaFilesDialog(files=files)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None
        media_data = dialog.media_data()
        first_audio = media_data["audio"][0]["path"]
        config_manager = _get_config_manager()
        config_manager.load_for_source(first_audio)
        try:
            ds = dataset_service.assemble_from_media(media_data)
        except Exception as exc:
            logger.exception("Could not assemble media dataset")
            QtWidgets.QMessageBox.warning(None, "Could not load media", str(exc))
            return None
        return PSV(
            ds,
            title=f"Media dataset - {Path(first_audio).name}",
            data_source=DataSource("media", first_audio),
            config_manager=config_manager,
        )

    @classmethod
    def from_file(
        cls,
        filename=None,
        app=None,
        qt_keycode=None,
        events_string="",
        spec_freq_min=None,
        spec_freq_max=None,
        target_samplingrate=None,
        skip_dialog: bool = False,
        is_das: bool = False,
        config_manager: Optional[gui_config.GuiConfigManager] = None,
    ):
        if not filename:
            # enable multiple filters: *.h5, *.npy, *.npz, *.wav, *.*
            file_filter = "Any file (*.*);;WAV files (*.wav);;HDF5 files (*.h5 *.hdf5);;NPY files (*.npy);;NPZ files (*.npz)"
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(parent=None, caption="Select file", filter=file_filter)
        if filename:
            if config_manager is None:
                config_manager = _get_config_manager()
                config_manager.load_for_source(filename)
            # infer loader from file name and set default in form
            # infer samplerate (catch error) and set default in form
            samplerate = None  # Hz
            datasets = [""]

            if filename.endswith(".npz"):
                try:
                    # TODO list variable for form
                    with np.load(filename) as file:
                        datasets = list(file.keys())
                        try:
                            samplerate = file["samplerate"]
                        except KeyError:
                            try:
                                samplerate = file["samplerate_Hz"]
                            except KeyError:
                                pass
                except KeyError:
                    logger.info(
                        f"{filename} no sample rate info in NPZ file.Need to save 'samplerate' variable with the audio data. Defaulting to {samplerate}"
                    )
            elif (
                filename.endswith(".h5")
                or filename.endswith(".hdfs")
                or filename.endswith(".hdf5")
                or filename.endswith(".mat")
            ):
                # infer data set (for hdf5) and populate form
                try:
                    # list all data sets in file and add to list
                    with h5py.File(filename, "r") as f:
                        datasets = []
                        f.visit(lambda name: datasets.append("/" + name))
                except:
                    pass
            else:
                try:  # to load as audio file
                    import librosa

                    samplerate = librosa.get_samplerate(filename)
                except:
                    pass

            dialog = YamlDialog(
                yaml_file=package_dir + "/gui/forms/from_file.yaml",
                title=f"Load {filename}",
            )

            dialog.form["target_samplingrate"] = 1_000
            if samplerate is None:
                samplerate = 10_000
                dialog.form["samplerate"] = samplerate
            else:
                samplerate = int(round(float(samplerate)))
                dialog.form["samplerate"] = samplerate
                dialog.form["spec_freq_max"] = samplerate / 2

            dialog.form.fields["data_set"].set_options(datasets)  # add datasets
            dialog.form.fields["data_set"].setValue(datasets[0])  # select first

            # set default filenames based on data file
            annotation_path = os.path.splitext(filename)[0] + "_annotations.csv"
            dialog.form.fields["annotation_path"].setText(annotation_path)  # select first
            definition_path = os.path.splitext(filename)[0] + "_definitions.csv"
            dialog.form.fields["definition_path"].setText(definition_path)  # select first

            _apply_dialog_config(dialog.form, config_manager, "from_file")

            # initialize form data with cli args
            if spec_freq_min is not None:
                dialog.form["spec_freq_min"] = spec_freq_min
            if spec_freq_max is not None:
                dialog.form["spec_freq_max"] = spec_freq_max
            if target_samplingrate is not None:
                dialog.form["target_samplingrate"] = target_samplingrate
            if len(events_string):
                dialog.form["events_string"] = events_string
            _apply_cli_bandpass_filter(dialog.form, spec_freq_min, spec_freq_max, skip_dialog)

            if not skip_dialog:
                dialog.show()
                result = dialog.exec_()
            else:
                result = QtWidgets.QDialog.Accepted

            if result == QtWidgets.QDialog.Accepted:
                form_data = dialog.form.get_form_data()  # why call this twice

                form_data = dialog.form.get_form_data()
                config_manager.remember_dialog("from_file", form_data)
                logger.info(f"Making new dataset from {filename}.")
                # if form_data['target_samplingrate'] is None:
                #     form_data['target_samplingrate'] = None

                ds = dataset_service.assemble_from_file(filename, form_data)
                return PSV(
                    ds,
                    title=filename,
                    fmin=dialog.form["spec_freq_min"],
                    fmax=dialog.form["spec_freq_max"],
                    data_source=DataSource("file", filename),
                    config_manager=config_manager,
                )

    @classmethod
    def from_dir(
        cls,
        dirname=None,
        app=None,
        qt_keycode=None,
        events_string="",
        spec_freq_min=None,
        spec_freq_max=None,
        target_samplingrate=None,
        box_size=None,
        pixel_size_mm=None,
        manifest: Optional[str] = None,
        skip_dialog: bool = False,
        is_das: bool = False,
    ):
        if not dirname:
            dirname = QtWidgets.QFileDialog.getExistingDirectory(parent=None, caption="Select data directory")
        if dirname:
            config_manager = _get_config_manager()
            config_manager.load_for_source(dirname)
            dialog = YamlDialog(
                yaml_file=package_dir + "/gui/forms/from_dir.yaml",
                title=f"Dataset from data directory {dirname}",
            )

            _apply_dialog_config(dialog.form, config_manager, "from_dir")

            # initialize form data with cli args
            if pixel_size_mm is not None:
                dialog.form["pixel_size_mm"] = pixel_size_mm  # and un-disable
            if spec_freq_min is not None:
                dialog.form["spec_freq_min"] = spec_freq_min
            if spec_freq_max is not None:
                dialog.form["spec_freq_max"] = spec_freq_max
            if box_size is not None:
                dialog.form["box_size_px"] = box_size
            if target_samplingrate is not None:
                dialog.form["target_samplingrate"] = target_samplingrate
            if len(events_string):
                dialog.form["init_annotations"] = True
                dialog.form["events_string"] = events_string
            _apply_cli_bandpass_filter(dialog.form, spec_freq_min, spec_freq_max, skip_dialog)

            if not skip_dialog:
                dialog.show()
                result = dialog.exec_()
            else:
                result = QtWidgets.QDialog.Accepted

            if result == QtWidgets.QDialog.Accepted:
                form_data = dialog.form.get_form_data()
                config_manager.remember_dialog("from_dir", form_data)
                logger.info(f"Making new dataset from directory {dirname}.")

                _, datename = os.path.split(os.path.normpath(dirname))  # normpath removes trailing pathsep
                assemble_kwargs = {"pixel_size_mm": pixel_size_mm}
                if manifest is not None:
                    assemble_kwargs["manifest"] = manifest
                ds = dataset_service.assemble_from_dir(dirname, form_data, **assemble_kwargs)

                # add video file
                vr = None
                try:
                    if dialog.form["video_filename"] != "":
                        try:
                            video_filename = dialog.form["video_filename"]
                            vr = modern_video.PyAVVideoReader(video_filename)
                        except:
                            pass
                    else:
                        try:
                            video_filename = os.path.join(dirname, datename + ".mp4")
                            vr = modern_video.PyAVVideoReader(video_filename)
                        except:
                            video_filename = os.path.join(dirname, datename + ".avi")
                            vr = modern_video.PyAVVideoReader(video_filename)
                    logger.info(vr)
                except FileNotFoundError:
                    logger.info(f'Video "{video_filename}" not found. Continuing without.')
                except:
                    logger.info("Something went wrong when loading the video. Continuing without.")

                return PSV(
                    ds,
                    title=dirname,
                    vr=vr,
                    fmin=dialog.form["spec_freq_min"],
                    fmax=dialog.form["spec_freq_max"],
                    frame_fliplr=dialog.form["frame_fliplr"],
                    frame_flipud=dialog.form["frame_flipud"],
                    box_size=dialog.form["box_size_px"],
                    data_source=DataSource("dir", dirname),
                    config_manager=config_manager,
                )

    @classmethod
    def from_zarr(
        cls,
        filename=None,
        app=None,
        qt_keycode=None,
        spec_freq_min=None,
        spec_freq_max=None,
        box_size=None,
        skip_dialog: bool = False,
        is_das: bool = False,
    ):
        if not filename:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(parent=None, caption="Select dataset")
        if filename:
            config_manager = _get_config_manager()
            config_manager.load_for_source(filename)
            dialog = YamlDialog(
                yaml_file=package_dir + "/gui/forms/from_zarr.yaml",
                title=f"Load dataset from zarr file {filename}",
            )

            _apply_dialog_config(dialog.form, config_manager, "from_zarr")

            # initialize form data with cli args
            if spec_freq_min is not None:
                dialog.form["spec_freq_min"] = spec_freq_min
            if spec_freq_max is not None:
                dialog.form["spec_freq_max"] = spec_freq_max
            if box_size is not None:
                dialog.form["box_size"] = box_size
            _apply_cli_bandpass_filter(dialog.form, spec_freq_min, spec_freq_max, skip_dialog)

            if not skip_dialog:
                dialog.show()
                result = dialog.exec_()
            else:
                result = QtWidgets.QDialog.Accepted

            if result == QtWidgets.QDialog.Accepted:
                form_data = dialog.form.get_form_data()
                config_manager.remember_dialog("from_zarr", form_data)
                logger.info(f"Loading {filename}.")
                ds = dataset_service.load_from_zarr(filename, form_data)
                vr = None
                try:
                    video_filename = ds.attrs["video_filename"]
                    vr = modern_video.PyAVVideoReader(video_filename)
                    logger.info(vr)
                except FileNotFoundError:
                    logger.info(f'Video "{video_filename}" not found. Continuing without.')
                except:
                    logger.info("Something went wrong when loading the video. Continuing without.")

                return PSV(
                    ds,
                    vr=vr,
                    title=filename,
                    fmin=dialog.form["spec_freq_min"],
                    fmax=dialog.form["spec_freq_max"],
                    box_size=dialog.form["box_size"],
                    data_source=DataSource("zarr", filename),
                    config_manager=config_manager,
                )

    def save_dataset(self, qt_keycode=None):
        try:
            savefilename = Path(
                self.ds.attrs["root"],
                self.ds.attrs["dat_path"],
                self.ds.attrs["datename"],
                f"{self.ds.attrs['datename']}.zarr",
            )
        except KeyError:
            savefilename = ""

        savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save dataset to",
            str(savefilename),
            filter="zarr files (*.zarr);;all files (*)",
        )

        if len(savefilename):
            file_exists = os.path.exists(savefilename)

            retval = QtWidgets.QMessageBox.Ignore
            if self.data_source.type == "zarr" and file_exists:
                retval = ZarrOverwriteWarning().exec_()

            if retval == QtWidgets.QMessageBox.Ignore:
                self.ds = dataset_service.prepare_for_save(
                    self.ds,
                    self.event_times,
                    original_spatial_units=getattr(self, "original_spatial_units", None),
                )

                logger.info(f"   Saving dataset to {savefilename}.")
                xb.save(savefilename, self.ds)
                logger.info("Done.")
            else:
                logger.info("Saving aborted.")


class PSV(MainWindow):
    MAX_AUDIO_AMP = 3.0

    def __init__(
        self,
        ds,
        vr=None,
        title="xb.ui",
        cmap_name: Optional[str] = None,
        box_size: Optional[int] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        data_source: Optional[DataSource] = None,
        frame_fliplr: Optional[bool] = None,
        frame_flipud: Optional[bool] = None,
        config_manager: Optional[gui_config.GuiConfigManager] = None,
        project_document: project_model.Project | None = None,
        recording_name: str | None = None,
    ):
        super().__init__(title=title, is_das=project_document is not None, project_document=project_document)
        if config_manager is not None:
            self.config_manager = config_manager
        config = self.config_manager.config
        viewer_config = config.get("viewer", {})
        video_config = viewer_config.get("video", {})
        spectrogram_config = viewer_config.get("spectrogram", {})
        audio_config = viewer_config.get("audio", {})
        annotation_config = viewer_config.get("annotations", {})
        threshold_config = viewer_config.get("thresholding", {})
        panel_config = config.get("window", {}).get("panels", {})
        self.setStyleSheet(WINDOW_STYLESHEET)
        pg.setConfigOptions(useOpenGL=False)  # appears to be faster that way
        try:
            import numba

            pg.setConfigOptions(useNumba=True)  # appears to be faster that way
        except ImportError:
            pass
        # build model:
        self.ds = ds
        self.data_source = data_source
        self.current_recording_name = recording_name
        self._audio_source_names = self._discover_audio_source_names()
        self._active_audio_source_name = self._audio_source_names[0] if self._audio_source_names else None
        self._audio_selector_items = []
        self._video_sources = self._discover_video_sources()
        self._active_video_name = next(iter(self._video_sources), None)
        if vr is None and self._active_video_name is not None:
            vr = self._video_reader_for_source(self._active_video_name)
        self.vr = vr

        # detect all event times
        self.event_times = dataset_service.event_times_from_dataset(ds)
        self.event_presets = self._initial_event_presets()
        self._merge_configured_event_types(config.get("event_types", []))
        self._current_event_name = self.event_times.names[-1] if self.event_times.names else None
        self._sync_event_colors_from_presets()
        self._saved_annotations = self._annotation_snapshot()

        self.box_size = int(video_config.get("box_size", 200) if box_size is None else box_size)
        self.fmin = spectrogram_config.get("fmin") if fmin is None else fmin
        self.fmax = spectrogram_config.get("fmax") if fmax is None else fmax
        self.ylim = None
        self.spec_denoise = bool(spectrogram_config.get("denoise", False))
        self.spec_levels = spectrogram_config.get("levels", [None, None])

        self.tmin = 0
        self.fs_song = self._source_sampling_rate(self._active_audio_source_name) or float(
            self.ds.attrs.get("target_sampling_rate_Hz", 1_000)
        )
        self.nb_channels = self._source_channel_count(self._active_audio_source_name)
        audio_length = self._source_length(self._active_audio_source_name)
        if audio_length is not None:
            self.tmax = audio_length
        elif "body_positions" in self.ds:
            self.tmax = len(self.ds.body_positions)
        elif "time" in self.ds:
            self.tmax = len(self.ds.time)
        else:
            self.tmax = 0

        self.crop = bool(video_config.get("crop", True))
        self.maintain_custom_crop = bool(video_config.get("maintain_custom_crop", False))
        try:
            self.pose_center_index = list(self.ds.poseparts).index("thorax")
        except:
            if "poseparts" in self.ds and len(list(self.ds.poseparts)) > 8:
                self.pose_center_index = 8
            else:  # fallback in case poses are a little different
                self.pose_center_index = 0

        self.show_dot = bool(video_config.get("show_dot", "body_positions" in self.ds))
        self.old_show_dot_state = self.show_dot
        self.dot_size = 2
        self.show_poses = bool(video_config.get("show_poses", False))
        self.move_poses = bool(video_config.get("move_poses", False))
        self.circle_size = 8

        self.nb_flies = _num_flies(self.ds)
        self.focal_fly = 0
        self.other_fly = 1 if self.nb_flies > 1 else 0

        if "poseparts" in self.ds:
            self.bodyparts = self.ds.poseparts.data
            self.nb_bodyparts = len(self.ds.poseparts)
        elif "bodyparts" in self.ds:
            self.bodyparts = self.ds.bodyparts.data
            self.nb_bodyparts = len(self.ds.bodyparts)
            self.track_center_index = 1
        else:
            self.nb_bodyparts = 1
            self.bodyparts = None

        self.fly_colors = utils.make_colors(self.nb_flies)
        self.bodypart_colors = utils.make_colors(self.nb_bodyparts)

        self.ds, original_spatial_units = dataset_service.prepare_for_display(self.ds)
        if original_spatial_units is not None:
            self.original_spatial_units = original_spatial_units

        if "swap_events" in self.ds.attrs:
            self.swap_events = self.ds.attrs["swap_events"]
        else:
            self.swap_events = []

        self.STOP = True
        self.show_spec = bool(panel_config.get("spectrogram", True))
        self.show_trace = bool(panel_config.get("waveform", True))
        self.show_tracks = bool(panel_config.get("tracks", False))
        self.show_movie = bool(panel_config.get("movie", True))
        self.show_sidebar = self.project is not None or bool(panel_config.get("sidebar", True))
        self.show_timeline = bool(panel_config.get("timeline", True))
        self.show_event_table = bool(panel_config.get("event_table", True))
        self.show_options = True
        self.show_event_text = bool(annotation_config.get("show_labels", True))
        self.spec_win = max(1, int(spectrogram_config.get("resolution", 200)))
        self.show_songevents = bool(annotation_config.get("show", True))
        self.movable_events = bool(annotation_config.get("movable", True))
        self.edit_only_current_events = bool(annotation_config.get("edit_only_current", False))
        self.audio_channel_settings = event_widgets.AudioChannelSettings(
            waveform_all=bool(audio_config.get("waveform_all", True)),
            events_all=bool(audio_config.get("events_all", True)),
            playback_all=bool(audio_config.get("playback_all", False)),
            scale_y_all=bool(audio_config.get("scale_y_all", True)),
        )
        self.show_all_channels = self.audio_channel_settings.waveform_all
        self.select_loudest_channel = bool(audio_config.get("select_loudest_channel", False))
        self.threshold_mode = bool(threshold_config.get("enabled", False))
        self.sinet0 = None
        self.sinet0_event_name = None

        self.frame_fliplr = bool(video_config.get("frame_fliplr", False) if frame_fliplr is None else frame_fliplr)
        self.frame_flipud = bool(video_config.get("frame_flipud", False) if frame_flipud is None else frame_flipud)

        self.thres_min_dist = float(threshold_config.get("min_distance", 0.020))
        self.thres_env_std = float(threshold_config.get("envelope_std", 0.002))
        self.thres_value = float(threshold_config.get("value", 0.0))
        self.thres_duration_enabled = bool(threshold_config.get("duration_enabled", False))
        self.thres_duration_min = float(threshold_config.get("duration_min", 0.0))
        self.thres_duration_max = float(threshold_config.get("duration_max", 1.0))
        self.thres_bandpass_enabled = bool(threshold_config.get("bandpass_enabled", False))
        self.thres_bandpass_low = float(threshold_config.get("bandpass_low", 0.0))
        self.thres_bandpass_high = threshold_config.get("bandpass_high")

        if "song_events" in self.ds:
            self.fs_other = self.ds.song_events.attrs["sampling_rate_Hz"]
        else:
            self.fs_other = float(ds.attrs.get("target_sampling_rate_Hz", self.fs_song))
        self._apply_active_audio_source()
        if self.thres_bandpass_high is None:
            self.thres_bandpass_high = self.fs_song / 2

        if self.vr is not None:
            self.frame_interval = self.fs_song * self._active_video_frame_seconds()
        else:
            self.frame_interval = self.fs_song / 1_000

        self._span = min(int(self.fs_song), self.tmax)
        self._t0 = 0
        self.step = 1
        self._is_playing = False
        self._is_slider_scrubbing = False
        self._playback_timer = QtCore.QTimer(self)
        self._playback_timer.setInterval(20)
        self._playback_timer.timeout.connect(self._on_playback_tick)
        self._playback_clock = QtCore.QElapsedTimer()
        self._playback_anchor_sample = 0.0
        self._playback_window_start = None
        self._playback_window_stop = None
        self._playback_audio_stop_sample = None
        self._window_audio_timer = QtCore.QTimer(self)
        self._window_audio_timer.setInterval(20)
        self._window_audio_timer.timeout.connect(self._on_window_audio_tick)
        self._window_audio_start_sample = None
        self._window_audio_stop_sample = None
        self._audio_output = None
        self._audio_player = None
        self._array_audio_sink = None
        self._array_audio_buffer = None
        self._array_audio_bytes = None
        self._array_audio_start_sample = None

        self.resize(1000, 800)

        # build UI/controller
        # MENU
        self.file_menu.clear()
        self._viewer_actions = {}
        if self.project is not None:
            self._add_keyed_menuitem(self.file_menu, "Open audio file", self.new_project_from_file)
            self._add_keyed_menuitem(self.file_menu, "Import folder as project", self.new_project_from_folder)
            self._add_keyed_menuitem(self.file_menu, "Open project", self.open_project)
            self._add_keyed_menuitem(self.file_menu, "Add recordings", self._add_project_recordings)
        else:
            self._add_keyed_menuitem(self.file_menu, "New from media files", self.from_media)
            self._viewer_actions["open_audio_annotations"] = self._add_keyed_menuitem(
                self.file_menu, "Open audio/annotations", self.from_file
            )
            self._add_keyed_menuitem(self.file_menu, "New from ethodrome folder", self.from_dir)
        self.file_menu.addSeparator()
        self._add_keyed_menuitem(self.file_menu, "Load dataset", self.from_zarr)
        self.file_menu.addSeparator()
        self._viewer_actions["import_annotations"] = self._add_keyed_menuitem(
            self.file_menu, "Import annotations...", self.import_annotations
        )
        self.file_menu.addSeparator()
        self._add_keyed_menuitem(self.file_menu, "Save swap files", self.save_swaps)
        self._viewer_actions["save_annotations"] = self._add_keyed_menuitem(self.file_menu, "Save", self.save_annotations)
        if self.project is not None:
            self._add_keyed_menuitem(self.file_menu, "Export annotations...", self.export_annotations)
            self._add_keyed_menuitem(self.file_menu, "Save Project As...", self.save_project_as)
        self.file_menu.addSeparator()
        self._add_keyed_menuitem(self.file_menu, "Save dataset", self.save_dataset)
        self.file_menu.addSeparator()
        self.file_menu.addAction("Save Configuration As...", self.save_gui_config_as)
        self.file_menu.addAction("Exit", self.close)

        view_play = self.bar.addMenu("Playback")
        self._add_keyed_menuitem(
            view_play,
            "Play video",
            self.toggle_playvideo,
            "Space",
            checkable=True,
            checked=not self.STOP,
        )
        view_play.addSeparator()
        (self._add_keyed_menuitem(view_play, " < Reverse one frame", self.single_frame_reverse, "Left"),)
        self._add_keyed_menuitem(view_play, "<< Reverse jump", self.jump_reverse, "A")
        self._add_keyed_menuitem(view_play, ">> Forward jump", self.jump_forward, "D")
        self._add_keyed_menuitem(view_play, " > Forward one frame", self.single_frame_advance, "Right")
        view_play.addSeparator()
        self._add_keyed_menuitem(view_play, "Move to previous annotation", self.set_prev_cuepoint, "K")
        self._add_keyed_menuitem(view_play, "Move to next annotation", self.set_next_cuepoint, "L")
        view_play.addSeparator()
        self._add_keyed_menuitem(view_play, "Zoom in song", self.zoom_in_song, "W")
        self._add_keyed_menuitem(view_play, "Zoom out song", self.zoom_out_song, "S")

        view_video = self.bar.addMenu("Video")
        self._add_keyed_menuitem(
            view_video,
            "Crop frame",
            partial(self.toggle, "crop"),
            "C",
            checkable=True,
            checked=self.crop,
        )
        self._add_keyed_menuitem(
            view_video,
            "Maintain custom crop",
            partial(self.toggle, "maintain_custom_crop"),
            None,
            checkable=True,
            checked=self.maintain_custom_crop,
        )
        self._add_keyed_menuitem(
            view_video,
            "Flip frame left-right",
            partial(self.toggle, "frame_fliplr"),
            None,
            checkable=True,
            checked=self.frame_fliplr,
        )
        self._add_keyed_menuitem(
            view_video,
            "Flip frame up-down",
            partial(self.toggle, "frame_flipud"),
            None,
            checkable=True,
            checked=self.frame_flipud,
        )
        self._add_keyed_menuitem(view_video, "Change focal fly", self.change_focal_fly, "F")
        self._add_keyed_menuitem(view_video, "Change other fly", self.change_other_fly, "Z")
        self._add_keyed_menuitem(view_video, "Swap flies", self.swap_flies, "X")
        view_video.addSeparator()
        self._add_keyed_menuitem(
            view_video,
            "Move poses",
            partial(self.toggle, "move_poses"),
            "B",
            checkable=True,
            checked=self.move_poses,
        )
        view_video.addSeparator()
        self._add_keyed_menuitem(
            view_video,
            "Show fly position",
            partial(self.toggle, "show_dot"),
            "O",
            checkable=True,
            checked=self.show_dot,
        )
        self._add_keyed_menuitem(
            view_video,
            "Show poses",
            partial(self.toggle, "show_poses"),
            "P",
            checkable=True,
            checked=self.show_poses,
        )

        view_audio = self.bar.addMenu("Audio")
        self._add_keyed_menuitem(view_audio, "Play waveform through speakers", self.play_audio, "E")
        view_audio.addSeparator()
        self._add_keyed_menuitem(
            view_audio,
            "Show all channels",
            partial(self.toggle, "show_all_channels"),
            None,
            checkable=True,
            checked=self.show_all_channels,
        )
        self._add_keyed_menuitem(
            view_audio,
            "Auto-select loudest channel",
            partial(self.toggle, "select_loudest_channel"),
            "Q",
            checkable=True,
            checked=self.select_loudest_channel,
        )
        self._add_keyed_menuitem(view_audio, "Select previous channel", self.set_next_channel, "Up")
        self._add_keyed_menuitem(view_audio, "Select next channel", self.set_prev_channel, "Down")
        view_audio.addSeparator()
        self._add_keyed_menuitem(
            view_audio,
            "Show spectrogram",
            partial(self.toggle, "show_spec"),
            None,
            checkable=True,
            checked=self.show_spec,
        )
        self._add_keyed_menuitem(view_audio, "Increase frequency resolution", self.inc_freq_res, "R")
        self._add_keyed_menuitem(view_audio, "Increase temporal resolution", self.dec_freq_res, "T")
        view_audio.addSeparator()

        view_annotations = self.bar.addMenu("Annotations")
        self._add_keyed_menuitem(
            view_annotations,
            "Show annotations",
            partial(self.toggle, "show_songevents"),
            "V",
            checkable=True,
            checked=self.show_songevents,
        )
        view_annotations.addSeparator()
        self._add_keyed_menuitem(
            view_annotations,
            "Allow moving annotations",
            partial(self.toggle, "movable_events"),
            "M",
            checkable=True,
            checked=self.movable_events,
        )
        self._add_keyed_menuitem(
            view_annotations,
            "Only edit active event",
            partial(self.toggle, "edit_only_current_events"),
            None,
            checkable=True,
            checked=self.edit_only_current_events,
        )
        self._add_keyed_menuitem(
            view_annotations,
            "Show event labels",
            partial(self.toggle, "show_event_text"),
            None,
            checkable=True,
            checked=self.show_event_text,
        )
        view_annotations.addSeparator()
        self._add_keyed_menuitem(
            view_annotations,
            "Delete active event in view",
            self.delete_current_events,
            "U",
        )
        self._add_keyed_menuitem(
            view_annotations,
            "Delete all events in view",
            self.delete_all_events,
            "Y",
        )
        view_annotations.addSeparator()
        self._viewer_actions["threshold_mode"] = self._add_keyed_menuitem(
            view_annotations,
            "Toggle thresholding mode",
            partial(self.toggle, "threshold_mode"),
            checkable=True,
            checked=self.threshold_mode,
        )
        self._add_keyed_menuitem(
            view_annotations,
            "Generate proposal by envelope thresholding",
            self.threshold,
            "I",
        )
        self._add_keyed_menuitem(view_annotations, "Adjust thresholding mode", self.set_envelope_computation)
        view_annotations.addSeparator()
        self._add_keyed_menuitem(
            view_annotations,
            "Approve proposals for active event in view",
            self.approve_active_proposals,
            "G",
        )
        self._add_keyed_menuitem(
            view_annotations,
            "Approve proposals for all events in view",
            self.approve_all_proposals,
            "H",
        )

        view_view = self.bar.addMenu("View")
        self._viewer_actions["show_sidebar"] = self._add_keyed_menuitem(
            view_view,
            "Show sidebar",
            partial(self.toggle, "show_sidebar"),
            None,
            checkable=True,
            checked=self.show_sidebar,
        )
        # TODO? only show these if tracks and/or video
        self._viewer_actions["show_spec"] = self._add_keyed_menuitem(
            view_view,
            "Show spectrogram",
            partial(self.toggle, "show_spec"),
            None,
            checkable=True,
            checked=self.show_spec,
        )
        self._viewer_actions["show_trace"] = self._add_keyed_menuitem(
            view_view,
            "Show waveform",
            partial(self.toggle, "show_trace"),
            None,
            checkable=True,
            checked=self.show_trace,
        )
        self._viewer_actions["show_timeline"] = self._add_keyed_menuitem(
            view_view,
            "Show event timeline",
            partial(self.toggle, "show_timeline"),
            None,
            checkable=True,
            checked=self.show_timeline,
        )
        self._viewer_actions["show_event_table"] = self._add_keyed_menuitem(
            view_view,
            "Show event table",
            partial(self.toggle, "show_event_table"),
            None,
            checkable=True,
            checked=self.show_event_table,
        )
        if "pose_positions_allo" in self.ds:
            self._add_keyed_menuitem(
                view_view,
                "Show tracks",
                partial(self.toggle, "show_tracks"),
                None,
                checkable=True,
                checked=self.show_tracks,
            )
        if self.vr is not None:
            self._viewer_actions["show_movie"] = self._add_keyed_menuitem(
                view_view,
                "Show movie",
                partial(self.toggle, "show_movie"),
                None,
                checkable=True,
                checked=self.show_movie,
            )

        view_audio.addSeparator()
        self.view_audio = view_audio
        self._build_annotation_toolbar()

        # TRACKS selector
        if "pose_positions_allo" in self.ds and self.bodyparts is not None:
            self.cb3 = utils.CheckableComboBox()
            items = [f"{b}, {c}" for b in self.bodyparts for c in ["x", "y"]]
            self.cb3.addItems(items)

            # color selected track parts in the combo box
            children = self.cb3.children()
            itemList = children[0]
            # repeat colors since we have x and y values for each
            # TODO: discriminate x/y
            self.tracks_colors = []
            for col in self.bodypart_colors:
                self.tracks_colors.append(col)
                self.tracks_colors.append(col)

            for ii, col in zip(range(0, itemList.rowCount()), self.tracks_colors):
                itemList.item(ii).setForeground(QtGui.QColor(*col))

            for index in range(self.cb3.model().rowCount()):
                item = self.cb3.model().item(index)
                item.setCheckState(QtCore.Qt.Unchecked)
            self.cb3.updateText()

            def on_tracksel_changed(source):
                if source.STOP:
                    source.update_xy()

            self.cb3.currentTextChanged.connect(lambda: on_tracksel_changed(self))

        self.movie_view = None
        self.movie_panel = None
        self.video_combo = None
        if self.vr is not None:
            self.movie_view = views.MovieView(model=self, callback=self.on_video_clicked)
            self.movie_panel = self._build_movie_panel()

        self.slice_view = event_widgets.WaveformPane(
            callback=self.on_trace_clicked,
            region_changed_callback=self.on_region_change_finished,
            position_changed_callback=self.on_position_change_finished,
        )
        self.tracks_view = views.TrackView(model=self, callback=self.on_trace_clicked)
        self.event_timeline = event_widgets.EventTimelineWidget(show_waveform=False)
        self.events_table = event_widgets.EventsTableWidget()
        self.cb2 = self.slice_view.channel_combo
        self.slice_view.set_audio_settings(self.audio_channel_settings)
        self._set_channel_selector_items()
        self.threshold_panel = event_widgets.ThresholdingPanel()
        self.threshold_panel.hide()
        self.preset_panel = event_widgets.EventPresetPanel()
        waveform_config = viewer_config.get("waveform", {})
        if "color" in waveform_config:
            self.slice_view.set_waveform_color(str(waveform_config["color"]))
        waveform_limits = waveform_config.get("y_limits")
        if isinstance(waveform_limits, (list, tuple)) and len(waveform_limits) == 2:
            self.slice_view.set_waveform_y_limits(tuple(waveform_limits))
        follow_table = annotation_config.get("table_audio_filter", annotation_config.get("table_audio_link", False))
        self.events_table.window_filter_checkbox.setChecked(bool(follow_table))
        for widget in (self.slice_view, self.tracks_view, self.event_timeline, self.events_table):
            widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.preset_panel.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self._syncing_event_selection = False
        self.slice_view.channel_changed.connect(self._on_audio_selection_changed)
        self.slice_view.audio_settings_changed.connect(self._set_audio_settings)
        self.events_table.window_filter_checkbox.toggled.connect(
            lambda checked: self._refresh_event_widgets(True, force_table=bool(checked))
        )
        self.threshold_panel.threshold_changed.connect(self._on_threshold_value_changed)
        self.threshold_panel.envelope_std_changed.connect(self._on_threshold_envelope_std_changed)
        self.threshold_panel.min_distance_changed.connect(self._on_threshold_min_distance_changed)
        self.threshold_panel.duration_filter_changed.connect(self._on_threshold_duration_filter_changed)
        self.threshold_panel.duration_range_changed.connect(self._on_threshold_duration_range_changed)
        self.threshold_panel.bandpass_filter_changed.connect(self._on_threshold_bandpass_filter_changed)
        self.threshold_panel.bandpass_range_changed.connect(self._on_threshold_bandpass_range_changed)
        self.threshold_panel.generate_requested.connect(lambda: self.threshold(None))
        self.slice_view.threshold_changed.connect(self._on_threshold_line_changed)
        self.preset_panel.selection_changed.connect(self._on_preset_selected)
        self.preset_panel.create_requested.connect(self._create_preset_from_panel)
        self.preset_panel.edit_requested.connect(self._edit_preset_from_panel)
        self.preset_panel.delete_requested.connect(self._delete_preset_from_panel)
        self.preset_panel.visibility_changed.connect(self._set_preset_visibility)
        self.preset_panel.editability_changed.connect(self._set_preset_editability)
        self.preset_panel.visibility_all_changed.connect(self._set_all_preset_visibility)
        self.preset_panel.editability_all_changed.connect(self._set_all_preset_editability)
        self.events_table.selection_changed.connect(self._on_events_table_selection)
        self.events_table.type_changed.connect(self._on_events_table_type_changed)
        self.events_table.time_changed.connect(self._on_events_table_time_changed)
        self.events_table.delete_requested.connect(self._on_events_table_delete)
        self.event_timeline.event_selected.connect(self._on_timeline_event_selected)
        self.event_timeline.event_created.connect(self._on_timeline_event_created)
        self.event_timeline.event_changed.connect(self._on_timeline_event_changed)
        self.spec_compression_ratio = float(spectrogram_config.get("compression", 0))
        self.spec_mel = bool(spectrogram_config.get("mel", False))
        self.spec_colormap = str(cmap_name or spectrogram_config.get("colormap", "turbo"))
        self.spec_view = views.SpecView(model=self, callback=self.on_trace_clicked, colormap=self.spec_colormap)
        self.spec_view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.ly = QtWidgets.QVBoxLayout()

        self.outer_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.outer_splitter.setHandleWidth(8)
        self.outer_splitter.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.center_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.center_splitter.setObjectName("centerWorkspace")
        self.center_splitter.setHandleWidth(8)
        self.center_splitter.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        splitter_sizes = []
        configured_sizes = config.get("window", {}).get("splitter_sizes", {})
        self._center_panel_names = []
        self._panel_widgets = {}
        self._last_panel_sizes = {
            name: max(1, int(size)) for name, size in configured_sizes.items() if isinstance(size, (int, float))
        }

        def add_splitter_panel(name: str, widget, default_size: int) -> None:
            size = max(1, int(configured_sizes.get(name, default_size)))
            self.center_splitter.addWidget(widget)
            self._center_panel_names.append(name)
            self._panel_widgets[name] = widget
            self._last_panel_sizes[name] = size
            splitter_sizes.append(size)

        if self.movie_panel is not None:
            add_splitter_panel("movie", self.movie_panel, 320)
        if "pose_positions_allo" in self.ds:
            add_splitter_panel("tracks", self.tracks_view, 150)
        add_splitter_panel("waveform", self.slice_view, 100)
        add_splitter_panel("spectrogram", self.spec_view, 400)
        add_splitter_panel("timeline", self.event_timeline, 100)
        add_splitter_panel("event_table", self.events_table, 200)

        for index in range(self.center_splitter.count()):
            self.center_splitter.setCollapsible(index, False)
            self.center_splitter.setStretchFactor(index, splitter_sizes[index])
        self.center_splitter.setSizes(splitter_sizes)

        self.left_sidebar = QtWidgets.QWidget()
        self.left_sidebar.setObjectName("leftSidebar")
        left_sidebar_layout = QtWidgets.QVBoxLayout(self.left_sidebar)
        left_sidebar_layout.setContentsMargins(0, 0, 0, 0)
        left_sidebar_layout.setSpacing(8)
        if self.project is not None:
            if self.project_panel is None:
                self.project_panel = self._make_project_panel()
            self.project_panel.set_project(self.project, self.current_recording_name)
            left_sidebar_layout.addWidget(self.project_panel, 1)
        left_sidebar_layout.addWidget(self.threshold_panel)
        left_sidebar_layout.addWidget(self.preset_panel, 1)

        self.outer_splitter.addWidget(self.left_sidebar)
        self.outer_splitter.addWidget(self.center_splitter)
        self.outer_splitter.setCollapsible(0, False)
        self.outer_splitter.setCollapsible(1, False)
        self.outer_splitter.setStretchFactor(0, 0)
        self.outer_splitter.setStretchFactor(1, 1)
        sidebar_size = max(1, int(configured_sizes.get("sidebar", 260)))
        workspace_size = max(1, int(configured_sizes.get("workspace", 1200)))
        self._last_panel_sizes.update({"sidebar": sidebar_size, "workspace": workspace_size})
        self.outer_splitter.setSizes([sidebar_size, workspace_size])

        self.ly.addWidget(self.outer_splitter)

        def edit_time_finished(source=None):
            try:
                self.t0 = self._seconds_sample(float(source.text()))
            except Exception as e:
                logger.debug(e)
            source.clearFocus()  # de-focus text field upon enter so we can continue annotating right away

        def edit_frame_finished(source=None):
            try:
                frame_number = float(source.text())
                frame_seconds = self._seconds_for_video_frame(frame_number)
                if frame_seconds is not None:
                    self.t0 = self._seconds_sample(frame_seconds)
            except Exception as e:
                print(e)

        self.transport_panel = QtWidgets.QWidget()
        self.transport_panel.setObjectName("transportPanel")
        transport_layout = QtWidgets.QHBoxLayout(self.transport_panel)
        transport_layout.setContentsMargins(0, 0, 0, 0)
        transport_layout.setSpacing(8)
        transport_layout.addWidget(self._build_transport(), stretch=10)

        self.edit_time = QtWidgets.QLineEdit()
        self.edit_time.editingFinished.connect(functools.partial(edit_time_finished, source=self.edit_time))
        transport_layout.addWidget(self.edit_time, stretch=1)
        edit_time_label = QtWidgets.QLabel("seconds")
        edit_time_label.setProperty("role", "muted")
        transport_layout.addWidget(edit_time_label)

        if self.vr is not None:
            self.edit_frame = QtWidgets.QLineEdit()
            self.edit_frame.editingFinished.connect(functools.partial(edit_frame_finished, source=self.edit_frame))
            transport_layout.addWidget(self.edit_frame, stretch=1)
            edit_frame_label = QtWidgets.QLabel("frame")
            edit_frame_label.setProperty("role", "muted")
            transport_layout.addWidget(edit_frame_label)

        self.ly.addWidget(self.transport_panel)
        self._setup_audio_clock(self._audio_source_path())
        self._sync_transport_controls()

        self.cw = pg.GraphicsLayoutWidget()
        self.cw.setLayout(self.ly)
        self.setCentralWidget(self.cw)

        self.update_eventtype_selector()
        self._apply_panel_visibility()
        self._sync_threshold_mode_ui()
        self._restore_window_geometry()

        self.show()
        self.update_xy()
        self.update_frame()
        self.t0 = self.t0 + 0.0000000001
        self.span = self.span + 0.0000000001
        logger.info("DAS gui initialized.")

        self.update_xy()
        self.app.processEvents()

    def _load_project_recording(self, ds, recording) -> None:
        self._pause_playback()
        self._stop_window_audio_playhead()
        self._clear_playback_window()

        event_names = self.event_times.names
        selected_event_name = self.current_event_name
        self.ds, original_spatial_units = dataset_service.prepare_for_display(ds)
        if original_spatial_units is None:
            self.__dict__.pop("original_spatial_units", None)
        else:
            self.original_spatial_units = original_spatial_units
        self.data_source = DataSource("project", str(recording.audio_path))
        self.current_recording_name = recording.name
        self._audio_source_names = self._discover_audio_source_names()
        self._active_audio_source_name = self._audio_source_names[0] if self._audio_source_names else None
        self._audio_selector_items = []
        self._video_sources = self._discover_video_sources()
        self._active_video_name = next(iter(self._video_sources), None)
        self.vr = self._video_reader_for_source(self._active_video_name) if self._active_video_name is not None else None

        self.event_times = dataset_service.event_times_from_dataset(self.ds)
        for name in event_names:
            if name not in self.event_times:
                self.event_times.add_name(name, category="event")
        event_types_changed = self.event_times.names != event_names
        for name in self.event_times.names:
            self._event_preset(name)
        self._current_event_name = selected_event_name if selected_event_name in self.event_times.names else None
        self._sync_event_colors_from_presets()
        self._saved_annotations = self._annotation_snapshot()

        self.tmin = 0
        self.fs_song = self._source_sampling_rate(self._active_audio_source_name) or float(
            self.ds.attrs.get("target_sampling_rate_Hz", 1_000)
        )
        self.nb_channels = self._source_channel_count(self._active_audio_source_name)
        audio_length = self._source_length(self._active_audio_source_name)
        self.tmax = audio_length if audio_length is not None else len(self.ds.time) if "time" in self.ds else 0
        self.nb_flies = _num_flies(self.ds)
        self.other_fly = 1 if self.nb_flies > 1 else 0
        if "poseparts" in self.ds:
            self.bodyparts = self.ds.poseparts.data
            self.nb_bodyparts = len(self.ds.poseparts)
        elif "bodyparts" in self.ds:
            self.bodyparts = self.ds.bodyparts.data
            self.nb_bodyparts = len(self.ds.bodyparts)
        else:
            self.nb_bodyparts = 1
            self.bodyparts = None
        self.fly_colors = utils.make_colors(self.nb_flies)
        self.bodypart_colors = utils.make_colors(self.nb_bodyparts)
        self.swap_events = self.ds.attrs.get("swap_events", [])
        self.fs_other = (
            self.ds.song_events.attrs["sampling_rate_Hz"]
            if "song_events" in self.ds
            else float(self.ds.attrs.get("target_sampling_rate_Hz", self.fs_song))
        )
        self._t0 = 0
        self._span = min(int(self.fs_song), self.tmax)
        self._apply_active_audio_source()
        self.frame_interval = self.fs_song * self._active_video_frame_seconds()

        self._set_channel_selector_items()
        self._setup_audio_clock(self._audio_source_path())
        self._sync_threshold_panel()
        self._sync_transport_controls()
        self._sync_channel_selector_overlay()
        if event_types_changed:
            self.update_eventtype_selector(selected_name=selected_event_name)
        self.project_panel.set_current_recording(recording.name)
        self.setWindowTitle(f"{_project_title(self.project)} - {recording.name}")
        self._force_next_event_table_refresh = True
        self.update_xy()
        self.update_frame()

    def save_annotations(self, qt_keycode=None):
        if self.project is None:
            return super().save_annotations(qt_keycode)
        self._capture_project_state()
        if self.project.is_saved or len(self.project.recordings) > 1:
            return self.save_project(qt_keycode)
        saved = super().save_annotations(qt_keycode)
        if saved:
            recording = self.project.recording(self.current_recording_name)
            recording.set_annotations(self.event_times)
            recording.mark_annotations_saved()
            self._refresh_project_panel()
        return saved

    def export_annotations(self, qt_keycode=None):
        return MainWindow.save_annotations(self, qt_keycode)

    def closeEvent(self, event):
        if self.project is None:
            super().closeEvent(event)
            return
        if self._switching_project:
            self._finish_close(event, save_global=False)
            return
        self._capture_project_state()
        if self.project.is_dirty:
            choice = QtWidgets.QMessageBox.warning(
                self,
                "Unsaved project changes",
                "Save changes before closing?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Save,
            )
            if choice == QtWidgets.QMessageBox.Cancel:
                event.ignore()
                return
            if choice == QtWidgets.QMessageBox.Save and not self.save_annotations():
                event.ignore()
                return
        save_global = not self.project.is_saved and len(self.project.recordings) <= 1
        self._finish_close(event, save_global=save_global)

    def _annotation_snapshot(self) -> pd.DataFrame:
        return self.event_times.to_df(preserve_empty=True, with_channels=True).copy(deep=True)

    def _annotations_changed(self) -> bool:
        return not self._annotation_snapshot().equals(self._saved_annotations)

    def _remember_splitter_sizes(self):
        if hasattr(self, "center_splitter"):
            for name, size in zip(self._center_panel_names, self.center_splitter.sizes()):
                if size > 0:
                    self._last_panel_sizes[name] = int(size)
        if hasattr(self, "outer_splitter"):
            for name, size in zip(("sidebar", "workspace"), self.outer_splitter.sizes()):
                if size > 0:
                    self._last_panel_sizes[name] = int(size)

    def _apply_panel_visibility(self):
        if not hasattr(self, "_panel_widgets"):
            return
        self._remember_splitter_sizes()
        visible = {
            "movie": self.show_movie,
            "tracks": self.show_tracks,
            "waveform": self.show_trace,
            "spectrogram": self.show_spec,
            "timeline": self.show_timeline,
            "event_table": self.show_event_table,
        }
        for name, widget in self._panel_widgets.items():
            widget.setVisible(bool(visible[name]))
        self.left_sidebar.setVisible(bool(self.show_sidebar))
        self.center_splitter.setSizes([self._last_panel_sizes[name] for name in self._center_panel_names])
        self.outer_splitter.setSizes(
            [self._last_panel_sizes.get("sidebar", 260), self._last_panel_sizes.get("workspace", 1200)]
        )
        self._sync_channel_selector_overlay()
        self._sync_view_action_checks()

    def _sync_channel_selector_overlay(self):
        if not hasattr(self, "cb2") or not hasattr(self, "slice_view") or not hasattr(self, "spec_view"):
            return
        show_selector = self.cb2.count() > 1 and (self.show_trace or self.show_spec)
        target = self.slice_view if self.show_trace or not self.show_spec else self.spec_view
        if self.cb2.parent() is not target:
            self.cb2.setParent(target)
            if target is self.spec_view:
                self.spec_view.channel_combo = self.cb2
        self.cb2.setVisible(show_selector)
        target._position_settings_button()
        self._sync_channel_selector_enabled()

    def _toolbar_icon(self, name: str) -> QtGui.QIcon:
        pixmap = QtGui.QPixmap(18, 18)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(
            QtGui.QPen(
                QtGui.QColor(TEXT_PRIMARY),
                1.6,
                QtCore.Qt.SolidLine,
                QtCore.Qt.RoundCap,
                QtCore.Qt.RoundJoin,
            )
        )
        painter.setBrush(QtCore.Qt.NoBrush)

        def line(x0, y0, x1, y1):
            painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))

        def path(points, close=False):
            item = QtGui.QPainterPath(QtCore.QPointF(*points[0]))
            for point in points[1:]:
                item.lineTo(QtCore.QPointF(*point))
            if close:
                item.closeSubpath()
            painter.drawPath(item)

        if name == "open_audio_annotations":
            path([(2.5, 6), (2.5, 14.5), (15.5, 14.5), (15.5, 7.5), (8.5, 7.5), (7, 5), (2.5, 5)])
            path([(5, 11.5), (6.5, 9), (8, 12), (10, 8.5), (12, 12), (13.5, 10)])
        elif name == "import_annotations":
            line(9, 3, 9, 10.5)
            path([(6.5, 8), (9, 10.5), (11.5, 8)])
            path([(4, 12), (4, 14.5), (14, 14.5), (14, 12)])
        elif name == "save_annotations":
            painter.drawRoundedRect(QtCore.QRectF(4, 3, 10, 12), 1.2, 1.2)
            line(6, 5.5, 11.5, 5.5)
            line(6, 12.5, 12, 12.5)
            line(6, 10.5, 12, 10.5)
        elif name == "show_trace":
            path([(2.5, 9.5), (4.5, 9.5), (6, 5), (8, 13), (10.5, 6), (12, 10.5), (15.5, 10.5)])
        elif name == "show_spec":
            painter.drawRoundedRect(QtCore.QRectF(3, 4, 12, 10), 1.0, 1.0)
            for x, height in ((5.5, 3), (8, 6), (10.5, 8), (13, 4)):
                line(x, 12.5, x, 12.5 - height)
        elif name == "show_timeline":
            line(3, 9, 15, 9)
            painter.drawRoundedRect(QtCore.QRectF(4, 6, 3.5, 6), 0.8, 0.8)
            painter.drawRoundedRect(QtCore.QRectF(10.5, 6, 3.5, 6), 0.8, 0.8)
        elif name == "show_event_table":
            painter.drawRoundedRect(QtCore.QRectF(3.5, 4, 11, 10), 1.0, 1.0)
            line(3.5, 7.5, 14.5, 7.5)
            line(3.5, 10.5, 14.5, 10.5)
            line(8, 4, 8, 14)
        elif name == "show_sidebar":
            painter.drawRoundedRect(QtCore.QRectF(3, 3.5, 12, 11), 1.0, 1.0)
            line(7, 3.5, 7, 14.5)
            line(9, 7, 13, 7)
            line(9, 10, 13, 10)
        elif name == "threshold_mode":
            line(3, 9, 15, 9)
            path([(4, 12.5), (6.5, 7), (8.5, 11.5), (11, 5.5), (14, 12.5)])
        elif name == "show_movie":
            painter.drawRoundedRect(QtCore.QRectF(3, 4, 12, 10), 1.0, 1.0)
            path([(7.5, 7), (7.5, 11), (11, 9)], close=True)

        painter.end()
        return QtGui.QIcon(pixmap)

    def _build_annotation_toolbar(self) -> None:
        actions = getattr(self, "_viewer_actions", {})
        toolbar = QtWidgets.QToolBar("Annotations", self)
        toolbar.setObjectName("annotationToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setIconSize(QtCore.QSize(18, 18))
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

        specs = [
            ("open_audio_annotations", "Open audio/annotations"),
            ("import_annotations", "Import annotations"),
            ("save_annotations", "Save project" if self.project is not None and self.project.is_saved else "Save annotations"),
            ("show_trace", "Show waveform"),
            ("show_spec", "Show spectrogram"),
            ("show_timeline", "Show event timeline"),
            ("show_event_table", "Show annotation table"),
            ("show_sidebar", "Show annotation type table"),
            ("threshold_mode", "Thresholding mode"),
            ("show_movie", "Show movie"),
        ]
        for key, tooltip in specs:
            action = actions.get(key)
            if action is None:
                continue
            action.setToolTip(tooltip)
            action.setStatusTip(tooltip)
            action.setIcon(self._toolbar_icon(key))
            toolbar.addAction(action)
            if key in {"save_annotations", "show_sidebar"}:
                toolbar.addSeparator()

        self.annotation_toolbar = toolbar
        self.addToolBar(toolbar)
        self._sync_view_action_checks()

    def _sync_view_action_checks(self) -> None:
        actions = getattr(self, "_viewer_actions", {})
        action_vars = {
            "show_trace": "show_trace",
            "show_spec": "show_spec",
            "show_timeline": "show_timeline",
            "show_event_table": "show_event_table",
            "show_sidebar": "show_sidebar",
            "threshold_mode": "threshold_mode",
            "show_movie": "show_movie",
        }
        for action_name, var_name in action_vars.items():
            action = actions.get(action_name)
            if action is not None:
                action.setChecked(bool(getattr(self, var_name, False)))

    def _set_imported_event_times(self, event_times) -> None:
        self.event_times = annot.Events(event_times)
        self.event_presets = {
            name: preset for name, preset in getattr(self, "event_presets", {}).items() if name in self.event_times.names
        }
        if getattr(self, "_current_event_name", None) not in self.event_times.names:
            self._current_event_name = self.event_times.names[-1] if self.event_times.names else None
        self._sync_event_colors_from_presets()
        self.update_eventtype_selector(selected_name=self._current_event_name)
        self._update_xy_with_event_table_refresh()

    def _choose_annotation_import_mode(self, filename: str) -> str:
        message = QtWidgets.QMessageBox(self)
        message.setWindowTitle("Import annotations")
        message.setText(f"Import annotations from {Path(filename).name}?")
        merge_button = message.addButton("Merge", QtWidgets.QMessageBox.AcceptRole)
        replace_button = message.addButton("Replace", QtWidgets.QMessageBox.DestructiveRole)
        message.addButton(QtWidgets.QMessageBox.Cancel)
        message.setDefaultButton(merge_button)
        message.exec_()
        clicked = message.clickedButton()
        if clicked is merge_button:
            return "merge"
        if clicked is replace_button:
            return "replace"
        return "cancel"

    def import_annotations(self, qt_keycode=None, filename: str = None, mode: str = None):
        del qt_keycode
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Import annotations",
                "",
                "Annotation files (*.csv *.txt *.zarr *.mat);;All files (*)",
            )
        if not filename:
            return

        try:
            imported = dataset_service.load_annotation_file(filename)
        except Exception as exc:
            logger.exception("Could not import annotations from %s", filename)
            QtWidgets.QMessageBox.warning(self, "Import annotations", str(exc))
            return

        if mode is None:
            mode = self._choose_annotation_import_mode(filename)
        if mode == "cancel":
            return
        if mode == "replace":
            updated = imported
        elif mode == "merge":
            updated = dataset_service.merge_event_times(self.event_times, imported)
        else:
            raise ValueError(f"Unknown annotation import mode {mode!r}.")

        self._set_imported_event_times(updated)
        logger.info("Imported annotations from %s with mode %s.", filename, mode)

    def _config_snapshot(self):
        self._remember_splitter_sizes()
        config = super()._config_snapshot()
        config["window"]["panels"] = {
            "sidebar": bool(self.show_sidebar),
            "movie": bool(self.show_movie),
            "tracks": bool(self.show_tracks),
            "waveform": bool(self.show_trace),
            "spectrogram": bool(self.show_spec),
            "timeline": bool(self.show_timeline),
            "event_table": bool(self.show_event_table),
        }
        config["window"]["splitter_sizes"] = dict(self._last_panel_sizes)
        config["viewer"] = {
            "waveform": {
                "color": self.slice_view.waveform_color,
                "y_limits": list(self.slice_view.waveform_y_limits) if self.slice_view.waveform_y_limits is not None else None,
            },
            "spectrogram": {
                "fmin": None if self.fmin is None else float(self.fmin),
                "fmax": None if self.fmax is None else float(self.fmax),
                "levels": [None if value is None else float(value) for value in self.spec_levels],
                "compression": float(self.spec_compression_ratio),
                "resolution": int(self.spec_win),
                "colormap": str(self.spec_colormap),
                "denoise": bool(self.spec_denoise),
                "mel": bool(self.spec_mel),
            },
            "video": {
                "box_size": int(self.box_size),
                "crop": bool(self.crop),
                "maintain_custom_crop": bool(self.maintain_custom_crop),
                "frame_fliplr": bool(self.frame_fliplr),
                "frame_flipud": bool(self.frame_flipud),
                "show_dot": bool(self.show_dot),
                "show_poses": bool(self.show_poses),
                "move_poses": bool(self.move_poses),
            },
            "audio": {
                "waveform_all": bool(self._audio_settings().waveform_all),
                "events_all": bool(self._audio_settings().events_all),
                "playback_all": bool(self._audio_settings().playback_all),
                "scale_y_all": bool(self._audio_settings().scale_y_all),
                "select_loudest_channel": bool(self.select_loudest_channel),
            },
            "annotations": {
                "show": bool(self.show_songevents),
                "movable": bool(self.movable_events),
                "edit_only_current": bool(self.edit_only_current_events),
                "show_labels": bool(self.show_event_text),
                "table_audio_link": bool(self.events_table.window_filter_enabled),
                "table_audio_filter": bool(self.events_table.window_filter_enabled),
            },
            "thresholding": {
                "enabled": bool(self.threshold_mode),
                "value": float(self.thres_value),
                "envelope_std": float(self.thres_env_std),
                "min_distance": float(self.thres_min_dist),
                "duration_enabled": bool(self.thres_duration_enabled),
                "duration_min": float(self.thres_duration_min),
                "duration_max": float(self.thres_duration_max),
                "bandpass_enabled": bool(self.thres_bandpass_enabled),
                "bandpass_low": float(self.thres_bandpass_low),
                "bandpass_high": float(self.thres_bandpass_high),
            },
        }
        return gui_config.sanitize_config(config)

    def _event_type_settings(self):
        return [
            {
                "name": preset.name,
                "fixed_duration": bool(preset.fixed_duration),
                "duration_seconds": float(preset.duration_seconds),
                "duration_editable": bool(preset.duration_editable),
                "color_hex": preset.color_hex,
                "visible": bool(preset.visible),
                "editable": bool(preset.editable),
            }
            for preset in self._event_presets_in_order()
        ]

    def _update_model(self):
        try:
            self.update_xy()
        except (AttributeError, ValueError) as e:
            logger.debug(e)

    @property
    def ylim(self):
        return self._ylim

    @ylim.setter
    def ylim(self, value: float):
        self._ylim = value
        if hasattr(self, "slice_view"):
            limits = None if value is None else (-abs(float(value)), abs(float(value)))
            self.slice_view.set_waveform_y_limits(limits)
        self._update_model()

    @property
    def fmin(self):
        return self._fmin

    @fmin.setter
    def fmin(self, value: float):
        self._fmin = value
        self._update_model()

    @property
    def fmax(self):
        return self._fmax

    @fmax.setter
    def fmax(self, value: float):
        self._fmax = value
        self._update_model()

    @property
    def spec_compression_ratio(self):
        return self._spec_compression_ratio

    @spec_compression_ratio.setter
    def spec_compression_ratio(self, value: float):
        self._spec_compression_ratio = value
        self._update_model()

    @property
    def spec_levels(self):
        return self._spec_levels

    @spec_levels.setter
    def spec_levels(self, value: bool):
        self._spec_levels = value
        self._update_model()

    @property
    def spec_denoise(self):
        return self._spec_denoise

    @spec_denoise.setter
    def spec_denoise(self, value: bool):
        self._spec_denoise = value
        self._update_model()

    @property
    def spec_mel(self):
        return self._spec_mel

    @spec_mel.setter
    def spec_mel(self, value: bool):
        self._spec_mel = value
        self._update_model()

    @property
    def spec_colormap(self):
        return self._spec_colormap

    @spec_colormap.setter
    def spec_colormap(self, value: str):
        self._spec_colormap = value
        try:
            self.spec_view.set_colormap(value)
        except AttributeError:
            pass

    @property
    def box_size(self):
        return self._box_size

    @box_size.setter
    def box_size(self, value: int):
        self._box_size = value
        try:
            self.update_frame()
        except:
            pass

    def _is_audio_dataarray(self, name: str, da) -> bool:
        if name in {"song", "non_song_raw", "song_events", "event_traces"}:
            return False
        if name.endswith("_video_path") or name.endswith("_frame_time"):
            return False
        dims = getattr(da, "dims", ())
        return (
            getattr(da, "ndim", 0) == 2
            and "sampling_rate_Hz" in getattr(da, "attrs", {})
            and len(dims) == 2
            and str(dims[1]).endswith("channels")
        )

    def _discover_audio_source_names(self) -> list[str]:
        names = []
        if "song_raw" in self.ds:
            names.append("song_raw")
        for name, da in self.ds.data_vars.items():
            if name not in names and self._is_audio_dataarray(name, da):
                names.append(name)
        if not names and "song" in self.ds:
            names.append("song")
        return names

    def _audio_dataarray_for_source(self, source_name: str | None = None):
        source_name = source_name or getattr(self, "_active_audio_source_name", None)
        if source_name is None:
            if "song_raw" in self.ds:
                source_name = "song_raw"
            elif "song" in self.ds:
                source_name = "song"
        if source_name == "song" and "song" in self.ds:
            return self.ds.song
        if hasattr(self.ds, "data_vars") and source_name in self.ds.data_vars:
            return self.ds[source_name]
        if source_name is not None and hasattr(self.ds, source_name):
            return getattr(self.ds, source_name)
        return None

    def _source_sampling_rate(self, source_name: str | None) -> float | None:
        da = self._audio_dataarray_for_source(source_name)
        if da is None:
            return None
        sampling_rate = da.attrs.get("sampling_rate_Hz")
        return None if sampling_rate is None else float(sampling_rate)

    def _source_length(self, source_name: str | None) -> int | None:
        da = self._audio_dataarray_for_source(source_name)
        return None if da is None else int(da.shape[0])

    def _source_channel_count(self, source_name: str | None) -> int | None:
        da = self._audio_dataarray_for_source(source_name)
        if da is None or getattr(da, "ndim", 0) < 2:
            return None
        return int(da.shape[1])

    def _audio_time_values(self) -> np.ndarray:
        if not hasattr(self, "ds"):
            return np.array([], dtype=float)
        da = self._audio_dataarray_for_source()
        if da is None:
            return np.array([], dtype=float)
        time_dim = da.dims[0]
        if time_dim in da.coords:
            return np.asarray(da[time_dim].data, dtype=float)
        if time_dim in self.ds.coords:
            return np.asarray(self.ds[time_dim].data, dtype=float)
        fs = self._source_sampling_rate(getattr(self, "_active_audio_source_name", None)) or self.fs_song
        return np.arange(da.shape[0], dtype=float) / fs

    def _sample_seconds(self, sample: float) -> float:
        times = self._audio_time_values()
        if len(times):
            return float(np.interp(float(sample), np.arange(len(times)), times))
        return float(sample) / self.fs_song

    def _seconds_sample(self, seconds: float) -> float:
        times = self._audio_time_values()
        if len(times):
            return float(np.interp(float(seconds), times, np.arange(len(times))))
        return float(seconds) * self.fs_song

    def _apply_active_audio_source(self, *, preserve_seconds: bool = False, old_seconds: float = None) -> None:
        if preserve_seconds and old_seconds is None and hasattr(self, "_t0"):
            old_seconds = self._sample_seconds(self.t0)
        old_seconds = 0.0 if old_seconds is None else float(old_seconds)
        da = self._audio_dataarray_for_source()
        if da is None:
            return
        self.fs_song = float(da.attrs["sampling_rate_Hz"])
        self.nb_channels = self._source_channel_count(getattr(self, "_active_audio_source_name", None))
        self.tmax = int(da.shape[0])
        if preserve_seconds:
            self._t0 = float(np.clip(self._seconds_sample(old_seconds), self.tmin, self.tmax_playhead))
        elif hasattr(self, "_t0"):
            self._t0 = float(np.clip(self._t0, self.tmin, self.tmax_playhead))

    def _combo_item(self, index: int | None = None):
        combo = getattr(self, "cb2", None)
        if combo is None or not hasattr(combo, "count") or not hasattr(combo, "itemData") or combo.count() == 0:
            return None
        if index is None:
            index = combo.currentIndex()
        item = combo.itemData(index)
        return item if isinstance(item, tuple) and len(item) == 3 else None

    def _set_channel_selector_items(self) -> None:
        labels = self._channel_labels()
        self.slice_view.set_channels(labels, show_selector=len(labels) > 1)
        for index, item in enumerate(self._audio_selector_items):
            self.cb2.setItemData(index, item)
        selected = self._combo_item()
        if selected is not None:
            self._active_audio_source_name = selected[0]

    def _combo_index_for_audio_item(self, source_name: str, data_index: int | None) -> int:
        for index, item in enumerate(getattr(self, "_audio_selector_items", [])):
            if item[0] == source_name and item[1] == data_index:
                return index
        return -1

    def _on_audio_selection_changed(self) -> None:
        selected = self._combo_item()
        if selected is None:
            self.update_xy()
            return
        old_source = getattr(self, "_active_audio_source_name", None)
        old_seconds = self._sample_seconds(self.t0)
        self._active_audio_source_name = selected[0]
        self._stop_window_audio_playhead()
        self._clear_playback_window()
        if getattr(self, "_is_playing", False):
            self._pause_playback()
        self._apply_active_audio_source(preserve_seconds=True, old_seconds=old_seconds)
        if selected[0] != old_source:
            self._setup_audio_clock(self._audio_source_path())
        if self.vr is not None:
            self.frame_interval = self.fs_song * self._active_video_frame_seconds()
        self._sync_threshold_panel()
        self._sync_transport_controls()
        self.update_xy()
        self.update_frame()

    def _discover_video_sources(self) -> dict[str, str]:
        sources = {}
        for name, da in self.ds.data_vars.items():
            if not name.endswith("_video_path") or getattr(da, "ndim", 0) != 0:
                continue
            try:
                value = da.item()
            except AttributeError:
                value = da.values.item()
            if isinstance(value, bytes):
                value = value.decode()
            source_name = name[: -len("_video_path")]
            sources[source_name] = str(value)
        video_filename = self.ds.attrs.get("video_filename")
        if video_filename and "camera" not in sources:
            sources["camera"] = str(video_filename)
        return sources

    def _video_reader_for_source(self, source_name: str):
        path = self._video_sources.get(source_name)
        if not path:
            return None
        try:
            return modern_video.PyAVVideoReader(path)
        except FileNotFoundError:
            logger.info(f'Video "{path}" not found. Continuing without.')
        except Exception:
            logger.info("Something went wrong when loading the video. Continuing without.")
        return None

    def _build_movie_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setObjectName("moviePanel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.video_combo = QtWidgets.QComboBox(panel)
        self.video_combo.setObjectName("videoSourceSelector")
        self.video_combo.setFixedHeight(24)
        for name, path in self._video_sources.items():
            self.video_combo.addItem(name, name)
            self.video_combo.setItemData(self.video_combo.count() - 1, path, QtCore.Qt.ToolTipRole)
        self.video_combo.setVisible(self.video_combo.count() > 1)
        self.video_combo.setEnabled(self.video_combo.count() > 1)
        if self._active_video_name is not None:
            index = self.video_combo.findData(self._active_video_name)
            self.video_combo.setCurrentIndex(max(0, index))
        self.video_combo.currentIndexChanged.connect(self._on_video_source_changed)
        layout.addWidget(self.video_combo)
        layout.addWidget(self.movie_view, 1)
        return panel

    def _on_video_source_changed(self) -> None:
        if self.video_combo is None:
            return
        source_name = self.video_combo.currentData()
        if not source_name or source_name == self._active_video_name:
            return
        reader = self._video_reader_for_source(source_name)
        if reader is None:
            return
        self._active_video_name = source_name
        self.vr = reader
        self.frame_interval = self.fs_song * self._active_video_frame_seconds()
        self.update_frame()

    def _active_video_frame_times(self) -> np.ndarray | None:
        name = getattr(self, "_active_video_name", None)
        if not name:
            return None
        frame_time_name = f"{name}_frame_time"
        if frame_time_name not in self.ds:
            return None
        return np.asarray(self.ds[frame_time_name].data, dtype=float)

    def _active_video_frame_seconds(self) -> float:
        frame_times = self._active_video_frame_times()
        if frame_times is not None and len(frame_times) > 1:
            return float(np.median(np.diff(frame_times)))
        return 1 / self.vr.frame_rate if self.vr is not None else 1 / 1_000

    def _seconds_for_video_frame(self, frame_number: float) -> float | None:
        frame_times = self._active_video_frame_times()
        frame_index = int(round(frame_number))
        if frame_times is not None and 0 <= frame_index < len(frame_times):
            return float(frame_times[frame_index])
        if "nearest_frame" in self.ds.coords:
            idx = np.argmax(self.ds.nearest_frame.data >= frame_number)
            return float(self.ds.nearest_frame[idx].time.values)
        return None

    def _build_transport(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setObjectName("transportPanel")
        layout = QtWidgets.QHBoxLayout(panel)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        transport_box = QtWidgets.QWidget(panel)
        transport_box.setObjectName("transportBox")
        transport_box_layout = QtWidgets.QHBoxLayout(transport_box)
        transport_box_layout.setContentsMargins(3, 2, 3, 2)
        transport_box_layout.setSpacing(2)

        buttons = (
            self._build_transport_button(
                "<<",
                "Fast Reverse",
                lambda: self._transport_fast_seek(-1),
            ),
            self._build_transport_button(
                "|<",
                "Reverse",
                lambda: self._transport_frame_seek(-1),
            ),
            self._build_transport_button(
                ">",
                "Play/Pause",
                self._toggle_playback,
                object_name="transportPlayButton",
            ),
            self._build_transport_button(
                "Loop",
                "Play current window (E)",
                lambda: self.play_audio("E"),
                object_name="transportLoopButton",
                width=40,
            ),
            self._build_transport_button(
                ">|",
                "Forward",
                lambda: self._transport_frame_seek(1),
            ),
            self._build_transport_button(
                ">>",
                "Fast Forward",
                lambda: self._transport_fast_seek(1),
            ),
        )
        self._play_button = buttons[2]
        self.playButton = self._play_button
        for button in buttons:
            transport_box_layout.addWidget(button)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self._slider.setRange(int(self.tmin), int(self.tmax_playhead))
        self._slider.setPageStep(max(1, int(round(max((self.tmax - self.tmin) / 100, self._span)))))
        self._slider.valueChanged.connect(self._on_transport_seek)
        self._slider.sliderPressed.connect(self._on_slider_scrub_started)
        self._slider.sliderReleased.connect(self._on_slider_scrub_finished)
        self._clock_label = QtWidgets.QLabel()
        self._clock_label.setObjectName("transportClock")
        self._clock_label.setMinimumWidth(150)

        layout.addWidget(transport_box)
        label = QtWidgets.QLabel("Playhead")
        label.setProperty("role", "muted")
        layout.addWidget(label)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._clock_label)
        self._set_play_button_state(playing=False)
        return panel

    def _build_transport_button(
        self,
        label: str,
        tooltip: str,
        callback: Callable[[], None],
        object_name: str = None,
        width: int = 22,
    ):
        button = QtWidgets.QToolButton()
        button.setProperty("role", "transport")
        if object_name is not None:
            button.setObjectName(object_name)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        button.setToolTip(tooltip)
        button.setText(label)
        button.setFixedSize(width, 22)
        button.clicked.connect(lambda _checked=False: callback())
        return button

    def _set_play_button_state(self, *, playing: bool) -> None:
        if not hasattr(self, "_play_button"):
            return
        self._play_button.setText("||" if playing else ">")
        self._play_button.setToolTip("Pause playback" if playing else "Start playback")

    def _channel_switching_locked(self) -> bool:
        return bool(getattr(self, "_is_playing", False)) or getattr(self, "_window_audio_start_sample", None) is not None

    def _sync_channel_selector_enabled(self) -> None:
        combo = getattr(self, "cb2", None)
        if combo is None or not hasattr(combo, "setEnabled"):
            return
        combo.setEnabled(not self._channel_switching_locked() and combo.count() > 1)

    def _format_seconds(self, seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        minutes, rem = divmod(seconds, 60.0)
        hours, minutes = divmod(int(minutes), 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{rem:06.3f}"
        return f"{minutes:02d}:{rem:06.3f}"

    def _sync_transport_controls(self) -> None:
        if hasattr(self, "_slider"):
            self._slider.blockSignals(True)
            self._slider.setValue(int(round(self.t0)))
            self._slider.blockSignals(False)
        if hasattr(self, "edit_time") and not self.edit_time.hasFocus():
            self.edit_time.setText(str(self._sample_seconds(self.t0)))
        if self.vr is not None and hasattr(self, "edit_frame") and not self.edit_frame.hasFocus():
            self.edit_frame.setText(str(self.framenumber))
        if hasattr(self, "_clock_label"):
            current = self._format_seconds(self._sample_seconds(self.t0))
            total = self._format_seconds(self._sample_seconds(self.tmax_playhead))
            self._clock_label.setText(f"{current} / {total}")

    def _sync_playhead_only(self) -> None:
        seconds = self._sample_seconds(self.t0)
        if hasattr(self, "slice_view") and hasattr(self.slice_view, "set_playhead"):
            self.slice_view.set_playhead(seconds)
        if hasattr(self, "event_timeline"):
            self.event_timeline.set_playhead(seconds)
        if hasattr(self, "spec_view") and hasattr(self.spec_view, "pos_line"):
            self.spec_view.pos_line.setValue(seconds)

    def _playhead_requires_view_refresh(self) -> bool:
        if not hasattr(self, "x") or len(self.x) == 0:
            return True
        seconds = self._sample_seconds(self.t0)
        return seconds < float(self.x[0]) or seconds > float(self.x[-1])

    def _clear_playback_window(self) -> None:
        self._playback_window_start = None
        self._playback_window_stop = None

    def _stop_window_audio_playhead(self) -> None:
        timer = getattr(self, "_window_audio_timer", None)
        if timer is not None:
            timer.stop()
        self._stop_qt_array_audio()
        self._window_audio_start_sample = None
        self._window_audio_stop_sample = None
        self._sync_channel_selector_enabled()

    def _set_playhead_sample(
        self,
        sample: float,
        *,
        refresh: bool = True,
        process_events: bool = True,
        preserve_playback_window: bool = False,
        force_refresh: bool = False,
    ) -> None:
        if not preserve_playback_window:
            self._stop_window_audio_playhead()
            self._clear_playback_window()
        old_t0 = getattr(self, "_t0", self.tmin)
        self._t0 = np.clip(sample, self.tmin, self.tmax_playhead)
        if force_refresh or not np.isclose(self._t0, old_t0, rtol=0.0, atol=1.0e-4):
            self._sync_transport_controls()
            if refresh or self._playhead_requires_view_refresh():
                self.update_xy()
                self.update_frame()
            else:
                self._sync_playhead_only()
            if process_events:
                self.app.processEvents()

    def _transport_fast_seek(self, direction: int) -> None:
        if direction:
            self._seek_playhead(self.t0 + direction * self.span / 2)

    def _transport_frame_seek(self, direction: int) -> None:
        if direction:
            seconds = self._sample_seconds(self.t0) + direction * self._active_video_frame_seconds()
            self._seek_playhead(self._seconds_sample(seconds))

    def _on_transport_seek(self, value: int) -> None:
        self._seek_playhead(value)

    def _on_slider_scrub_started(self) -> None:
        self._is_slider_scrubbing = True

    def _on_slider_scrub_finished(self) -> None:
        self._is_slider_scrubbing = False
        self._seek_audio_to_playhead()
        self.update_xy()
        self.update_frame()

    def _seek_playhead(self, sample: float) -> None:
        self._set_playhead_sample(sample, refresh=True)
        self._playback_anchor_sample = float(self.t0)
        self._playback_clock.restart()
        self._seek_audio_to_playhead()

    def _audio_source_path(self):
        audio_suffixes = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".m4a"}
        da = self._audio_dataarray_for_source()
        if da is not None and da.attrs.get("sampling_rate_overridden", False):
            return None
        value = da.attrs.get("source_path") if da is not None else None
        if value:
            path = Path(str(value)).expanduser()
            if path.suffix.lower() in audio_suffixes and path.exists():
                return path.resolve()
        for key in ("filename", "audio_filename", "filepath_daq", "source_audio"):
            value = self.ds.attrs.get(key) if hasattr(self.ds, "attrs") else None
            if not value:
                continue
            path = Path(str(value)).expanduser()
            if path.suffix.lower() in audio_suffixes and path.exists():
                return path.resolve()
        return None

    def _setup_audio_clock(self, media_path: Path | None) -> None:
        player = getattr(self, "_audio_player", None)
        if player is not None:
            try:
                player.stop()
            except Exception:
                pass
        self._audio_output = None
        self._audio_player = None
        if media_path is None or QAudioOutput is None or QMediaPlayer is None or QUrl is None:
            if hasattr(self, "_clock_label"):
                self._clock_label.setToolTip("Timer-backed playback; install PySide6 for QMediaPlayer audio.")
            if QMediaPlayer is None:
                logger.warning("QMediaPlayer unavailable. Install PySide6 to enable synced transport audio.")
            return
        try:
            self._audio_output = QAudioOutput(self)
            self._audio_player = QMediaPlayer(self)
            self._audio_player.setAudioOutput(self._audio_output)
            self._audio_player.setSource(QUrl.fromLocalFile(str(media_path)))
            self._audio_player.positionChanged.connect(self._on_audio_position_changed)
            self._audio_player.durationChanged.connect(self._on_audio_duration_changed)
            self._audio_player.playbackStateChanged.connect(self._on_audio_playback_state_changed)
            self._audio_player.mediaStatusChanged.connect(self._on_audio_media_status_changed)
            self._audio_player.errorOccurred.connect(self._on_audio_error)
            if hasattr(self, "_clock_label"):
                self._clock_label.setToolTip(f"QMediaPlayer audio ({media_path.name})")
        except Exception as exc:
            logger.debug("Could not initialize Qt audio playback: %s", exc)
            self._audio_output = None
            self._audio_player = None
            if hasattr(self, "_clock_label"):
                self._clock_label.setToolTip("Timer-backed playback")

    def _materialize_audio_data(self, data):
        try:
            data = data.compute()
        except AttributeError:
            pass
        return np.array(data)

    def _audio_window_data(self, window_start: int, window_stop: int, *, all_channels: bool):
        da = self._audio_dataarray_for_source()
        if da is None:
            return None
        if getattr(da, "ndim", 0) == 1:
            return self._materialize_audio_data(da.data[window_start:window_stop])
        if all_channels:
            return self._materialize_audio_data(da.data[window_start:window_stop, :])
        channel = self._current_channel_data_index()
        if channel is None:
            return None
        return self._materialize_audio_data(da.data[window_start:window_stop, channel])

    def _transport_uses_qmedia_audio(self) -> bool:
        return self._audio_player is not None

    def _transport_uses_qt_array_audio(self) -> bool:
        return getattr(self, "_array_audio_sink", None) is not None

    def _stop_qt_array_audio(self) -> None:
        sink = getattr(self, "_array_audio_sink", None)
        if sink is not None:
            sink.stop()
        buffer = getattr(self, "_array_audio_buffer", None)
        if buffer is not None:
            buffer.close()
        self._array_audio_sink = None
        self._array_audio_buffer = None
        self._array_audio_bytes = None
        self._array_audio_start_sample = None

    def _qt_array_audio_format(self, y) -> tuple[object, np.ndarray] | tuple[None, None]:
        if QAudioFormat is None or QAudioSink is None or QMediaDevices is None:
            return None, None
        y = np.asarray(y)
        if y.ndim == 1:
            channel_count = 1
        elif y.ndim == 2:
            channel_count = int(y.shape[1])
        else:
            return None, None
        pcm = y.astype(np.float32, copy=False)
        peak = np.nanmax(np.abs(pcm)) if pcm.size else 0.0
        if np.isfinite(peak) and peak > 1.0:
            pcm = pcm / peak * 0.95
        pcm = np.nan_to_num(pcm, copy=False)
        pcm = np.ascontiguousarray(pcm)

        audio_format = QAudioFormat()
        audio_format.setSampleRate(int(round(self.fs_song)))
        audio_format.setChannelCount(channel_count)
        audio_format.setSampleFormat(QAudioFormat.SampleFormat.Float)
        device = QMediaDevices.defaultAudioOutput()
        if not device.isFormatSupported(audio_format):
            return None, None
        return audio_format, pcm

    def _start_qt_array_audio_window(self, window_start: int, window_stop: int, *, all_channels: bool) -> bool:
        if getattr(self, "_disable_qt_array_audio", False):
            return False
        y = self._audio_window_data(window_start, window_stop, all_channels=all_channels)
        if y is None or len(y) == 0:
            return False
        audio_format, pcm = self._qt_array_audio_format(y)
        if audio_format is None:
            return False
        self._stop_qt_array_audio()
        try:
            self._array_audio_bytes = QtCore.QByteArray(pcm.tobytes())
            self._array_audio_buffer = QtCore.QBuffer(self)
            self._array_audio_buffer.setData(self._array_audio_bytes)
            self._array_audio_buffer.open(QtCore.QIODevice.OpenModeFlag.ReadOnly)
            self._array_audio_sink = QAudioSink(audio_format, self)
            self._array_audio_start_sample = float(window_start)
            self._array_audio_sink.start(self._array_audio_buffer)
            return True
        except Exception as exc:
            logger.debug("Could not start Qt array audio playback: %s", exc)
            self._stop_qt_array_audio()
            return False

    def _transport_array_stop_sample(self) -> int:
        if self._audio_playback_all_channels():
            return int(self._playback_window_stop)
        return int(self.tmax)

    def _start_transport_array_audio(self) -> bool:
        window_start = int(self._playback_window_start)
        window_stop = self._transport_array_stop_sample()
        self._playback_audio_stop_sample = None
        if self._start_qt_array_audio_window(
            window_start,
            window_stop,
            all_channels=self._audio_playback_all_channels(),
        ):
            self._playback_audio_stop_sample = window_stop
            return True
        logger.info("Could not start Qt array audio playback.")
        return False

    def _seek_audio_to_playhead(self) -> None:
        if self._transport_uses_qmedia_audio():
            self._audio_player.setPosition(int(round(self.t0 / self.fs_song * 1000)))

    def _toggle_playback(self) -> None:
        if self._is_playing:
            self._pause_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        if self.t0 >= self.tmax_playhead:
            self._seek_playhead(self.tmin)
        page_start = self.time0
        self._start_playback_page(page_start)

    def _set_playback_window(self, page_start: float) -> None:
        max_start = max(0, self.tmax - self.span)
        page_start = int(np.clip(page_start, self.tmin, max_start))
        page_stop = int(min(self.tmax, page_start + self.span))
        self._playback_window_start = page_start
        self._playback_window_stop = page_stop

    def _start_playback_page(self, page_start: float) -> None:
        self._stop_window_audio_playhead()
        self._set_playback_window(page_start)
        self.STOP = False
        self._is_playing = True
        self._sync_channel_selector_enabled()
        self._set_playhead_sample(
            self._playback_window_start,
            refresh=True,
            preserve_playback_window=True,
            force_refresh=True,
        )
        self._playback_anchor_sample = float(self.t0)
        self._seek_audio_to_playhead()
        self._set_play_button_state(playing=True)
        if self._transport_uses_qmedia_audio():
            self._playback_clock.restart()
            self._playback_timer.start()
            self._audio_player.play()
        else:
            if self._start_transport_array_audio():
                self._playback_anchor_sample = float(self.t0)
                self._playback_clock.restart()
                self._playback_timer.start()
            else:
                self._pause_playback()

    def _advance_playback_window(self, target_sample: float) -> bool:
        changed = False
        while self._playback_window_stop is not None and target_sample >= self._playback_window_stop:
            next_start = self._playback_window_stop
            if next_start >= self.tmax_playhead:
                break
            old_start = self._playback_window_start
            old_stop = self._playback_window_stop
            self._set_playback_window(next_start)
            changed = True
            if self._playback_window_start == old_start and self._playback_window_stop == old_stop:
                break
        return changed

    def _pause_playback(self) -> None:
        self.STOP = True
        self._is_playing = False
        self._playback_timer.stop()
        self._set_play_button_state(playing=False)
        if self._transport_uses_qmedia_audio():
            self._audio_player.pause()
        else:
            self._stop_qt_array_audio()
            self._playback_audio_stop_sample = None
        self._sync_channel_selector_enabled()

    def _start_window_audio_playhead(self, window_start: int, window_stop: int, *, all_channels: bool) -> bool:
        if self._is_playing:
            self._pause_playback()
        self._playback_window_start = int(window_start)
        self._playback_window_stop = int(window_stop)
        self._window_audio_start_sample = float(window_start)
        self._window_audio_stop_sample = float(max(window_start, min(window_stop - 1, self.tmax_playhead)))
        self._sync_channel_selector_enabled()
        self._set_playhead_sample(
            self._window_audio_start_sample,
            refresh=True,
            process_events=False,
            preserve_playback_window=True,
            force_refresh=True,
        )
        if not self._start_qt_array_audio_window(window_start, window_stop, all_channels=all_channels):
            self._stop_window_audio_playhead()
            return False
        self._window_audio_timer.start()
        return True

    def _on_window_audio_tick(self) -> None:
        start_sample = self._window_audio_start_sample
        stop_sample = self._window_audio_stop_sample
        if start_sample is None or stop_sample is None:
            self._stop_window_audio_playhead()
            return
        if not self._transport_uses_qt_array_audio():
            self._stop_window_audio_playhead()
            return
        target_sample = start_sample + self._array_audio_sink.processedUSecs() / 1e6 * self.fs_song
        if target_sample >= stop_sample:
            self._set_playhead_sample(
                stop_sample,
                refresh=False,
                process_events=False,
                preserve_playback_window=True,
            )
            self._stop_window_audio_playhead()
            return
        self._set_playhead_sample(
            target_sample,
            refresh=False,
            process_events=False,
            preserve_playback_window=True,
        )

    def _on_playback_tick(self) -> None:
        if not self._is_playing:
            return
        if self._transport_uses_qmedia_audio():
            target_sample = self._audio_player.position() / 1000 * self.fs_song
        elif self._transport_uses_qt_array_audio():
            target_sample = self._array_audio_start_sample + self._array_audio_sink.processedUSecs() / 1e6 * self.fs_song
        else:
            target_sample = self._playback_anchor_sample + self._playback_clock.nsecsElapsed() / 1e9 * self.fs_song
        if self._playback_window_start is not None:
            target_sample = max(float(self._playback_window_start), target_sample)
        page_stop = self._playback_window_stop if self._playback_window_stop is not None else self.tmax
        if target_sample >= page_stop:
            if page_stop >= self.tmax_playhead:
                self._set_playhead_sample(self.tmax_playhead, refresh=True, preserve_playback_window=True)
                self._pause_playback()
                return
            if self._advance_playback_window(target_sample):
                self._set_playhead_sample(
                    target_sample,
                    refresh=True,
                    process_events=False,
                    preserve_playback_window=True,
                    force_refresh=True,
                )
                if not self._transport_uses_qmedia_audio():
                    buffer_stop = getattr(self, "_playback_audio_stop_sample", None)
                    if buffer_stop is None or target_sample >= buffer_stop:
                        if self._start_transport_array_audio():
                            self._playback_anchor_sample = float(self.t0)
                            self._playback_clock.restart()
            return
        if target_sample >= self.tmax_playhead:
            self._set_playhead_sample(self.tmax_playhead, refresh=True, preserve_playback_window=True)
            self._pause_playback()
            return
        self._set_playhead_sample(target_sample, refresh=False, process_events=False, preserve_playback_window=True)

    def _on_audio_position_changed(self, position_ms: int) -> None:
        if not self._transport_uses_qmedia_audio():
            return
        if self._is_playing and self._playback_timer.isActive():
            return
        self._set_playhead_sample(
            position_ms / 1000 * self.fs_song,
            refresh=True,
            preserve_playback_window=getattr(self, "_is_playing", False),
        )

    def _on_audio_playback_state_changed(self, state) -> None:
        if QMediaPlayer is None or self._audio_player is None or not self._transport_uses_qmedia_audio():
            return
        stopped_state = getattr(QMediaPlayer, "StoppedState", None)
        if stopped_state is None and hasattr(QMediaPlayer, "PlaybackState"):
            stopped_state = QMediaPlayer.PlaybackState.StoppedState
        if state == stopped_state and self._is_playing:
            self._pause_playback()

    def _on_audio_duration_changed(self, duration_ms: int) -> None:
        if duration_ms > 0:
            self.tmax = max(self.tmax, int(round(duration_ms / 1000 * self.fs_song)))
            if hasattr(self, "_slider"):
                self._slider.setMaximum(int(self.tmax_playhead))

    def _on_audio_media_status_changed(self, status) -> None:
        logger.debug("QMediaPlayer media status changed: %s", status)

    def _on_audio_error(self, *args) -> None:
        logger.warning("Qt audio playback error: %s", args)
        self._audio_player = None
        self._audio_output = None
        if self._is_playing:
            self._playback_anchor_sample = float(self.t0)
            self._playback_clock.restart()

    @property
    def fs_ratio(self):
        return self.fs_song / self.fs_other

    @property
    def time0(self):
        max_start = max(0, self.tmax - self.span)
        playback_start = getattr(self, "_playback_window_start", None)
        if playback_start is None:
            start = np.clip(self.t0 - self.span / 2, 0, max_start)
        else:
            start = np.clip(playback_start, 0, max_start)
        return int(int(start / self.fs_ratio) * self.fs_ratio)

    @property
    def time1(self):
        stop = min(self.tmax, self.time0 + self.span)
        return int(int(stop / self.fs_ratio) * self.fs_ratio)

    @property
    def tmax_playhead(self):
        return max(getattr(self, "tmin", 0), self.tmax - 1)

    @property
    def trange(self):
        return np.array([self._sample_seconds(self.time0), self._sample_seconds(max(self.time0, self.time1 - 1))])

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, val: float):
        self._set_playhead_sample(val, refresh=True)
        if getattr(self, "_is_playing", False):
            self._playback_anchor_sample = float(self.t0)
            self._playback_clock.restart()
            self._seek_audio_to_playhead()

    @property
    def framenumber(self):
        frame_times = self._active_video_frame_times()
        if frame_times is not None and len(frame_times):
            seconds = self._sample_seconds(self.t0)
            return int(utils.find_nearest_idx(frame_times, seconds))
        if "nearest_frame" in self.ds.coords:
            try:  # in case nearest_frame is nan
                t = self._sample_seconds(self.t0)
                return int(self.ds.nearest_frame.sel(time=t, method="nearest"))
            except:
                pass

        return None

    @property
    def span(self):
        return self._span

    @span.setter
    def span(self, val):
        # HACK fixes weird offset/jump error - probably arises from self.fs_song / self.fs_other
        self._span = min(max(200, val), self.tmax)
        self.update_xy()

    @property
    def current_event_index(self):
        name = getattr(self, "_current_event_name", None)
        if name is None or name not in self.event_times.names:
            return None
        return self.event_times.names.index(name)

    @property
    def current_event_name(self):
        index = self.current_event_index
        if index is None:
            return None
        return self.event_times.names[index]

    def _initial_event_presets(self):
        colors = utils.make_colors(max(1, len(self.event_times.names)))
        presets = {}
        for index, name in enumerate(self.event_times.names):
            color_hex = event_widgets.color_hex_from_rgb(colors[index % len(colors)])
            values = np.asarray(self.event_times[name])
            finite = values[np.all(np.isfinite(values[:, :2]), axis=1)] if values.size else np.zeros((0, 3))
            durations = np.abs(finite[:, 1] - finite[:, 0]) if len(finite) else np.zeros((0,))
            fixed_duration = not np.any(durations > 1e-9)
            duration_seconds = 0.0 if fixed_duration else float(np.nanmedian(durations))
            presets[name] = event_widgets.EventTypePreset(
                name=name,
                fixed_duration=fixed_duration,
                duration_seconds=duration_seconds,
                duration_editable=not fixed_duration,
                color_hex=color_hex,
            )
        return presets

    def _merge_configured_event_types(self, configured_presets):
        configured_names = []
        for values in configured_presets:
            name = values["name"]
            if name in configured_names:
                logger.warning("Ignoring duplicate configured event type %s", name)
                continue
            configured_names.append(name)
            if name not in self.event_times:
                self.event_times.add_name(name, category="event")
            base = self.event_presets.get(name, event_widgets.EventTypePreset(name=name))
            color_hex = str(values.get("color_hex", base.color_hex))
            if not QtGui.QColor(color_hex).isValid():
                color_hex = base.color_hex
            self.event_presets[name] = event_widgets.EventTypePreset(
                name=name,
                fixed_duration=bool(values.get("fixed_duration", base.fixed_duration)),
                duration_seconds=max(0.0, float(values.get("duration_seconds", base.duration_seconds))),
                duration_editable=bool(values.get("duration_editable", base.duration_editable)),
                color_hex=color_hex,
                visible=bool(values.get("visible", base.visible)),
                editable=bool(values.get("editable", base.editable)),
            )

        if configured_names:
            ordered_names = configured_names + [name for name in self.event_times.names if name not in configured_names]
            self.event_times = annot.Events({name: self.event_times[name] for name in ordered_names})

    def _sync_event_colors_from_presets(self):
        self.nb_eventtypes = len(self.event_times.names)
        self.eventtype_colors = np.array(
            [self._event_preset(name).color_tuple() for name in self.event_times.names],
            dtype=int,
        )

    def _event_preset(self, name: str):
        if not hasattr(self, "event_presets"):
            self.event_presets = {}
        if name not in self.event_presets:
            index = self.event_times.names.index(name) if name in self.event_times.names else len(self.event_presets)
            colors = utils.make_colors(max(1, index + 1))
            self.event_presets[name] = event_widgets.EventTypePreset(
                name=name,
                color_hex=event_widgets.color_hex_from_rgb(colors[index % len(colors)]),
            )
        return self.event_presets[name]

    def _event_presets_in_order(self):
        return [self._event_preset(name) for name in self.event_times.names]

    def _event_type_visible(self, name: str) -> bool:
        if not name:
            return False
        return bool(self._event_preset(name).visible)

    def _event_type_editable(self, name: str) -> bool:
        if not name:
            return False
        return bool(self._event_preset(name).editable)

    def _event_type_can_edit(self, name: str) -> bool:
        return self._event_type_visible(name) and self._event_type_editable(name)

    def _visible_event_names(self) -> list[str]:
        return [name for name in self.event_times.names if self._event_type_visible(name)]

    def _editable_visible_event_names(self) -> list[str]:
        return [name for name in self.event_times.names if self._event_type_can_edit(name)]

    def _event_times_for_names(self, names):
        names = [name for name in names if name in self.event_times]
        data = {name: self.event_times[name].copy() for name in names}
        categories = {name: "event" for name in names}
        return annot.Events(data, categories=categories, add_names_from_categories=False)

    def _visible_event_times(self):
        return self._event_times_for_names(self._visible_event_names())

    def _locked_event_type_names(self):
        return [name for name in self.event_times.names if not self._event_type_editable(name)]

    def _locked_duration_record_ids(self, start_seconds: float | None = None, stop_seconds: float | None = None):
        locked = []
        for record in event_widgets.records_from_events(
            self.event_times,
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
            channel_filter=self._audio_event_channel_filter(),
        ):
            preset = self._event_preset(record.name)
            if preset.fixed_duration and not preset.duration_editable:
                locked.append(record.id)
        return locked

    def _set_preset_visibility(self, name: str, visible: bool):
        if name not in self.event_times.names:
            return
        self.event_presets[name] = self._event_preset(name).with_visibility(visible)
        self._after_preset_layer_change()

    def _set_preset_editability(self, name: str, editable: bool):
        if name not in self.event_times.names:
            return
        self.event_presets[name] = self._event_preset(name).with_editability(editable)
        self._after_preset_layer_change()

    def _set_all_preset_visibility(self, visible: bool):
        for name in self.event_times.names:
            self.event_presets[name] = self._event_preset(name).with_visibility(visible)
        self._after_preset_layer_change()

    def _set_all_preset_editability(self, editable: bool):
        for name in self.event_times.names:
            self.event_presets[name] = self._event_preset(name).with_editability(editable)
        self._after_preset_layer_change()

    def _after_preset_layer_change(self):
        self._refresh_preset_panel(selected_name=self.current_event_name)
        if getattr(self, "STOP", True):
            self._update_xy_with_event_table_refresh()

    def _clamp_event_bounds(self, start_seconds: float, stop_seconds: float):
        start = max(0.0, float(start_seconds))
        stop = max(0.0, float(stop_seconds))
        max_seconds = self._sample_seconds(self.tmax_playhead) if getattr(self, "fs_song", 0) else None
        if max_seconds is not None:
            start = min(start, max_seconds)
            stop = min(stop, max_seconds)
        return start, stop

    def _bounds_for_event_creation(self, name: str, start_seconds: float, stop_seconds: float = None):
        preset = self._event_preset(name)
        if preset.fixed_duration:
            duration = max(0.0, float(preset.duration_seconds))
            if stop_seconds is None:
                center = float(start_seconds)
            else:
                center = (float(start_seconds) + float(stop_seconds)) / 2
            start = center - duration / 2
            stop = center + duration / 2
            max_seconds = self._sample_seconds(self.tmax_playhead) if getattr(self, "fs_song", 0) else None
            if max_seconds is not None and duration <= max_seconds:
                if start < 0:
                    stop -= start
                    start = 0.0
                if stop > max_seconds:
                    start -= stop - max_seconds
                    stop = max_seconds
            return self._clamp_event_bounds(start, stop)
        if stop_seconds is None:
            stop_seconds = start_seconds
        start, stop = sorted([float(start_seconds), float(stop_seconds)])
        return self._clamp_event_bounds(start, stop)

    def _clear_pending_event_creation(self):
        self.sinet0 = None
        self.sinet0_event_name = None

    def _pending_event_creation(self):
        start_seconds = getattr(self, "sinet0", None)
        name = getattr(self, "sinet0_event_name", None)
        if start_seconds is None or name is None:
            return None
        if name != self.current_event_name or not self._event_type_can_edit(name):
            return None
        if self._event_preset(name).fixed_duration:
            return None
        return name, float(start_seconds)

    def _bounds_for_event_edit(
        self,
        name: str,
        old_start_seconds: float,
        old_stop_seconds: float,
        start_seconds: float,
        stop_seconds: float,
        changed_edge: str = "move",
    ):
        preset = self._event_preset(name)
        start = float(start_seconds)
        stop = float(stop_seconds)
        if preset.fixed_duration and not preset.duration_editable:
            duration = max(0.0, float(preset.duration_seconds))
            if changed_edge == "stop":
                stop = float(stop_seconds)
                start = stop - duration
            else:
                start = float(start_seconds)
                stop = start + duration
        else:
            start, stop = sorted([start, stop])
        return self._clamp_event_bounds(start, stop)

    def _refresh_preset_panel(self, selected_name: str = None):
        if not hasattr(self, "preset_panel"):
            return
        if selected_name is None:
            selected_name = self.current_event_name
        self.preset_panel.set_presets(self._event_presets_in_order(), selected_name=selected_name)

    def _on_preset_selected(self, name: str):
        if name not in self.event_times.names:
            return
        if getattr(self, "_current_event_name", None) == name:
            return
        self._clear_pending_event_creation()
        self._current_event_name = name
        self.update_xy()

    def _create_preset_from_panel(self):
        dialog = event_widgets.EventTypePresetDialog(
            title="Create Event",
            used_names=self.event_times.names,
            used_color_hexes=[preset.color_hex for preset in self._event_presets_in_order()],
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        preset = dialog.value()
        if preset is None:
            return
        self.event_times.add_name(preset.name, category="event")
        self.event_presets[preset.name] = preset
        self._sync_after_event_type_change(selected_name=preset.name)

    def _edit_preset_from_panel(self, name: str):
        preset = self._event_preset(name)
        dialog = event_widgets.EventTypePresetDialog(
            title="Edit Event",
            preset=preset,
            used_names=self.event_times.names,
            used_color_hexes=[item.color_hex for item in self._event_presets_in_order()],
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        updated = dialog.value()
        if updated is None:
            return
        if updated.name != name:
            if self.project is not None:
                current = self.project.recording(self.current_recording_name)
                current.set_annotations(self.event_times)
                self.project.rename_event_type(name, updated.name)
                self.event_times = annot.Events(current.annotations)
            else:
                self.event_times[updated.name] = self.event_times.pop(name)
                self.event_times.categories.pop(name, None)
                self.event_times.categories[updated.name] = "event"
            self.event_presets.pop(name, None)
        self.event_presets[updated.name] = updated
        self._sync_after_event_type_change(selected_name=updated.name)

    def _delete_preset_from_panel(self, name: str):
        if name not in self.event_times.names:
            return
        confirmed = QtWidgets.QMessageBox.question(
            self,
            "Delete Event",
            f"Delete event '{name}' and all of its annotations?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if confirmed != QtWidgets.QMessageBox.Yes:
            return
        if self.project is not None:
            current = self.project.recording(self.current_recording_name)
            current.set_annotations(self.event_times)
            self.project.delete_event_type(name)
            self.event_times = annot.Events(current.annotations)
        else:
            del self.event_times[name]
            self.event_times.categories.pop(name, None)
        self.event_presets.pop(name, None)
        selected_name = self.event_times.names[0] if self.event_times.names else None
        self._sync_after_event_type_change(selected_name=selected_name)

    def _sync_after_event_type_change(self, selected_name: str = None):
        self._sync_event_colors_from_presets()
        self.update_eventtype_selector(selected_name=selected_name)
        self._refresh_preset_panel(selected_name=selected_name)
        self._update_xy_with_event_table_refresh()

    @property
    def current_channel_name(self):
        return self.cb2.currentText()

    @property
    def current_channel_index(self):
        item = self._combo_item()
        if item is not None:
            return item[2]
        if self.current_channel_name != "Merged channels":
            return int(self.current_channel_name.split(" ")[-1])  # "Channel XX"
        return None

    def _current_channel_data_index(self):
        item = self._combo_item()
        if item is not None:
            return item[1]
        return self.current_channel_index

    @property
    def current_audio_source_name(self):
        item = self._combo_item()
        if item is not None:
            return item[0]
        return getattr(self, "_active_audio_source_name", None)

    def _audio_settings(self):
        if hasattr(self, "audio_channel_settings"):
            return self.audio_channel_settings
        return event_widgets.AudioChannelSettings(waveform_all=getattr(self, "show_all_channels", True))

    def _set_audio_settings(self, settings):
        if getattr(self, "_is_playing", False):
            self._pause_playback()
        self.audio_channel_settings = settings
        self.show_all_channels = bool(settings.waveform_all)
        if hasattr(self, "slice_view"):
            self.slice_view.set_audio_settings(settings)
        if getattr(self, "STOP", True):
            self.update_xy()

    def _audio_event_channel_filter(self):
        if self._audio_settings().events_all:
            return None
        channel = self.current_channel_index
        return -1 if channel is None else channel

    def _filter_event_rows_for_audio_channel(self, rows):
        channel_filter = self._audio_event_channel_filter()
        if channel_filter is None or len(rows) == 0:
            return rows
        channels = np.full(rows.shape[0], -1, dtype=int)
        if rows.shape[1] > 2:
            finite_channels = np.isfinite(rows[:, 2])
            channels[finite_channels] = rows[finite_channels, 2].astype(int)
        return rows[channels == int(channel_filter)]

    def _audio_playback_all_channels(self) -> bool:
        return bool(self._audio_settings().playback_all)

    def _channel_labels(self) -> list[str]:
        labels = []
        self._audio_selector_items = []
        multi_source = len(self._audio_source_names) > 1
        for source_name in self._audio_source_names:
            if source_name == "song":
                labels.append(f"{source_name}: Merged channels" if multi_source else "Merged channels")
                self._audio_selector_items.append((source_name, None, None))
                continue
            if source_name == "song_raw" and "song" in self.ds:
                labels.append(f"{source_name}: Merged channels" if multi_source else "Merged channels")
                self._audio_selector_items.append(("song", None, None))
            da = self._audio_dataarray_for_source(source_name)
            if da is None or getattr(da, "ndim", 0) < 2:
                continue
            channel_dim = da.dims[1]
            if channel_dim in da.coords:
                channel_values = np.asarray(da[channel_dim].data)
            elif channel_dim in self.ds.coords:
                channel_values = np.asarray(self.ds[channel_dim].data)
            else:
                channel_values = np.arange(da.shape[1])
            for data_index, channel_value in enumerate(channel_values):
                try:
                    channel_value = int(channel_value)
                except (TypeError, ValueError):
                    channel_value = str(channel_value)
                label = f"Channel {channel_value}"
                labels.append(f"{source_name}: {label}" if multi_source else label)
                self._audio_selector_items.append((source_name, data_index, channel_value))
        return labels

    @property
    def index_other(self):
        current_time = self.ds.time.sel(time=self._sample_seconds(self.t0), method="nearest")
        index_other = np.where(self.ds.time == current_time)[0]
        return int(index_other)

    def _event_color_map(self):
        colors = {}
        for index, name in enumerate(self.event_times.names):
            if index < len(self.eventtype_colors):
                colors[name] = tuple(int(v) for v in self.eventtype_colors[index])
        return colors

    def _refresh_event_widgets(self, sync_table_to_view: bool = False, force_table: bool = False):
        if not hasattr(self, "event_timeline") or not hasattr(self, "events_table"):
            return
        selected_ids = self.events_table.selected_record_ids()
        colors = self._event_color_map()
        visible_events = self._visible_event_times()
        locked_event_names = self._locked_event_type_names()
        start_seconds = None
        stop_seconds = None
        if sync_table_to_view and hasattr(self, "x") and len(self.x):
            start_seconds = float(self.x[0])
            stop_seconds = float(self.x[-1])
        table_follows_view = sync_table_to_view and self.events_table.window_filter_enabled
        table_is_empty = self.events_table.table.model().rowCount() == 0
        update_table = force_table or table_follows_view or not sync_table_to_view or table_is_empty
        table_start_seconds = start_seconds if table_follows_view else None
        table_stop_seconds = stop_seconds if table_follows_view else None
        channel_filter = self._audio_event_channel_filter()
        self.event_timeline.set_events(
            visible_events,
            colors=colors,
            selected_ids=selected_ids,
            locked_duration_ids=self._locked_duration_record_ids(start_seconds, stop_seconds),
            locked_event_names=locked_event_names,
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
            channel_filter=channel_filter,
        )
        if update_table:
            self.events_table.set_events(
                visible_events,
                selected_ids=selected_ids,
                locked_event_names=locked_event_names,
                start_seconds=table_start_seconds,
                stop_seconds=table_stop_seconds,
                channel_filter=channel_filter,
            )
        self.event_timeline.set_selected_ids(self.events_table.selected_record_ids())
        self.event_timeline.set_playhead(self._sample_seconds(self.t0))
        if table_follows_view and not self._syncing_event_selection:
            try:
                self._syncing_event_selection = True
                self.events_table.select_overlapping_range(float(self.x[0]), float(self.x[-1]))
            finally:
                self._syncing_event_selection = False

    def _after_event_edit(self):
        for name in self.event_times.names:
            self._event_preset(name)
        self._sync_event_colors_from_presets()
        self.update_eventtype_selector()
        self._refresh_preset_panel()
        self._update_xy_with_event_table_refresh()

    def _update_xy_with_event_table_refresh(self):
        self._force_next_event_table_refresh = True
        self.update_xy()

    def _on_events_table_selection(self, records):
        if not hasattr(self, "event_timeline"):
            return
        selected_ids = [record.id for record in records]
        self.event_timeline.set_selected_ids(selected_ids)
        if self._syncing_event_selection or not records or not self.events_table.sync_enabled:
            return
        start = min(record.start_seconds for record in records)
        stop = max(record.stop_seconds for record in records)
        center = (start + stop) / 2
        try:
            self._syncing_event_selection = True
            width_samples = max(200, int((stop - start) * self.fs_song * 1.25))
            if width_samples > self.span:
                self.span = width_samples
            self.t0 = self._seconds_sample(center)
        finally:
            self._syncing_event_selection = False

    def _on_timeline_event_selected(self, records):
        if not hasattr(self, "events_table"):
            return
        try:
            self._syncing_event_selection = True
            self.events_table.select_ids([record.id for record in records])
            self.event_timeline.set_selected_ids([record.id for record in records])
        finally:
            self._syncing_event_selection = False

    def _on_events_table_type_changed(self, records, new_name: str):
        records = [record for record in records if self._event_type_can_edit(record.name)]
        if records and self._event_type_can_edit(new_name):
            self._move_event_records(records, new_name=new_name)

    def _on_events_table_time_changed(self, record, start_seconds: float, stop_seconds: float, changed_edge: str):
        if (
            self._event_type_can_edit(record.name)
            and record.name in self.event_times
            and record.index < len(self.event_times[record.name])
        ):
            start_seconds, stop_seconds = self._bounds_for_event_edit(
                record.name,
                record.start_seconds,
                record.stop_seconds,
                start_seconds,
                stop_seconds,
                changed_edge=changed_edge,
            )
            self.event_times[record.name][record.index, :2] = [start_seconds, stop_seconds]
            self._after_event_edit()

    def _on_events_table_delete(self, records):
        by_name = {}
        for record in records:
            if not self._event_type_can_edit(record.name):
                continue
            by_name.setdefault(record.name, []).append(record.index)
        for name, indices in by_name.items():
            if name not in self.event_times:
                continue
            for index in sorted(indices, reverse=True):
                if index < len(self.event_times[name]):
                    self.event_times[name] = np.delete(self.event_times[name], index, axis=0)
        self._after_event_edit()

    def _on_timeline_event_created(self, name: str, start_seconds: float, stop_seconds: float):
        if not self._event_type_can_edit(name):
            return
        preset = self._event_preset(name)
        if not preset.fixed_duration:
            if np.isclose(start_seconds, stop_seconds):
                pending = self._pending_event_creation()
                if pending is None or pending[0] != name:
                    self.sinet0 = float(start_seconds)
                    self.sinet0_event_name = name
                    logger.info(f"  Started {name} at t={start_seconds:1.4f} seconds.")
                    self.update_xy()
                    return
                _name, pending_start = pending
                start_seconds, stop_seconds = pending_start, float(start_seconds)
            else:
                self._clear_pending_event_creation()
        channel = self.current_channel_index
        if channel is None:
            channel = -1
        start_seconds, stop_seconds = self._bounds_for_event_creation(name, start_seconds, stop_seconds)
        self._clear_pending_event_creation()
        self.event_times.add_time(name, start_seconds, stop_seconds, category="event", channel=channel)
        self._after_event_edit()

    def _on_timeline_event_changed(self, record, new_name: str, start_seconds: float, stop_seconds: float):
        if not self._event_type_can_edit(record.name) or not self._event_type_can_edit(new_name):
            return
        if new_name != record.name:
            self._move_event_records([record], new_name=new_name, start_seconds=start_seconds, stop_seconds=stop_seconds)
            return
        if record.name in self.event_times and record.index < len(self.event_times[record.name]):
            start_seconds, stop_seconds = self._bounds_for_event_edit(
                record.name,
                record.start_seconds,
                record.stop_seconds,
                start_seconds,
                stop_seconds,
                changed_edge="move",
            )
            self.event_times[record.name][record.index, :2] = [start_seconds, stop_seconds]
            self._after_event_edit()

    def _update_event_time_by_bounds(
        self, name: str, old_start: float, old_stop: float, new_start: float, new_stop: float
    ) -> bool:
        if name not in self.event_times:
            return False
        rows = self.event_times[name]
        hits = np.isclose(rows[:, 0], old_start) & np.isclose(rows[:, 1], old_stop)
        if not np.any(hits):
            return False
        rows[hits, :2] = [new_start, new_stop]
        return True

    def _move_event_records(self, records, new_name: str, start_seconds: float = None, stop_seconds: float = None):
        if not self._event_type_can_edit(new_name):
            return
        if new_name not in self.event_times:
            self.event_times.add_name(new_name, category="event")
        by_name = {}
        for record in records:
            if not self._event_type_can_edit(record.name):
                continue
            by_name.setdefault(record.name, []).append(record)
        for old_name, grouped in by_name.items():
            if old_name not in self.event_times:
                continue
            for record in sorted(grouped, key=lambda item: item.index, reverse=True):
                if record.index >= len(self.event_times[old_name]):
                    continue
                row = self.event_times[old_name][record.index].copy()
                self.event_times[old_name] = np.delete(self.event_times[old_name], record.index, axis=0)
                row[0] = record.start_seconds if start_seconds is None else start_seconds
                row[1] = record.stop_seconds if stop_seconds is None else stop_seconds
                row[0], row[1] = self._bounds_for_event_edit(
                    new_name,
                    record.start_seconds,
                    record.stop_seconds,
                    row[0],
                    row[1],
                    changed_edge="move",
                )
                self.event_times.add_time(new_name, row[0], row[1], category="event", channel=int(row[2]))
        self._after_event_edit()

    def _add_keyed_menuitem(
        self,
        parent,
        label: str,
        callback,
        qt_keycode=None,
        checkable=False,
        checked=True,
    ):
        """Add new action to menu and register key press."""
        menuitem = parent.addAction(label)
        menuitem.setCheckable(checkable)
        menuitem.setChecked(checked)
        if qt_keycode is not None:
            menuitem.setShortcut(qt_keycode)
        menuitem.triggered.connect(lambda: callback(qt_keycode))
        return menuitem

    def change_event_type(self, qt_keycode):
        """Select event to annotate using key presses (0-nb_events)."""
        key_pressed = QtGui.QKeySequence(qt_keycode).toString()  # numeric key code to actual char pressed
        old_event_name = getattr(self, "_current_event_name", None)
        try:
            key_index = int(key_pressed)
        except ValueError:  # if non-int pressed or int too large for index
            return
        if key_index == 0:
            self._current_event_name = None
        elif 0 < key_index <= len(self.eventList):
            self._current_event_name = self.eventList[key_index - 1][1]
        else:
            return
        if getattr(self, "_current_event_name", None) != old_event_name:
            self._clear_pending_event_creation()
        self._refresh_preset_panel(selected_name=self.current_event_name)
        self.update_xy()

    def toggle(self, var_name, qt_keycode):
        try:
            self.__dict__[var_name] = not self.__dict__[var_name]
            if var_name == "show_all_channels":
                current = self._audio_settings()
                self.audio_channel_settings = event_widgets.AudioChannelSettings(
                    waveform_all=bool(self.show_all_channels),
                    events_all=current.events_all,
                    playback_all=current.playback_all,
                    scale_y_all=current.scale_y_all,
                )
            if var_name == "threshold_mode":
                if self.threshold_mode:
                    self.show_sidebar = True
                self._sync_threshold_mode_ui()
            if self.STOP:
                self.update_frame()
                self.update_xy()
            self._apply_panel_visibility()
        except KeyError as e:
            logger.exception(e)

    def _sync_threshold_mode_ui(self) -> None:
        enabled = bool(getattr(self, "threshold_mode", False))
        if hasattr(self, "threshold_panel"):
            self.threshold_panel.setVisible(enabled)
            self._sync_threshold_panel()
        if not enabled and hasattr(self, "slice_view"):
            self.slice_view.set_threshold_data(None, None, enabled=False)

    def _sync_threshold_panel(self) -> None:
        if not hasattr(self, "threshold_panel"):
            return
        self.threshold_panel.set_limits(
            duration_max=self._threshold_duration_limit(),
            frequency_max=self.fs_song / 2,
        )
        self.threshold_panel.set_values(
            threshold=float(getattr(self, "thres_value", 0.0)),
            envelope_std=float(getattr(self, "thres_env_std", 0.0)),
            min_distance=float(getattr(self, "thres_min_dist", 0.0)),
            duration_enabled=bool(getattr(self, "thres_duration_enabled", False)),
            duration_range=self._threshold_duration_bounds(),
            bandpass_enabled=bool(getattr(self, "thres_bandpass_enabled", False)),
            bandpass_range=self._threshold_bandpass_bounds(),
        )

    def _on_threshold_value_changed(self, value: float) -> None:
        self.thres_value = max(0.0, float(value))
        if hasattr(self, "slice_view"):
            self.slice_view.set_threshold_value(self.thres_value)

    def _on_threshold_line_changed(self, value: float) -> None:
        self.thres_value = max(0.0, float(value))
        if hasattr(self, "threshold_panel"):
            self.threshold_panel.set_threshold(self.thres_value)

    def _on_threshold_envelope_std_changed(self, value: float) -> None:
        self.thres_env_std = max(float(value), 1 / self.fs_song)
        if self.STOP:
            self.update_xy()

    def _on_threshold_min_distance_changed(self, value: float) -> None:
        self.thres_min_dist = max(float(value), 1 / self.fs_song)
        self._sync_threshold_panel()

    def _on_threshold_duration_filter_changed(self, enabled: bool) -> None:
        self.thres_duration_enabled = bool(enabled)

    def _on_threshold_duration_range_changed(self, value) -> None:
        self.thres_duration_min, self.thres_duration_max = self._sorted_pair(value)
        self._sync_threshold_panel()

    def _on_threshold_bandpass_filter_changed(self, enabled: bool) -> None:
        self.thres_bandpass_enabled = bool(enabled)
        if self.STOP:
            self.update_xy()

    def _on_threshold_bandpass_range_changed(self, value) -> None:
        self.thres_bandpass_low, self.thres_bandpass_high = self._sorted_pair(value)
        self._sync_threshold_panel()
        if self.STOP and self.thres_bandpass_enabled:
            self.update_xy()

    def _sorted_pair(self, value) -> tuple[float, float]:
        low, high = (float(v) for v in value)
        return (low, high) if low <= high else (high, low)

    def _threshold_duration_limit(self) -> float:
        duration = self._sample_seconds(self.tmax_playhead) if getattr(self, "fs_song", 0) else 1.0
        return min(max(1.0, float(duration)), 100.0)

    def _threshold_duration_bounds(self) -> tuple[float, float]:
        low, high = self._sorted_pair(
            (
                getattr(self, "thres_duration_min", 0.0),
                getattr(self, "thres_duration_max", 1.0),
            )
        )
        limit = self._threshold_duration_limit()
        low = min(max(0.0, low), limit)
        high = max(low, min(high, limit))
        return low, high

    def _threshold_bandpass_bounds(self) -> tuple[float, float]:
        high_default = self.fs_song / 2 if getattr(self, "fs_song", 0) else 1.0
        high_value = getattr(self, "thres_bandpass_high", high_default)
        if high_value is None:
            high_value = high_default
        low, high = self._sorted_pair((getattr(self, "thres_bandpass_low", 0.0), high_value))
        nyquist = self.fs_song / 2
        low = min(max(0.0, low), nyquist)
        high = max(low, min(high, nyquist))
        return low, high

    def delete_current_events(self, qt_keycode):
        if self.current_event_index is not None:
            if not self._event_type_can_edit(self.current_event_name):
                logger.info(f"   Event type {self.current_event_name} is hidden or locked. Not deleting anything.")
                return
            deleted_events = self.event_times.delete_range(
                self.current_event_name,
                self._sample_seconds(self.time0),
                self._sample_seconds(max(self.time0, self.time1 - 1)),
            )
            nb_deleted_events = len(deleted_events)
            if nb_deleted_events:
                logger.info(f"   Deleted {nb_deleted_events} annotation(s) of type {self.current_event_name}.")
                if self.STOP:
                    self._update_xy_with_event_table_refresh()
        else:
            logger.info("   No event type selected. Not deleting anything.")

    def delete_all_events(self, qt_keycode):
        for event_name in self.event_times.names:
            if not self._event_type_can_edit(event_name):
                continue
            deleted_events = self.event_times.delete_range(
                event_name,
                self._sample_seconds(self.time0),
                self._sample_seconds(max(self.time0, self.time1 - 1)),
            )
            nb_deleted_events = len(deleted_events)
            if nb_deleted_events:
                logger.info(f"   Deleted {nb_deleted_events} annotation(s) of type {event_name}.")

        if self.STOP:
            self._update_xy_with_event_table_refresh()

    def threshold(self, qt_keycode):
        if self.STOP and self.current_event_name is not None and self._event_type_can_edit(self.current_event_name):
            if self.envelope is None:
                self.envelope = self.get_envelope()
            if self.envelope is None or len(self.envelope) == 0:
                return
            self.thres_value = self.slice_view.threshold
            if self.thres_duration_enabled:
                intervals = self._threshold_interval_proposals()
                for start, stop in intervals:
                    self.event_times.add_time(self.current_event_name, start, stop)
                    logger.info(f"   Added {self.current_event_name} from t={start:1.4f} to {stop:1.4f} seconds.")
            else:
                min_dist = max(1, int(round(self.thres_min_dist * self.fs_song)))
                indexes, _ = scipy.signal.find_peaks(
                    self.envelope,
                    height=self.thres_value,
                    distance=min_dist,
                )
                for t in self.x[indexes]:
                    self.event_times.add_time(self.current_event_name, t)
                    logger.info(f"   Added {self.current_event_name} at t={t:1.4f} seconds.")
            old_len = self.event_times[self.current_event_name].shape[0]
            self.event_times[self.current_event_name] = np.unique(self.event_times[self.current_event_name], axis=0)
            new_len = self.event_times[self.current_event_name].shape[0]
            if new_len != old_len:
                logger.info(f"   Removed {old_len - new_len} duplicates in {self.current_event_name}.")

            self._update_xy_with_event_table_refresh()

    def _threshold_interval_proposals(self) -> list[tuple[float, float]]:
        above = np.asarray(self.envelope) >= self.thres_value
        if above.size == 0 or not np.any(above):
            return []
        padded = np.concatenate(([False], above, [False]))
        changes = np.diff(padded.astype(int))
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        runs = self._merge_threshold_runs(list(zip(starts, stops)))
        min_duration, max_duration = self._threshold_duration_bounds()
        sample_period = 1 / self.fs_song
        intervals = []
        for start_index, stop_index in runs:
            start = float(self.x[start_index])
            last_index = min(stop_index - 1, len(self.x) - 1)
            stop = min(float(self.x[-1]), float(self.x[last_index]) + sample_period)
            duration = max(0.0, stop - start)
            if duration < min_duration or duration > max_duration:
                continue
            intervals.append((start, stop))
        return intervals

    def _merge_threshold_runs(self, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not runs:
            return []
        gap_samples = max(0, int(round(self.thres_min_dist * self.fs_song)))
        merged = [runs[0]]
        for start, stop in runs[1:]:
            prev_start, prev_stop = merged[-1]
            if start - prev_stop <= gap_samples:
                merged[-1] = (prev_start, stop)
            else:
                merged.append((start, stop))
        return merged

    def get_envelope(self):
        y = self._threshold_signal()
        std = self.thres_env_std * self.fs_song
        win = scipy.signal.windows.gaussian(int(std * 6), std)
        win /= np.sum(win)
        env = np.sqrt(np.convolve(y**2, win, mode="same"))
        return env

    def _threshold_signal(self) -> np.ndarray:
        y = np.asarray(self.y, dtype=float)
        if not self.thres_bandpass_enabled:
            return y
        low, high = self._threshold_bandpass_bounds()
        nyquist = self.fs_song / 2
        if high <= low or (low <= 0 and high >= nyquist):
            return y
        if low <= 0:
            btype = "lowpass"
            cutoff = high / nyquist
        elif high >= nyquist:
            btype = "highpass"
            cutoff = low / nyquist
        else:
            btype = "bandpass"
            cutoff = [low / nyquist, high / nyquist]
        sos = scipy.signal.butter(4, cutoff, btype=btype, output="sos")
        try:
            return scipy.signal.sosfiltfilt(sos, y)
        except ValueError:
            return scipy.signal.sosfilt(sos, y)

    def set_prev_channel(self, qt_keycode):
        if self._channel_switching_locked():
            return
        idx = self.cb2.currentIndex()
        idx -= 1
        idx = idx % self.cb2.count()

        old_status = self.select_loudest_channel
        self.select_loudest_channel = False
        self.cb2.setCurrentIndex(idx)
        self.select_loudest_channel = old_status

    def set_next_channel(self, qt_keycode):
        if self._channel_switching_locked():
            return
        idx = self.cb2.currentIndex()
        idx += 1
        idx = idx % self.cb2.count()

        old_status = self.select_loudest_channel
        self.select_loudest_channel = False
        self.cb2.setCurrentIndex(idx)
        self.select_loudest_channel = old_status

    def inc_freq_res(self, qt_keycode):
        self.spec_win = int(self.spec_win * 2)
        if self.STOP:
            # need to update twice to fix axis limits for some reason
            self.update_xy()

    def dec_freq_res(self, qt_keycode):
        self.spec_win = int(max(2, self.spec_win // 2))
        if self.STOP:
            # need to update twice to fix axis limits for some reason
            self.update_xy()

    def toggle_playvideo(self, qt_keycode=None):
        self._toggle_playback()

    def change_focal_fly(self, qt_keycode):
        tmp = (self.focal_fly + 1) % self.nb_flies
        if tmp == self.other_fly:  # swap focal and other fly if same
            self.other_fly, self.focal_fly = self.focal_fly, self.other_fly
        else:
            self.focal_fly = tmp
        if self.STOP:
            self.update_frame()

    def change_other_fly(self, qt_keycode):
        tmp = (self.other_fly + 1) % self.nb_flies
        if tmp == self.focal_fly:  # skip focal fly if same
            tmp = tmp + 1
        self.other_fly = tmp
        if self.STOP:
            self.update_frame()

    def set_prev_cuepoint(self, qt_keycode):
        # if self.edit_only_current_events:  # of the currently active type
        #     names = [self.current_event_name]
        # else:  # of any type
        #     names = self.event_times.names
        names = [self.current_event_name]
        t = self._sample_seconds(self.t0 - 1)
        nxt = self.event_times.find_prev(t, names)

        if nxt is not None:
            self.t0 = self._seconds_sample(nxt)

    def set_next_cuepoint(self, qt_keycode):
        # if self.edit_only_current_events:  # of the currently active type
        #     names = [self.current_event_name]
        # else:  # of any type
        #     names = self.event_times.names
        names = [self.current_event_name]
        t = self._sample_seconds(self.t0 + 1)
        nxt = self.event_times.find_next(t, names)

        if nxt is not None:
            self.t0 = self._seconds_sample(nxt)

    def zoom_in_song(self, qt_keycode):
        self.span /= 2

    def zoom_out_song(self, qt_keycode):
        self.span *= 2

    def single_frame_reverse(self, qt_keycode):
        self._transport_frame_seek(-1)

    def single_frame_advance(self, qt_keycode):
        self._transport_frame_seek(1)

    def jump_reverse(self, qt_keycode):
        self.t0 -= self.span / 2

    def jump_forward(self, qt_keycode):
        self.t0 += self.span / 2

    def set_envelope_computation(self, qt_keycode):
        dialog = YamlDialog(
            yaml_file=package_dir + "/gui/forms/envelope_computation.yaml",
            title="Set options for envelope computation",
        )

        dialog.form["thres_min_dist"] = self.thres_min_dist
        dialog.form["thres_env_std"] = self.thres_env_std

        dialog.show()
        result = dialog.exec_()

        if result == QtWidgets.QDialog.Accepted:
            form_data = dialog.form.get_form_data()
            # fix these to be at least 1/fs audio
            self.thres_min_dist = max(form_data["thres_min_dist"], 1 / self.fs_song)
            self.thres_env_std = max(form_data["thres_env_std"], 1 / self.fs_song)
            logger.info("Setting parameters for envelope computation:")
            logger.info(f"     Minimal distance between events: {self.thres_min_dist} seconds")
            logger.info(f"     Smoothing window for envelope: {self.thres_env_std} seconds")
            self._sync_threshold_panel()
            self.update_xy()

    def update_xy(self):
        da = self._audio_dataarray_for_source()
        if da is None:
            return
        self.x = self._audio_time_values()[self.time0 : self.time1]
        self.step = int(max(1, np.ceil(len(self.x) / self.fs_song / 2)))  # make sure step is >= 1
        self.y_other = None

        if getattr(da, "ndim", 0) == 1:
            self.y = da.data[self.time0 : self.time1]
        else:
            # load song for current channel
            try:
                y_all = da.data[self.time0 : self.time1, :].compute()
            except AttributeError:
                y_all = da.data[self.time0 : self.time1, :]

            channel_index = self._current_channel_data_index()
            if channel_index is None:
                return
            self.y = y_all[:, channel_index]
            if self._audio_settings().waveform_all:
                channel_list = np.delete(np.arange(self.nb_channels), channel_index)
                self.y_other = y_all[:, channel_list]

            if self.select_loudest_channel and not self._channel_switching_locked():
                self.loudest_channel = np.argmax(np.max(y_all, axis=0))
                combo_index = self._combo_index_for_audio_item(self.current_audio_source_name, int(self.loudest_channel))
                if combo_index >= 0:
                    self.cb2.setCurrentIndex(combo_index)

        if self.threshold_mode:
            self.envelope = self.get_envelope()
        else:
            self.envelope = None

        if hasattr(self, "preset_panel"):
            self.preset_panel.set_current_name(self.current_event_name)

        if self.show_trace:
            self.slice_view.set_waveform(
                self.x,
                self.y,
                y_other=self.y_other,
                scale_y_all=self._audio_settings().scale_y_all,
            )
            self.slice_view.set_threshold_data(
                self.x,
                self.envelope,
                enabled=self.threshold_mode,
                threshold=self.thres_value,
            )
            self.slice_view.set_playhead(self._sample_seconds(self.t0))
            self.slice_view.clear_annotations()
            self.slice_view.show()
        else:
            self.slice_view.set_threshold_data(None, None, enabled=False)
            self.slice_view.clear_annotations()
            self.slice_view.hide()

        if hasattr(self, "event_timeline"):
            self.event_timeline.set_waveform(
                self.x,
                self.y,
                y_other=self.y_other,
                scale_y_all=self._audio_settings().scale_y_all,
            )

        if "pose_positions_allo" in self.ds:
            if self.show_tracks:
                # make this part of the callback?
                sel_parts = self.cb3.currentData()
                self.track_sel_names = []
                self.track_sel_coords = []
                for part in sel_parts:
                    self.track_sel_names.append(self.bodyparts.tolist().index(part[:-3]))
                    self.track_sel_coords.append(0 if part[-1] == "x" else 1)

                i0 = int(self.time0 / self.fs_ratio)
                i1 = int(self.time1 / self.fs_ratio)

                self.x_tracks = self.ds.time.data[i0:i1]
                self.y_tracks = self.ds.pose_positions_allo.data[
                    i0:i1, self.focal_fly, self.track_sel_names, self.track_sel_coords
                ]
                self.tracks_view.update_trace()
                self.tracks_view.show()
            else:
                self.tracks_view.clear()
                self.tracks_view.hide()

        self.spec_view.clear_annotations()
        if self.show_spec:
            self.spec_view.update_spec(self.x, self.y)
            self.spec_view.show()
        else:
            self.spec_view.clear()
            self.spec_view.hide()

        if self.show_songevents and (self.show_tracks or self.show_spec or self.show_trace):
            self.plot_song_events(self.x)

        force_event_table = bool(getattr(self, "_force_next_event_table_refresh", False))
        self._force_next_event_table_refresh = False
        self._refresh_event_widgets(sync_table_to_view=True, force_table=force_event_table)

    def update_frame(self):
        if self.movie_view is not None:
            if self.show_movie:
                self.movie_view.update_frame()
                self.movie_view.show()
            else:
                self.movie_view.hide()

    def plot_song_events(self, x):
        for event_index in range(self.nb_eventtypes):
            event_name = self.event_times.names[event_index]
            if not self._event_type_visible(event_name):
                continue
            movable = self.STOP and self.movable_events and self._event_type_editable(event_name)
            if self.edit_only_current_events:
                movable = movable and self.current_event_index == event_index

            event_pen = pg.mkPen(color=self.eventtype_colors[event_index], width=3)
            event_brush = pg.mkBrush(color=[*self.eventtype_colors[event_index], 25])
            events_in_view = self.event_times.filter_range(event_name, x[0], x[-1], strict=False)

            if self.show_event_text:
                event_text = event_name
            else:
                event_text = None

            events_in_view = self._filter_event_rows_for_audio_channel(events_in_view)
            point_like = events_in_view[:, 0] == events_in_view[:, 1] if len(events_in_view) else []
            interval_events = events_in_view[~point_like] if len(events_in_view) else []
            point_events = events_in_view[point_like] if len(events_in_view) else []
            if len(interval_events):
                for onset, offset in zip(interval_events[:, 0], interval_events[:, 1]):
                    if self.show_trace:
                        self.slice_view.add_segment(
                            onset,
                            offset,
                            event_index,
                            brush=event_brush,
                            pen=event_pen,
                            movable=movable,
                            text=event_text,
                        )
                    if self.show_tracks:
                        self.tracks_view.add_segment(
                            onset,
                            offset,
                            event_index,
                            brush=event_brush,
                            pen=event_pen,
                            movable=movable,
                            text=event_text,
                        )
                    if self.show_spec:
                        self.spec_view.add_segment(
                            onset,
                            offset,
                            event_index,
                            brush=event_brush,
                            pen=event_pen,
                            movable=movable,
                            text=event_text,
                        )
            if len(point_events):
                if self.show_trace:
                    self.slice_view.add_event(
                        point_events[:, 0],
                        event_index,
                        event_pen,
                        movable=movable,
                        text=event_text,
                    )
                if self.show_tracks:
                    self.tracks_view.add_event(
                        point_events[:, 0],
                        event_index,
                        event_pen,
                        movable=movable,
                        text=event_text,
                    )
                if self.show_spec:
                    self.spec_view.add_event(
                        point_events[:, 0],
                        event_index,
                        event_pen,
                        movable=movable,
                        text=event_text,
                    )
        self._plot_pending_event_boundary(x)

    def _plot_pending_event_boundary(self, x):
        pending = self._pending_event_creation()
        if pending is None or len(x) == 0:
            return
        event_name, seconds = pending
        if seconds < x[0] or seconds > x[-1]:
            return
        event_index = self.event_times.names.index(event_name)
        event_pen = pg.mkPen(color=self.eventtype_colors[event_index], width=3)
        event_text = event_name if self.show_event_text else None
        xx = np.array([seconds], dtype=float)
        if self.show_trace:
            self.slice_view.add_event(xx, event_index, event_pen, movable=False, text=event_text)
        if self.show_tracks:
            self.tracks_view.add_event(xx, event_index, event_pen, movable=False, text=event_text)
        if self.show_spec:
            self.spec_view.add_event(xx, event_index, event_pen, movable=False, text=event_text)

    def on_region_change_finished(self, region):
        """Called when dragging an interval event - will change its bounds."""
        if self.edit_only_current_events and self.current_event_index != region.event_index:
            return

        event_name_to_move = self.current_event_name
        if self.current_event_index != region.event_index:
            event_name_to_move = self.event_times.names[region.event_index]
        if not self._event_type_can_edit(event_name_to_move):
            return

        new_region = region.getRegion()
        changed_edge = "move"
        old_delta = region.bounds[1] - region.bounds[0]
        new_delta = new_region[1] - new_region[0]
        if not np.isclose(old_delta, new_delta):
            if abs(new_region[0] - region.bounds[0]) >= abs(new_region[1] - region.bounds[1]):
                changed_edge = "start"
            else:
                changed_edge = "stop"
        new_region = self._bounds_for_event_edit(
            event_name_to_move,
            region.bounds[0],
            region.bounds[1],
            new_region[0],
            new_region[1],
            changed_edge=changed_edge,
        )
        if not self._update_event_time_by_bounds(
            event_name_to_move,
            region.bounds[0],
            region.bounds[1],
            new_region[0],
            new_region[1],
        ):
            self.event_times.move_time(event_name_to_move, region.bounds, new_region)
        logger.info(
            f"  Moved {event_name_to_move} from t=[{region.bounds[0]:1.4f}:{region.bounds[1]:1.4f}] to [{new_region[0]:1.4f}:{new_region[1]:1.4f}] seconds."
        )

        self._update_xy_with_event_table_refresh()

    def on_position_change_finished(self, position):
        """Called when dragging an event-like song_event - will change time."""
        if self.edit_only_current_events and self.current_event_index != position.event_index:
            return
        event_name_to_move = self.current_event_name
        if self.current_event_index != position.event_index:
            event_name_to_move = self.event_times.names[position.event_index]
        if not self._event_type_can_edit(event_name_to_move):
            return
        new_point = position.pos()
        new_position = new_point.x() if hasattr(new_point, "x") else new_point[0]
        new_position, new_stop = self._bounds_for_event_edit(
            event_name_to_move,
            position.position,
            position.position,
            new_position,
            new_position,
            changed_edge="move",
        )
        if not self._update_event_time_by_bounds(
            event_name_to_move,
            position.position,
            position.position,
            new_position,
            new_stop,
        ):
            self.event_times.move_time(event_name_to_move, position.position, new_position)
        logger.info(f"  Moved {event_name_to_move} from t={position.position:1.4f} to {new_position:1.4f} seconds.")

        self._update_xy_with_event_table_refresh()

    def on_position_dragged(self, fly, pos, offset):
        """Called when dragging a fly body position - will change that pos."""
        if hasattr(self.ds, "pose_positions_allo"):
            pos0 = self.ds.pose_positions_allo.data[self.index_other, fly, self.pose_center_index]
            try:
                pos1 = [pos.y(), pos.x()]
            except:
                pos1 = pos
            self.ds.pose_positions_allo.data[self.index_other, fly, :] += pos1 - pos0
            logger.info(f"   Moved fly from {pos0} to {pos1}.")
            self.update_frame()

    def on_poses_dragged(self, ind, pos, offset):
        """Called when dragging a fly body position - will change that pos."""
        if hasattr(self.ds, "pose_positions_allo"):
            fly, part = np.unravel_index(ind, (self.nb_flies, self.nb_bodyparts))
            pos0 = self.ds.pose_positions_allo.data[self.index_other, fly, part]
            try:
                pos1 = [pos.y(), pos.x()]
            except:
                pos1 = pos
            self.ds.pose_positions_allo.data[self.index_other, fly, part] += pos1 - pos0
            logger.info(f"   Moved {self.ds.poseparts[part].data} of fly {fly} from {pos0} to {pos1}.")
            self.update_frame()

    def on_video_clicked(self, mouseX, mouseY, event):
        """Called when clicking the video - will select the focal fly."""
        if hasattr(self.ds, "pose_positions_allo"):
            if event.modifiers() == QtCore.Qt.ControlModifier and self.focal_fly is not None:
                self.on_position_dragged(self.focal_fly, pos=[mouseY, mouseX], offset=None)
            else:
                fly_pos = self.ds.pose_positions_allo.data[self.index_other, :, self.pose_center_index, :]
                fly_pos = np.array(fly_pos)  # in case this is a dask.array
                if self.crop:  # transform fly pos to coordinates of the cropped box
                    box_center = (
                        self.ds.pose_positions_allo.data[self.index_other, self.focal_fly, self.pose_center_index]
                        + self.box_size / 2
                    )
                    box_center = np.array(box_center)  # in case this is a dask.array
                    fly_pos = fly_pos - box_center
                fly_dist = np.sum((fly_pos - np.array([mouseY, mouseX])) ** 2, axis=-1)
                fly_dist[self.focal_fly] = np.inf  # ensure that other_fly is not focal_fly
                self.other_fly = np.argmin(fly_dist)
                logger.debug(f"Selected {self.other_fly}.")
            self.update_frame()

    def on_trace_clicked(self, mouseT, mouseButton):
        """Called when traceview or specview have been clicked - will add new
        song event at click position.
        """
        if self.current_event_index is None:
            msgbox = NoEventsRegisteredWarning(parent=self)
            msgbox.exec()
            if msgbox.clickedButton() == msgbox.button:
                self._create_preset_from_panel()

        modifiers = QtWidgets.QApplication.keyboardModifiers()

        if mouseButton == QtCore.Qt.MouseButton.LeftButton and modifiers == QtCore.Qt.ControlModifier:  # change event type
            self._clear_pending_event_creation()

            if not self.edit_only_current_events or not self._event_type_can_edit(self.current_event_name):
                return
            else:
                current_event_name = self.current_event_name

            editable_events = self._event_times_for_names(self._editable_visible_event_names())
            old_name = editable_events._get_name_of_nearest(
                mouseT,
                min_time=self._sample_seconds(self.time0),
                max_time=self._sample_seconds(max(self.time0, self.time1 - 1)),
            )
            if old_name is None:
                return
            changed_time, old_name, new_name = self.event_times.change_name(
                time=mouseT,
                new_name=current_event_name,
                tol=0.05,
                min_time=self._sample_seconds(self.time0),
                max_time=self._sample_seconds(max(self.time0, self.time1 - 1)),
                old_name=old_name,
            )
            if changed_time is not None:
                logger.info(f"  Changed event at {changed_time[0]:1.4f}:{changed_time[1]:1.4f} from {old_name} to {new_name}.")
                self._update_xy_with_event_table_refresh()
        elif mouseButton == QtCore.Qt.MouseButton.LeftButton:  # add event
            if self.current_event_index is not None and self._event_type_can_edit(self.current_event_name):
                preset = self._event_preset(self.current_event_name)
                if preset.fixed_duration:
                    self._clear_pending_event_creation()
                    start_seconds, stop_seconds = self._bounds_for_event_creation(self.current_event_name, mouseT)
                elif self._pending_event_creation() is None:
                    self.sinet0 = float(mouseT)
                    self.sinet0_event_name = self.current_event_name
                    logger.info(f"  Started {self.current_event_name} at t={mouseT:1.4f} seconds.")
                    self.update_xy()
                    return
                else:
                    _name, pending_start = self._pending_event_creation()
                    start_seconds, stop_seconds = self._bounds_for_event_creation(
                        self.current_event_name, pending_start, mouseT
                    )
                    self._clear_pending_event_creation()
                self.event_times.add_time(
                    self.current_event_name,
                    start_seconds=start_seconds,
                    stop_seconds=stop_seconds,
                    channel=self.current_channel_index,
                )
                logger.info(
                    f"  Added {self.current_event_name} on channel {self.current_channel_index} "
                    f"at t={start_seconds:1.4f}:{stop_seconds:1.4f} seconds."
                )
                self._update_xy_with_event_table_refresh()
            else:
                self._clear_pending_event_creation()
        elif mouseButton == QtCore.Qt.MouseButton.RightButton:  # delete nearest event
            self.spec_view.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            self.slice_view.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            self._clear_pending_event_creation()

            if not self.edit_only_current_events:
                editable_events = self._event_times_for_names(self._editable_visible_event_names())
                current_event_name = editable_events._get_name_of_nearest(
                    mouseT,
                    min_time=self._sample_seconds(self.time0),
                    max_time=self._sample_seconds(max(self.time0, self.time1 - 1)),
                )
            else:
                current_event_name = self.current_event_name

            if current_event_name is None or not self._event_type_can_edit(current_event_name):
                return
            deleted_name, deleted_time = self.event_times.delete_time(
                time=mouseT,
                name=current_event_name,
                tol=0.05,
                min_time=self._sample_seconds(self.time0),
                max_time=self._sample_seconds(max(self.time0, self.time1 - 1)),
            )
            if len(deleted_time):
                logger.info(f"  Deleted {deleted_name} at t={deleted_time[0]:1.4f}:{deleted_time[1]:1.4f} seconds.")
            self._update_xy_with_event_table_refresh()

    def play_audio(self, qt_keycode):
        """Play the visible audio window using Qt audio."""
        if self._audio_dataarray_for_source() is not None:
            window_start = self.time0
            window_stop = self.time1
            self._start_window_audio_playhead(
                window_start,
                window_stop,
                all_channels=self._audio_playback_all_channels(),
            )
        else:
            logger.info("Could not play sound - no sound data in the dataset.")

    def swap_flies(self, qt_keycode):
        if self.vr is not None:
            swap_time = float(self.ds.time[self.index_other])
            logger.info(f"   Swapping flies {self.focal_fly} & {self.other_fly} at {swap_time} seconds.")

            # save swap info
            # if already in there remove - swapping a second time would negate first swap
            if [self.t0, self.focal_fly, self.other_fly] in self.swap_events:
                self.swap_events.remove([swap_time, self.focal_fly, self.other_fly])
            else:
                self.swap_events.append([swap_time, self.focal_fly, self.other_fly])

            # swap flies
            self.ds = ld.swap_flies(self.ds, [swap_time], self.focal_fly, self.other_fly)

            self.update_frame()

    def approve_active_proposals(self, qt_keycode):
        self.approve_proposals(appprove_only_active_event=True)

    def approve_all_proposals(self, qt_keycode):
        self.approve_proposals(appprove_only_active_event=False)

    def approve_proposals(self, appprove_only_active_event: bool = False):
        audio_times = self._audio_time_values()
        t0 = audio_times[self.time0]
        t1 = audio_times[min(self.time1, self.tmax_playhead)]

        proposal_suffix = "_proposals"
        logger.info("Approving:")
        for name in self.event_times.names:
            if appprove_only_active_event and name != self.current_event_name:
                continue

            if name.endswith(proposal_suffix):
                # get event times within range
                within_range_times = self.event_times.filter_range(name, t0, t1, strict=False)
                # delete from `songtype_proposals`, add to `songtype`
                self.event_times.add_name(
                    name=name[: -len(proposal_suffix)],
                    category="event",
                    times=within_range_times,
                    append=True,
                    overwrite=False,
                )
                self.event_times.delete_range(name, t0, t1, strict=False)
                if len(within_range_times):
                    logger.info(f"   {len(within_range_times)} events of {name} to {name[: -len(proposal_suffix)]}")
        # update event selector in case the event did not exist yet
        for name in self.event_times.names:
            self._event_preset(name)
        self._sync_event_colors_from_presets()
        self.update_eventtype_selector()

        logger.info("Done.")
        self._update_xy_with_event_table_refresh()

    def update_eventtype_selector(self, selected_name: str = None):
        old_event_name = getattr(self, "_current_event_name", None)
        if selected_name is None:
            try:
                selected_name = self.current_event_name
            except Exception:
                selected_name = None

        if hasattr(self, "event_times"):
            self.eventList = [(cnt, evt) for cnt, evt in enumerate(self.event_times.names)]
            self.eventList = sorted(self.eventList)
        else:
            self.eventList = []

        names = [event_name for _event_index, event_name in self.eventList]
        if selected_name in names:
            self._current_event_name = selected_name
        elif getattr(self, "_current_event_name", None) not in names:
            self._current_event_name = names[-1] if names else None
        if getattr(self, "_current_event_name", None) != old_event_name:
            self._clear_pending_event_creation()

        # update menus
        # remove associated menu items
        if not hasattr(self, "event_items"):
            self.event_items = []
        else:
            for event_item in self.event_items:
                try:
                    self.view_audio.removeAction(event_item)
                except ValueError:
                    logger.warning("item not in actions")  # item not in actions

        # add new ones (make this function)
        self.event_items = []
        menu_labels = ["No annotation", *[f"Add {event_name}" for event_name in names]]
        for ii, label in enumerate(menu_labels):
            key = str(ii) if ii < 10 else None
            key_label = f"({key})" if key is not None else ""
            menu_item = self._add_keyed_menuitem(self.view_audio, f"{label} {key_label}", self.change_event_type, key)
            self.event_items.append(menu_item)

        self._refresh_preset_panel(selected_name=self.current_event_name)


def main(
    source: str = "",
    *,
    config: Optional[str] = None,
    events_string: str = "",
    target_samplingrate: Optional[float] = None,
    spec_freq_min: Optional[float] = None,
    spec_freq_max: Optional[float] = None,
    box_size: int = 200,
    pixel_size_mm: Optional[float] = None,
    manifest: Optional[str] = None,
    skip_dialog: bool = False,
    is_das: bool = False,
):
    """
    Args:
        source (str): Data source to load.
            Optional - will open an empty GUI if omitted.
            Source can be the path to:
            - an audio file,
            - a numpy file (npy or npz),
            - an h5 file
            - an xarray-behave dataset constructed from an ethodrome data folder saved as a zarr file,
            - an ethodrome data folder (e.g. 'dat/localhost-xxx').
        config (Optional[str]): Additional GUI configuration file. Overrides global and local configuration.
        events_string (str): Initialize event names for annotations.
                             String of the form "event_name;event_name".
                             Avoid spaces or trailing ';'.
                             Need to wrap the string in "..." in the terminal
                             "event_name" can be any string w/o space, ",", or ";"
        target_samplingrate (Optional[float]): [description]. If 0, will use frame times. Defaults to None.
                                     Only used if source is a data folder or a wav audio file.
        spec_freq_min (Optional[float]): Smallest frequency displayed in the spectrogram view.
                                       With skip_dialog, also sets the lower bandpass cutoff. Defaults to 0 Hz.
        spec_freq_max (Optional[float]): Largest frequency displayed in the spectrogram view.
                                       With skip_dialog, also sets the upper bandpass cutoff. Defaults to samplerate/2.
        box_size (int): Crop size around tracked fly. Not used for wav audio files (no videos).
        pixel_size_mm (Optional[float]): Size of a pixel (in mm) in the video. Used to convert tracking data to mm.
        manifest (Optional[str]): YAML manifest for discovering files when source is a data folder.
        skip_dialog (bool): If True, skips the loading dialog and goes straight to the data view.
        is_das (bool): reduced GUI for audio only data
    """
    app = pg.mkQApp()
    config_manager = gui_config.GuiConfigManager(config)
    config_manager.load_for_source(source or None)
    _set_config_manager(config_manager)

    mainwin = None
    if not len(source):
        mainwin = MainWindow(media_manifest=manifest, is_das=is_das)
        mainwin.show()
    elif not os.path.exists(source):
        logger.info(f"{source} does not exist - skipping.")
    elif source.lower().endswith(project_model.PROJECT_SUFFIX):
        mainwin = MainWindow.open_project(filename=source)
    elif is_das and Path(source).suffix.lower() in project_model.AUDIO_SUFFIXES:
        mainwin = MainWindow.new_project_from_file(
            filename=source,
            events_string=events_string,
            spec_freq_min=spec_freq_min,
            spec_freq_max=spec_freq_max,
        )
    elif (
        source.lower().endswith(".wav")
        or source.lower().endswith(".npz")
        or source.lower().endswith(".h5")
        or source.lower().endswith(".mmap")
        or source.endswith(".hdf5")
        or source.lower().endswith(".mat")
    ):
        mainwin = MainWindow.from_file(
            filename=source,
            events_string=events_string,
            target_samplingrate=target_samplingrate,
            spec_freq_min=spec_freq_min,
            spec_freq_max=spec_freq_max,
            skip_dialog=skip_dialog,
            is_das=is_das,
        )
    elif source.endswith(".zarr"):
        mainwin = MainWindow.from_zarr(
            filename=source,
            box_size=box_size,
            spec_freq_min=spec_freq_min,
            spec_freq_max=spec_freq_max,
            skip_dialog=skip_dialog,
            is_das=is_das,
        )
    elif os.path.isdir(source):
        if is_das and project_model.audio_files_in_folder(source):
            mainwin = MainWindow.new_project_from_folder(
                dirname=source,
                events_string=events_string,
                spec_freq_min=spec_freq_min,
                spec_freq_max=spec_freq_max,
            )
        else:
            mainwin = MainWindow.from_dir(
                dirname=source,
                events_string=events_string,
                target_samplingrate=target_samplingrate,
                box_size=box_size,
                spec_freq_min=spec_freq_min,
                spec_freq_max=spec_freq_max,
                pixel_size_mm=pixel_size_mm,
                manifest=manifest,
                skip_dialog=skip_dialog,
                is_das=is_das,
            )
    app._xarray_behave_mainwin = mainwin

    # # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtWidgets.QApplication.instance().exec_()


def main_das(
    source: str = "",
    *,
    config: Optional[str] = None,
    song_types_string: str = "",
    spec_freq_min: Optional[float] = None,
    spec_freq_max: Optional[float] = None,
    skip_dialog: bool = False,
):
    """GUI for annotating song and training and using das networks.

    Args:
        source (str): Data source to load.
            Optional - will open an empty GUI if omitted.
            Source can be the path to:
            - an audio file,
            - a folder containing wav files,
            - a numpy file (npy or npz),
            - an h5 file
            - an xarray-behave dataset constructed from an ethodrome data folder saved as a zarr file,
            - an ethodrome data folder (e.g. 'dat/localhost-xxx').
        config (Optional[str]): Additional GUI configuration file. Overrides global and local configuration.
        song_types_string (str): Initialize event names for annotations.
                             String of the form "event_name;event_name".
                             Avoid spaces or trailing ';'.
                             Need to wrap the string in "..." in the terminal
                             "event_name" can be any string w/o space, ",", or ";"
        spec_freq_min (Optional[float]): Smallest frequency displayed in the spectrogram view.
                                       With skip_dialog, also sets the lower bandpass cutoff. Defaults to 0 Hz.
        spec_freq_max (Optional[float]): Largest frequency displayed in the spectrogram view.
                                       With skip_dialog, also sets the upper bandpass cutoff. Defaults to samplerate/2.
        skip_dialog (bool): If True, skips the loading dialog and goes straight to the data view.
    """
    main(
        source,
        config=config,
        events_string=song_types_string,
        spec_freq_min=spec_freq_min,
        spec_freq_max=spec_freq_max,
        skip_dialog=skip_dialog,
        is_das=True,
    )


def cli():
    import warnings

    warnings.filterwarnings("ignore")
    # enforce log level
    try:  # py38+
        logging.basicConfig(level=logging.INFO)
    except ValueError:  # <py38
        logger.getLogger().setLevel(logging.INFO)

    defopt.run(main, show_defaults=False)


if __name__ == "__main__":
    # main_das()
    cli()
