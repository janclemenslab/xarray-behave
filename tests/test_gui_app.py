from types import SimpleNamespace

import numpy as np
import xarray as xr

from xarray_behave import annot
from xarray_behave.gui import app as gui_app, gui_config, project


class _FakeForm:
    def __init__(self):
        self.data = {
            "target_samplingrate": 1000,
            "ignore_tracks": False,
            "fix_fly_indices": True,
            "video_filename": "",
            "frame_fliplr": False,
            "frame_flipud": False,
            "box_size_px": 200,
            "pixel_size_mm": None,
            "daq_filename": "",
            "ignore_song": False,
            "spec_freq_min": None,
            "spec_freq_max": None,
            "annotation_path": "",
            "init_annotations": False,
            "events_string": "",
            "filter_song": "no",
        }

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def get_form_data(self):
        return dict(self.data)

    def set_form_data(self, data):
        self.data.update(data)


class _FakeDialog:
    def __init__(self, *args, **kwargs):
        self.form = _FakeForm()


def test_main_das_folder_imports_unsaved_project(tmp_path, monkeypatch):
    (tmp_path / "b.wav").write_bytes(b"")
    (tmp_path / "a.wav").write_bytes(b"")
    captured = {}
    fake_app = SimpleNamespace(
        exec_=lambda: captured.setdefault("event_loop", True),
        processEvents=lambda: None,
    )
    replacement = object()

    def fake_from_folder(cls, **kwargs):
        captured.update(kwargs)
        return replacement

    monkeypatch.setattr(gui_app.pg, "mkQApp", lambda: fake_app)
    monkeypatch.setattr(gui_app.QtWidgets.QApplication, "instance", staticmethod(lambda: fake_app))
    monkeypatch.setattr(gui_app.MainWindow, "new_project_from_folder", classmethod(fake_from_folder))
    monkeypatch.setattr(gui_app.MainWindow, "from_dir", classmethod(lambda cls, **kwargs: (_ for _ in ()).throw(AssertionError())))

    gui_app.main(str(tmp_path), is_das=True)

    assert captured["dirname"] == str(tmp_path)
    assert fake_app._xarray_behave_mainwin is replacement
    assert captured["event_loop"] is True


def test_new_audio_file_creates_ephemeral_project_with_cli_settings(tmp_path, monkeypatch):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"")
    (tmp_path / "clip_annotations.csv").write_text(
        "name,start_seconds,stop_seconds,channel\npulse,0.1,0.1,-1\n", encoding="utf-8"
    )
    manager = gui_config.GuiConfigManager(home=tmp_path / "home")
    captured = {}

    def fake_from_project(cls, document, **kwargs):
        captured["document"] = document
        captured.update(kwargs)
        return "window"

    monkeypatch.setattr(gui_app, "_get_config_manager", lambda: manager)
    monkeypatch.setattr(gui_app.MainWindow, "from_project", classmethod(fake_from_project))

    result = gui_app.MainWindow.new_project_from_file(
        filename=str(audio), events_string="pulse;noise", spec_freq_min=50, spec_freq_max=1_000
    )

    document = captured["document"]
    assert result == "window"
    assert document.path is None
    assert not document.document_changed
    assert document.recordings[0].annotations.names == ["pulse"]
    assert document.settings["viewer"]["spectrogram"] == {"fmin": 50, "fmax": 1_000}
    assert [item["name"] for item in document.settings["event_types"]] == ["pulse", "noise"]


def test_from_project_assembles_selected_recording(monkeypatch, tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"")
    document = project.project_from_audio_files([audio], {"version": 1})
    manager = gui_config.GuiConfigManager(home=tmp_path / "home")

    monkeypatch.setattr(gui_app.dataset_service, "assemble_project_recording", lambda recording: xr.Dataset())
    monkeypatch.setattr(gui_app, "PSV", lambda ds, **kwargs: SimpleNamespace(ds=ds, kwargs=kwargs))

    result = gui_app.MainWindow.from_project(document, "clip", manager)

    assert result.kwargs["project_document"] is document
    assert result.kwargs["recording_name"] == "clip"
    assert result.kwargs["data_source"].type == "project"


def test_capture_project_state_keeps_recording_annotations_in_memory(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"")
    document = project.project_from_audio_files([audio], {"version": 1})
    window = SimpleNamespace(
        project=document,
        current_recording_name="clip",
        event_times=annot.Events({"pulse": np.array([[0.2, 0.2, -1]])}),
        _config_snapshot=lambda: {"version": 1},
        _refresh_project_panel=lambda: None,
    )

    gui_app.MainWindow._capture_project_state(window)

    assert document.recording("clip").annotations["pulse"].tolist() == [[0.2, 0.2, -1.0]]
    assert document.recording("clip").annotations_changed


def test_from_dir_cli_spec_bounds_enable_bandpass_when_skipping_dialog(monkeypatch):
    captured = {}

    def fake_assemble_from_dir(dirname, form_data, pixel_size_mm=None):
        captured["dirname"] = dirname
        captured["form_data"] = form_data
        captured["pixel_size_mm"] = pixel_size_mm
        return xr.Dataset()

    def fake_psv(ds, **kwargs):
        captured["psv"] = SimpleNamespace(ds=ds, kwargs=kwargs)
        return captured["psv"]

    def fake_video_reader(filename):
        raise FileNotFoundError(filename)

    monkeypatch.setattr(gui_app, "YamlDialog", _FakeDialog)
    monkeypatch.setattr(gui_app.dataset_service, "assemble_from_dir", fake_assemble_from_dir)
    monkeypatch.setattr(gui_app.modern_video, "PyAVVideoReader", fake_video_reader)
    monkeypatch.setattr(gui_app, "PSV", fake_psv)

    result = gui_app.MainWindow.from_dir(
        "/data/root/dat/session",
        spec_freq_min=50,
        spec_freq_max=1000,
        skip_dialog=True,
    )

    assert result is captured["psv"]
    assert captured["dirname"] == "/data/root/dat/session"
    assert captured["form_data"]["filter_song"] == "yes"
    assert captured["form_data"]["f_low"] == 50
    assert captured["form_data"]["f_high"] == 1000
    assert captured["psv"].kwargs["fmin"] == 50
    assert captured["psv"].kwargs["fmax"] == 1000


def test_from_dir_applies_local_config_before_cli_overrides(monkeypatch, tmp_path):
    captured = {}
    source = tmp_path / "recording"
    source.mkdir()
    gui_config.write_config(
        source / ".das.yaml",
        {
            "version": 1,
            "load_dialogs": {
                "from_dir": {
                    "target_samplingrate": 500,
                    "frame_fliplr": True,
                    "spec_freq_min": 25,
                }
            },
        },
    )
    manager = gui_config.GuiConfigManager(home=tmp_path / "home")

    def fake_assemble_from_dir(dirname, form_data, pixel_size_mm=None):
        captured["form_data"] = form_data
        captured["pixel_size_mm"] = pixel_size_mm
        return xr.Dataset()

    monkeypatch.setattr(gui_app, "_get_config_manager", lambda: manager)
    monkeypatch.setattr(gui_app, "YamlDialog", _FakeDialog)
    monkeypatch.setattr(gui_app.dataset_service, "assemble_from_dir", fake_assemble_from_dir)
    monkeypatch.setattr(gui_app.modern_video, "PyAVVideoReader", lambda filename: (_ for _ in ()).throw(FileNotFoundError(filename)))
    monkeypatch.setattr(gui_app, "PSV", lambda ds, **kwargs: SimpleNamespace(ds=ds, kwargs=kwargs))

    result = gui_app.MainWindow.from_dir(
        str(source),
        target_samplingrate=2_000,
        spec_freq_max=900,
        skip_dialog=True,
    )

    assert captured["form_data"]["target_samplingrate"] == 2_000
    assert captured["form_data"]["frame_fliplr"] is True
    assert captured["form_data"]["spec_freq_min"] == 25
    assert captured["form_data"]["spec_freq_max"] == 900
    assert captured["pixel_size_mm"] is None
    assert result.kwargs["config_manager"] is manager


def test_from_dir_passes_manifest_to_dataset_service(monkeypatch):
    captured = {}

    def fake_assemble_from_dir(dirname, form_data, pixel_size_mm=None, manifest=None):
        captured["dirname"] = dirname
        captured["manifest"] = manifest
        return xr.Dataset()

    monkeypatch.setattr(gui_app, "YamlDialog", _FakeDialog)
    monkeypatch.setattr(gui_app.dataset_service, "assemble_from_dir", fake_assemble_from_dir)
    monkeypatch.setattr(gui_app.modern_video, "PyAVVideoReader", lambda filename: (_ for _ in ()).throw(FileNotFoundError(filename)))
    monkeypatch.setattr(gui_app, "PSV", lambda ds, **kwargs: SimpleNamespace(ds=ds, kwargs=kwargs))

    gui_app.MainWindow.from_dir(
        "/data/root/dat/session",
        manifest="/tmp/files.yaml",
        skip_dialog=True,
    )

    assert captured["dirname"] == "/data/root/dat/session"
    assert captured["manifest"] == "/tmp/files.yaml"


def test_from_media_opens_dataset_from_dialog(monkeypatch, tmp_path):
    captured = {}
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")

    class FakeMediaDialog:
        def __init__(self, files=None):
            captured["files"] = files

        def exec_(self):
            return gui_app.QtWidgets.QDialog.Accepted

        def media_data(self):
            return {"audio": [{"name": "audio", "path": str(audio), "offset_seconds": 0.0}], "video": []}

    manager = gui_config.GuiConfigManager(home=tmp_path / "home")

    def fake_assemble(media):
        captured["media"] = media
        return xr.Dataset()

    monkeypatch.setattr(gui_app, "MediaFilesDialog", FakeMediaDialog)
    monkeypatch.setattr(gui_app, "_get_config_manager", lambda: manager)
    monkeypatch.setattr(gui_app.dataset_service, "assemble_from_media", fake_assemble)
    monkeypatch.setattr(gui_app, "PSV", lambda ds, **kwargs: SimpleNamespace(ds=ds, kwargs=kwargs))

    result = gui_app.MainWindow.from_media()

    assert captured["files"] is None
    assert captured["media"]["audio"][0]["path"] == str(audio)
    assert result.kwargs["data_source"].type == "media"
    assert result.kwargs["data_source"].name == str(audio)


def test_from_media_prefills_manifest(monkeypatch, tmp_path):
    captured = {}
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    discovered = {"audio": {"main": {"path": str(audio)}}}

    class FakeMediaDialog:
        def __init__(self, files=None):
            captured["files"] = files

        def exec_(self):
            return gui_app.QtWidgets.QDialog.Accepted

        def media_data(self):
            return {"audio": [{"name": "main", "path": str(audio), "offset_seconds": 0.0}], "video": []}

    def fake_discover(datename, root="", dat_path="dat", res_path="res", manifest=None):
        captured["discover"] = (datename, root, dat_path, res_path, manifest)
        return discovered

    manager = gui_config.GuiConfigManager(home=tmp_path / "home")
    monkeypatch.setattr(gui_app, "MediaFilesDialog", FakeMediaDialog)
    monkeypatch.setattr(gui_app, "_get_config_manager", lambda: manager)
    monkeypatch.setattr(gui_app.v2_api, "discover", fake_discover)
    monkeypatch.setattr(gui_app.dataset_service, "assemble_from_media", lambda media: xr.Dataset())
    monkeypatch.setattr(gui_app, "PSV", lambda ds, **kwargs: SimpleNamespace(ds=ds, kwargs=kwargs))

    gui_app.MainWindow.from_media(manifest="/tmp/files.yaml", datename="session", root="/data")

    assert captured["discover"] == ("session", "/data", "dat", "res", "/tmp/files.yaml")
    assert captured["files"] is discovered
