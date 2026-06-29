from types import SimpleNamespace

import xarray as xr

from xarray_behave.gui import app as gui_app, gui_config


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
