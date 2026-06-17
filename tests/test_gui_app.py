from types import SimpleNamespace

import xarray as xr

from xarray_behave.gui import app as gui_app


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
