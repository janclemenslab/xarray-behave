import numpy as np
import xarray as xr

from xarray_behave import _dataset_service as dataset_service, annot


def test_assemble_from_file_maps_form_data(monkeypatch):
    calls = {}
    ds = xr.Dataset()

    def fake_assemble(**kwargs):
        calls["kwargs"] = kwargs
        return ds

    monkeypatch.setattr(dataset_service.xb, "assemble", fake_assemble)

    form_data = {
        "annotation_path": "/tmp/audio_annotations.csv",
        "definition_path": "/tmp/audio_definitions.csv",
        "samplerate": 10_000,
        "target_samplingrate": 1_000,
        "data_set": "audio",
        "filter_song": "no",
        "f_low": None,
        "f_high": None,
    }

    result = dataset_service.assemble_from_file("/tmp/audio.wav", form_data)

    assert calls["kwargs"] == {
        "filepath_daq": "/tmp/audio.wav",
        "filepath_annotations": "/tmp/audio_annotations.csv",
        "filepath_definitions": "/tmp/audio_definitions.csv",
        "audio_sampling_rate": 10_000,
        "target_sampling_rate": 1_000,
        "audio_dataset": "audio",
    }
    assert result.attrs["filename"] == "/tmp/audio.wav"
    assert result.attrs["filebase"] == "/tmp/audio"
    assert result.attrs["datename"] == ""


def test_assemble_from_dir_maps_form_data(monkeypatch):
    calls = {}
    ds = xr.Dataset()

    def fake_assemble(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return ds

    monkeypatch.setattr(dataset_service.xb, "assemble", fake_assemble)

    form_data = {
        "target_samplingrate": 500,
        "filter_song": "no",
        "ignore_tracks": False,
        "ignore_song": False,
        "annotation_path": "",
        "video_filename": "/tmp/custom.mp4",
        "daq_filename": "",
        "fix_fly_indices": True,
        "init_annotations": False,
        "events_string": "",
        "f_low": None,
        "f_high": None,
    }

    dataset_service.assemble_from_dir("/data/root/dat/session", form_data, pixel_size_mm=0.05)

    assert calls["args"] == ("session", "/data/root", "dat")
    assert calls["kwargs"] == {
        "res_path": "res",
        "filepath_annotations": None,
        "filepath_video": "/tmp/custom.mp4",
        "filepath_daq": None,
        "fix_fly_indices": True,
        "include_song": -1,
        "target_sampling_rate": 500,
        "resample_video_data": True,
        "pixel_size_mm": 0.05,
        "lazy_load_song": True,
        "include_tracks": True,
        "include_poses": True,
    }


def test_prepare_for_save_updates_traces_units_and_event_arrays(monkeypatch):
    calls = []
    ds = xr.Dataset(
        {
            "song_events": xr.DataArray(
                np.zeros((2, 1)),
                dims=["time", "event_types"],
                coords={"time": [0.0, 1.0], "event_types": ["pulse"], "event_categories": ("event_types", ["event"])},
            ),
            "event_times": xr.DataArray(np.zeros((1, 3)), dims=["index", "event_time"]),
            "event_names": xr.DataArray(["old"], dims=["index"]),
        }
    )
    event_times = annot.Events({"pulse": np.array([[0.0, 0.0]])}, categories={"pulse": "event"})

    def fake_eventtimes_to_traces(dataset, updated_event_times):
        calls.append(("traces", updated_event_times))
        return dataset

    def fake_convert_spatial_units(dataset, to_units):
        calls.append(("units", to_units))
        return dataset

    monkeypatch.setattr(dataset_service.event_utils, "eventtimes_to_traces", fake_eventtimes_to_traces)
    monkeypatch.setattr(dataset_service.xb, "convert_spatial_units", fake_convert_spatial_units)

    result = dataset_service.prepare_for_save(ds, event_times, original_spatial_units="mm")

    assert calls == [("traces", event_times), ("units", "mm")]
    assert result.event_names.values.tolist() == ["pulse"]
    np.testing.assert_allclose(result.event_times.values[:, :2], [[0.0, 0.0]])
