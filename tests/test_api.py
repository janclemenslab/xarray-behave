import numpy as np
import h5py
import pytest
import soundfile as sf
import xarray_behave as xb
from xarray_behave.gui import modern_video


def _write_timestamps(path, rows):
    np.savetxt(path, rows, delimiter=",", header="index,timestamp", comments="")


def test_discover_default_manifest_finds_current_scheme():
    files = xb.discover("localhost-20210629_171532", root="tests/data")

    assert files["audio"]["main"]["path"].endswith("_daq.h5")
    assert files["video"]["camera"]["path"].endswith(".mp4")
    assert files["video"]["ball"]["path"].endswith("_ball.mp4")
    assert files["tracks"]["main"]["path"].endswith("_tracks.csv")


def test_assemble_splits_audio_from_manifest(tmp_path):
    audio_path = tmp_path / "recording.wav"
    data = np.arange(40, dtype=np.float32).reshape(10, 4)
    sf.write(audio_path, data, samplerate=10_000, subtype="FLOAT")

    files = {
        "audio": {
            "main": {
                "path": str(audio_path),
                "splits": {
                    "audio": "0:2",
                    "other": "2:4",
                },
            }
        }
    }

    ds = xb.assemble(files)

    assert "song_raw" not in ds
    assert ds.audio.dims == ("audio_time", "audio_channels")
    assert ds.other.dims == ("other_time", "other_channels")
    np.testing.assert_array_equal(ds.audio_channels, [0, 1])
    np.testing.assert_array_equal(ds.other_channels, [2, 3])


def test_assemble_splits_all_ethodrome_h5_channels(tmp_path):
    audio_path = tmp_path / "recording_daq.h5"
    data = np.arange(200, dtype=np.float32).reshape(10, 20)
    with h5py.File(audio_path, "w") as file:
        file.create_dataset("samples", data=data)
        file.attrs["rate"] = 10_000

    ds = xb.assemble(
        {
            "audio": {
                "main": {
                    "path": str(audio_path),
                    "splits": {
                        "audio": "0:16",
                        "other": "16:18",
                    },
                }
            }
        }
    )

    assert ds.audio.shape == (10, 16)
    assert ds.other.shape == (10, 2)
    np.testing.assert_array_equal(ds.other_channels, [16, 17])


def test_assemble_rejects_overlapping_audio_splits(tmp_path):
    audio_path = tmp_path / "recording.wav"
    sf.write(audio_path, np.zeros((10, 4), dtype=np.float32), samplerate=10_000, subtype="FLOAT")

    with pytest.raises(ValueError, match="overlaps"):
        xb.assemble(
            {
                "audio": {
                    "main": {
                        "path": str(audio_path),
                        "splits": {
                            "audio": "0:3",
                            "other": "2:4",
                        },
                    }
                }
            }
        )


def test_discover_custom_manifest_multiple_videos(tmp_path):
    (tmp_path / "cam.mp4").write_bytes(b"")
    (tmp_path / "side.avi").write_bytes(b"")
    manifest = {
        "video": {
            "camera": {"path": "{root}/cam.mp4"},
            "side": {"path": "{root}/side.avi"},
        }
    }

    files = xb.discover(root=str(tmp_path), manifest=manifest)

    assert set(files["video"]) == {"camera", "side"}


def test_assemble_media_sources_align_to_first_audio_clock(tmp_path):
    master_path = tmp_path / "master.wav"
    aux_path = tmp_path / "aux.wav"
    sf.write(master_path, np.zeros((10, 1), dtype=np.float32), samplerate=10, subtype="FLOAT")
    sf.write(aux_path, np.zeros((10, 1), dtype=np.float32), samplerate=10, subtype="FLOAT")
    master_timestamps = tmp_path / "master_timestamps.csv"
    aux_timestamps = tmp_path / "aux_timestamps.csv"
    camera_timestamps = tmp_path / "camera_timestamps.csv"
    _write_timestamps(master_timestamps, [[0, 100.0], [9, 100.9]])
    _write_timestamps(aux_timestamps, [[0, 100.25], [9, 101.15]])
    _write_timestamps(camera_timestamps, [[0, 100.5], [1, 100.7]])

    ds = xb.assemble(
        {
            "audio": {
                "master": {
                    "path": str(master_path),
                    "timestamp_path": str(master_timestamps),
                    "splits": {"master": ":"},
                },
                "aux": {
                    "path": str(aux_path),
                    "timestamp_path": str(aux_timestamps),
                    "offset_seconds": 0.1,
                    "splits": {"aux": ":"},
                },
            },
            "video": {
                "camera": {
                    "path": str(tmp_path / "camera.mp4"),
                    "timestamp_path": str(camera_timestamps),
                    "offset_seconds": -0.1,
                }
            },
        }
    )

    assert ds.attrs["ref_time"] == 100.0
    np.testing.assert_allclose(ds.master_time, np.arange(10) / 10)
    np.testing.assert_allclose(ds.aux_time, 0.35 + np.arange(10) / 10)
    np.testing.assert_allclose(ds.camera_frame_time, [0.4, 0.6])


def test_assemble_media_uses_rate_overrides_and_embedded_daq_timestamps(monkeypatch, tmp_path):
    daq_path = tmp_path / "recording_daq.h5"
    with h5py.File(daq_path, "w") as file:
        file.create_dataset("samples", data=np.zeros((10, 1), dtype=np.float32))
        file.create_dataset("systemtime", data=np.array([[20.0], [20.5], [21.0]]))
        file.create_dataset("samplenumber", data=np.array([[0], [5], [5]]))
        file.attrs["rate"] = 10

    array_path = tmp_path / "other.npy"
    np.save(array_path, np.zeros((10, 1), dtype=np.float32))

    class FakeVideoReader:
        number_of_frames = 3
        frame_rate = 20

        def __init__(self, path):
            self.path = path

    monkeypatch.setattr(modern_video, "PyAVVideoReader", FakeVideoReader)
    ds = xb.assemble(
        {
            "audio": {
                "daq": {"path": str(daq_path), "splits": {"daq": ":"}},
                "other": {
                    "path": str(array_path),
                    "sampling_rate_Hz": 20,
                    "offset_seconds": 2.0,
                    "splits": {"other": ":"},
                },
            },
            "video": {"side": {"path": str(tmp_path / "side.mp4"), "frame_rate_Hz": 5}},
        }
    )

    assert ds.attrs["ref_time"] == 20.0
    np.testing.assert_allclose(ds.daq_time, np.arange(10) / 10)
    np.testing.assert_allclose(ds.other_time, 2.0 + np.arange(10) / 20)
    assert ds.other.attrs["sampling_rate_overridden"] is True
    np.testing.assert_allclose(ds.side_frame_time, [0.0, 0.2, 0.4])


def test_resample_aligns_native_tracks():
    files = xb.discover("localhost-20210629_171532", root="tests/data")
    ds = xb.assemble(files, lazy_load_audio=True)

    assert ds.body_positions.dims[0] == "frame_number"

    resampled = xb.resample(ds, target_sampling_rate=100)

    assert resampled.body_positions.dims[0] == "time"
    assert "nearest_frame" in resampled.coords
    assert np.isclose(resampled.attrs["target_sampling_rate_Hz"], 100)


def test_v1_assemble_import():
    from xarray_behave.v1 import assemble

    assert assemble.__module__ == "xarray_behave.xarray_behave"


def test_save_accepts_v2_time_dimension_names(tmp_path):
    audio_path = tmp_path / "recording.wav"
    sf.write(audio_path, np.zeros((10, 2), dtype=np.float32), samplerate=10_000, subtype="FLOAT")
    ds = xb.assemble({"audio": {"main": {"path": str(audio_path), "splits": {"audio": "0:1", "other": "1:2"}}}})

    xb.save(tmp_path / "recording.zarr", ds)
