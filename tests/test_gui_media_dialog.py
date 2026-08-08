import numpy as np

import xarray_behave  # noqa: F401 - sets QT_API before qtpy imports
from qtpy import QtWidgets

from xarray_behave.gui.media_dialog import MediaFilesDialog


def _app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_media_dialog_adds_files_with_names_sidecars_and_dataset_keys(tmp_path):
    _app()
    audio_path = tmp_path / "Mic Left.wav"
    video_path = tmp_path / "Mic Left.mp4"
    audio_path.write_bytes(b"")
    video_path.write_bytes(b"")
    (tmp_path / "Mic Left_timestamps.csv").write_text("index,timestamp\n0,0\n")
    array_path = tmp_path / "array.npz"
    np.savez(array_path, signal=np.zeros(4), samplerate=1_000)

    dialog = MediaFilesDialog()
    dialog.add_audio_files([str(audio_path), str(array_path)])
    dialog.add_video_files([str(video_path)])

    assert dialog.audio_table.item(0, 0).text() == "mic_left"
    assert dialog.video_table.item(0, 0).text() == "mic_left_2"
    assert dialog.audio_table.item(0, 2).text().endswith("Mic Left_timestamps.csv")
    assert dialog.audio_table.item(1, 5).text() == "signal"

    dialog.audio_table.item(1, 3).setText("0.25")
    dialog.audio_table.item(1, 4).setText("2000")
    media = dialog.media_data()
    assert media["audio"][1]["offset_seconds"] == 0.25
    assert media["audio"][1]["sampling_rate_Hz"] == 2_000


def test_media_dialog_prepopulates_from_manifest_mapping(tmp_path):
    _app()
    audio_path = tmp_path / "main.npz"
    video_path = tmp_path / "camera.mp4"
    audio_timestamp = tmp_path / "main_timestamps.csv"
    video_timestamp = tmp_path / "camera_timestamps.csv"
    np.savez(audio_path, samples=np.zeros(4), samplerate=1_000)
    video_path.write_bytes(b"")
    audio_timestamp.write_text("index,timestamp\n0,0\n")
    video_timestamp.write_text("index,timestamp\n0,0\n")

    dialog = MediaFilesDialog(
        files={
            "audio": {
                "main_audio": {
                    "path": str(audio_path),
                    "timestamp_path": str(audio_timestamp),
                    "offset_seconds": 0.25,
                    "sampling_rate_Hz": 2_000,
                    "audio_dataset": "samples",
                }
            },
            "video": {
                "camera": {
                    "path": str(video_path),
                    "offset_seconds": -0.5,
                    "frame_rate_Hz": 30,
                }
            },
            "timestamps": {"camera": {"path": str(video_timestamp)}},
        }
    )

    assert dialog.audio_table.item(0, 0).text() == "main_audio"
    assert dialog.audio_table.item(0, 2).text() == str(audio_timestamp)
    assert dialog.audio_table.item(0, 5).text() == "samples"
    assert dialog.video_table.item(0, 2).text() == str(video_timestamp)

    media = dialog.media_data()
    assert media["audio"][0]["sampling_rate_Hz"] == 2_000
    assert media["video"][0]["offset_seconds"] == -0.5
    assert media["video"][0]["frame_rate_Hz"] == 30


def test_media_dialog_validates_and_removes_rows(tmp_path):
    _app()
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"")
    dialog = MediaFilesDialog()

    try:
        dialog.media_data()
    except ValueError as exc:
        assert "audio" in str(exc).lower()
    else:
        raise AssertionError("Expected an audio validation error")

    dialog.add_audio_files([str(audio_path)])
    dialog.audio_table.item(0, 4).setText("0")
    try:
        dialog.media_data()
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected a rate validation error")

    dialog.audio_table.item(0, 4).setText("")
    dialog.audio_table.selectRow(0)
    dialog._remove_selected(dialog.audio_table)
    assert dialog.audio_table.rowCount() == 0
