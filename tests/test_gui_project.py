import numpy as np
import pandas as pd
import pytest
import yaml

from xarray_behave import annot
from xarray_behave.gui import project


def _events():
    return annot.Events(
        {
            "pulse": np.array([[0.125, 0.125, -1]]),
            "sine": np.array([[0.5, 0.85, 0]]),
        }
    )


def test_project_round_trip_embeds_literal_csv_and_paths(tmp_path):
    audio = tmp_path / "media" / "clip.wav"
    video = tmp_path / "media" / "clip.mp4"
    audio.parent.mkdir()
    audio.write_bytes(b"")
    video.write_bytes(b"")
    recording = project.Recording(
        name="clip",
        audio={"path": str(audio), "offset_seconds": 0.0},
        videos=[{"name": "camera", "path": str(video), "offset_seconds": 0.25}],
        annotations=_events(),
    )
    document = project.Project(
        recordings=[recording],
        settings={"version": 1, "viewer": {"spectrogram": {"colormap": "magma"}}},
        document_changed=True,
    )

    path = project.write_project(tmp_path / "songs", document)
    text = path.read_text(encoding="utf-8")

    assert path.name == "songs.xbp.yaml"
    assert "data: |" in text
    assert "pulse,0.125,0.125,-1.0" in text
    assert "path: media/clip.wav" in text
    loaded = project.read_project(path)
    assert loaded.recordings[0].audio_path == audio
    assert loaded.recordings[0].videos[0]["path"] == str(video)
    pd.testing.assert_frame_equal(
        loaded.recordings[0].annotations.to_df(preserve_empty=False),
        recording.annotations.to_df(preserve_empty=False),
    )
    assert loaded.settings["viewer"]["spectrogram"]["colormap"] == "magma"
    assert not loaded.is_dirty


def test_project_keeps_external_paths_absolute(tmp_path):
    project_dir = tmp_path / "project"
    external = tmp_path / "external.wav"
    external.write_bytes(b"")
    document = project.project_from_audio_files([external], {"version": 1})

    mapping = project.project_mapping(document, project_dir / "songs.xbp.yaml")

    assert mapping["recordings"][0]["audio"]["path"] == str(external)


def test_project_rejects_duplicates_and_bad_annotation_columns(tmp_path):
    payload = {
        "version": 1,
        "recordings": [
            {
                "name": "clip",
                "audio": {"path": "missing.wav"},
                "annotations": {"format": "csv", "data": "name,start_seconds\npulse,0.1\n"},
            }
        ],
        "settings": {"version": 1},
    }
    path = tmp_path / "bad.xbp.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(project.ProjectError, match="missing columns"):
        project.read_project(path)

    payload["recordings"][0]["annotations"]["data"] = "name,start_seconds,stop_seconds,channel\n"
    payload["recordings"].append(dict(payload["recordings"][0]))
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(project.ProjectError, match="Duplicate recording"):
        project.read_project(path)


def test_project_validates_media_metadata_and_annotation_values(tmp_path):
    payload = {
        "version": 1,
        "recordings": [
            {
                "name": "clip",
                "audio": {"path": "missing.wav", "sampling_rate_Hz": -1},
                "annotations": {
                    "format": "csv",
                    "data": "name,start_seconds,stop_seconds,channel\npulse,wrong,0.1,-1\n",
                },
            }
        ],
        "settings": {"version": 1},
    }
    path = tmp_path / "bad.xbp.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(project.ProjectError, match="positive number"):
        project.read_project(path)

    payload["recordings"][0]["audio"].pop("sampling_rate_Hz")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(project.ProjectError, match="must be numeric"):
        project.read_project(path)


def test_project_normalizes_plain_yaml_suffix(tmp_path):
    document = project.Project([], {"version": 1})

    saved = project.write_project(tmp_path / "songs.yaml", document)

    assert saved.name == "songs.xbp.yaml"


def test_recording_auto_imports_sidecars_and_folder_falls_back_to_recursive(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    audio = nested / "clip.wav"
    video = nested / "clip.MP4"
    timestamps = nested / "clip_timestamps.h5"
    audio.write_bytes(b"")
    video.write_bytes(b"")
    timestamps.write_bytes(b"")
    (nested / "clip_annotations.csv").write_text(
        "name,start_seconds,stop_seconds,channel\npulse,0.1,0.1,-1\n", encoding="utf-8"
    )

    assert project.audio_files_in_folder(tmp_path) == [audio]
    recording = project.recording_from_audio(audio)
    assert recording.videos[0]["path"] == str(video)
    assert recording.audio["timestamp_path"] == str(timestamps)
    assert recording.sidecar_path == nested / "clip_annotations.csv"
    assert recording.annotations.names == ["pulse"]


def test_project_tracks_annotations_and_project_wide_type_changes(tmp_path):
    first = project.Recording("first", {"path": str(tmp_path / "first.wav")}, annotations=_events())
    second = project.Recording("second", {"path": str(tmp_path / "second.wav")}, annotations=_events())
    document = project.Project([first, second], {"version": 1})

    document.rename_event_type("pulse", "click")
    assert all("click" in recording.annotations and "pulse" not in recording.annotations for recording in document.recordings)
    assert document.is_dirty

    document.mark_saved()
    document.delete_event_type("click")
    assert all("click" not in recording.annotations for recording in document.recordings)
    assert document.is_dirty


def test_failed_project_write_leaves_existing_file_and_dirty_state(tmp_path, monkeypatch):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"")
    document = project.project_from_audio_files([audio], {"version": 1}, document_changed=True)
    path = tmp_path / "songs.xbp.yaml"
    path.write_text("original", encoding="utf-8")
    monkeypatch.setattr(project.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("nope")))

    with pytest.raises(OSError, match="nope"):
        project.write_project(path, document)

    assert path.read_text(encoding="utf-8") == "original"
    assert document.is_dirty


def test_saved_project_updates_embedded_annotations_without_touching_sidecar(tmp_path):
    audio = tmp_path / "clip.wav"
    sidecar = tmp_path / "clip_annotations.csv"
    audio.write_bytes(b"")
    original = "name,start_seconds,stop_seconds,channel\npulse,0.1,0.1,-1\n"
    sidecar.write_text(original, encoding="utf-8")
    document = project.project_from_audio_files([audio], {"version": 1})
    path = project.write_project(tmp_path / "songs.xbp.yaml", document)

    document.recording("clip").annotations.add_time("pulse", 0.2, 0.2, channel=0)
    project.write_project(path, document)

    assert sidecar.read_text(encoding="utf-8") == original
    assert project.read_project(path).recording("clip").annotations["pulse"].tolist() == [
        [0.1, 0.1, -1.0],
        [0.2, 0.2, 0.0],
    ]
