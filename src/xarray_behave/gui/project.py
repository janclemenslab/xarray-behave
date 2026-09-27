"""Qt-free project storage for the media GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from numbers import Real
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .. import annot
from . import gui_config


PROJECT_VERSION = 1
PROJECT_SUFFIX = ".xbp.yaml"
AUDIO_SUFFIXES = frozenset(
    {".wav", ".aif", ".aiff", ".mp3", ".flac", ".ogg", ".m4a", ".h5", ".hdf5", ".hdfs", ".npy", ".npz", ".mmap"}
)
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")
ANNOTATION_COLUMNS = ["name", "start_seconds", "stop_seconds", "channel"]


class ProjectError(ValueError):
    pass


class _LiteralString(str):
    pass


class _ProjectDumper(yaml.SafeDumper):
    pass


def _represent_literal(dumper, value):
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")


_ProjectDumper.add_representer(_LiteralString, _represent_literal)


def _events_frame(events) -> pd.DataFrame:
    frame = annot.Events(events).to_df(preserve_empty=False, with_channels=True)
    return frame.sort_values("start_seconds", ignore_index=True) if len(frame) else frame


def _events_csv(events) -> str:
    return _events_frame(events).to_csv(index=False, lineterminator="\n")


def _events_from_csv(value: str, recording_name: str) -> annot.Events:
    try:
        frame = pd.read_csv(StringIO(value))
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ProjectError(f"Invalid annotations for recording {recording_name!r}: {exc}") from exc
    missing = set(ANNOTATION_COLUMNS) - set(frame.columns)
    if missing:
        raise ProjectError(
            f"Invalid annotations for recording {recording_name!r}: missing columns {', '.join(sorted(missing))}"
        )
    frame = frame[ANNOTATION_COLUMNS].copy()
    if frame["name"].isna().any() or (frame["name"].astype(str).str.strip() == "").any():
        raise ProjectError(f"Invalid annotations for recording {recording_name!r}: names must not be empty")
    try:
        numeric_columns = ["start_seconds", "stop_seconds", "channel"]
        frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric)
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"Invalid annotations for recording {recording_name!r}: times and channels must be numeric") from exc
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise ProjectError(f"Invalid annotations for recording {recording_name!r}: times and channels must be finite")
    return annot.Events.from_df(frame)


def _annotation_snapshot(events) -> pd.DataFrame:
    return _events_frame(events).copy(deep=True)


@dataclass
class Recording:
    name: str
    audio: dict[str, Any]
    videos: list[dict[str, Any]] = field(default_factory=list)
    annotations: annot.Events = field(default_factory=annot.Events)
    sidecar_path: Path | None = None
    _saved_annotations: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.annotations = annot.Events(self.annotations)
        self._saved_annotations = _annotation_snapshot(self.annotations)

    @property
    def audio_path(self) -> Path:
        return Path(self.audio["path"])

    @property
    def available(self) -> bool:
        return self.audio_path.is_file()

    @property
    def annotations_changed(self) -> bool:
        return not _annotation_snapshot(self.annotations).equals(self._saved_annotations)

    def set_annotations(self, events) -> None:
        self.annotations = annot.Events(events)

    def mark_annotations_saved(self) -> None:
        self._saved_annotations = _annotation_snapshot(self.annotations)


@dataclass
class Project:
    recordings: list[Recording]
    settings: dict[str, Any]
    path: Path | None = None
    document_changed: bool = False

    def __post_init__(self) -> None:
        self.settings = gui_config.sanitize_config(self.settings)

    @property
    def is_saved(self) -> bool:
        return self.path is not None

    @property
    def is_dirty(self) -> bool:
        return self.document_changed or any(recording.annotations_changed for recording in self.recordings)

    def recording(self, name: str) -> Recording:
        for recording in self.recordings:
            if recording.name == name:
                return recording
        raise KeyError(name)

    def set_settings(self, settings: Mapping[str, Any]) -> None:
        clean = gui_config.sanitize_config(settings)
        if clean != self.settings:
            self.settings = clean
            self.document_changed = True

    def add_recordings(self, paths) -> list[Recording]:
        known_paths = {recording.audio_path.resolve() for recording in self.recordings}
        names = {recording.name for recording in self.recordings}
        added = []
        for path in paths:
            path = Path(path).expanduser().resolve()
            if path in known_paths:
                continue
            recording = recording_from_audio(path, names=names)
            self.recordings.append(recording)
            known_paths.add(path)
            names.add(recording.name)
            added.append(recording)
        if added:
            self.document_changed = True
        return added

    def remove_recording(self, name: str) -> None:
        self.recordings.remove(self.recording(name))
        self.document_changed = True

    def rename_event_type(self, old_name: str, new_name: str) -> None:
        if old_name == new_name:
            return
        for recording in self.recordings:
            if old_name not in recording.annotations:
                continue
            rows = recording.annotations[old_name].copy()
            recording.annotations.delete_name(old_name)
            recording.annotations.add_name(new_name, times=rows, append=new_name in recording.annotations)

    def delete_event_type(self, name: str) -> None:
        for recording in self.recordings:
            recording.annotations.delete_name(name)

    def mark_saved(self) -> None:
        self.document_changed = False
        for recording in self.recordings:
            recording.mark_annotations_saved()


def _require_mapping(value, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectError(f"{label} must be a mapping")
    return dict(value)


def _validate_media_entry(entry, label: str, *, video: bool) -> dict[str, Any]:
    entry = _require_mapping(entry, label)
    path = entry.get("path")
    if not isinstance(path, str) or not path:
        raise ProjectError(f"{label}.path must be a non-empty string")
    timestamp = entry.get("timestamp_path")
    if timestamp is not None and (not isinstance(timestamp, str) or not timestamp):
        raise ProjectError(f"{label}.timestamp_path must be a non-empty string or null")
    offset = entry.get("offset_seconds")
    if offset is not None:
        if isinstance(offset, bool) or not isinstance(offset, Real):
            raise ProjectError(f"{label}.offset_seconds must be a number or null")
        entry["offset_seconds"] = float(offset)
    rate_key = "frame_rate_Hz" if video else "sampling_rate_Hz"
    rate = entry.get(rate_key)
    if rate is not None:
        if isinstance(rate, bool) or not isinstance(rate, Real) or rate <= 0:
            raise ProjectError(f"{label}.{rate_key} must be a positive number or null")
        entry[rate_key] = float(rate)
    if not video and entry.get("audio_dataset") is not None and not isinstance(entry["audio_dataset"], str):
        raise ProjectError(f"{label}.audio_dataset must be a string or null")
    if video and (not isinstance(entry.get("name"), str) or not entry["name"].strip()):
        raise ProjectError(f"{label}.name must be a non-empty string")
    return entry


def _resolve_media_entry(entry, base: Path, label: str, *, video: bool) -> dict[str, Any]:
    entry = _validate_media_entry(entry, label, video=video)
    entry["path"] = _resolve_path(entry["path"], base)
    timestamp = entry.get("timestamp_path")
    if timestamp is not None:
        entry["timestamp_path"] = _resolve_path(timestamp, base)
    return entry


def _resolve_path(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def _stored_path(value: str | Path, base: Path) -> str:
    path = Path(value).expanduser().resolve()
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def _stored_media_entry(entry: Mapping[str, Any], base: Path, label: str, *, video: bool) -> dict[str, Any]:
    stored = _validate_media_entry(entry, label, video=video)
    stored["path"] = _stored_path(stored["path"], base)
    if stored.get("timestamp_path"):
        stored["timestamp_path"] = _stored_path(stored["timestamp_path"], base)
    return stored


def read_project(path: str | Path) -> Project:
    project_path = Path(path).expanduser().resolve()
    try:
        with project_path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ProjectError(f"Could not read project {project_path}: {exc}") from exc
    payload = _require_mapping(payload, "project")
    if payload.get("version") != PROJECT_VERSION:
        raise ProjectError(f"Project version must be {PROJECT_VERSION}")
    raw_recordings = payload.get("recordings", [])
    if not isinstance(raw_recordings, list):
        raise ProjectError("recordings must be a list")

    recordings = []
    names = set()
    for index, value in enumerate(raw_recordings):
        item = _require_mapping(value, f"recordings[{index}]")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ProjectError(f"recordings[{index}].name must be a non-empty string")
        if name in names:
            raise ProjectError(f"Duplicate recording name {name!r}")
        names.add(name)
        audio = _resolve_media_entry(item.get("audio"), project_path.parent, f"recordings[{index}].audio", video=False)
        raw_videos = item.get("videos", [])
        if not isinstance(raw_videos, list):
            raise ProjectError(f"recordings[{index}].videos must be a list")
        videos = [
            _resolve_media_entry(
                video,
                project_path.parent,
                f"recordings[{index}].videos[{video_index}]",
                video=True,
            )
            for video_index, video in enumerate(raw_videos)
        ]
        video_names = [video["name"] for video in videos]
        if len(video_names) != len(set(video_names)):
            raise ProjectError(f"recordings[{index}].videos must have unique names")
        annotations = _require_mapping(item.get("annotations", {}), f"recordings[{index}].annotations")
        if annotations.get("format") != "csv" or not isinstance(annotations.get("data"), str):
            raise ProjectError(f"recordings[{index}].annotations must contain CSV data")
        recordings.append(
            Recording(
                name=name,
                audio=audio,
                videos=videos,
                annotations=_events_from_csv(annotations["data"], name),
            )
        )

    settings = payload.get("settings", {"version": gui_config.CONFIG_VERSION})
    try:
        settings = gui_config.sanitize_config(settings)
    except gui_config.ConfigError as exc:
        raise ProjectError(f"Invalid project settings: {exc}") from exc
    return Project(recordings=recordings, settings=settings, path=project_path)


def project_mapping(project: Project, path: str | Path) -> dict[str, Any]:
    base = Path(path).expanduser().resolve().parent
    names = [recording.name for recording in project.recordings]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ProjectError("Recording names must be non-empty strings")
    if len(names) != len(set(names)):
        raise ProjectError("Recording names must be unique")
    for recording in project.recordings:
        video_names = [video.get("name") for video in recording.videos]
        if len(video_names) != len(set(video_names)):
            raise ProjectError(f"Recording {recording.name!r} videos must have unique names")
    return {
        "version": PROJECT_VERSION,
        "recordings": [
            {
                "name": recording.name,
                "audio": _stored_media_entry(recording.audio, base, f"recording {recording.name!r}.audio", video=False),
                "videos": [
                    _stored_media_entry(
                        video,
                        base,
                        f"recording {recording.name!r}.videos[{index}]",
                        video=True,
                    )
                    for index, video in enumerate(recording.videos)
                ],
                "annotations": {"format": "csv", "data": _LiteralString(_events_csv(recording.annotations))},
            }
            for recording in project.recordings
        ],
        "settings": gui_config.sanitize_config(project.settings),
    }


def write_project(path: str | Path, project: Project) -> Path:
    project_path = Path(path).expanduser()
    if not str(project_path).lower().endswith(PROJECT_SUFFIX):
        if project_path.suffix.lower() in {".yaml", ".yml"}:
            project_path = project_path.with_suffix("")
        project_path = project_path.with_name(project_path.name + PROJECT_SUFFIX)
    project_path = project_path.resolve()
    project_path.parent.mkdir(parents=True, exist_ok=True)
    payload = project_mapping(project, project_path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=project_path.parent, prefix=f".{project_path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            yaml.dump(payload, handle, Dumper=_ProjectDumper, sort_keys=False, allow_unicode=False)
        os.replace(temporary_path, project_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    project.path = project_path
    project.mark_saved()
    return project_path


def _unique_name(stem: str, names: set[str]) -> str:
    base = stem or "recording"
    name = base
    suffix = 2
    while name in names:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


def _timestamp_sidecar(path: Path) -> Path | None:
    for suffix in ("_timestamps.h5", "_timeStamps.h5", "_timestamps.csv"):
        candidate = path.with_name(path.stem + suffix)
        if candidate.is_file():
            return candidate.resolve()
    return None


def _sidecar_annotations(path: Path) -> tuple[annot.Events, Path | None]:
    sidecar = path.with_name(path.stem + "_annotations.csv")
    if not sidecar.is_file():
        return annot.Events(), sidecar.resolve()
    try:
        frame = pd.read_csv(sidecar)
        return annot.Events.from_df(frame), sidecar.resolve()
    except (OSError, ValueError, pd.errors.ParserError):
        return annot.Events(), sidecar.resolve()


def recording_from_audio(path: str | Path, *, names: set[str] | None = None) -> Recording:
    audio_path = Path(path).expanduser().resolve()
    names = set() if names is None else names
    name = _unique_name(audio_path.stem, names)
    audio: dict[str, Any] = {"path": str(audio_path), "offset_seconds": 0.0}
    timestamp = _timestamp_sidecar(audio_path)
    if timestamp is not None:
        audio["timestamp_path"] = str(timestamp)

    videos = []
    matching_videos = sorted(
        (
            candidate
            for candidate in audio_path.parent.iterdir()
            if candidate.is_file()
            and candidate.stem.casefold() == audio_path.stem.casefold()
            and candidate.suffix.lower() in VIDEO_SUFFIXES
        ),
        key=lambda candidate: candidate.name.casefold(),
    )
    for candidate in matching_videos:
        video: dict[str, Any] = {
            "name": "camera" if not videos else f"camera_{len(videos) + 1}",
            "path": str(candidate.resolve()),
            "offset_seconds": 0.0,
        }
        video_timestamp = _timestamp_sidecar(candidate)
        if video_timestamp is not None:
            video["timestamp_path"] = str(video_timestamp)
        videos.append(video)

    events, sidecar = _sidecar_annotations(audio_path)
    return Recording(name=name, audio=audio, videos=videos, annotations=events, sidecar_path=sidecar)


def audio_files_in_folder(folder: str | Path) -> list[Path]:
    folder = Path(folder).expanduser()

    def is_audio(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES and not path.stem.casefold().endswith("_timestamps")

    direct = sorted(
        (path.resolve() for path in folder.iterdir() if is_audio(path)),
        key=lambda path: path.name.casefold(),
    )
    if direct:
        return direct
    return sorted(
        (path.resolve() for path in folder.rglob("*") if is_audio(path)),
        key=lambda path: path.relative_to(folder.resolve()).as_posix().casefold(),
    )


def project_from_audio_files(paths, settings: Mapping[str, Any], *, document_changed: bool = False) -> Project:
    project = Project(recordings=[], settings=dict(settings), document_changed=document_changed)
    project.add_recordings(paths)
    project.document_changed = bool(document_changed)
    return project
