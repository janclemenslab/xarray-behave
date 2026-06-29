"""Persistent, Qt-free configuration helpers for the GUI."""

from __future__ import annotations

from copy import deepcopy
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

CONFIG_VERSION = 1
CONFIG_FILENAME = ".das.yaml"

SOURCE_SPECIFIC_FIELDS = frozenset(
    {
        "annotation_path",
        "daq_filename",
        "data_set",
        "definition_path",
        "pixel_size_mm",
        "samplerate",
        "video_filename",
    }
)

LOAD_DIALOG_FIELDS = {
    "from_file": frozenset(
        {
            "box_size",
            "f_high",
            "f_low",
            "filter_song",
            "generate_event_traces",
            "ignore_tracks",
            "spec_freq_max",
            "spec_freq_min",
            "target_samplingrate",
        }
    ),
    "from_dir": frozenset(
        {
            "box_size_px",
            "events_string",
            "f_high",
            "f_low",
            "filter_song",
            "fix_fly_indices",
            "frame_fliplr",
            "frame_flipud",
            "generate_event_traces",
            "ignore_song",
            "ignore_tracks",
            "init_annotations",
            "spec_freq_max",
            "spec_freq_min",
            "target_samplingrate",
        }
    ),
    "from_zarr": frozenset(
        {
            "box_size",
            "f_high",
            "f_low",
            "filter_song",
            "lazy",
            "load_event_traces",
            "spec_freq_max",
            "spec_freq_min",
        }
    ),
}

WINDOW_FIELDS = {
    "geometry": frozenset({"x", "y", "width", "height", "maximized"}),
    "panels": frozenset(
        {
            "sidebar",
            "movie",
            "tracks",
            "waveform",
            "spectrogram",
            "timeline",
            "event_table",
        }
    ),
    "splitter_sizes": frozenset(
        {
            "sidebar",
            "workspace",
            "movie",
            "tracks",
            "waveform",
            "spectrogram",
            "timeline",
            "event_table",
        }
    ),
}

VIEWER_FIELDS = {
    "waveform": frozenset({"color", "y_limits"}),
    "spectrogram": frozenset({"fmin", "fmax", "levels", "compression", "resolution", "colormap", "denoise", "mel"}),
    "video": frozenset(
        {"box_size", "crop", "maintain_custom_crop", "frame_fliplr", "frame_flipud", "show_dot", "show_poses", "move_poses"}
    ),
    "audio": frozenset({"waveform_all", "events_all", "playback_all", "scale_y_all", "select_loudest_channel"}),
    "annotations": frozenset({"show", "movable", "edit_only_current", "show_labels", "table_audio_link"}),
    "thresholding": frozenset(
        {
            "enabled",
            "value",
            "envelope_std",
            "min_distance",
            "duration_enabled",
            "duration_min",
            "duration_max",
            "bandpass_enabled",
            "bandpass_low",
            "bandpass_high",
        }
    ),
}

EVENT_TYPE_FIELDS = frozenset(
    {"name", "fixed_duration", "duration_seconds", "duration_editable", "color_hex", "visible", "editable"}
)
TOP_LEVEL_FIELDS = frozenset({"version", "load_dialogs", "window", "viewer", "event_types"})


class ConfigError(ValueError):
    """Raised when a GUI configuration file is invalid."""


def _to_builtin(value: Any) -> Any:
    """Convert NumPy and container subclasses to YAML-safe Python values."""
    if isinstance(value, np.generic):
        return _to_builtin(value.item())
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


def global_config_path(home: Optional[Path] = None) -> Path:
    return (Path.home() if home is None else Path(home)).expanduser() / CONFIG_FILENAME


def local_config_path(source: str | os.PathLike[str]) -> Path:
    source_path = Path(source).expanduser()
    if source_path.suffix.lower() == ".zarr" or not source_path.is_dir():
        source_path = source_path.parent
    return source_path / CONFIG_FILENAME


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def filter_dialog_data(kind: str, data: Mapping[str, Any]) -> dict[str, Any]:
    allowed = LOAD_DIALOG_FIELDS.get(kind, frozenset())
    return {key: deepcopy(value) for key, value in data.items() if key in allowed and key not in SOURCE_SPECIFIC_FIELDS}


def _known_mapping(data: Any, allowed: frozenset[str], section: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{section} must be a mapping")
    unknown = set(data) - allowed
    if unknown:
        logger.warning("Ignoring unknown GUI config keys in %s: %s", section, ", ".join(sorted(map(str, unknown))))
    return {key: deepcopy(value) for key, value in data.items() if key in allowed}


def _require_types(value: Any, expected, field: str, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, expected):
        raise ConfigError(f"{field} has an invalid value")


def _validate_values(config: Mapping[str, Any]) -> None:
    dialog_bools = {
        "fix_fly_indices",
        "frame_fliplr",
        "frame_flipud",
        "generate_event_traces",
        "ignore_song",
        "ignore_tracks",
        "init_annotations",
        "lazy",
        "load_event_traces",
    }
    dialog_strings = {"events_string", "filter_song"}
    for kind, values in config.get("load_dialogs", {}).items():
        for field, value in values.items():
            if field in dialog_bools:
                _require_types(value, bool, f"load_dialogs.{kind}.{field}")
            elif field in dialog_strings:
                _require_types(value, str, f"load_dialogs.{kind}.{field}")
            else:
                _require_types(value, (int, float), f"load_dialogs.{kind}.{field}", allow_none=True)

    window = config.get("window", {})
    for field, value in window.get("geometry", {}).items():
        expected = bool if field == "maximized" else (int, float)
        _require_types(value, expected, f"window.geometry.{field}")
    for field, value in window.get("panels", {}).items():
        _require_types(value, bool, f"window.panels.{field}")
    for field, value in window.get("splitter_sizes", {}).items():
        _require_types(value, (int, float), f"window.splitter_sizes.{field}")

    viewer = config.get("viewer", {})
    waveform = viewer.get("waveform", {})
    if "color" in waveform:
        _require_types(waveform["color"], str, "viewer.waveform.color")
    if waveform.get("y_limits") is not None:
        limits = waveform["y_limits"]
        if not isinstance(limits, (list, tuple)) or len(limits) != 2 or not all(isinstance(v, (int, float)) for v in limits):
            raise ConfigError("viewer.waveform.y_limits must contain two numbers or be null")

    spectrogram = viewer.get("spectrogram", {})
    for field in ("fmin", "fmax"):
        if field in spectrogram:
            _require_types(spectrogram[field], (int, float), f"viewer.spectrogram.{field}", allow_none=True)
    if "levels" in spectrogram:
        levels = spectrogram["levels"]
        if (
            not isinstance(levels, (list, tuple))
            or len(levels) != 2
            or not all(v is None or isinstance(v, (int, float)) for v in levels)
        ):
            raise ConfigError("viewer.spectrogram.levels must contain two numbers or null values")
    for field in ("compression", "resolution"):
        if field in spectrogram:
            _require_types(spectrogram[field], (int, float), f"viewer.spectrogram.{field}")
    if "colormap" in spectrogram:
        _require_types(spectrogram["colormap"], str, "viewer.spectrogram.colormap")
    for field in ("denoise", "mel"):
        if field in spectrogram:
            _require_types(spectrogram[field], bool, f"viewer.spectrogram.{field}")

    video = viewer.get("video", {})
    if "box_size" in video:
        _require_types(video["box_size"], (int, float), "viewer.video.box_size")
    for field in set(video) - {"box_size"}:
        _require_types(video[field], bool, f"viewer.video.{field}")

    for section in ("audio", "annotations"):
        for field, value in viewer.get(section, {}).items():
            _require_types(value, bool, f"viewer.{section}.{field}")

    thresholding = viewer.get("thresholding", {})
    threshold_bools = {"enabled", "duration_enabled", "bandpass_enabled"}
    for field, value in thresholding.items():
        if field in threshold_bools:
            _require_types(value, bool, f"viewer.thresholding.{field}")
        else:
            _require_types(value, (int, float), f"viewer.thresholding.{field}", allow_none=field == "bandpass_high")

    for index, preset in enumerate(config.get("event_types", [])):
        for field in ("fixed_duration", "duration_editable", "visible", "editable"):
            if field in preset:
                _require_types(preset[field], bool, f"event_types[{index}].{field}")
        if "duration_seconds" in preset:
            _require_types(preset["duration_seconds"], (int, float), f"event_types[{index}].duration_seconds")
        if "color_hex" in preset:
            _require_types(preset["color_hex"], str, f"event_types[{index}].color_hex")


def sanitize_config(data: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ConfigError("GUI config must contain a mapping at the top level")
    data = _to_builtin(data)
    if data.get("version") != CONFIG_VERSION:
        raise ConfigError(f"GUI config version must be {CONFIG_VERSION}")

    unknown = set(data) - TOP_LEVEL_FIELDS
    if unknown:
        logger.warning("Ignoring unknown GUI config sections: %s", ", ".join(sorted(map(str, unknown))))

    result: dict[str, Any] = {"version": CONFIG_VERSION}

    if "load_dialogs" in data:
        dialogs = _known_mapping(data["load_dialogs"], frozenset(LOAD_DIALOG_FIELDS), "load_dialogs")
        result["load_dialogs"] = {}
        for kind, values in dialogs.items():
            if not isinstance(values, Mapping):
                raise ConfigError(f"load_dialogs.{kind} must be a mapping")
            unknown_fields = set(values) - LOAD_DIALOG_FIELDS[kind]
            if unknown_fields:
                logger.warning(
                    "Ignoring unknown GUI config keys in load_dialogs.%s: %s",
                    kind,
                    ", ".join(sorted(map(str, unknown_fields))),
                )
            result["load_dialogs"][kind] = filter_dialog_data(kind, values)

    if "window" in data:
        window = _known_mapping(data["window"], frozenset(WINDOW_FIELDS), "window")
        result["window"] = {}
        for section, values in window.items():
            result["window"][section] = _known_mapping(values, WINDOW_FIELDS[section], f"window.{section}")

    if "viewer" in data:
        viewer = _known_mapping(data["viewer"], frozenset(VIEWER_FIELDS), "viewer")
        result["viewer"] = {}
        for section, values in viewer.items():
            result["viewer"][section] = _known_mapping(values, VIEWER_FIELDS[section], f"viewer.{section}")

    if "event_types" in data:
        if not isinstance(data["event_types"], list):
            raise ConfigError("event_types must be a list")
        result["event_types"] = []
        for index, item in enumerate(data["event_types"]):
            preset = _known_mapping(item, EVENT_TYPE_FIELDS, f"event_types[{index}]")
            if not isinstance(preset.get("name"), str) or not preset["name"]:
                raise ConfigError(f"event_types[{index}].name must be a non-empty string")
            result["event_types"].append(preset)

    _validate_values(result)
    return result


def read_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read GUI config {config_path}: {exc}") from exc
    try:
        return sanitize_config(data)
    except ConfigError as exc:
        raise ConfigError(f"Invalid GUI config {config_path}: {exc}") from exc


def write_config(path: str | os.PathLike[str], data: Mapping[str, Any]) -> Path:
    config_path = Path(path).expanduser()
    clean = sanitize_config(data)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=config_path.parent, prefix=f".{config_path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            yaml.safe_dump(clean, handle, sort_keys=False, allow_unicode=False)
        os.replace(temporary_path, config_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return config_path


class GuiConfigManager:
    def __init__(self, explicit_path: Optional[str] = None, home: Optional[Path] = None) -> None:
        self.explicit_path = Path(explicit_path).expanduser() if explicit_path else None
        self.home = Path(home).expanduser() if home is not None else None
        self.config: dict[str, Any] = {"version": CONFIG_VERSION}
        self.source: Optional[str] = None

    @property
    def global_path(self) -> Path:
        return global_config_path(self.home)

    def load_for_source(self, source: Optional[str] = None) -> dict[str, Any]:
        paths: list[tuple[Path, bool]] = [(self.global_path, False)]
        if source:
            local_path = local_config_path(source)
            if local_path != self.global_path:
                paths.append((local_path, False))
        if self.explicit_path is not None:
            paths.append((self.explicit_path, True))

        merged: dict[str, Any] = {"version": CONFIG_VERSION}
        seen: set[Path] = set()
        for path, explicit in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if not path.exists():
                if explicit:
                    raise ConfigError(f"Explicit GUI config does not exist: {path}")
                continue
            try:
                merged = deep_merge(merged, read_config(path))
            except ConfigError as exc:
                if explicit:
                    raise
                logger.warning("Ignoring invalid automatic GUI config %s: %s", path, exc)

        self.source = source
        self.config = sanitize_config(merged)
        return deepcopy(self.config)

    def dialog_values(self, kind: str) -> dict[str, Any]:
        return deepcopy(self.config.get("load_dialogs", {}).get(kind, {}))

    def remember_dialog(self, kind: str, data: Mapping[str, Any]) -> None:
        dialogs = self.config.setdefault("load_dialogs", {})
        dialogs[kind] = filter_dialog_data(kind, data)

    def save_global(self, data: Mapping[str, Any]) -> Path:
        return write_config(self.global_path, data)

    def save_as(self, path: str | os.PathLike[str], data: Mapping[str, Any]) -> Path:
        return write_config(path, data)
