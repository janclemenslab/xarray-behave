"""Manifest-based v2 assembly API."""

from __future__ import annotations

from glob import glob
from importlib import resources
from typing import Any, Mapping, Optional, Union

import logging
import numpy as np
import xarray as xr
import yaml

from . import annot, event_utils, io, loaders as ld
from .io.samplestamps import SampStamp
from .xarray_behave import add_time, align_time, convert_spatial_units, interp_duplicates

logger = logging.getLogger(__name__)


def discover(
    datename: str = "",
    root: str = "",
    manifest: Optional[Union[str, Mapping[str, Any]]] = None,
    dat_path: str = "dat",
    res_path: str = "res",
    **context,
) -> dict:
    """Discover files using a simple YAML manifest."""

    manifest_data = _load_manifest(manifest)
    format_context = {
        "datename": datename,
        "root": root,
        "dat_path": dat_path,
        "res_path": res_path,
        **context,
    }
    found: dict[str, Any] = {
        "_context": format_context,
        "_manifest": str(manifest) if manifest is not None and not isinstance(manifest, Mapping) else "default",
    }

    for group, streams in manifest_data.items():
        if not isinstance(streams, Mapping):
            continue
        group_found = {}
        for name, spec in streams.items():
            spec = _coerce_spec(spec)
            matches = _matches(spec, format_context)
            if not matches:
                continue
            entry = {key: value for key, value in spec.items() if key not in {"path", "paths", "patterns", "mode"}}
            if spec.get("mode") == "all":
                entry["paths"] = matches
            else:
                entry["path"] = matches[0]
            group_found[name] = entry
        if group_found:
            found[group] = group_found

    return found


def assemble(
    files: Mapping[str, Any],
    *,
    audio_sampling_rate: Optional[float] = None,
    audio_dataset: Optional[str] = None,
    lazy_load_audio: bool = False,
    annotation_column: str = "Annotation",
    pixel_size_mm: Optional[float] = None,
) -> xr.Dataset:
    """Assemble a native-timeline Dataset from discovered files.

    Audio and video entries may specify ``timestamp_path`` and
    ``offset_seconds``. Audio entries also accept ``sampling_rate_Hz`` and
    ``audio_dataset``; video entries accept ``frame_rate_Hz``.
    """

    data_vars: dict[str, xr.DataArray] = {}
    attrs = dict(files.get("_context", {}))
    attrs["pixel_size_mm"] = np.nan if pixel_size_mm is None else pixel_size_mm

    audio_info = _load_audio_vars(files, audio_sampling_rate, audio_dataset, lazy_load_audio)
    data_vars.update(audio_info["vars"])
    attrs.update(audio_info["attrs"])

    stamps = _load_sample_stamps(files, attrs, audio_info)
    data_vars.update(
        _load_video_vars(
            files,
            stamps.get("camera"),
            attrs.get("ref_time", 0.0),
            audio_info["reference_is_absolute"],
        )
    )
    data_vars.update(_load_native_arrays(files, stamps, attrs["pixel_size_mm"]))
    data_vars.update(_load_event_vars(files, annotation_column))

    return xr.Dataset(data_vars, attrs=attrs)


def resample(
    ds: xr.Dataset,
    *,
    target_sampling_rate: Optional[float] = 1_000,
    resample_video_data: bool = True,
    make_event_traces: bool = False,
    convert_to_mm: bool = True,
) -> xr.Dataset:
    """Return a copy with frame-indexed arrays aligned to a uniform time grid."""

    ss, sampling_rate, ref_time = _stamp_from_dataset(ds, target_sampling_rate)
    if target_sampling_rate == 0 or target_sampling_rate is None:
        resample_video_data = False

    frame_data = [ds[name] for name in ("body_positions", "pose_positions", "pose_positions_allo") if name in ds]
    if frame_data:
        first_frame = int(min(da.frame_number.data[0] for da in frame_data))
        last_frame = int(max(da.frame_number.data[-1] for da in frame_data))
    else:
        camera_frame = _camera_frame_coord(ds)
        first_frame = int(camera_frame[0])
        last_frame = int(camera_frame[-1])

    if not resample_video_data:
        frame_numbers = np.arange(first_frame, last_frame)
        target_samples = interp_duplicates(ss.sample(frame_numbers))
        target_sampling_rate = 1 / np.nanmedian(np.diff(ss.frame_time(frame_numbers)))
    else:
        step = sampling_rate / target_sampling_rate
        last_sample = _last_sample(ds, sampling_rate)
        last_sample_with_frame = np.min((last_sample, ss.sample(last_frame - 1))).astype(np.intp)
        target_samples = np.arange(0, last_sample_with_frame, step, dtype=np.uintp)

    time = ss.sample_time(target_samples) - ref_time
    out = ds.copy()
    out = _align_if_present(out, "body_positions", ss, target_samples, time, ref_time, target_sampling_rate)
    out = _align_if_present(out, "pose_positions", ss, target_samples, time, ref_time, target_sampling_rate)
    out = _align_if_present(out, "pose_positions_allo", ss, target_samples, time, ref_time, target_sampling_rate)
    out = _align_aux_if_present(
        out, "balltracks", "frame_number_ball", "_ball", target_samples, time, ref_time, target_sampling_rate
    )
    out = _align_aux_if_present(
        out, "movieparams", "frame_number_movie", "_movie", target_samples, time, ref_time, target_sampling_rate
    )

    out = out.assign_coords(
        {
            "time": time,
            "nearest_frame": (("time"), ss.times2frames(time + ref_time).astype(np.intp)),
        }
    )
    out.attrs["sampling_rate_Hz"] = sampling_rate
    out.attrs["target_sampling_rate_Hz"] = target_sampling_rate
    out.attrs["ref_time"] = ref_time

    if make_event_traces and "event_times" in out and "event_names" in out:
        out["event_traces"] = _make_event_traces(out, time, target_sampling_rate)

    if convert_to_mm:
        out = convert_spatial_units(out, to_units="mm", names=["body_positions", "pose_positions", "pose_positions_allo"])

    return out


def _load_manifest(manifest):
    if manifest is None:
        path = resources.files("xarray_behave").joinpath("manifests/default.yaml")
        with path.open() as handle:
            return yaml.safe_load(handle)
    if isinstance(manifest, Mapping):
        return dict(manifest)
    with open(manifest) as handle:
        return yaml.safe_load(handle)


def _coerce_spec(spec):
    if isinstance(spec, str):
        return {"path": [spec]}
    spec = dict(spec)
    if "path" in spec and isinstance(spec["path"], str):
        spec["path"] = [spec["path"]]
    return spec


def _matches(spec, context):
    patterns = spec.get("path", spec.get("paths", spec.get("patterns", [])))
    matches = []
    for pattern in patterns:
        matches.extend(sorted(glob(pattern.format(**context))))
        if matches and spec.get("mode") != "all":
            break
    return matches


def _entry_path(entry):
    if isinstance(entry, str):
        return entry
    return entry.get("path")


def _entry_paths(entry):
    if isinstance(entry, str):
        return [entry]
    if "paths" in entry:
        return entry["paths"]
    if "path" in entry:
        return [entry["path"]]
    return []


def _load_audio_vars(files, audio_sampling_rate, audio_dataset, lazy_load_audio):
    data_vars = {}
    attrs = {}
    reference_is_absolute = False
    reference_set = False
    ref_time = 0.0
    for source_name, entry in files.get("audio", {}).items():
        path = _entry_path(entry)
        if not path:
            continue
        entry = entry if isinstance(entry, Mapping) else {}
        loader = _loader_for_path("audio", path) or io.audio.AudioFile(path)
        song_channels = _all_source_channels(loader, path)
        source_dataset = entry.get("audio_dataset", audio_dataset)
        data, _, sampling_rate = loader.load(
            path,
            song_channels=song_channels,
            return_nonsong_channels=False,
            lazy=lazy_load_audio,
            audio_dataset=source_dataset,
        )
        sampling_rate_override = entry.get("sampling_rate_Hz")
        if sampling_rate_override is not None:
            sampling_rate = float(sampling_rate_override)
        elif sampling_rate is None:
            sampling_rate = audio_sampling_rate
        timestamp_path = _timestamp_path(files, source_name, entry)
        timestamp_data = _load_audio_timestamps(path, timestamp_path)
        if sampling_rate is None and timestamp_data is not None:
            indices, timestamps = timestamp_data
            if len(indices) > 1:
                sampling_rate = float(np.median(np.diff(indices)) / np.median(np.diff(timestamps)))
        if sampling_rate is None:
            raise ValueError(f"No sampling rate for audio file {path}.")
        if sampling_rate <= 0:
            raise ValueError(f"Sampling rate for audio file {path} must be positive.")
        data = data[:, np.newaxis] if getattr(data, "ndim", 2) == 1 else data
        raw_times, has_absolute_time = _audio_sample_times(data.shape[0], sampling_rate, timestamp_data)
        if not reference_set:
            reference_is_absolute = has_absolute_time
            ref_time = float(raw_times[0]) if has_absolute_time and len(raw_times) else 0.0
            reference_set = True
        if has_absolute_time:
            origin = ref_time if reference_is_absolute else float(raw_times[0])
            source_times = raw_times - origin
        else:
            source_times = raw_times
        offset_seconds = float(entry.get("offset_seconds", 0.0))
        source_times = source_times + offset_seconds
        attrs.setdefault("sampling_rate_Hz", sampling_rate)
        attrs.setdefault("last_sample_number", data.shape[0] - 1)

        splits = entry.get("splits", {"audio": ":"})
        used_channels = set()
        for var_name, channel_spec in splits.items():
            channels = _parse_channels(channel_spec, data.shape[1])
            overlap = used_channels.intersection(channels.tolist())
            if overlap:
                raise ValueError(f"Audio split {var_name!r} overlaps source channels {sorted(overlap)}.")
            used_channels.update(channels.tolist())
            if var_name in data_vars:
                raise ValueError(f"Duplicate audio split name {var_name!r}.")
            time_name = f"{var_name}_time"
            channel_name = f"{var_name}_channels"
            data_vars[var_name] = xr.DataArray(
                data=data[:, channels],
                dims=[time_name, channel_name],
                coords={
                    time_name: source_times,
                    channel_name: channels,
                },
                attrs={
                    "description": f"Audio split {var_name!r} from {source_name!r}.",
                    "sampling_rate_Hz": sampling_rate,
                    "time_units": "seconds",
                    "amplitude_units": "volts",
                    "source_audio": source_name,
                    "source_path": path,
                    "offset_seconds": offset_seconds,
                    "sampling_rate_overridden": sampling_rate_override is not None,
                },
            )
            if timestamp_path:
                data_vars[var_name].attrs["timestamp_path"] = str(timestamp_path)
            if source_dataset:
                data_vars[var_name].attrs["audio_dataset"] = str(source_dataset)
    attrs["ref_time"] = ref_time
    return {
        "vars": data_vars,
        "attrs": attrs,
        "reference_is_absolute": reference_is_absolute,
    }


def _timestamp_path(files, source_name, entry):
    if isinstance(entry, Mapping) and entry.get("timestamp_path"):
        return entry["timestamp_path"]
    return _entry_path(files.get("timestamps", {}).get(source_name, {}))


def _load_audio_timestamps(path, timestamp_path):
    if timestamp_path:
        if str(timestamp_path).lower().endswith(".csv"):
            indices, timestamps = io.timestamps.CsvStamps().load(timestamp_path)
        else:
            indices, timestamps = io.timestamps.DaqStamps().load(timestamp_path)
        return np.asarray(indices), np.asarray(timestamps)
    if str(path).lower().endswith("_daq.h5"):
        import h5py

        with h5py.File(path, "r") as file:
            if "systemtime" in file and "samplenumber" in file:
                indices, timestamps = io.timestamps.DaqStamps().load(path)
                return np.asarray(indices), np.asarray(timestamps)
    return None


def _audio_sample_times(nb_samples, sampling_rate, timestamp_data):
    if timestamp_data is None:
        return np.arange(nb_samples) / sampling_rate, False
    sample_numbers, timestamps = timestamp_data
    if len(timestamps) == 0:
        return np.arange(nb_samples) / sampling_rate, False
    if len(timestamps) == 1:
        return timestamps[0] + np.arange(nb_samples) / sampling_rate, True
    stamp = SampStamp(
        sample_times=np.asarray(timestamps, dtype=float).copy(),
        sample_numbers=np.asarray(sample_numbers, dtype=float),
    )
    return np.asarray(stamp.sample_time(np.arange(nb_samples)), dtype=float), True


def _all_source_channels(loader, path):
    if isinstance(loader, io.audio.Ethodrome):
        import h5py

        with h5py.File(path, "r") as file:
            return np.arange(file["samples"].shape[1])
    return None


def _parse_channels(spec, nb_channels):
    if spec in (None, ":"):
        channels = np.arange(nb_channels)
    elif isinstance(spec, int):
        channels = np.array([spec])
    elif isinstance(spec, (list, tuple)):
        channels = np.asarray(spec, dtype=int)
    elif ":" in spec:
        start, stop = spec.split(":", 1)
        start = int(start) if start else 0
        stop = int(stop) if stop else nb_channels
        channels = np.arange(start, stop)
    else:
        channels = np.array([int(spec)])
    if np.any(channels < 0) or np.any(channels >= nb_channels):
        raise ValueError(f"Audio channel spec {spec!r} is outside 0:{nb_channels}.")
    return channels


def _load_sample_stamps(files, attrs, audio_info):
    stamps = {}
    audio_path = None
    if files.get("audio"):
        audio_path = _entry_path(next(iter(files["audio"].values())))
    timestamp_entries = files.get("timestamps", {})
    camera_timestamp = _entry_path(timestamp_entries.get("camera", timestamp_entries.get("main", {})))
    if audio_info["vars"]:
        var = next(iter(audio_info["vars"].values()))
        sampling_rate = var.attrs["sampling_rate_Hz"]
        sample_times = np.asarray(var[var.dims[0]].data, dtype=float)
        if audio_info["reference_is_absolute"]:
            sample_times = sample_times + attrs["ref_time"]
        attrs["last_sample_number"] = var.shape[0] - 1
        attrs.setdefault("sampling_rate_Hz", sampling_rate)
        if camera_timestamp:
            frame_times = _load_frame_times(camera_timestamp)
            if not audio_info["reference_is_absolute"]:
                frame_times = frame_times - frame_times[0]
            stamps["camera"] = SampStamp(sample_times=sample_times, frame_times=frame_times)
    else:
        attrs.setdefault("ref_time", 0.0)

    ball_timestamp = _entry_path(timestamp_entries.get("ball", {}))
    if audio_path and audio_path.endswith(".h5") and ball_timestamp:
        stamps["ball"], _, _ = ld.load_times(ball_timestamp, audio_path)

    movie_timestamp = _entry_path(timestamp_entries.get("movie", {}))
    if audio_path and audio_path.endswith(".h5") and movie_timestamp:
        stamps["movie"], _, _ = ld.load_movietimes(movie_timestamp, audio_path)

    return stamps


def _load_video_vars(files, ss, ref_time, reference_is_absolute):
    data_vars = {}
    timestamp_entries = files.get("timestamps", {})
    for name, entry in files.get("video", {}).items():
        path = _entry_path(entry)
        if not path:
            continue
        entry = entry if isinstance(entry, Mapping) else {}
        data_vars[f"{name}_video_path"] = xr.DataArray(str(path))
        timestamp = entry.get("timestamp_path") or _entry_path(timestamp_entries.get(name, {}))
        offset_seconds = float(entry.get("offset_seconds", 0.0))
        frame_rate_override = entry.get("frame_rate_Hz")
        if frame_rate_override is not None and float(frame_rate_override) <= 0:
            raise ValueError(f"Frame rate for video file {path} must be positive.")
        frame_times = None
        if timestamp:
            frame_times = _load_frame_times(timestamp)
            origin = ref_time if reference_is_absolute else frame_times[0]
            frame_times = frame_times - origin
        elif name == "camera" and ss is not None:
            frame_times = ss.frames2times.y - ref_time
        if frame_times is None:
            try:
                from .gui.modern_video import PyAVVideoReader

                reader = PyAVVideoReader(path)
                frame_rate = float(frame_rate_override or reader.frame_rate)
                frame_times = np.arange(reader.number_of_frames) / frame_rate
            except Exception:
                frame_times = None
        if frame_times is not None:
            frame_times = np.asarray(frame_times, dtype=float) + offset_seconds
            frame_dim = f"{name}_frame"
            data_vars[f"{name}_frame_time"] = xr.DataArray(
                frame_times,
                dims=[frame_dim],
                coords={frame_dim: np.arange(len(frame_times))},
                attrs={
                    "time_units": "seconds",
                    "source_path": path,
                    "offset_seconds": offset_seconds,
                },
            )
            if timestamp:
                data_vars[f"{name}_frame_time"].attrs["timestamp_path"] = str(timestamp)
            if frame_rate_override is not None:
                data_vars[f"{name}_frame_time"].attrs["frame_rate_Hz"] = float(frame_rate_override)
    return data_vars


def _load_frame_times(path):
    if str(path).endswith(".csv"):
        _, frame_times = io.timestamps.CsvStamps().load(path)
    else:
        _, frame_times = io.timestamps.CamStamps().load(path)
    return np.asarray(frame_times)


def _load_native_arrays(files, stamps, pixel_size_mm):
    data_vars = {}
    if "tracks" in files:
        path = _entry_path(next(iter(files["tracks"].values())))
        loader = _loader_for_path("tracks", path)
        if loader:
            try:
                tracks = loader.make(path)
                tracks.attrs["pixel_size_mm"] = pixel_size_mm
                if "camera" in stamps:
                    tracks = add_time(tracks, stamps["camera"], dim="frame_number")
                data_vars["body_positions"] = tracks
            except Exception:
                logger.exception("Loading tracks from %s failed.", path)

    if "poses" in files:
        path = _entry_path(next(iter(files["poses"].values())))
        loader = _loader_for_path("poses", path)
        if loader:
            try:
                poses, poses_allo = loader.make(path)
                poses.attrs["pixel_size_mm"] = pixel_size_mm
                poses_allo.attrs["pixel_size_mm"] = pixel_size_mm
                if "camera" in stamps:
                    poses = add_time(poses, stamps["camera"], dim="frame_number")
                    poses_allo = add_time(poses_allo, stamps["camera"], dim="frame_number")
                data_vars["pose_positions"] = poses
                data_vars["pose_positions_allo"] = poses_allo
            except Exception:
                logger.exception("Loading poses from %s failed.", path)

    if "balltracks" in files and "ball" in stamps:
        path = _entry_path(next(iter(files["balltracks"].values())))
        loader = _loader_for_path("balltracks", path)
        if loader:
            try:
                data_vars["balltracks"] = add_time(loader.make(path), stamps["ball"], dim="frame_number_ball", suffix="_ball")
            except Exception:
                logger.exception("Loading balltracks from %s failed.", path)

    if "movieparams" in files and "movie" in stamps:
        path = _entry_path(next(iter(files["movieparams"].values())))
        loader = _loader_for_path("movieparams", path)
        if loader:
            try:
                data_vars["movieparams"] = add_time(
                    loader.make(path), stamps["movie"], dim="frame_number_movie", suffix="_movie"
                )
            except Exception:
                logger.exception("Loading movieparams from %s failed.", path)

    return data_vars


def _load_event_vars(files, annotation_column):
    event_seconds = {}
    event_categories = {}
    for entry in files.get("annotations", {}).values():
        for path in _entry_paths(entry):
            loader = _loader_for_path("annotations", path)
            if loader is None:
                loader = _loader_for_path("annotations_manual", path)
            if loader is None:
                continue
            if isinstance(loader, io.annotations_manual.RavenPro):
                loaded, categories = loader.load(path, annotation_column=annotation_column)
            else:
                loaded, categories = loader.load(path)
            event_seconds.update(loaded)
            event_categories.update(categories)

    for entry in files.get("definitions", {}).values():
        for path in _entry_paths(entry):
            loader = _loader_for_path("definitions_manual", path)
            if loader:
                loaded, categories = loader.load(path)
                event_seconds.update({key: val for key, val in loaded.items() if key not in event_seconds})
                event_categories.update(categories)

    event_seconds = ld.fix_keys(event_seconds)
    event_categories = dict.fromkeys(ld.fix_keys(event_categories), "event")
    for name in event_seconds:
        event_categories.setdefault(name, "event")

    event_ds = annot.Events(event_seconds, categories=event_categories).to_dataset()
    return {"event_times": event_ds.event_times, "event_names": event_ds.event_names}


def _loader_for_path(kind, path):
    if not path:
        return None
    loader = io.get_loader(kind=kind, basename=path, basename_is_full_name=True)
    if isinstance(loader, list):
        return loader[0] if loader else None
    return loader


def _stamp_from_dataset(ds, target_sampling_rate):
    ref_time = float(ds.attrs.get("ref_time", 0.0))
    if "audio" in ds:
        time_name = ds["audio"].dims[0]
        sample_times = np.asarray(ds[time_name].data) + ref_time
        sample_numbers = np.arange(sample_times.shape[0])
        sampling_rate = float(ds["audio"].attrs["sampling_rate_Hz"])
    else:
        frame_times = np.asarray(ds["camera_frame_time"].data)
        sampling_rate = 10 * (target_sampling_rate or 1 / np.nanmedian(np.diff(frame_times)))
        sample_times = np.arange(0, frame_times[-1], 1 / sampling_rate)
        sample_numbers = np.arange(sample_times.shape[0])

    frame_times = np.asarray(ds["camera_frame_time"].data) + ref_time
    ss = SampStamp(sample_times=sample_times, frame_times=frame_times, sample_numbers=sample_numbers)
    return ss, sampling_rate, ref_time


def _camera_frame_coord(ds):
    if "camera_frame_time" in ds:
        dim = ds["camera_frame_time"].dims[0]
        return ds[dim].data
    raise ValueError("Dataset has no camera frame timeline.")


def _last_sample(ds, sampling_rate):
    if "audio" in ds:
        return ds["audio"].shape[0] - 1
    return int(ds["camera_frame_time"].data[-1] * sampling_rate)


def _align_if_present(out, name, ss, target_samples, time, ref_time, target_sampling_rate):
    if name not in out:
        return out
    aligned = align_time(
        out[name], ss, target_samples, ref_time=ref_time, target_time=time, extrapolate=name == "body_positions"
    )
    aligned.attrs["sampling_rate_Hz"] = target_sampling_rate
    out[name] = aligned
    return out


def _align_aux_if_present(out, name, dim, suffix, target_samples, time, ref_time, target_sampling_rate):
    if name not in out:
        return out
    ss = _aux_stamp(out[name], dim, suffix, ref_time)
    aligned = align_time(
        out[name],
        ss,
        target_samples,
        target_time=time,
        dim=dim,
        suffix=suffix,
        ref_time=ref_time,
        extrapolate=name == "movieparams",
    )
    aligned.attrs["sampling_rate_Hz"] = target_sampling_rate
    out[name] = aligned
    return out


def _aux_stamp(da, dim, suffix, ref_time):
    frame_times = np.asarray(da[f"frametimes{suffix}"].data)
    frame_samples = np.asarray(da[f"framesamples{suffix}"].data)
    return SampStamp(
        sample_times=frame_times,
        frame_times=frame_times,
        sample_numbers=frame_samples,
        frame_numbers=np.asarray(da[dim].data),
    )


def _make_event_traces(ds, time, sampling_rate):
    names = ds.event_names.data.tolist()
    traces = xr.DataArray(
        np.zeros((len(time), len(names)), dtype=np.int16),
        dims=["time", "event_types"],
        coords={"time": time, "event_types": names},
        attrs={"sampling_rate_Hz": sampling_rate, "event_times": annot.Events.from_dataset(ds)},
    )
    traces_ds = event_utils.eventtimes_to_traces(traces.to_dataset(name="song_events"), traces.attrs["event_times"])
    event_traces = traces_ds.song_events.rename("event_traces")
    event_traces.attrs.pop("event_times", None)
    return event_traces
