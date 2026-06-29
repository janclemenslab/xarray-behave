"""Internal dataset helpers used by the GUI without depending on Qt."""

import logging
import os
from typing import Optional

import numpy as np
import scipy.signal as ss

from . import annot, api, event_utils, io, v1 as xb

logger = logging.getLogger(__name__)


def assemble_from_file(filename: str, form_data: dict):
    filter_song_requested = form_data["filter_song"] == "yes"
    ds = xb.assemble(
        filepath_daq=filename,
        filepath_annotations=form_data["annotation_path"],
        filepath_definitions=form_data["definition_path"],
        audio_sampling_rate=form_data["samplerate"],
        target_sampling_rate=form_data["target_samplingrate"],
        audio_dataset=form_data["data_set"],
        lazy_load_song=not filter_song_requested,
        make_song_events=form_data.get("generate_event_traces", False),
    )

    if filter_song_requested:
        ds = filter_song(ds, form_data["f_low"], form_data["f_high"])

    ds.attrs["filename"] = filename
    ds.attrs["filebase"] = os.path.splitext(filename)[0]
    ds.attrs["datename"] = ""
    ds.attrs["res_path"] = ""
    ds.attrs["dat_path"] = ""
    return ds


def assemble_from_dir(dirname: str, form_data: dict, pixel_size_mm: Optional[float] = None, manifest: Optional[str] = None):
    if form_data["target_samplingrate"] == 0 or form_data["target_samplingrate"] is None:
        resample_video_data = False
    else:
        resample_video_data = True

    filter_song_requested = form_data["filter_song"] == "yes"
    include_tracks = not form_data["ignore_tracks"]
    include_poses = not form_data["ignore_tracks"]
    lazy_load_song = not filter_song_requested
    base, datename = os.path.split(os.path.normpath(dirname))
    root, dat_path = os.path.split(base)
    annotation_path = None if not len(form_data["annotation_path"]) else form_data["annotation_path"]
    filepath_video = None if not len(form_data["video_filename"]) else form_data["video_filename"]
    filepath_daq = None if not len(form_data["daq_filename"]) else form_data["daq_filename"]
    discovered = None
    if manifest is not None:
        discovered = api.discover(datename, root=root, dat_path=dat_path, res_path="res", manifest=manifest)
        filepath_video = filepath_video or _discovered_path(discovered, "video", "camera")
        filepath_daq = filepath_daq or _discovered_path(discovered, "audio", "main")
        annotation_path = (
            annotation_path
            or _discovered_path(discovered, "annotations", "manual")
            or _discovered_path(discovered, "annotations", "auto")
        )

    assemble_kwargs = {
        "res_path": "res",
        "filepath_annotations": annotation_path,
        "filepath_video": filepath_video,
        "filepath_daq": filepath_daq,
        "fix_fly_indices": form_data["fix_fly_indices"],
        "include_song": ~int(form_data["ignore_song"]),
        "target_sampling_rate": form_data["target_samplingrate"],
        "resample_video_data": resample_video_data,
        "pixel_size_mm": pixel_size_mm,
        "lazy_load_song": lazy_load_song,
        "include_tracks": include_tracks,
        "include_poses": include_poses,
        "make_song_events": form_data.get("generate_event_traces", False),
    }
    if discovered is not None:
        assemble_kwargs.update(
            {
                "filepath_timestamps": _discovered_path(discovered, "timestamps", "camera"),
                "filepath_timestamps_ball": _discovered_path(discovered, "timestamps", "ball"),
                "filepath_tracks": _discovered_path(discovered, "tracks", "main"),
                "filepath_poses": _discovered_path(discovered, "poses", "main"),
                "filepath_definitions": _discovered_path(discovered, "definitions", "main"),
            }
        )
        assemble_kwargs = {key: value for key, value in assemble_kwargs.items() if value is not None}

    ds = xb.assemble(datename, root, dat_path, **assemble_kwargs)

    if filter_song_requested:
        ds = filter_song(ds, form_data["f_low"], form_data["f_high"])

    event_names = []
    if form_data["init_annotations"] and len(form_data["events_string"]):
        for pair in form_data["events_string"].split(";"):
            items = pair.strip().split(",")
            if len(items) > 0 and len(items[0].strip()):
                event_names.append(items[0].strip())

    ds = ensure_event_categories(ds)

    if "song_events" not in ds or len(ds.event_types) == 0:
        cats = {event_name: "event" for event_name in event_names}
        ds.attrs["event_times"] = annot.Events(categories=cats)

    return ds


def _discovered_path(files: dict, group: str, name: str):
    entry = files.get(group, {}).get(name)
    if entry is None:
        entries = files.get(group, {})
        entry = next(iter(entries.values()), None)
    if isinstance(entry, str):
        return entry
    if not entry:
        return None
    if "path" in entry:
        return entry["path"]
    paths = entry.get("paths", [])
    return paths[0] if len(paths) == 1 else None


def load_from_zarr(filename: str, form_data: dict):
    ds = xb.load(filename, lazy=True, use_temp=True)
    load_event_traces = form_data.get("load_event_traces", False)
    has_event_table = "event_times" in ds and "event_names" in ds and len(ds["event_names"]) > 0
    if "song_events" in ds and (load_event_traces or not has_event_table):
        ds.song_events.load()
    if not form_data["lazy"]:
        logger.info("   Loading data from ds.")
        if "song" in ds:
            ds.song.load()
        if "pose_positions_allo" in ds:
            ds.pose_positions_allo.load()
        if "sampletime" in ds:
            ds.sampletime.load()
        if "song_raw" in ds:
            ds.song_raw.load()

    if form_data["filter_song"] == "yes":
        ds = filter_song(ds, form_data["f_low"], form_data["f_high"])

    ds = ensure_event_categories(ds)
    logger.info(ds)
    return ds


def load_annotation_file(filename: str):
    loader = io.get_loader("annotations_manual", filename, basename_is_full_name=True)
    if loader is None:
        raise ValueError(f"No annotation loader found for {filename}.")
    event_times, categories = loader.load(filename)
    return annot.Events(event_times, categories=categories)


def _event_row_key(name: str, row) -> tuple[str, float, float, int]:
    channel = row[2] if len(row) > 2 else -1
    channel = int(channel) if np.isfinite(channel) else -1
    return str(name), float(row[0]), float(row[1]), channel


def merge_event_times(existing, imported):
    merged = annot.Events(existing)
    imported = annot.Events(imported)
    for name in imported.names:
        imported_rows = np.asarray(imported[name])
        if name not in merged:
            merged.add_name(name, times=imported_rows.copy(), overwrite=True)
            continue

        imported_keys = {_event_row_key(name, row) for row in imported_rows}
        existing_rows = np.asarray(merged[name])
        if imported_keys and len(existing_rows):
            keep = [_event_row_key(name, row) not in imported_keys for row in existing_rows]
            existing_rows = existing_rows[np.asarray(keep, dtype=bool)]
        if len(imported_rows):
            merged[name] = np.vstack([existing_rows, imported_rows])
        else:
            merged[name] = existing_rows.copy()
        merged.categories[name] = "event"

    for name in merged.names:
        merged.categories[name] = "event"
    merged.sort()
    return annot.Events(merged)


def filter_song(ds, f_low, f_high):
    if f_low is None:
        f_low = 1.0
    if "song_raw" in ds:
        if f_high is None:
            f_high = ds.song_raw.attrs["sampling_rate_Hz"] / 2 - 1
        else:
            f_high = min(f_high, ds.song_raw.attrs["sampling_rate_Hz"] / 2 - 1)
        sos_bp = ss.butter(
            5,
            [f_low, f_high],
            "bandpass",
            output="sos",
            fs=ds.song_raw.attrs["sampling_rate_Hz"],
        )
        logger.info(f"Filtering `song_raw` between {f_low} and {f_high} Hz.")
        ds.song_raw.data = ss.sosfiltfilt(sos_bp, ds.song_raw.data, axis=0)
    return ds


def ensure_event_categories(ds):
    if "song_events" in ds and "event_categories" not in ds:
        event_categories = ["event" for _evt in ds.event_types.values]
        ds = ds.assign_coords({"event_categories": (("event_types"), event_categories)})
    elif "song_events" in ds and "event_categories" in ds:
        event_categories = ["event" for _evt in ds.event_types.values]
        ds = ds.assign_coords({"event_categories": (("event_types"), event_categories)})
    return ds


def event_times_from_dataset(ds):
    if "event_times" in ds and "event_names" in ds and len(ds["event_names"]) > 0:
        event_times = annot.Events.from_dataset(ds)
    elif "event_times" in ds.attrs:
        event_times = ds.attrs["event_times"].copy()
    elif "song_events" in ds:
        event_times = event_utils.detect_events(ds)
    else:
        event_times = dict()
    return annot.Events(event_times)


def prepare_for_display(ds):
    original_spatial_units = None
    for name in ["body_positions", "pose_positions", "pose_positions_allo"]:
        if name in ds:
            original_spatial_units = ds[name].attrs["spatial_units"]
    ds = xb.convert_spatial_units(ds, to_units="pixels")
    return ds, original_spatial_units


def prepare_for_save(ds, event_times, original_spatial_units=None, generate_event_traces: bool = False):
    if "song_events" in ds and generate_event_traces:
        logger.info("   Updating song events")
        ds = event_utils.eventtimes_to_traces(ds, event_times)

    if original_spatial_units is not None:
        logger.info(f"Converting spatial units back to {original_spatial_units} if required.")
        ds = xb.convert_spatial_units(ds, to_units=original_spatial_units)

    event_times = annot.Events(event_times)
    ds_event_times = event_times.to_dataset()
    if "index" in ds.dims and "event_time" in ds.dims:
        ds = ds.drop_dims(["index", "event_time"])
        ds = ds.combine_first(ds_event_times)

    return ds
