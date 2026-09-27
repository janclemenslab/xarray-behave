import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def detect_events(ds):
    """Transform ds.song_events into dict of event start/stop times.
    Args:
        ds ([xarray.Dataset]): dataset with song_events

    Returns:
        dict: with event start/stop times.
    """
    event_times = dict()
    ds.song_events.data = ds.song_events.data.astype(float)  # make sure this is non-bool so diff works
    event_names = ds.song_events.event_types.data
    logger.info("Extracting event times from song_events:")
    for event_idx, event_name in enumerate(event_names):
        logger.info(f"   {event_name}")
        event_times[event_name] = _trace_to_start_stop(ds.song_events[:, event_idx].data, ds.song_events.time.data)
    return event_times


def _trace_to_start_stop(trace, times=None):
    trace = np.asarray(trace).astype(bool).ravel()
    if times is None:
        times = np.arange(trace.shape[0])
    times = np.asarray(times)
    if trace.size == 0 or not np.any(trace):
        return np.zeros((0, 2))

    padded = np.concatenate(([False], trace, [False]))
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    stops_exclusive = np.where(changes == -1)[0]
    rows = []
    for start, stop_exclusive in zip(starts, stops_exclusive):
        stop = max(start, stop_exclusive - 1)
        start_time = times[min(start, len(times) - 1)]
        stop_time = times[min(stop, len(times) - 1)]
        rows.append([start_time, stop_time])
    return np.asarray(rows, dtype=float)


def _as_start_stop_rows(event_data):
    rows = np.asarray(event_data, dtype=float)
    if rows.size == 0:
        return np.zeros((0, 2), dtype=float)
    if rows.ndim == 1:
        rows = rows[:, np.newaxis]
    if rows.shape[1] == 1:
        rows = np.concatenate((rows, rows), axis=1)
    return rows[:, :2]


def infer_event_categories_from_traces(data):
    """Return legacy event categories for trace columns.

    Args:
        data ([type]): binary matrix [samples x events]
    """
    return ["event"] * data.shape[1]


def update_traces(ds, event_times):
    """Slightly redundant with eventtimes_to_traces but
    will add populate new events.
    """
    ## event_times to ds.song_events
    # make new song_events DataArray
    if "song_events" in ds:
        old_values = ds.song_events.values.copy()
        attrs = ds.song_events.attrs.copy()
    else:
        old_values = np.zeros((ds.time.shape[0], 0))
        attrs = {"sampling_rate_Hz": ds.attrs["target_sampling_rate_Hz"]}

    new_values = np.zeros_like(old_values, shape=(old_values.shape[0], len(event_times)))
    fs = attrs["sampling_rate_Hz"]
    event_categories = {event_name: "event" for event_name in event_times.keys()}
    # populate with data:
    for cnt, (event_name, event_data) in enumerate(event_times.items()):
        event_rows = _as_start_stop_rows(event_data)
        logger.info(f"   {event_name} ({event_rows.shape[0]} instances)")
        for onset, offset in event_rows:
            if not np.isfinite(onset) or not np.isfinite(offset):
                continue
            onset_idx = int(round(min(onset, offset) * fs))
            offset_idx = int(round(max(onset, offset) * fs))
            onset_idx = max(0, min(onset_idx, len(new_values) - 1))
            offset_idx = max(0, min(offset_idx, len(new_values) - 1))
            if onset_idx == offset_idx:
                new_values[onset_idx, cnt] = 1
            else:
                new_values[onset_idx : offset_idx + 1, cnt] = 1

    # rebuild dataset
    song_events = xr.DataArray(
        data=new_values,
        dims=["time", "event_types"],
        coords={
            "time": ds.time,
            "event_types": list(event_times.keys()),
            "event_categories": (("event_types"), list(event_categories.values())),
        },
        attrs=attrs,
    )
    # delete old
    if "song_events" in ds:
        del ds["song_events"]
        del ds.coords["event_types"]
        if "event_categories" in ds.coords:
            del ds.coords["event_categories"]
    # add new
    ds = xr.merge((ds, song_events.to_dataset(name="song_events")))
    return ds


def eventtimes_to_traces(ds, event_times):
    """Update events in ds.song_events from dict.

    Does not add new events (events that exist in event_times but not in ds.song_events)!!

    Args:
        ds ([xarray.Dataset]): dataset with song_events
        event_times ([dict]): event start/stop times.

    Returns:
        xarray.Dataset
    """
    event_names = ds.song_events.event_types.data
    for event_idx, event_name in enumerate(event_names):
        logger.info(f"   {event_name}")
        ds.song_events.sel(event_types=event_name).data[:] = 0  # delete all events
        if event_name not in event_times:
            continue
        for onset, offset in _as_start_stop_rows(event_times[event_name]):
            if not np.isfinite(onset) or not np.isfinite(offset):
                continue
            start = min(onset, offset)
            stop = max(onset, offset)
            if start == stop:
                time = ds.song_events.time.sel(time=start, method="nearest").data
                idx = np.where(ds.time == time)[0]
                ds.song_events[idx, event_idx] = 1
            else:
                ds.song_events.sel(time=slice(start, stop), event_types=event_name).data[:] = 1
    if "event_categories" in ds.song_events.coords:
        ds = ds.assign_coords({"event_categories": (("event_types"), ["event"] * len(event_names))})
    return ds
