import logging
import numpy as np
import xarray as xr
import pandas as pd

def detect_events(ds):
    """Transform ds.song_events into dict of event (on/offset) times.
    Args:
        ds ([xarray.Dataset]): dataset with song_events

    Returns:
        dict: with event times or segment on/offsets.
    """
    event_times = dict()
    ds.song_events.data = ds.song_events.data.astype(np.float)  # make sure this is non-bool so diff works
    event_names = ds.song_events.event_types.data
    event_categories = ds.song_events.event_categories.data
    logging.info('Extracting event times from song_events:')
    for event_idx, (event_name, event_category) in enumerate(zip(event_names, event_categories)):
        logging.info(f'   {event_name}')
        if event_category == 'event':
            event_times[event_name] = ds.song_events.time.where(ds.song_events[:, event_idx] == 1, drop=True).data.ravel()
        elif event_category == 'segment':
            onsets = ds.song_events.time.where(ds.song_events[:, event_idx].diff(dim='time') == 1, drop=True).data
            offsets = ds.song_events.time.where(ds.song_events[:, event_idx].diff(dim='time') == -1, drop=True).data
            if len(onsets) and len(offsets):
                # ensure onsets and offsets match
                offsets = offsets[offsets>np.min(onsets)]
                onsets = onsets[onsets<np.max(offsets)]
            if len(onsets) != len(offsets):
                logging.warning('Inconsistent segment onsets or offsets - ignoring all on- and offsets.')
                onsets = []
                offsets = []

            if not len(onsets) and not len(offsets):
                event_times[event_name] = np.zeros((0,2))
            else:
                event_times[event_name] = np.stack((onsets, offsets)).T
    return event_times


def infer_class_info_from_df(df: pd.DataFrame):
    """Based on difference between start_seconds/stop_seconds

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    class_names, first_indices = np.unique(df['name'], return_index=True)
    class_names = list(class_names)
    class_names.insert(0, 'noise')

    # infer class type - event if start and end are the same
    class_types = ['segment']
    for first_index in first_indices:
        if df.loc[first_index]['start_seconds']==df.loc[first_index]['stop_seconds']:
            class_types.append('event')
        else:
            class_types.append('segment')
    return class_names, class_types


def infer_event_categories_from_traces(data):
    """Infer event type (event/segment) based on median interval between
    event times.

    Args:
        data ([type]): binary matrix [samples x events]
    """
    event_categories = []
    for evt in range(data.shape[1]):
        if np.any(data[:, evt]):
            dt = np.nanmedian(np.diff(np.where(data[:, evt])[0]))
        else:
            dt = 10  # fall back to events w/o annotations

        if dt > 1:
            event_categories.append('event')
        else:
            event_categories.append('segment')
    return event_categories


def infer_event_categories_from_shape(event_times):
    """Based on shape of time lists."""
    event_categories = {}
    for event_name, event_data in event_times.items():
        event_categories[event_name] = 'event' if event_data.ndim==1 else 'segment'
    return event_categories


def update_traces(ds, event_times):
    """Slightly redundant with eventtimes_to_traces but
    will add populate new events.
    """
    ## event_times to ds.song_events
    # make new song_events DataArray
    if 'song_events' in ds:
        old_values = ds.song_events.values.copy()
        attrs = ds.song_events.attrs.copy()
    else:
        old_values = np.zeros((ds.time.shape[0], 0))
        attrs = {'sampling_rate_Hz': ds.attrs['target_sampling_rate_Hz']}

    new_values = np.zeros_like(old_values, shape=(old_values.shape[0], len(event_times)))
    fs = ds.song_events.attrs['sampling_rate_Hz']
    event_categories = infer_event_categories_from_shape(event_times)
    # populate with data:
    logging.info(f'Updating:')
    for cnt, (event_name, event_data) in enumerate(event_times.items()):
        logging.info(f'   {event_name} ({event_data.shape[0]} instances')
        if event_categories[event_name] == 'event':
            event_indices = (event_data * fs).astype(np.uintp)
            event_indices = event_indices[event_indices>=0]
            event_indices = event_indices[event_indices<len(new_values)]
            new_values[event_indices, cnt] = 1
        elif event_categories[event_name] == 'segment':
            for onset, offset in event_data:
                onset_idx = (onset * fs).astype(np.uintp)
                offset_idx = (offset * fs).astype(np.uintp)
                within_bounds = onset_idx < len(new_values) and offset_idx < len(new_values)
                greater_zero = onset_idx >= 0 and offset_idx >= 0
                if within_bounds and greater_zero:
                    new_values[onset_idx:offset_idx, cnt] = 1
    logging.info(f'Done:')

    # rebuild dataset
    song_events = xr.DataArray(data=new_values,
                            dims=['time', 'event_types'],
                            coords={'time': ds.time,
                                    'event_types': list(event_times.keys()),
                                    'event_categories': (('event_types'), list(event_categories.values()))},
                            attrs=attrs)
    # delete old
    if 'song_events' in ds:
        del ds['song_events']
        del ds.coords['event_types']
        del ds.coords['event_categories']
    # add new
    ds = xr.merge((ds, song_events.to_dataset(name='song_events')))
    return ds


def eventtimes_to_traces(ds, event_times):
    """Update events in ds.song_events from dict.

    Does not add new events (events that exist in event_times but not in ds.song_events)!!

    Args:
        ds ([xarray.Dataset]): dataset with song_events
        event_times ([dict]): event times or segment on/offsets.

    Returns:
        xarray.Dataset
    """
    event_names = ds.song_events.event_types.data
    event_categories = ds.song_events.event_categories.data
    logging.info('Updating song_events from event_times:')
    for event_idx, (event_name, event_category) in enumerate(zip(event_names, event_categories)):
        logging.info(f'   {event_name}')
        ds.song_events.sel(event_types=event_name).data[:] = 0  # delete all events
        if event_category == 'event':
            # times = ds.song_events.time.sel(time=event_times[event_name][:, 0], method='nearest').data
            times = ds.song_events.time.sel(time=event_times[event_name].ravel(), method='nearest').data
            times = np.unique(times)

            # this is sloooooow
            for time in times:
                idx = np.where(ds.time==time)[0]
                ds.song_events[idx, event_idx] = 1
        elif event_category == 'segment':
            if event_times[event_name].shape[0] > 0:
                for onset, offset in zip(event_times[event_name][:, 0], event_times[event_name][:, 1]):
                    ds.song_events.sel(time=slice(onset, offset), event_types=event_name).data[:] = 1
    logging.info(f'Done.')
    return ds


def traces_to_eventtimes(traces, event_names, event_categories):
    """[summary]

    Args:
        traces ([type]): list of numpy arrays with the binary traces for each event/segment
        event_names ([type]): [description]
        event_categories ([type]): [description]

    Returns:
        [type]: [description]
    """
    assert len(traces) == len(event_names)
    assert len(traces) == len(event_categories)

    event_times = dict()

    logging.info('Extracting event times from song_events:')
    for event_idx, (event_name, event_category) in enumerate(zip(event_names, event_categories)):
        logging.info(f'   {event_name}')
        if event_category == 'event':
            event_times[event_name] = np.where(traces[event_idx] == 1)[0]
        elif event_category == 'segment':
            onsets = np.where(np.diff(traces[event_idx]) == 1)[0]
            offsets = np.where(np.diff(traces[event_idx]) == -1)[0]
            if len(onsets) and len(offsets):
                # ensure onsets and offsets match
                offsets = offsets[offsets>np.min(onsets)]
                onsets = onsets[onsets<np.max(offsets)]
            if len(onsets) != len(offsets):
                logging.warning('Inconsistent segment onsets or offsets - ignoring all on- and offsets.')
                onsets = []
                offsets = []

            if not len(onsets) and not len(offsets):
                event_times[event_name] = np.zeros((0,2))
            else:
                event_times[event_name] = np.stack((onsets, offsets)).T
    return event_times
