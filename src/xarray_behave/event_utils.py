import logging
import numpy as np
import xarray as xr


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
                print('Inconsistent segment onsets or offsets - ignoring all on- and offsets.')
                onsets = []
                offsets = []

            if not len(onsets) and not len(offsets):
                event_times[event_name] = np.zeros((0,2))
            else:
                event_times[event_name] = np.stack((onsets, offsets)).T
    return event_times


def infer_event_categories(event_times):
    """Based on shape."""
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
    event_categories = infer_event_categories(event_times)
    # populate with data:
    logging.info(f'Updating:')
    for cnt, (event_name, event_data) in enumerate(event_times.items()):
        logging.info(f'   {event_name} ({event_data.shape[0]} instances')
        if event_categories[event_name] == 'event':
            new_values[(event_data * fs).astype(np.uintp), cnt] = 1
        elif event_categories[event_name] == 'segment':
            for onset, offset in event_data:
                onset_idx = (onset * fs).astype(np.uintp)
                offset_idx = (offset * fs).astype(np.uintp)
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
            times = ds.song_events.time.sel(time=event_times[event_name].ravel(), method='nearest').data
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
            event_times[event_name] = np.where(traces[:, event_idx] == 1)[0]
        elif event_category == 'segment':
            breakpoint()
            onsets = np.where(np.diff(traces[event_idx]) == 1)[0]
            offsets = np.where(np.diff(traces[event_idx]) == -1)[0]
            if len(onsets) and len(offsets):
                # ensure onsets and offsets match
                offsets = offsets[offsets>np.min(onsets)]
                onsets = onsets[onsets<np.max(offsets)]
            if len(onsets) != len(offsets):
                print('Inconsistent segment onsets or offsets - ignoring all on- and offsets.')
                onsets = []
                offsets = []

            if not len(onsets) and not len(offsets):
                event_times[event_name] = np.zeros((0,2))
            else:
                event_times[event_name] = np.stack((onsets, offsets)).T
    return event_times

def eventtimes_delete(eventtimes, which):
    return eventtimes

def eventtimes_add(eventtimes, which, resort=False):
    return eventtimes

def eventtimes_replace(eventtimes, which_old, which_new, resort=False):
    eventtimes = eventtimes_delete(eventtimes, which_old)
    eventtimes = eventtimes_add(eventtimes, which_new)
    return eventtimes