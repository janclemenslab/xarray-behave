import logging
import numpy as np


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


# FACTOR OUT TO NEW MODULE FOR INDEPENDENCE FROM GUI
def eventtimes_to_traces(ds, event_times):
    """Convert dict of event (on/offset) times into song_events.
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

def eventtimes_delete(eventtimes, which):
    return eventtimes

def eventtimes_add(eventtimes, which, resort=False):
    return eventtimes

def eventtimes_replace(eventtimes, which_old, which_new, resort=False):
    eventtimes = eventtimes_delete(eventtimes, which_old)
    eventtimes = eventtimes_add(eventtimes, which_new)
    return eventtimes