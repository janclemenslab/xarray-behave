"""Utils for handling song data."""
import numpy as np


def waveforms(eventtimes, trace, waveform_duration: float = 100) -> np.ndarray:
    """Cuts out waveforms of specified duration from trace around eventtimes."""
    pass


def events_to_mask(event_trace, max_interval) -> np.ndarray:
    """Make continuous mask from binary trace of events.
    Events closer than max_interval will be connected in the mask.

    Args:
        event_trace ([type]): [description]
        max_interval ([type]): in samples
        mutually_exclusive (bool, optional): [description]. Defaults to True.

    Returns:
        np.ndarray: boolean array
    """
    event_times = np.where(event_trace)[0]
    event_mask = np.full_like(event_trace, False, dtype=np.bool)

    for pre, post in zip(event_times[:-1], event_times[1:]):
        event_interval = post - pre
        if event_interval <= max_interval:
            event_mask[pre:post+1] = 1
    return event_mask


def combine_masks(masks, max_interval=None):
    """Events take priority based on their order in `masks`. For instance, if masks[:,0] and masks[:,1] are both true, the combined mask will be set to event 0

    CAUTION: Only tested for masks with two event types (masks.shape[1]==2). Weird things could happen for inputs with more event types.
    Args:
        masks ([type]): [samples, event_types] boolean matrix
        max_interval ([type], optional): Fills gaps <max_interval between events of different types. Defaults to None.

    Returns:
        [type]: [samples,], "0" if no event_type true, otherwise 1+index of event_type in masks (e.g. output will have value 1 for samples with mask[:,0] True.)
    """
    masks = np.concatenate((np.zeros_like(masks[:, :1]), masks), axis=-1)  # prepend zeros for non-events    
    combined_mask = np.argmax(masks, axis=-1)
    if max_interval is not None:
        event_onsets = np.where(np.diff(combined_mask) > 0)[0]+1
        event_offsets = np.where(np.diff(combined_mask) < 0)[0]

        for off in event_offsets:
            event_type = combined_mask[off]
            nearest_onset = event_onsets - off
            nearest_onset = nearest_onset[np.logical_and(nearest_onset > 0, nearest_onset <= max_interval)] + off
            if len(nearest_onset):
                combined_mask[off+1:nearest_onset[0]] = event_type
    return combined_mask
