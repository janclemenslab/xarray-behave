"""Load files created by various analysis programs created."""
import numpy as np
import pandas as pd
import h5py
import flammkuchen as dd_io
from typing import Sequence, Union
import logging

import scipy.interpolate
import scipy.ndimage
from scipy.io import loadmat
import scipy.signal
import scipy.stats
from scipy.ndimage import maximum_filter1d

from samplestamps import SampStamp
import xarray as xr
import zarr
import math
from . import xarray_behave as xb
from . import (event_utils,
               annot)



def rotate_point(pos, degrees, origin=(0, 0)):
    """Rotate point.

    Args:
        pos (tuple): (x.y) position
        degrees ([type]): degree by which to rotate
        origin (tuple, optional): point (x,y) around which to rotate point. Defaults to (0, 0).

    Returns:
        tuple: (x, y) rotated
    """
    x, y = pos
    radians = degrees / 180 * np.pi
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return qx, qy


def rotate_pose(positions, degree, origin=(0, 0)):
    """Rotate point set.

    Args:
        pos (np.ndarray): [points, x/y]
        degrees (float): degree by which to rotate
        origin (Tuple[float, float], optional): point (x,y) around which to rotate point. Defaults to (0, 0).

    Returns:
        tuple: (x, y) rotated
    """
    radians = degree / 180 * np.pi
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    rot = np.array([[cos_rad, sin_rad], [-sin_rad, cos_rad]])
    positions_centered = positions - origin
    return rot.dot(positions_centered.T).T + origin


def find_nearest(array, values):
    """Find nearest occurrence of each item of values in array.

    Args:
        array: find nearest in this list
        values: queries

    Returns:
        val: nearest val in array to each item in values
        idx: index of nearest val in array to each item in values
        dist: distance to nearest val in array for each item in values
        NOTE: Returns nan-arrays of the same size as values if `array` is empty.
    """
    if len(values) and len(array):  # only do this if boh inputs are non-empty lists
        values = np.atleast_1d(values)
        abs_dist = np.abs(np.subtract.outer(array, values))  # this eats a lot of memory - maybe use nearest neighbour interpolation for upsampling?
        idx = abs_dist.argmin(0)
        dist = abs_dist.min(0)
        val = array[idx]
    else:
        idx = np.full_like(values, fill_value=np.nan)
        dist = np.full_like(values, fill_value=np.nan)
        val = np.full_like(values, fill_value=np.nan)
    return val, idx, dist


def interpolate_binary_old(x0, y0, x1):
    """Interpolate a binary trace, preserving all True/1 events.
    More accurate than `interpolate_binary` for irregularly spaced samples but uses too much memory.

    Args:
        x0 ([type]): sample times for each point in y0
        y0 ([type]): binary sequence (True, False) or 0, 1
        x1 ([type]): new sample times

    Returns:
        y1: y0 values at x1
    """
    if y0.ndim == 1:
        y0 = y0[..., np.newaxis]

    y1 = np.zeros((len(x1), y0.shape[1]))
    for idx in range(y0.shape[1]):
        tt = x0[np.where(y0[:, idx])[0]]
        _, ii, _ = find_nearest(x1, tt)
        y1[ii.astype(np.uintp), idx] = 1

    return y1


def interpolate_binary(x0, y0, x1):
    """Interpolate a binary trace, preserving all True/1 events.

    Args:
        x0 ([type]): sample times for each point in y0
        y0 ([type]): binary sequence (True, False) or 0, 1
        x1 ([type]): new sample times

    Returns:
        y1: y0 values at x1
    """
    fs0 = 1/np.mean(np.diff(x0))
    fs1 = 1/np.mean(np.diff(x1))
    ratio = fs0/fs1

    if ratio > 1:  # we are downsampling
        # if we downsample - spread out events so we catch all of them
        y0 = scipy.ndimage.maximum_filter(y0, size=(ratio, 1))
        # use 'nearest' here - why? could we just use linear, too?
        interpolator = scipy.interpolate.interp1d(x0, y0, axis=0, kind='nearest', bounds_error=False, fill_value=np.nan)
    else:  # we are upsampling
        # using linear here prevents spreading out of events when upsampling
        interpolator = scipy.interpolate.interp1d(x0, y0, axis=0, kind='linear', bounds_error=False, fill_value=np.nan)
    y1 = interpolator(x1).astype(np.uintp)
    return y1


def merge_channels(data, sampling_rate, filter_data: bool = True):
    """Merge channels based on a running maximum.

    Args:
        data (ndarray): [samples, channels]
        sampling_rate (num): in Hz

    Returns:
        ndarray: merged across
    """
    data = np.array(data)  # ensure data is an np.array (and not dask) - otherwise np.interp will fail
    mask = ~np.isfinite(data)  # remove all nan/inf data
    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
    # band-pass filter out noise on each channel
    b, a = scipy.signal.butter(6, (25, 1500), btype='bandpass', fs=sampling_rate)
    data = scipy.signal.filtfilt(b, a, data, axis=0, method='pad')
    # find loudest channel in 101-sample windows
    if filter_data:
        sng_max = maximum_filter1d(np.abs(data), size=101, axis=0)
        loudest_channel = np.argmax(sng_max, axis=-1)
    # get linear index and merge channels
    idx = np.ravel_multi_index((np.arange(sng_max.shape[0]), loudest_channel), data.shape)
    data_merged_max = data.ravel()[idx]
    data_merged_max = data_merged_max[:, np.newaxis]  # shape needs to be [nb_samples, 1]
    return data_merged_max


def fill_gaps(sine_pred, gap_dur=100):
    onsets = np.where(np.diff(sine_pred.astype(np.int))==1)[0]
    offsets = np.where(np.diff(sine_pred.astype(np.int))==-1)[0]
    if len(onsets) and len(offsets):
        onsets = onsets[onsets<offsets[-1]]
        offsets = offsets[offsets>onsets[0]]
        durations = offsets - onsets
        for idx, (onset, offset, duration) in enumerate(zip(onsets, offsets, durations)):
            if idx>0 and offsets[idx-1]>onsets[idx]-gap_dur:
                sine_pred[offsets[idx-1]:onsets[idx]+1] = 1
    return sine_pred


def remove_short(sine_pred, min_len=100):
    # remove too short sine songs
    onsets = np.where(np.diff(sine_pred.astype(np.int))==1)[0]
    offsets = np.where(np.diff(sine_pred.astype(np.int))==-1)[0]
    if len(onsets) and len(offsets):
        onsets = onsets[onsets<offsets[-1]]
        offsets = offsets[offsets>onsets[0]]
        durations = offsets - onsets
        for cnt, (onset, offset, duration) in enumerate(zip(onsets, offsets, durations)):
            if duration<min_len:
                sine_pred[onset:offset+1] = 0
    return sine_pred


def load_swap_indices(filepath):
    a = np.loadtxt(filepath, dtype=np.uintp)
    indices, flies1, flies2 = np.split(a, [1, 2], axis=-1)  # split columns in file into variables
    return indices, flies1, flies2


def swap_flies(dataset, indices, flies1=0, flies2=1):
    """Swap flies in dataset.

    Caution: datavariables are currently hard-coded!
    Caution: Swap *may* be in place so *might* will alter original dataset.

    Args:
        dataset ([type]): Dataset for which to swap flies
        indices ([type]): List of indices at which to swap flies.
        flies1 (int or list/tuple, optional): Either a single value for all indices or a list with one value per item in indices. Defaults to 0.
        flies2 (int or list/tuple, optional): Either a single value for all indices or a list with one value per item in indices. Defaults to 1.

    Returns:
        dataset with swapped indices ()
    """
    for cnt, index in enumerate(indices):
        if isinstance(flies1, (list, tuple)) and isinstance(flies2, (list, tuple)):
            fly1, fly2 = flies1[cnt], flies2[cnt]
        else:
            fly1, fly2 = flies1, flies2

        if 'pose_positions_allo' in dataset:
            dataset.pose_positions_allo.values[index:, [fly2, fly1], ...] = dataset.pose_positions_allo.values[index:, [fly1, fly2], ...]
        if 'pose_positions' in dataset:
            dataset.pose_positions.values[index:, [fly2, fly1], ...] = dataset.pose_positions.values[index:, [fly1, fly2], ...]
        if 'body_positions' in dataset:
            dataset.body_positions.values[index:, [fly2, fly1], ...] = dataset.body_positions.values[index:, [fly1, fly2], ...]

    return dataset


def load_segmentation_matlab(filepath):
    """Load output produced by FlySongSegmenter."""
    res = dict()
    try:
        d = loadmat(filepath)
        res['pulse_times_samples'] = d['pInf'][0, 0]['wc']
        res['pulse_labels'] = d['pInf'][0, 0]['pulseLabels']
        res['song_mask'] = d['bInf'][0, 0]['Mask'].T  # 0 - silence, 1 - pulse, 2 - sine
    except NotImplementedError:
        with h5py.File(filepath, 'r') as f:
            res['pulse_times_samples'] = f['pInf/wc'][:].T
            res['pulse_labels'] = f['pInf/pulseLabels'][:].T
            res['song_mask'] = f['bInf/Mask'][:]

    sine_song = res['song_mask'] == 2
    fs = 10_000  # Hz

    res['event_names'] = ['song_pulse_any_fss', 'song_pulse_slow_fss', 'song_pulse_fast_fss', 'sine_fss']
    res['event_categories'] = ['event', 'event', 'event', 'segment']
    res['event_indices'] = [res['pulse_times_samples'],
                            res['pulse_times_samples'][res['pulse_labels'] == 1],
                            res['pulse_times_samples'][res['pulse_labels'] == 0],
                            np.where(sine_song == 2)[0]]
    # extract event_seconds
    event_seconds = {'song_pulse_any_fss': res['pulse_times_samples'] / fs,
                     'song_pulse_slow_fss': res['pulse_times_samples'][res['pulse_labels'] == 1] / fs,
                     'song_pulse_fast_fss': res['pulse_times_samples'][res['pulse_labels'] == 0] / fs,
                     'sine_fss': np.where(sine_song == 2)[0] / fs}
    # event_categories
    event_categories = {}
    for cat, typ in zip(event_seconds.keys(), ['event', 'event', 'event', 'segment']):
        event_categories[cat] = typ
    return event_seconds, event_categories


def load_segmentation(filepath):
    """Load output produced by DeepSongSegmenter.

    File should have at least 'event_names' and 'event_indices' datasets."""
    res = dd_io.load(filepath)
    # extract event_seconds
    song_seconds = {}
    song_categories = {}

    for event_indices, event_name in zip(res['event_indices'], res['event_names']):
        song_seconds[event_name] = event_indices / res['samplerate_Hz']
        song_categories[event_name] = 'event'

    # post process segments ?
    if 'sine' in res['segment_names']:
        logging.info('   Found sine song - attempting to post process ')
        sine_index = res['segment_names'].index('sine')
        res['segment_labels'][sine_index] = fill_gaps(res['segment_labels'][sine_index],
                                                        gap_dur = 20 / 1000 * res['samplerate_Hz'])
        res['segment_labels'][sine_index] = remove_short(res['segment_labels'][sine_index],
                                                            min_len = 20 / 1000 * res['samplerate_Hz'])

    # extract segment on- and offsets
    segment_indices = event_utils.traces_to_eventtimes(res['segment_labels'], res['segment_names'], ['segment'])
    for segment_name, segment_indices in segment_indices.items():
        song_seconds[segment_name] = segment_indices / res['samplerate_Hz']
        song_categories[segment_name] = 'segment'

    return song_seconds, song_categories


def load_manual_annotation_csv(filepath):
    """Load output produced by xb."""
    df = pd.read_csv(filepath)

    if not all([item in df.columns for item in ['name','start_seconds', 'stop_seconds']]):
        logging.error(f"Malformed CSV file {filepath} - needs to have these columns: ['name','start_seconds', 'stop_seconds']. Returning empty results")
        event_seconds = dict()
        event_categories = dict()
    else:
        event_seconds = annot.Events.from_df(df)
        event_categories = event_seconds.categories
    return event_seconds, event_categories


def load_manual_annotation_zarr(filepath):
    """Load output produced by xb (legacy format)."""
    manual_events_ds = xb.load(filepath)

    if 'event_categories' not in manual_events_ds:
        event_categories_List = event_utils.infer_event_categories_from_traces(manual_events_ds.song_events.data)

        # force these to be the correct types even if they are empty and
        # the event_cat inference does not work
        for cnt, event_type in enumerate(manual_events_ds.event_types.data):
            if event_type in ['pulse_manual', 'vibration_manual', 'aggression_manual']:
                event_categories_List[cnt] = 'event'
            if event_type == 'sine_manual':
                event_categories_List[cnt] = 'segment'

        manual_events_ds = manual_events_ds.assign_coords(
            {'event_categories': (('event_types'), event_categories_List)})

    event_seconds = event_utils.detect_events(manual_events_ds)

    event_categories = {}
    for typ, cat in zip(manual_events_ds.event_types.data, manual_events_ds.event_categories.data):
        event_categories[typ] = cat

    return event_seconds, event_categories


def load_manual_annotation_matlab(filepath):
    """Load output produced by the matlab ManualSegmenter."""
    try:
        mat_data = loadmat(filepath)
    except NotImplementedError:
        with h5py.File(filepath, 'r') as f:
            mat_data = dict()
            for key, val in f.items():
                mat_data[key.lower()] = val[:].T

    events_seconds = dict()
    event_categories = dict()
    for key, val in mat_data.items():
        if len(val) and hasattr(val, 'ndim') and val.ndim == 2 and not key.startswith('_'):  # ignore matfile metadata
            events_seconds[key.lower() + '_manual'] = np.sort(val[:, 1:])
            if val.shape[1] == 2:  # pulse times
                event_categories[key.lower() + '_manual'] = 'event'
            else:  # sine on and offset
                event_categories[key.lower() + '_manual'] = 'segment'
    return events_seconds, event_categories


def load_tracks(filepath):
    """Load tracker data"""
    with h5py.File(filepath, 'r') as f:
        if 'data' in f.keys():  # in old-style or unfixes tracks, everything is in the 'data' group
            data = dd_io.load(filepath)
            chbb = data['chambers_bounding_box'][:]
            heads = data['lines'][:, 0, :, 0, :]   # nframe, fly id, coordinates
            tails = data['lines'][:, 0, :, 1, :]   # nframe, fly id, coordinates
            box_centers = data['centers'][:, 0, :, :]   # nframe, fly id, coordinates
            background = data['background'][:]
            first_tracked_frame = data['start_frame']
            last_tracked_frame = data['frame_count']
        else:
            chbb = f['chambers_bounding_box'][:]
            heads = f['lines'][:, 0, :, 0, :]   # nframe, fly id, coordinates
            tails = f['lines'][:, 0, :, 1, :]   # nframe, fly id, coordinates
            box_centers = f['centers'][:, 0, :, :]   # nframe, fly id, coordinates
            background = f['background'][:]
            first_tracked_frame = f.attrs['start_frame']
            last_tracked_frame = f.attrs['frame_count']
    # everything to frame coords
    heads = heads[..., ::-1]
    tails = tails[..., ::-1]
    heads = heads + chbb[1][0][:]
    tails = tails + chbb[1][0][:]
    box_centers = box_centers + chbb[1][0][:]
    body_parts = ['head', 'center', 'tail']
    # first_tracked_frame, last_tracked_frame = data['start_frame'], data['frame_count']
    x = np.stack((heads, box_centers, tails), axis=2)
    x = x[first_tracked_frame:last_tracked_frame, ...]
    return x, body_parts, first_tracked_frame, last_tracked_frame, background


def load_poses_leap(filepath):
    """Load pose tracks estimated using LEAP.

    Args:
        filepath ([type]): [description]

    Returns:
        poses, poses_allo, body_parts, first_pose_frame, last_pose_frame
    """

    with h5py.File(filepath, 'r') as f:
        box_offset = f['box_centers'][:]
        box_angle = f['fixed_angles'][:].astype(np.float64)
        box_size = (f['box_size'].attrs['i0'], f['box_size'].attrs['i1'])
        poses = f['positions'][:]
        nb_flies = int(np.max(f['fly_id']) + 1)
    body_parts = ['head', 'neck', 'front_left_leg', 'middle_left_leg', 'back_left_leg', 'front_right_leg',
                  'middle_right_leg', 'back_right_leg', 'thorax', 'left_wing', 'right_wing', 'tail']
    # need to offset and rotate
    last_pose_index = np.argmin(poses[:, 0, 0] > 0)
    first_pose_index = np.argmin(poses[:, 0, 0] == 0)
    box_offset = box_offset[first_pose_index:last_pose_index, ...]
    box_angle = box_angle[first_pose_index:last_pose_index, ...]
    poses = poses[first_pose_index:last_pose_index, ...]
    poses = poses.astype(np.float64)

    # transform poses back to frame coordinates
    origin = [b / 2 for b in box_size]
    poses_allo = np.zeros_like(poses)
    for cnt, ((x, y), ang, p_ego) in enumerate(zip(box_offset, box_angle, poses)):
        # tmp = [rotate_point(pt, a, origin) for pt in p_ego]  # rotate
        tmp = rotate_pose(p_ego, float(ang), origin)
        p_allo = np.stack(tmp) + np.array((x, y)) - origin  # translate
        poses_allo[cnt, ...] = p_allo

    # center poses around thorax
    thorax_idx = 8
    thorax_pos = poses[:, thorax_idx, :]
    poses = poses - thorax_pos[:, np.newaxis, :]  # subtract remaining thorax position
    origin = (0, 0)

    # rotate poses such that the head is at 0 degrees
    head_idx = 0
    head_pos = poses[:,  head_idx, :]
    head_thorax_angle = 90 + np.arctan2(head_pos[:, 0], head_pos[:, 1]) * 180 / np.pi
    for cnt, (a, p_ego) in enumerate(zip(head_thorax_angle, poses)):
        # poses[cnt, ...] = [rotate_point(pt, -a, origin) for pt in p_ego]
        poses[cnt, ...] = rotate_pose(p_ego, -a, origin)

    box_offset = box_offset.reshape((-1, nb_flies, *box_offset.shape[1:]))  # "unfold" fly dimension
    box_angle = box_angle.reshape((-1, nb_flies, *box_angle.shape[1:]))  # "unfold" fly dimension
    poses = poses.reshape((-1, nb_flies, *poses.shape[1:]))  # "unfold" fly dimension
    poses_allo = poses_allo.reshape((-1, nb_flies, *poses_allo.shape[1:]))  # "unfold" fly dimension

    first_pose_frame = int(first_pose_index / nb_flies)
    last_pose_frame = int(last_pose_index / nb_flies)

    return poses, poses_allo, body_parts, first_pose_frame, last_pose_frame


def load_poses_deepposekit(filepath):
    """Load pose tracks estimated using DeepPoseKit.

    Args:
        filepath (str): name of the file produced by DeepPoseKit

    Returns:
        poses_ego (xarray.DataArray): poses in EGOcentric (centered around thorax, head aligned straight upwards), [frames, flies, bodypart, x/y]
        poses_allo (xarray.DataArray): poses in ALLOcentric (frame) coordinates, [frames, flies, bodypart, x/y]
        partnames (List[str]): list of names for each body part (is already part of the poses_ego/allo xrs)
        first_pose_frame, last_pose_frame (int): frame corresponding to first and last item in the poses_ego/allo arrays (could attach this info as xr dim)
    """
    with zarr.ZipStore(filepath, mode='r') as zarr_store:
        ds = xr.open_zarr(zarr_store).load()  # the final `load()` prevents
    nb_flies = len(ds.flies)
    box_size = np.array(ds.attrs['box_size'])

    poses_allo = ds.poses + ds.box_centers - box_size/2

    first_pose_frame = int(np.argmin(np.isnan(ds.poses.data[:, 0, 0, 0]).data))
    last_pose_frame = int(np.argmin(~np.isnan(np.array(ds.poses.data[first_pose_frame:, 0, 0, 0]).data)) + first_pose_frame)
    if last_pose_frame == 0:
        last_pose_frame = ds.poses.shape[0]

    # CUT to first/last frame with poses
    poses_ego = ds.poses[first_pose_frame:last_pose_frame, ...]
    poses_allo = poses_allo[first_pose_frame:last_pose_frame, ...]

    poses_ego = poses_ego - poses_ego.sel(poseparts='thorax')  # CENTER egocentric poses around thorax

    # ROTATE egocentric poses such that the angle between head and thorax is 0 degrees (straight upwards)
    head_thorax_angle = 270 + np.arctan2(poses_ego.sel(poseparts='head', coords='y'),
                                         poses_ego.sel(poseparts='head', coords='x')) * 180 / np.pi
    for cnt, (a, p_ego) in enumerate(zip(head_thorax_angle.data, poses_ego.data)):
        for fly in range(nb_flies):
            # poses_ego.data[cnt, fly, ...] = [rotate_point(pt, -a[fly]) for pt in p_ego[fly]]
            poses_ego.data[cnt, fly, ...] = rotate_pose(p_ego[fly], -a[fly])

    return poses_ego, poses_allo, ds.poseparts, first_pose_frame, last_pose_frame


def load_poses_sleap(filepath):
    with h5py.File(filepath, 'r') as f:
        pose_parts = f['node_names']
        track_names = f['track_names']
        track_occupancy = f['track_occupancy'][:]
        tracks = f['tracks'][:]

    poses_allo = tracks.transpose([3, 0, 2, 1])
    return poses_ego, poses_allo, ds.poseparts, first_pose_frame, last_pose_frame


def load_raw_song(filepath_daq: str, song_channels: Union[Sequence[int], None] = None,
                  return_nonsong_channels: bool = False, lazy: bool = False):
    """[summary]

    Args:
        filepath_daq ([type]): [description]
        song_channels (List[int], optional): Sequence of integers as indices into 'samples' datasaet.
                                             Defaults to [0,..., 15].
        return_nonsong_channels (bool, optional): will return the data not in song_channels as separate array. Defaults to False
        lazy (bool, optional): If True, will load song as dask.array, which allows lazy indexing.
                               Otherwise, will load the full recording from disk (slow). Defaults to False.

    Returns:
        [type]: [description]
    """
    if song_channels is None:  # the first 16 channels in the data are the mic recordings
        song_channels = np.arange(16)

    if lazy:
        f = h5py.File(filepath_daq, 'r')
        # convert to dask array since this allows lazily evaluated indexing...
        import dask.array as daskarray
        da = daskarray.from_array(f['samples'], chunks=(10000, 1))
        nb_channels = f['samples'].shape[1]
        song_channels = song_channels[song_channels < nb_channels]
        song = da[:, song_channels]
        if return_nonsong_channels:
            non_song_channels = list(set(list(range(nb_channels))) - set(song_channels))
            non_song = da[:, non_song_channels]
    else:
        with h5py.File(filepath_daq, 'r') as f:
            nb_channels = f['samples'].shape[1]
            song = f['samples'][:, song_channels]
            if return_nonsong_channels:
                non_song_channels = list(set(list(range(nb_channels))) - set(song_channels))
                non_song = f['samples'][:, non_song_channels]

    if return_nonsong_channels:
        return song, non_song
    else:
        return song


def load_times(filepath_timestamps, filepath_daq):
    """Load daq and cam time stamps, create muxer"""
    with h5py.File(filepath_timestamps, 'r') as f:
        cam_stamps = f['timeStamps'][:]

    # time stamps at idx 0 can be a little wonky - so use the information embedded in the image
    if cam_stamps.shape[1] == 2:  # time stamps from Spinnaker cam
        shutter_times = cam_stamps[:, 1]
    else:  # time stamps from point grey cam
        shutter_times = cam_stamps[:, 1] + cam_stamps[:, 2]/1_000_000  # time of "Shutter OFF"

    # DAQ time stamps
    with h5py.File(filepath_daq, 'r') as f:
        daq_stamps = f['systemtime'][:]
        daq_sampleinterval = f['samplenumber'][:]

    # remove trailing zeros - may be left over if recording didn't finish properly
    last_valid_idx = np.argmax(daq_stamps == 0)
    daq_samplenumber = np.cumsum(daq_sampleinterval)[:last_valid_idx, np.newaxis]
    last_sample = daq_samplenumber[-1, 0]

    last_frame_idx = np.argmax(shutter_times==0) - 1
    shutter_times = shutter_times[:last_frame_idx]
    last_frame = shutter_times[-1]

    nb_seconds_per_interval, _ = scipy.stats.mode(np.diff(daq_stamps[:last_valid_idx, 0]))  # seconds - using mode here to be more robust
    nb_seconds_per_interval = nb_seconds_per_interval[0]
    nb_samples_per_interval = np.mean(np.diff(daq_samplenumber[:last_valid_idx, 0]))
    sampling_rate_Hz = np.around(nb_samples_per_interval / nb_seconds_per_interval, -3)  # round to 1000s of Hz

    ss = SampStamp(sample_times=daq_stamps[:last_valid_idx, 0], frame_times=shutter_times, sample_numbers=daq_samplenumber[:, 0], auto_monotonize=False)
    # different refs:
    # s0 = ss.sample_time(0)  # first sample is 0 seconds
    # ss = SampStamp(sample_times=daq_stamps[:, 0] - s0, frame_times=cam_stamps[:, 0] - s0, sample_numbers=daq_samplenumber[:, 0])
    # f0 = ss.frame_time(0)  # first frame is 0 seconds - for no-resample-video-data
    # ss = SampStamp(sample_times=daq_stamps[:, 0] - f0, frame_times=cam_stamps[:, 0] - f0, sample_numbers=daq_samplenumber[:, 0])

    return ss, last_sample, sampling_rate_Hz


def initialize_manual_song_events(ds: xr.Dataset, from_segmentation: bool = False, force_overwrite: bool = False,
                                  new_manual_event_types=['sine_manual', 'pulse_manual', 'vibration_manual', 'aggression_manual'],
                                  new_manual_event_categories=['segment', 'event', 'event', 'event']) -> xr.Dataset:
    """[summary]

    Args:
        ds (xarray.Dataset): [description]
        from_segmentation (bool, optional): Init manual events from automatic events with same name.
                                            Otherwise they initialized as empty.
                                            If force_overwrite: will *ADD* existing manual events.
                                            otherwise: will *ADD* auto events to existing manual events.
                                            Defaults to False.
        force_overwrite (bool, optional): Overwrite existing manual events.
                                          Defaults to False.
        new_manual_event_types = ['sine_manual', 'pulse_manual', 'vibration_manual', 'aggression_manual']
        new_manual_event_categories = ['segment', 'event', 'event', 'event']
    Returns:
        xarray.Dataset: [description]
    """

    # only add new ones
    if 'song_events' in ds:
        new_manual_event_categories = [cat for evt, cat in zip(new_manual_event_types, new_manual_event_categories)
                                       if evt not in ds.song_events.event_types]
        new_manual_event_types = [evt for evt in new_manual_event_types if evt not in ds.song_events.event_types]

    song_events_manual = None
    if 'song_events' not in ds or new_manual_event_types:
        new_manual_events = np.zeros((ds.time.shape[0],
                                      len(new_manual_event_types)),
                                     dtype=np.bool)
        song_events_manual = xr.DataArray(data=new_manual_events,
                                          dims=['time', 'event_types'],
                                          coords={'time': ds.time,
                                                  'event_types': new_manual_event_types,
                                                  'event_categories': (('event_types'), new_manual_event_categories),
                                                  'nearest_frame': (('time'), ds.nearest_frame), },
                                          attrs={'description': 'Event times as boolean arrays.',
                                                 'sampling_rate_Hz': ds.attrs['target_sampling_rate_Hz'],
                                                 'time_units': 'seconds', })

    # if song_events_manual is not None 'song_events' in ds and not force_overwrite:
    if song_events_manual is not None:
        if not force_overwrite and 'song_events' in ds:
            combined = xr.concat([ds.song_events, song_events_manual], dim='event_types')
        else:
            combined = song_events_manual

        if 'song_events' in ds:
            ds.drop('song_events')

        new_ds = combined.to_dataset(name='song_events')
        attrs = ds.attrs  # for some reason, attrs are not preserved during merge...
        ds = xr.merge((ds, new_ds))
        ds.attrs = attrs

    if from_segmentation:
        for evt in new_manual_event_types:
            auto_key = evt.strip('_manual')
            if auto_key in ds.song_events.event_types:
                if force_overwrite:  # overwrite manual events with the corresponding auto event
                    ds.song_events.loc[:, evt] = ds.song_events.loc[:, auto_key]
                else:  # add auto events to the corresponding manual event
                    ds.song_events.loc[:, evt] = np.logical_or((ds.song_events.loc[:, evt], ds.song_events.loc[:, auto_key]))
    return ds


def fix_keys(d):
    d_new = dict()
    # HACK zarr (or xarray) cuts off long string keys in event-types
    fix_dict = {'aggression_manu': 'aggression_manual', 'vibration_manua': 'vibration_manual'}
    # make this a function!!
    for eventtype in d.keys():
        # convert all from b'..' to str
        try:
            d_new[eventtype.decode()] = d[eventtype]
        except AttributeError:
            d_new[eventtype] = d[eventtype]

    for eventtype in d_new:
        if eventtype in fix_dict.keys():
            # logging.info(f'   Renaming {eventtype} to {fix_dict[eventtype]}.')
            d_new[fix_dict[eventtype]] = d_new.pop(eventtype)
    return d_new
