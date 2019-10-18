"""Load files created by various analysis programs created."""
import numpy as np
import h5py
import deepdish as dd

import scipy.interpolate
from scipy.io import loadmat
import scipy.signal
from scipy.ndimage import maximum_filter1d

from samplestamps import SampStamp
import xarray as xr
import zarr
import math
from . import xarray_behave as xb
from typing import Sequence


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


def merge_channels(data, sampling_rate):
    """Merge channels based on a running maximum.

    Args:
        data (ndarray): [samples, channels]
        sampling_rate (num): in Hz

    Returns:
        ndarray: merged across
    """

    # remove all nan/inf data
    mask = ~np.isfinite(data)
    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
    # band-pass filter out noise on each channel
    b, a = scipy.signal.butter(6, (25, 1500), btype='bandpass', fs=sampling_rate)
    data = scipy.signal.filtfilt(b, a, data, axis=0, method='pad')
    # find loudest channel in 101-sample windows
    sng_max = maximum_filter1d(np.abs(data), size=101, axis=0)
    loudest_channel = np.argmax(sng_max, axis=-1)
    # get linear index and merge channels
    idx = np.ravel_multi_index((np.arange(sng_max.shape[0]), loudest_channel), data.shape)
    data_merged_max = data.ravel()[idx]
    data_merged_max = data_merged_max[:, np.newaxis]  # shape needs to be [nb_samples, 1]
    return data_merged_max


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


def load_segmentation(filepath):
    """Load output produced by FlySongSegmenter."""
    res = dict()
    try:
        d = loadmat(filepath)
        res['pulse_times_samples'] = d['pInf'][0, 0]['wc']
        res['pulse_labels'] = d['pInf'][0, 0]['pulseLabels']
        res['song_mask'] = d['bInf'][0, 0]['Mask'].T  # 0 - silence, 1 - pulse, 2 - sine
        res['song'] = d['Song'].T
    except NotImplementedError:
        with h5py.File(filepath, 'r') as f:
            res['pulse_times_samples'] = f['pInf/wc'][:].T
            res['pulse_labels'] = f['pInf/pulseLabels'][:].T
            res['song_mask'] = f['bInf/Mask'][:]
            res['song'] = f['Song'][:]
    res['song'] = res['song'].astype(np.float32) / 1000
    return res


def load_manual_annotation(filepath):
    """Load output produced by the python ManualSegmenter."""
    manual_events = xb.load(filepath)
    return manual_events


def load_manual_annotation_matlab(filepath):
    """Load output produced by the matlab ManualSegmenter."""
    try:
        mat_data = loadmat(filepath)
    except NotImplementedError:
        with h5py.File(filepath, 'r') as f:
            mat_data = dict()
            for key, val in f.items():
                mat_data[key.lower()] = val[:].T

    manual_events_seconds = dict()
    for key, val in mat_data.items():
        if len(val) and hasattr(val, 'ndim') and val.ndim == 2 and not key.startswith('_'):  # ignore matfile metadata
            manual_events_seconds[key.lower() + '_manual'] = np.sort(val[:, 1:])
    return manual_events_seconds


def load_tracks(filepath):
    """Load tracker data"""
    with h5py.File(filepath, 'r') as f:
        if 'data' in f.keys():  # in old-style or unfixes tracks, everything is in the 'data' group
            data = dd.io.load(filepath)
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
    for cnt, ((x, y), a, p_ego) in enumerate(zip(box_offset, box_angle, poses)):
        tmp = [rotate_point(pt, a, origin) for pt in p_ego]  # rotate
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
        poses[cnt, ...] = [rotate_point(pt, -a, origin) for pt in p_ego]

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
        poses_ego (xarray.DataArray): poses in EGOcentric (centered around thorax, head aligned straight upwards
        poses_allo (xarray.DataArray): poses in ALLOcentric (frame) coordinates
        partnames (List[str]): list of names for each body part (is already part of the poses_ego/allo xrs)
        first_pose_frame, last_pose_frame (int): frame corresponding to first and last item in the poses_ego/allo arrays (could attach this info as xr dim)
    """
    with zarr.ZipStore(filepath, mode='r') as zarr_store:
        ds = xr.open_zarr(zarr_store).load()  # the final `load()` prevents
    nb_flies = len(ds.flies)
    box_size = np.array(ds.attrs['box_size'])

    poses_allo = ds.poses + ds.box_centers - box_size/2

    first_pose_frame = int(np.argmin(np.isnan(ds.poses[:, 0, 0, 0])))
    last_pose_frame = int(np.argmin(~np.isnan(ds.poses[first_pose_frame:, 0, 0, 0])) + first_pose_frame)

    # CUT to first/last frame with poses
    poses_ego = ds.poses[first_pose_frame:last_pose_frame, ...]
    poses_allo = poses_allo[first_pose_frame:last_pose_frame, ...]

    poses_ego = poses_ego - poses_ego.sel(poseparts='thorax')  # CENTER egocentric poses around thorax

    # ROTATE egocentric poses such that the angle between head and thorax is 0 degrees (straight upwards)
    head_thorax_angle = 270 + np.arctan2(poses_ego.sel(poseparts='head', coords='y'),
                                         poses_ego.sel(poseparts='head', coords='x')) * 180 / np.pi
    for cnt, (a, p_ego) in enumerate(zip(head_thorax_angle.data, poses_ego.data)):
        for fly in range(nb_flies):
            poses_ego.data[cnt, fly, ...] = [rotate_point(pt, -a[fly]) for pt in p_ego[fly]]

    return poses_ego, poses_allo, ds.poseparts, first_pose_frame, last_pose_frame


def load_raw_song(filepath_daq, song_channels: Sequence[int] = None, lazy=False):
    """[summary]

    Args:
        filepath_daq ([type]): [description]
        song_channels (List[int], optional): Sequence of integers as indices into 'samples' datasaet.
                                             Defaults to [0,..., 15].
        lazy (bool, optional): If True, will load song as dask.array, which allows lazy indexing.
                               Otherwise, will load the full recording from disk (slow). Defaults to False.

    Returns:
        [type]: [description]
    """
    if song_channels is None:  # the first 16 channels in the data are the mic recordings
        song_channels = list(range(16))

    if lazy:
        f = h5py.File(filepath_daq, 'r')
        # convert to dask array since this allows lazily evaluated indexing...
        import dask.array as daskarray
        da = daskarray.from_array(f['samples'], chunks=(10000, 1))
        song = da[:, song_channels]
    else:
        with h5py.File(filepath_daq, 'r') as f:
            song = f['samples'][:, song_channels]

    return song


def load_times(filepath_timestamps, filepath_daq):
    """Load daq and cam time stamps, create muxer"""
    with h5py.File(filepath_timestamps, 'r') as f:
        cam_stamps = f['timeStamps'][:]

    # DAQ time stamps
    with h5py.File(filepath_daq, 'r') as f:
        daq_stamps = f['systemtime'][:]
        daq_sampleinterval = f['samplenumber'][:]
    daq_samplenumber = np.cumsum(daq_sampleinterval)[:, np.newaxis]
    last_sample = daq_samplenumber[-1, 0]
    interval = np.mean(np.diff(daq_stamps[:np.argmax(daq_stamps <= 0), 0]))  # seconds
    nb_samples = np.mean(np.diff(daq_samplenumber[:np.argmax(daq_stamps <= 0), 0]))
    sampling_rate_Hz = np.round(interval * nb_samples)
    ss = SampStamp(sample_times=daq_stamps[:, 0], frame_times=cam_stamps[:, 0], sample_numbers=daq_samplenumber[:, 0])
    return ss, last_sample, sampling_rate_Hz


def initialize_manual_song_events(ds: xr.Dataset, from_segmentation: bool = False, force_overwrite: bool = False) -> xr.Dataset:
    """[summary]
    
    Args:
        ds (xarray.Dataset): [description]
        from_segmentation (bool, optional): Init manual events from automatic events with same name. Otherwise they inited as empty.
                                            If force_overwrite: will *ADD* existing manual events.
                                            otherwise: will *ADD* auto events to existing manual events.
                                            Defaults to False.
        force_overwrite (bool, optional): Overwrite existing manual events. 
                                          Defaults to False.
    
    Returns:
        xarray.Dataset: [description]
    """
    new_manual_event_types = ['sine_manual', 'pulse_manual', 'vibration_manual', 'aggression_manual']
    if 'song_events' in ds:
        new_manual_event_types = [evt for evt in new_manual_event_types
                                      if evt not in ds.song_events.event_types]
    # else:
    #     # set time axis at target_sampling_rate
    #     ds['time'] = np.arange(ds.sampletime[0], ds.sampletime[-1], 1/ds.attrs['target_sampling_rate'])

    song_events_manual = None
    if 'song_events' not in ds or new_manual_event_types:
        new_manual_events = np.zeros((ds.time.shape[0], len(new_manual_event_types)), 
                                        dtype=np.bool)
        song_events_manual = xr.DataArray(data=new_manual_events,
                                            dims=['time', 'event_types'],
                                            coords={'time': ds.time,
                                                    'event_types': new_manual_event_types,
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
           
             