import numpy as np
import h5py
# import deepdish as dd
import scipy.interpolate
from scipy.io import loadmat
import scipy.signal
from scipy.ndimage import maximum_filter1d
   
from samplestamps import *
import xarray as xr
import zarr
import math
import logging
from pathlib import Path


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
    idx = np.ravel_multi_index((np.arange(sng_max.shape[0]),loudest_channel), data.shape)
    data_merged_max = data.ravel()[idx]
    data_merged_max = data_merged_max[:, np.newaxis]  # shape needs to be [nb_samples, 1]
    return data_merged_max


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
    """Load output produced by ManualSegmenter."""
    try:
    mat_data = loadmat(filepath)
    except NotImplementedError:
        with h5py.File(filepath, 'r') as f:
            mat_data = dict()
            for key, val in f.items():            
                mat_data[key.lower()] = val[:].T

    manual_events_seconds = dict()
    for key, val in mat_data.items():
        if len(val) and val.ndim==2 and not key.startswith('_'):  # ignore matfile metadata
            manual_events_seconds[key.lower() + '_manual'] = np.sort(val[:, 1])
    return manual_events_seconds


def load_tracks(filepath):
    """Load tracker data"""
    with h5py.File(filepath, 'r') as f:
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


def load_poses(filepath):
    """Load pose tracks"""
    with h5py.File(filepath, 'r') as f:
        box_offset = f['box_centers'][:]
        box_angle = f['fixed_angles'][:].astype(np.float64)
        box_size = (f['box_size'].attrs['i0'], f['box_size'].attrs['i1'])
        poses = f['positions'][:]
        nb_flies = int(np.max(f['fly_id']) + 1)
    body_parts = ['head', 'neck', 'front_left_leg', 'middle_left_leg', 'back_left_leg', 'front_right_leg', 'middle_right_leg', 'back_right_leg', 'thorax', 'left_wing', 'right_wing', 'tail']
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
    thorax_pos = poses[:,  thorax_idx,:]
    poses = poses - thorax_pos[:, np.newaxis, :]  # subtract remaining thorax position
    origin = (0, 0)
    
    # rotate poses such that the head is at 0 degrees
    head_idx = 0
    head_pos = poses[:,  head_idx,:]
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

    poses_allo = ds.poses + ds.box_centers.data - box_size/2

    first_pose_frame = int(np.argmin(np.isnan(ds.poses[:, 0, 0, 0])))
    last_pose_frame = int(np.argmin(~np.isnan(ds.poses[first_pose_frame:, 0, 0, 0])) + first_pose_frame)

    # CUT to first/last frame with poses
    poses_ego = ds.poses[first_pose_frame:last_pose_frame, ...]
    poses_allo = poses_allo[first_pose_frame:last_pose_frame, ...]

    poses_ego = poses_ego - poses_ego.sel(poseparts='thorax')  # CENTER egocentric poses around thorax

    # ROTATE egocentric poses such that the angle between head and thorax is 0 degrees (straight upwards)
    head_thorax_angle = 270 + np.arctan2(poses_ego.sel(poseparts='head', coords='y'), poses_ego.sel(poseparts='head', coords='x')) * 180 / np.pi
    for cnt, (a, p_ego) in enumerate(zip(head_thorax_angle.data, poses_ego.data)):
        for fly in range(nb_flies):
            poses_ego.data[cnt, fly, ...] = [rotate_point(pt, -a[fly]) for pt in p_ego[fly]]
    
    return poses_ego, poses_allo, ds.poseparts, first_pose_frame, last_pose_frame


def load_raw_song(filepath_daq, song_channels=None, sampling_rate=10_000):
    """Load daq and merge channels"""
    from scipy.ndimage import maximum_filter1d
    if song_channels is None:
        song_channels = list(range(16))
    
    with h5py.File(filepath_daq, 'r') as f:
        song = f['samples'][:, song_channels]
    
    song_merged_max = merge_channels(song, sampling_rate)
    return song_merged_max


def load_times(filepath_timestamps, filepath_daq):
    """Load daq and cam time stamps, create muxer"""
    with h5py.File(filepath_timestamps) as f:
        cam_stamps = f['timeStamps'][:]

    # DAQ time stamps
    with h5py.File(filepath_daq) as f:
        daq_stamps = f['systemtime'][:]
        daq_sampleinterval = f['samplenumber'][:]
    daq_samplenumber = np.cumsum(daq_sampleinterval)[:, np.newaxis]

    interval = np.mean(np.diff(daq_stamps[:np.argmax(daq_stamps<=0),0]))  # seconds
    nb_samples = np.mean(np.diff(daq_samplenumber[:np.argmax(daq_stamps<=0),0]))
    sampling_rate_Hz = np.round(interval * nb_samples)    
    ss = SampStamp(sample_times=daq_stamps[:, 0], frame_times=cam_stamps[:, 0], sample_numbers=daq_samplenumber[:, 0])
    return ss, daq_samplenumber[-1, 0], sampling_rate_Hz


def assemble(datename, root='', dat_path='dat', res_path='res', target_sampling_rate=1000):
    """Assemble data set containing song and video data.

    Synchronizes A/V, resamples annotations (pose, song) to common sampling grid.

    Args:
        datename
        root = ''
        dat_path = 'dat'
        res_path = 'res'
        target_sampling_rate [float=1000] Sampling rate in Hz for pose and annotation data.

    Returns
        xarray Dataset containing
            [DESCRIPTION]
    """

    # load RECORDING and TIMING INFORMATION
    filepath_timestamps = Path(root, dat_path, datename, f'{datename}_timeStamps.h5')
    filepath_daq = Path(root, dat_path, datename, f'{datename}_daq.h5')
    ss, last_sample_number, sampling_rate = load_times(filepath_timestamps, filepath_daq)

    # LOAD TRACKS
    filepath_tracks = Path(root, res_path, datename, f'{datename}_tracks_fixed.h5')
    body_pos, body_parts, first_tracked_frame, last_tracked_frame, background = load_tracks(filepath_tracks)

    # LOAD POSES from DEEPPOSEKIT
    with_poses = False
    poses_from = None
    filepath_poses = Path(root, res_path, datename, f'{datename}_densenet-bgsub_scaled_poses.zarr')
    try:
        pose_pos, pose_pos_allo, pose_parts, first_pose_frame, last_pose_frame = load_poses_deepposekit(filepath_poses)
        with_poses = True
        poses_from = 'DeepPoseKit'
    except Exception as e:
        logging.warning(f'could not load pose from {filepath_poses}')
        logging.debug(e)
    
    # LOAD POSES from LEAP
    if not with_poses:
    filepath_poses = Path(root, res_path, datename, f'{datename}_poses.h5')
    try:
            pose_pos, pose_pos_allo, pose_parts, first_pose_frame, last_pose_frame = load_poses_leap(filepath_poses)
        with_poses = True
            poses_from = 'LEAP'
    except Exception as e:
        logging.warning(f'could not load pose from {filepath_poses}')
        logging.debug(e)

    # load AUTOMATIC SEGMENTATION - currently produced by matlab
    with_segmentation = False
    with_song = False
    filepath_segmentation = Path(root, res_path, datename, f'{datename}_song.mat')
    try:
        res = load_segmentation(filepath_segmentation)
        with_segmentation = True
        with_song = True
    except Exception as e:
        logging.warning(f'could not load segmentation from {filepath_segmentation}')
        logging.debug(e)

    # load RAW song traces
    if not with_segmentation:
        try:
            logging.warning(f'trying to read and merge recording from {filepath_daq}')
            song = load_raw_song(filepath_daq)
            res = {'song': song}
            with_song = True
        except Exception as e:
            logging.warning(f'could not load song from {filepath_daq}')
            logging.debug(e)
    

    # load MANUAL SONG ANNOTATIONS
    with_segmentation_manual = False
    filepath_segmentation_manual = Path(root, res_path, datename, f'{datename}_songmanual.mat')
    try:
        manual_events_seconds = load_manual_annotation(filepath_segmentation_manual)
        with_segmentation_manual = True
    except Exception as e:
        logging.warning(f'could not load manual segmentation from {filepath_segmentation_manual}')
        logging.debug(e)

    last_sample_with_frame = np.min((last_sample_number, ss.sample(frame=last_tracked_frame - 1))).astype(np.intp)
    first_sample = 0
    last_sample = int(last_sample_with_frame)

    step = int(sampling_rate / target_sampling_rate)  # ms - will resample song annotations and tracking data to 1000Hz
    target_samples = np.arange(first_sample, last_sample, step, dtype=np.uintp)  # construct desired sample grid for resampled data
    time = target_samples / sampling_rate  # time in seconds for each sample in the song annotation data
    sampletime = np.arange(first_sample, last_sample) / sampling_rate  # time in seconds for each sample in the song recording

    # get nearest frame for each sample in the resampled grid
    frame_numbers = np.arange(first_tracked_frame, last_tracked_frame)
    frame_samples = ss.sample(frame_numbers)
    interpolator = scipy.interpolate.interp1d(frame_samples, frame_numbers, kind='nearest', bounds_error=False, fill_value=np.nan)
    nearest_frame_time = interpolator(target_samples).astype(np.uintp)

    dataset_data = dict()
    # RAW SONG RECORDING
    if with_song:
        merged_song_recording = res['song'][first_sample:last_sample, 0]  # cut recording to match new grid
        song = xr.DataArray(data=merged_song_recording,
                            dims=['sampletime'],
                            coords={'sampletime': sampletime, },
                            attrs={'description': 'Song signal merged across all recording channels.',
                                   'sampling_rate_Hz': sampling_rate,
                                   'time_units': 'seconds',
                                   'amplitude_units': 'volts'})
        dataset_data['song'] = song
        
    if with_segmentation:
        # SONG LABELS
        song_labels = xr.DataArray(data=res['song_mask'][first_sample:last_sample:step, 0].astype(np.uint8),
                                   dims=['time'],
                                   coords={'time': time,
                                           'nearest_frame': (('time'), nearest_frame_time), },
                                   attrs={'description': 'Song label for each sample - 0: silence, 1: pulse, 2: sine.',
                                          'sampling_rate_Hz': sampling_rate / step,
                                          'time_units': 'seconds', })
        dataset_data['song_labels'] = song_labels


    # SONG EVENTS
    if with_segmentation_manual:
        manual_events_samples = {key: val * sampling_rate for key, val in manual_events_seconds.items()}
        events = manual_events_samples
    else:
        events = dict()

    if with_segmentation:        
        events['song_pulse_any'] = res['pulse_times_samples']
        events['song_pulse_slow'] = res['pulse_times_samples'][res['pulse_labels'] == 1]
        events['song_pulse_fast'] = res['pulse_times_samples'][res['pulse_labels'] == 0]
        events['sine'] = song_labels == 2

    if with_segmentation_manual or with_segmentation:
        eventtypes = [*events.keys()]
        nb_eventtypes = len(eventtypes)
        song_events_np = np.zeros((len(time), nb_eventtypes), dtype=np.bool)  # pre-allocate grid holding event data
        for cnt, key in enumerate(events.keys()):
            event_times = (events[key] / step).astype(np.uintp)
            event_times = event_times[event_times < last_sample_with_frame / step]
            song_events_np[event_times, cnt] = True

        song_events = xr.DataArray(data=song_events_np,
                                   dims=['time', 'event_types'],
                                   coords={'time': time,
                                           'event_types': eventtypes,
                                           'nearest_frame': (('time'), nearest_frame_time), },
                                   attrs={'description': 'Event times as boolean arrays.',
                                          'sampling_rate_Hz': sampling_rate / step,
                                          'time_units': 'seconds', })
        dataset_data['song_events'] = song_events

    # BODY POSITION
    frame_numbers = np.arange(first_tracked_frame, last_tracked_frame)
    frame_samples = ss.sample(frame_numbers)  # get sample numbers for each frame
    interpolator = scipy.interpolate.interp1d(frame_samples, body_pos, axis=0, bounds_error=False, fill_value=np.nan)
    body_pos_re = interpolator(target_samples)
    positions = xr.DataArray(data=body_pos_re,
                             dims=['time', 'flies', 'bodyparts', 'coords'],
                             coords={'time': time,
                                     'bodyparts': body_parts,
                                     'nearest_frame': (('time'), nearest_frame_time),
                                         'coords': ['y', 'x']},
                             attrs={'description': 'coords are "allocentric" - rel. to the full frame',
                                    'sampling_rate_Hz': sampling_rate / step,
                                    'time_units': 'seconds',
                                    'spatial_units': 'pixels',
                                    'background': background})
    dataset_data['body_positions'] = positions

    # POSES
    if with_poses:
        # resample to common grid at 1000Hz.
        frame_numbers = np.arange(first_pose_frame, last_pose_frame)
        frame_samples = ss.sample(frame_numbers)  # get sample numbers for each frame
        interpolator = scipy.interpolate.interp1d(frame_samples, pose_pos, axis=0, kind='linear', bounds_error=False, fill_value=np.nan)
        pose_pos_re = interpolator(target_samples)
        interpolator = scipy.interpolate.interp1d(frame_samples, pose_pos_allo, axis=0, kind='linear', bounds_error=False, fill_value=np.nan)
        pose_pos_allo_re = interpolator(target_samples)

        # make DataArray
        poses = xr.DataArray(data=pose_pos_re,
                             dims=['time', 'flies', 'poseparts', 'coords'],
                             coords={'time': time,
                                     'poseparts': pose_parts,
                                     'nearest_frame': (('time'), nearest_frame_time),
                                     'coords': ['y', 'x']},
                             attrs={'description': 'coords are "egocentric" - rel. to box',
                                    'sampling_rate_Hz': sampling_rate / step,
                                    'time_units': 'seconds',
                                    'spatial_units': 'pixels',
                                    'poses_from': poses_from})
        # make DataArray
        poses_allo = xr.DataArray(data=pose_pos_allo_re,
                             dims=['time', 'flies', 'poseparts', 'coords'],
                             coords={'time': time,
                                     'poseparts': pose_parts,
                                     'nearest_frame': (('time'), nearest_frame_time),
                                     'coords': ['y', 'x']},
                             attrs={'description': 'coords are "allocentric" - rel. to frame',
                                    'sampling_rate_Hz': sampling_rate / step,
                                    'time_units': 'seconds',
                                    'spatial_units': 'pixels',
                                    'poses_from': poses_from})
        dataset_data['pose_positions'] = poses
        dataset_data['pose_positions_allo'] = poses_allo

    # MAKE THE DATASET
    dataset = xr.Dataset(dataset_data, attrs={})
    # save command line args
    dataset.attrs = {'video_filename': str(Path(root, dat_path, datename, f'{datename}.mp4')),
                     'datename': datename, 'root': root, 'dat_path': dat_path, 'res_path': res_path}
    return dataset


def save(savepath, dataset):
    with zarr.ZipStore(savepath, mode='w') as zarr_store:
        dataset.to_zarr(store=zarr_store, compute=True)


def load(savepath):
    with zarr.ZipStore(savepath, mode='r') as zarr_store:
        dataset = xr.open_zarr(zarr_store).load()  # the final `load()` prevents lazy loading
    return dataset
