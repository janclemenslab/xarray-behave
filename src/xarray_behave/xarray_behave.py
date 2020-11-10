"""Create self-documenting xarray dataset from behavioral recordings and annotations."""
import numpy as np
import scipy.interpolate
import scipy.stats
import xarray as xr
import zarr
import logging
import os.path
from pathlib import Path
import pandas as pd

from . import (loaders as ld,
               metrics as mt,
               event_utils,
               annot)


def assemble(datename, root='', dat_path='dat', res_path='res', target_sampling_rate=1_000,
             keep_multi_channel: bool = False, resample_video_data: bool = True,
             include_song: bool = True, include_tracks: bool = True, include_poses: bool = True,
             make_mask: bool = False, fix_fly_indices: bool = True) -> xr.Dataset:
    """[summary]

    Args:
        datename ([type]): [description]
        root (str, optional): [description]. Defaults to ''.
        dat_path (str, optional): [description]. Defaults to 'dat'.
        res_path (str, optional): [description]. Defaults to 'res'.
        target_sampling_rate (int, optional): Sampling rate in Hz for pose and annotation data. Defaults to 1000.
        keep_multi_channel (bool, optional): Add multi-channel data (otherwise will only add merged traces). Defaults to False.
        resample_video_data (bool, optional): Or keep video with original frame times. Defaults to True.
        include_song (bool, optional): [description]. Defaults to True.
        include_tracks (bool, optional): [description]. Defaults to True.
        include_poses (bool, optional): [description]. Defaults to True.
        make_mask (bool, optional): ..., Defaults to False.
        fix_fly_indices (bool, optional): Will attempt to load swap info and fix fly id's accordingly, Defaults to True.

    Returns:
        xarray.Dataset
    """

    # load RECORDING and TIMING INFORMATION
    filepath_timestamps = Path(root, dat_path, datename, f'{datename}_timeStamps.h5')
    filepath_daq = Path(root, dat_path, datename, f'{datename}_daq.h5')
    ss, last_sample_number, sampling_rate = ld.load_times(filepath_timestamps, filepath_daq)
    if not resample_video_data:
        logging.info(f'  setting targetsamplingrate to avg. fps.')
        target_sampling_rate = 1 / np.mean(np.diff(ss.frames2times.y))

    # LOAD TRACKS
    with_tracks = False
    with_fixed_tracks = False
    if include_tracks:
        logging.info('Attempting to load automatic tracking:')
        filepath_tracks = Path(root, res_path, datename, f'{datename}_tracks_fixed.h5')
        filepath_tracks_nonfixed = Path(root, res_path, datename, f'{datename}_tracks.h5')
        if os.path.exists(filepath_tracks):
        # try:
            body_pos, body_parts, first_tracked_frame, last_tracked_frame, background = ld.load_tracks(filepath_tracks)
            with_tracks = True
            with_fixed_tracks = True
            logging.debug(f'  {filepath_tracks} loaded.')
        elif os.path.exists(filepath_tracks_nonfixed):
        # except (FileNotFoundError, OSError) as e:
            logging.debug(f'  {filepath_tracks} not found.')
            # logging.debug(e)
            # try:
            #     logging.info(f'Trying non-fixed tracks at {filepath_tracks_nonfixed}.')
            body_pos, body_parts, first_tracked_frame, last_tracked_frame, background = ld.load_tracks(filepath_tracks_nonfixed)
            with_tracks = True
            logging.debug(f'  {filepath_tracks_nonfixed} loaded.')
        else:
            logging.debug(f'  {filepath_tracks_nonfixed} not found.')
            logging.info(f'   Could not load automatic tracking from {filepath_tracks} or {filepath_tracks_nonfixed}.')

            # except (FileNotFoundError, OSError) as e:
            #     logging.info(f'   This failed, too:')
            #     logging.debug(e)
    if not with_tracks:
        first_tracked_frame = int(ss.frame(0))
        last_tracked_frame = int(ss.frame(last_sample_number))
        logging.info(f'No tracks - setting first/last tracked frame numbers to those of the first/last sample in the recording ({first_tracked_frame}, {last_tracked_frame}).')
    else:
        logging.info(f'Tracked frame {first_tracked_frame} to {last_tracked_frame}.')

    # LOAD POSES from DEEPPOSEKIT
    with_poses = False
    poses_from = None
    if include_poses:
        logging.info(f'Attempting to load poses:')
        filepath_poses = Path(root, res_path, datename, f'{datename}_poses_dpk.zarr')
        filepath_poses_leap = Path(root, res_path, datename, f'{datename}_poses.h5')
        if os.path.exists(filepath_poses):
            pose_pos, pose_pos_allo, pose_parts, first_pose_frame, last_pose_frame = ld.load_poses_deepposekit(filepath_poses)
            with_poses = pose_pos.shape[0]>0  # ensure non-empty poses
            poses_from = 'DeepPoseKit'
            logging.debug(f'  {filepath_poses} loaded.')
        elif os.path.exists(filepath_poses_leap):
            logging.debug(f'  {filepath_poses} not found.')
            pose_pos, pose_pos_allo, pose_parts, first_pose_frame, last_pose_frame = ld.load_poses_leap(filepath_poses_leap)
            with_poses = pose_pos.shape[0]>0  # ensure non-empty poses
            poses_from = 'LEAP'
        else:
            logging.debug(f'  {filepath_poses_leap} not found.')
            logging.info(f'   Could not load pose from {filepath_poses} or {filepath_poses_leap} or .')

    # instead of these flags - just check for existence of dict keys??
    with_segmentation = False
    with_segmentation_manual = False
    with_segmentation_manual_matlab = False
    with_song = False
    with_song_raw = False
    merge_raw_recording = False
    auto_event_seconds = {}
    auto_event_categories = {}
    manual_event_seconds = {}
    manual_event_categories = {}

    if include_song:
        logging.info('Attempting to load automatic segmentation:')
        # load AUTOMATIC SEGMENTATION - currently produced by matlab
        filepath_segmentation_matlab = Path(root, res_path, datename, f'{datename}_song.mat')
        filepath_segmentation = Path(root, res_path, datename, f'{datename}_song.h5')
        if os.path.exists(filepath_segmentation):
            try:
                auto_event_seconds, auto_event_categories = ld.load_segmentation(filepath_segmentation)
                with_segmentation = True
                merge_raw_recording = False  # True
                logging.debug(f'   {filepath_segmentation} loaded.')
            except Exception as e:
                logging.debug(f'   {filepath_segmentation} failed.')
                logging.exception(e)
                with_segmentation = False
                merge_raw_recording = False

        elif os.path.exists(filepath_segmentation_matlab):
            logging.debug(f'   {filepath_segmentation} not found.')
            auto_event_seconds, auto_event_categories  = ld.load_segmentation_matlab(filepath_segmentation_matlab)
            with_segmentation = True
            with_segmentation_matlab = True
            merge_raw_recording = False  # True
            with_song = True
            logging.debug(f'   {filepath_segmentation_matlab} loaded.')
        else:
            logging.debug(f'   {filepath_segmentation_matlab} not found.')
            logging.info(f'   Could not load automatic segmentation from {filepath_segmentation} or {filepath_segmentation_matlab}.')

        # load MANUAL SONG ANNOTATIONS
        # first try PYTHON, then matlab
        logging.info('Attempting to load manual segmentation:')
        filepath_segmentation_manual_csv = Path(root, res_path, datename, f'{datename}_songmanual.csv')
        filepath_segmentation_manual_zarr = Path(root, res_path, datename, f'{datename}_songmanual.zarr')
        filepath_segmentation_manual_matlab = Path(root, res_path, datename, f'{datename}_songmanual.mat')
        if os.path.exists(filepath_segmentation_manual_csv):
            manual_event_seconds, manual_event_categories = ld.load_manual_annotation_csv(filepath_segmentation_manual_csv)  # need to extract event_seconds and categories from that one
            with_segmentation_manual = True
            logging.debug(f'   {filepath_segmentation_manual_csv} loaded.')
        elif os.path.exists(filepath_segmentation_manual_zarr):
            manual_event_seconds, manual_event_categories = ld.load_manual_annotation_zarr(filepath_segmentation_manual_zarr)  # need to extract event_seconds and categories from that one
            with_segmentation_manual = True
            logging.debug(f'   {filepath_segmentation_manual_zarr} loaded.')
        elif os.path.exists(filepath_segmentation_manual_matlab):
            manual_event_seconds, manual_event_categories = ld.load_manual_annotation_matlab(filepath_segmentation_manual_matlab)
            with_segmentation_manual = True
            with_segmentation_manual_matlab = True
            logging.debug(f'   {filepath_segmentation_manual_matlab} loaded.')
        else:
            logging.info(f'Could not load manual segmentation from {filepath_segmentation_manual_csv} or {filepath_segmentation_manual_zarr} or {filepath_segmentation_manual_matlab}.')

        # load RAW song traces
        res = dict()
        if keep_multi_channel or merge_raw_recording:
            try:
                logging.info(f'Reading recording from {filepath_daq}.')
                song_raw, non_song_raw = ld.load_raw_song(filepath_daq, return_nonsong_channels=True, lazy=True)
                if keep_multi_channel:
                    res['song_raw'] = song_raw
                    res['non_song_raw'] = non_song_raw
                    with_song_raw = True
                if merge_raw_recording:
                    res['song'] = ld.merge_channels(song_raw, sampling_rate)
                    with_song = True
            except (FileNotFoundError, OSError) as e:
                logging.info(f'Could not load song from {filepath_daq}.')
                logging.debug(e)

    # merge manual and auto events -
    # this will overwrite thing with manual overwriting existing auto events with the same key
    # could add optional merging were values for identical keys are combined
    event_seconds = auto_event_seconds.copy()
    event_seconds.update(manual_event_seconds)
    event_categories = auto_event_categories.copy()
    event_categories.update(manual_event_categories)

    event_seconds = ld.fix_keys(event_seconds)
    event_categories = ld.fix_keys(event_categories)

    # PREPARE sample/time/framenumber grids
    last_sample_with_frame = np.min((last_sample_number, ss.sample(frame=last_tracked_frame - 1))).astype(np.intp)
    first_sample = 0
    last_sample = int(last_sample_with_frame)
    ref_time = ss.sample_time(0)  # 0 seconds = first DAQ sample
    # time in seconds for each sample in the song recording
    sampletime = ss.sample_time(np.arange(first_sample, last_sample)) - ref_time

    # build interpolator to get neareast frame for each sample in the new grid
    frame_numbers = np.arange(first_tracked_frame, last_tracked_frame)
    frame_samples = ss.sample(frame_numbers)
    interpolator = scipy.interpolate.interp1d(frame_samples, frame_numbers,
                                              kind='nearest', bounds_error=False, fill_value=np.nan)
    # construct desired sample grid for data
    step = int(sampling_rate / target_sampling_rate)
    if resample_video_data:
        target_samples = np.arange(first_sample, last_sample, step, dtype=np.uintp)
    else:
        target_samples = frame_samples
        resample_video_data = True  # <- this means we can remove a lot of code below (needs some more testing though)
    time = ss.sample_time(target_samples) - ref_time
    # get neareast frame for each sample in the sample grid
    nearest_frame = interpolator(target_samples).astype(np.uintp)

    # PREPARE DataArrays
    dataset_data = dict()
    if with_song and merge_raw_recording:  # MERGED song recording
        song = xr.DataArray(data=res['song'][first_sample:last_sample, 0],  # cut recording to match new grid
                            dims=['sampletime'],
                            coords={'sampletime': sampletime, },
                            attrs={'description': 'Song signal merged across all recording channels.',
                                   'sampling_rate_Hz': sampling_rate,
                                   'time_units': 'seconds',
                                   'amplitude_units': 'volts'})
        dataset_data['song'] = song

    if with_song_raw:
        if 0 not in res['song_raw'].shape:  # xr fails saving zarr files with 0-size along any dim
            song_raw = xr.DataArray(data=res['song_raw'][first_sample:last_sample, :],  # cut recording to match new grid
                                    dims=['sampletime', 'channels'],
                                    coords={'sampletime': sampletime, },
                                    attrs={'description': 'Raw song recording (multi channel).',
                                        'sampling_rate_Hz': sampling_rate,
                                        'time_units': 'seconds',
                                        'amplitude_units': 'volts'})
            dataset_data['song_raw'] = song_raw
        if 0 not in res['non_song_raw'].shape:  # xr fails saving zarr files with 0-size along any dim
            non_song_raw = xr.DataArray(data=res['non_song_raw'][first_sample:last_sample, :],  # cut recording to match new grid
                                        dims=['sampletime', 'no_song_channels'],
                                        coords={'sampletime': sampletime, },
                                        attrs={'description': 'Non song (stimulus) data.',
                                            'sampling_rate_Hz': sampling_rate,
                                            'time_units': 'seconds',
                                            'amplitude_units': 'volts'})
            dataset_data['non_song_raw'] = non_song_raw


    if not resample_video_data:
        logging.info(f'Resampling event data to match frame times.')
        frame_times = ss.frame_time(frame_numbers) - ref_time
        time = frame_times

        len_list = [time.shape[0]]
        if with_tracks:
            len_list.append(body_pos.shape[0])
        # if with_segmentation_manual or with_segmentation:
        #     len_list.append(song_events.shape[0])
        if with_poses:
            len_list.append(pose_pos.shape[0])
        min_len = min(len_list)
        logging.info(f'Cutting all data to {min_len} frames.')

    logging.info('Making all datasets:')
    if with_segmentation_manual or with_segmentation_manual_matlab or with_segmentation:
        logging.info('   segmentations')
        song_events = np.zeros((len(time), len(event_seconds)), dtype=np.int16)

        if not resample_video_data:
            time = time[:min_len]
            song_events = song_events[:min_len]
            nearest_frame = nearest_frame[:min_len]

        song_events = xr.DataArray(data=song_events,  # start with empty song_events matrix
                                   dims=['time', 'event_types'],
                                   coords={'time': time,
                                           'event_types': list(event_seconds.keys()),
                                           'event_categories': (('event_types'), list(event_categories.values())),
                                           'nearest_frame': (('time'), nearest_frame), },
                                   attrs={'description': 'Event times as boolean arrays.',
                                          'sampling_rate_Hz': sampling_rate / step,
                                          'time_units': 'seconds',
                                          'event_times': event_seconds})

        # now populate from event_times attribute
        song_events_ds = event_utils.eventtimes_to_traces(song_events.to_dataset(name='song_events'),
                                                        song_events.attrs['event_times'])
        dataset_data['song_events'] = song_events_ds.song_events
        try:
            ds_eventtimes = annot.Events(event_seconds).to_dataset()
            dataset_data['event_times'] = ds_eventtimes.event_times
            dataset_data['event_names'] = ds_eventtimes.event_names
            dataset_data['event_names'].attrs['possible_event_names'] = ds_eventtimes.attrs['possible_event_names']
        except Exception as e:
            logging.error('Failed to generate event_times data arrays:')
            logging.exception(e)

    # BODY POSITION
    fps = None
    if with_tracks:
        logging.info('   tracking')
        frame_numbers = np.arange(first_tracked_frame, last_tracked_frame)
        frame_times = ss.frame_time(frame_numbers) - ref_time
        fps = 1 / np.nanmean(np.diff(frame_times))

        if resample_video_data:  # resample to common grid at target_sampling_rate.
            frame_numbers = np.arange(first_tracked_frame, last_tracked_frame)
            frame_samples = ss.sample(frame_numbers)  # get sample numbers for each frame

            interpolator = scipy.interpolate.interp1d(
                frame_samples, body_pos, axis=0, bounds_error=False, fill_value=np.nan)
            body_pos = interpolator(target_samples)
        else:
            time = frame_times
            nearest_frame = frame_numbers
            time = time[:min_len]
            nearest_frame = nearest_frame[:min_len]
            body_pos = body_pos[:min_len]

        positions = xr.DataArray(data=body_pos,
                                 dims=['time', 'flies', 'bodyparts', 'coords'],
                                 coords={'time': time,
                                         'bodyparts': body_parts,
                                         'nearest_frame': (('time'), nearest_frame),
                                         'coords': ['y', 'x']},
                                 attrs={'description': 'coords are "allocentric" - rel. to the full frame',
                                        'sampling_rate_Hz': sampling_rate / step,
                                        'time_units': 'seconds',
                                        'video_fps': fps,
                                        'spatial_units': 'pixels',
                                        'background': background,
                                        'tracks_fixed': with_fixed_tracks})
        dataset_data['body_positions'] = positions

    # POSES
    if with_poses:
        logging.info('   poses')
        frame_numbers = np.arange(first_pose_frame, last_pose_frame)
        frame_times = ss.frame_time(frame_numbers) - ref_time
        fps = 1/np.nanmean(np.diff(frame_times))

        if resample_video_data:  # resample to common grid at target_sampling_rate.
            frame_samples = ss.sample(frame_numbers)  # get sample numbers for each frame
            interpolator = scipy.interpolate.interp1d(
                frame_samples, pose_pos, axis=0, kind='linear', bounds_error=False, fill_value=np.nan)
            pose_pos = interpolator(target_samples)

            interpolator = scipy.interpolate.interp1d(
                frame_samples, pose_pos_allo, axis=0, kind='linear', bounds_error=False, fill_value=np.nan)
            pose_pos_allo = interpolator(target_samples)
        else:
            time = frame_times
            nearest_frame = frame_numbers
            # ensure all is equal length
            time = time[:min_len]
            nearest_frame = nearest_frame[:min_len]
            pose_pos = pose_pos[:min_len]
            pose_pos_allo = pose_pos_allo[:min_len]

        # poses in EGOCENTRIC coordinates
        poses = xr.DataArray(data=pose_pos,
                             dims=['time', 'flies', 'poseparts', 'coords'],
                             coords={'time': time,
                                     'poseparts': pose_parts,
                                     'nearest_frame': (('time'), nearest_frame),
                                     'coords': ['y', 'x']},
                             attrs={'description': 'coords are "egocentric" - rel. to box',
                                    'sampling_rate_Hz': sampling_rate / step,
                                    'time_units': 'seconds',
                                    'video_fps': fps,
                                    'spatial_units': 'pixels',
                                    'poses_from': poses_from})
        dataset_data['pose_positions'] = poses

        # poses in ALLOcentric (frame-relative) coordinates
        poses_allo = xr.DataArray(data=pose_pos_allo,
                                  dims=['time', 'flies', 'poseparts', 'coords'],
                                  coords={'time': time,
                                          'poseparts': pose_parts,
                                          'nearest_frame': (('time'), nearest_frame),
                                          'coords': ['y', 'x']},
                                  attrs={'description': 'coords are "allocentric" - rel. to frame',
                                         'sampling_rate_Hz': sampling_rate / step,
                                         'time_units': 'seconds',
                                         'video_fps': fps,
                                         'spatial_units': 'pixels',
                                         'poses_from': poses_from})
        dataset_data['pose_positions_allo'] = poses_allo
    # MAKE THE DATASET
    logging.info('   assembling')
    dataset = xr.Dataset(dataset_data, attrs={})
    if 'time' not in dataset:
        dataset.coords['time'] = time
    if 'nearest_frame' not in dataset:
        dataset.coords['nearest_frame'] = (('time'), nearest_frame)

    if target_sampling_rate is None:
        target_sampling_rate = fps
    # save command line args
    dataset.attrs = {'video_filename': str(Path(root, dat_path, datename, f'{datename}.mp4')),
                     'datename': datename, 'root': root, 'dat_path': dat_path, 'res_path': res_path,
                     'target_sampling_rate_Hz': target_sampling_rate}

    if fix_fly_indices:
        logging.info('   applying fly identity fixes')
        try:
            filepath_swap = Path(root, res_path, datename, f'{datename}_idswaps.txt')
            indices, flies1, flies2 = ld.load_swap_indices(filepath_swap)
            dataset = ld.swap_flies(dataset, indices, flies1=0, flies2=1)
            logging.info(f'  Fixed fly identities using info from {filepath_swap}.')
        except (FileNotFoundError, OSError) as e:
            logging.debug(f'  Could not load fly identities using info from {filepath_swap}.')
            logging.debug(e)
    logging.info('Done.')

    return dataset


def assemble_metrics(dataset, make_abs: bool = True, make_rel: bool = True, smooth_positions: bool = True):
    """[summary]

    Args:
        dataset ([type]): [description]
        make_abs (bool, optional): [description]. Defaults to True.
        make_rel (bool, optional): [description]. Defaults to True.
        smooth_positions (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
        xarray.Dataset: containing these features:
                angles [time,flies]
                vels [time,flies,forward/lateral]
                chamber_vels [time,flies,y/x]
                rotational_speed [time,flies]
                accelerations [time,flies,forward/lateral]
                chamber_acc [time,flies,y/x]
                rotational_acc [time,flies]
                wing_angle_left [time,flies]
                wing_angle_right [time,flies]
                wing_angle_sum [time,flies]
                relative angle [time,flies,flies]
                (relative orientation) [time,flies,flies]
                (relative velocities) [time,flies,flies,y/x]
    """

    time = dataset.time
    nearest_frame = dataset.nearest_frame
    sampling_rate = dataset.pose_positions.attrs['sampling_rate_Hz']
    frame_rate = dataset.pose_positions.attrs['video_fps']

    thoraces = dataset.pose_positions_allo.loc[:, :, 'thorax', :].values.astype(np.float32)
    heads = dataset.pose_positions_allo.loc[:, :, 'head', :].values.astype(np.float32)
    wing_left = dataset.pose_positions_allo.loc[:, :, 'left_wing', :].values.astype(np.float32)
    wing_right = dataset.pose_positions_allo.loc[:, :, 'right_wing', :].values.astype(np.float32)

    if smooth_positions:
        # Smoothing window should span 2 frames to get smooth acceleration traces.
        # Since std of Gaussian used for smoothing has std of winlen/8, winlen should span 16 frames.
        winlen = np.ceil(16 / frame_rate * sampling_rate)
        thoraces = mt.smooth(thoraces, winlen)
        heads = mt.smooth(heads, winlen)
        wing_left = mt.smooth(wing_left, winlen)
        wing_right = mt.smooth(wing_right, winlen)

    angles = mt.angle(thoraces, heads)
    chamber_vels = mt.velocity(thoraces, ref='chamber')

    ds_dict = dict()
    if make_abs:
        vels = mt.velocity(thoraces, heads)
        accelerations = mt.acceleration(thoraces, heads)
        chamber_acc = mt.acceleration(thoraces, ref='chamber')

        vels_x = chamber_vels[..., 1]
        vels_y = chamber_vels[..., 0]
        vels_forward = vels[..., 0]
        vels_lateral = vels[..., 1]
        vels_mag = np.linalg.norm(vels, axis=2)
        accs_x = chamber_acc[..., 1]
        accs_y = chamber_acc[..., 0]
        accs_forward = accelerations[..., 0]
        accs_lateral = accelerations[..., 1]
        accs_mag = np.linalg.norm(accelerations, axis=2)

        rotational_speed = mt.rot_speed(thoraces, heads)
        rotational_acc = mt.rot_acceleration(thoraces, heads)

        # wing_angle_left = mt.angle(heads, thoraces) - mt.angle(thoraces, wing_left)
        # wing_angle_right = -(mt.angle(heads, thoraces) - mt.angle(thoraces, wing_right))
        # wing_angle_sum = mt.internal_angle(wing_left,thoraces,wing_right)
        wing_angle_left = 180-mt.internal_angle(wing_left,thoraces,heads)
        wing_angle_right = 180-mt.internal_angle(wing_right,thoraces,heads)
        wing_angle_sum = wing_angle_left + wing_angle_right

        list_absolute = [
            angles, rotational_speed, rotational_acc,
            vels_mag, vels_x, vels_y, vels_forward, vels_lateral,
            accs_mag, accs_x, accs_y, accs_forward, accs_lateral,
            wing_angle_left, wing_angle_right, wing_angle_sum
        ]

        abs_feature_names = [
            'angles', 'rotational_speed', 'rotational_acceleration',
            'velocity_magnitude', 'velocity_x', 'velocity_y', 'velocity_forward', 'velocity_lateral',
            'acceleration_mag', 'acceleration_x', 'acceleration_y', 'acceleration_forward', 'acceleration_lateral',
            'wing_angle_left', 'wing_angle_right', 'wing_angle_sum'
        ]

        absolute = np.stack(list_absolute, axis=2)

        ds_dict['abs_features'] = xr.DataArray(data=absolute,
                                               dims=['time', 'flies', 'absolute_features'],
                                               coords={'time': time,
                                                       'absolute_features': abs_feature_names,
                                                       'nearest_frame': (('time'), nearest_frame)},
                                               attrs={'description': 'coords are "egocentric" - rel. to box',
                                                      'sampling_rate_Hz': sampling_rate,
                                                      'time_units': 'seconds',
                                                      'spatial_units': 'pixels'})
    if make_rel:
        # RELATIVE FEATURES #
        dis = mt.distance(thoraces)
        rel_angles = mt.relative_angle(thoraces, heads)
        rel_orientation = angles[:, np.newaxis, :] - angles[:, :, np.newaxis]
        rel_velocities_y = chamber_vels[..., 0][:, np.newaxis, :] - chamber_vels[..., 0][:, :, np.newaxis]
        rel_velocities_x = chamber_vels[..., 1][:, np.newaxis, :] - chamber_vels[..., 1][:, :, np.newaxis]
        rel_velocities_lateral, rel_velocities_forward = mt.project_velocity(rel_velocities_x, rel_velocities_y, np.radians(angles))
        rel_velocities_mag = np.sqrt(rel_velocities_forward**2 + rel_velocities_lateral**2)

        list_relative = [
            dis, rel_angles, rel_orientation,
            rel_velocities_mag, rel_velocities_forward, rel_velocities_lateral
        ]

        rel_feature_names = [
            'distance', 'relative_angle', 'relative_orientation',
            'relative_velocity_mag', 'relative_velocity_forward', 'relative_velocity_lateral'
        ]

        relative = np.stack(list_relative, axis=3)
        ds_dict['rel_features'] = xr.DataArray(data=relative,
                                               dims=['time', 'flies', 'relative_flies', 'relative_features'],
                                               coords={'time': time,
                                                       'relative_features': rel_feature_names,
                                                       'nearest_frame': (('time'), nearest_frame)},
                                               attrs={'description': 'coords are "egocentric" - rel. to box',
                                                      'sampling_rate_Hz': sampling_rate,
                                                      'time_units': 'seconds',
                                                      'spatial_units': 'pixels'})

    # MAKE ONE DATASET
    feature_dataset = xr.Dataset(ds_dict, attrs={})
    return feature_dataset


def save(savepath, dataset):
    """[summary]

    Args:
        savepath ([type]): [description]
        dataset ([type]): [description]
    """
    # xarray can't save attrs with dict values.
    # Delete event_times prior to saving.
    # Should make event_times a sparse xr.DataArray in the future.
    if 'song_events' in dataset:
        if 'event_times' in dataset.song_events.attrs:
            del dataset.song_events.attrs['event_times']

    with zarr.ZipStore(savepath, mode='w') as zarr_store:
        ## re-chunking does not seem to help with IO speed upon lazy loading
        # chunks = dict(dataset.dims)
        # chunks['time'] = 100_000
        # chunks['sampletime'] = 100_000
        # dataset = dataset.chunk(chunks)
        dataset.to_zarr(store=zarr_store, compute=True)


def _normalize_strings(dataset):
    """Ensure all keys in coords are proper (unicode?) python strings, not byte strings."""
    dn = dict()

    for key, val in dataset.coords.items():
        if val.dtype == 'S16':
            dataset[key] = [v.decode() for v in val.data]
    return dataset


def load(savepath, lazy: bool = False, normalize_strings: bool = True,
         use_temp: bool = False):
    """[summary]

    Args:
        savepath ([type]): [description]
        lazy (bool, optional): [description]. Defaults to True.
        normalize_strings (bool, optional): [description]. Defaults to True.
        use_temp (bool, optional): Unpack zip to temp file - potentially speeds up loading and allows overwriting existing zarr file.
                                   Defaults to True.
    Returns:
        [type]: [description]
    """
    zarr_store = zarr.ZipStore(savepath, mode='r')
    if use_temp:
        dest = zarr.TempStore()
        zarr.copy_store(zarr_store, dest)
        zarr_store.close()
        zarr_store = dest
    dataset = xr.open_zarr(zarr_store)
    if not lazy:
        dataset.load()
        zarr_store.close()

    if normalize_strings:
        dataset = _normalize_strings(dataset)
    print(dataset)
    return dataset


def from_wav(filepath, target_samplerate=None,
             event_names=[], event_categories=[],
             annotation_path=None):
    import soundfile
    logging.info(f"Loading data from {filepath}.")
    data, sampling_rate = soundfile.read(filepath)

    if data.ndim==1:
        logging.info("   Data is 1d so prolly single-channel audio - appending singleton dimension.")
        data = data[:, np.newaxis]

    ds = from_data(filepath, data, sampling_rate, target_samplerate, event_names, event_categories, annotation_path)
    return ds


def from_hdf5(filepath, data_set, sampling_rate, target_samplerate=None,
              event_names=[], event_categories=[],
              annotation_path=None):
    import h5py
    logging.info(f"Loading dataset 'data_set' from {filepath}.")
    with h5py.File(filepath, 'r') as f:
        data = f[data_set][:]

    if data.ndim==1:
        logging.info("   Dataset is 1d so prolly single-channel data - appending singleton dimension.")
        data = data[:, np.newaxis]

    if data.shape[0] < data.shape[1]:
        logging.info("   Dataset should be [time x channels] but is of shape {data.shape} - transposing to {data.T.shape}.")
        data = data.T

    ds = from_data(filepath, data, sampling_rate, target_samplerate, event_names, event_categories, annotation_path)
    return ds


def from_data(filepath, data, sampling_rate, target_samplerate=None,
              event_names=[], event_categories=[],
              annotation_path=None):

    if event_names and not event_categories:
        logging.info('No event_categories specified - defaulting to segments')
        event_categories = ['segment'] * len(event_names)

    if len(event_names) != len(event_categories):
        raise ValueError(f'event_names and event_categories need to have same length - have {len(event_names)} and {len(event_categories)}.')

    if target_samplerate is None:
        target_samplerate = sampling_rate

    dataset_data = dict()

    sampletime = np.arange(len(data)) / sampling_rate
    time = np.arange(sampletime[0], sampletime[-1], 1 / target_samplerate)

    song_raw = xr.DataArray(data=data,  # cut recording to match new grid
                        dims=['sampletime', 'channels'],
                        coords={'sampletime': sampletime, },
                        attrs={'description': 'Song signal merged across all recording channels.',
                               'sampling_rate_Hz': sampling_rate,
                               'time_units': 'seconds',
                               'amplitude_units': 'volts'})
    dataset_data['song_raw'] = song_raw

    if event_names is not None and event_categories is not None:
        song_events_data = np.zeros((len(time), len(event_names)), dtype=np.uint)
        song_events = xr.DataArray(data=song_events_data,
                    dims=['time', 'event_types'],
                    coords={'time': time,
                            'event_types': event_names,
                            'event_categories': (('event_types'), event_categories)},
                    attrs={'description': 'Song annotations',
                            'sampling_rate_Hz': sampling_rate,
                            'time_units': 'seconds',
                           })
        dataset_data['song_events'] = song_events

    if annotation_path is not None:
        try:
            df = pd.read_csv(annotation_path)
            ds_eventtimes = annot.Events.from_df(df).to_dataset()
            dataset_data['event_times'] = ds_eventtimes.event_times
            dataset_data['event_names'] = ds_eventtimes.event_names
        except Exception as e:
            logging.exception(e)

    # MAKE THE DATASET
    ds = xr.Dataset(dataset_data, attrs={})
    ds.coords['time'] = time
    # ds.coords['nearest_frame'] = ('time', (time/100).astype(np.uint))  # do we need this?

    # save command line args
    ds.attrs = {'video_filename': '',
                'datename': filepath,
                'root': '', 'dat_path': '', 'res_path': '',
                'sampling_rate_Hz': sampling_rate,
                'target_sampling_rate_Hz': target_samplerate}
    logging.info(ds)
    return ds
