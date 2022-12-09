"""Load files created by various analysis programs created."""
import numpy as np
import pandas as pd
import h5py
from typing import Sequence, Optional, Tuple
import scipy.interpolate
import scipy.ndimage
import scipy.signal
import scipy.stats
from scipy.ndimage import maximum_filter1d
from samplestamps import SampStamp
import math


def rotate_point(point: Tuple[float, float], degrees: float, origin: Tuple[float, float] = (0, 0)) -> Tuple[float, float]:
    """Rotates 2D point around another point.

    Args:
        point (Tuple[float, float]): (x,y) coordinates of the point
        degrees (float): angle in degrees by which to rotate
        origin (Tuple[float, float], optional): point (x,y) around which to rotate point. Defaults to (0, 0).

    Returns:
        Tuple[float, float]: Rotated point
    """
    x, y = point
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
    """Rotates point set.

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
        # this eats a lot of memory - maybe use nearest neighbour interpolation for upsampling?
        abs_dist = np.abs(np.subtract.outer(array, values))
        idx = abs_dist.argmin(0)
        dist = abs_dist.min(0)
        val = array[idx]
    else:
        idx = np.full_like(values, fill_value=np.nan)
        dist = np.full_like(values, fill_value=np.nan)
        val = np.full_like(values, fill_value=np.nan)
    return val, idx, dist


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
    onsets = np.where(np.diff(sine_pred.astype(np.int)) == 1)[0]
    offsets = np.where(np.diff(sine_pred.astype(np.int)) == -1)[0]
    if len(onsets) and len(offsets):
        onsets = onsets[onsets < offsets[-1]]
        offsets = offsets[offsets > onsets[0]]
        durations = offsets - onsets
        for idx, (onset, offset, duration) in enumerate(zip(onsets, offsets, durations)):
            if idx > 0 and offsets[idx - 1] > onsets[idx] - gap_dur:
                sine_pred[offsets[idx - 1]:onsets[idx] + 1] = 1
    return sine_pred


def remove_short(sine_pred, min_len=100):
    # remove too short sine songs
    onsets = np.where(np.diff(sine_pred.astype(np.int)) == 1)[0]
    offsets = np.where(np.diff(sine_pred.astype(np.int)) == -1)[0]
    if len(onsets) and len(offsets):
        onsets = onsets[onsets < offsets[-1]]
        offsets = offsets[offsets > onsets[0]]
        durations = offsets - onsets
        for cnt, (onset, offset, duration) in enumerate(zip(onsets, offsets, durations)):
            if duration < min_len:
                sine_pred[onset:offset + 1] = 0
    return sine_pred


def load_swap_indices(filepath):
    a = np.loadtxt(filepath)  # dtype=np.uintp)
    indices, flies1, flies2 = np.split(a, [1, 2], axis=-1)  # split columns in file into variables
    if indices.ndim == 2:  # is sometimes 2d for some reason
        indices = indices[:, 0]
    return indices, flies1.astype(np.uintp), flies2.astype(np.uintp)


def swap_flies(dataset, swap_times, flies1=0, flies2=1):
    """Swap flies in dataset.

    Caution: datavariables are currently hard-coded!
    Caution: Swap *may* be in place so *might* will alter original dataset.

    Args:
        dataset ([type]): Dataset for which to swap flies
        swap_times ([type]): List of times (in seconds) at which to swap flies.
        flies1 (int or list/tuple, optional): Either a single value for all indices or a list with one value per item in indices. Defaults to 0.
        flies2 (int or list/tuple, optional): Either a single value for all indices or a list with one value per item in indices. Defaults to 1.

    Returns:
        dataset with swapped indices ()
    """
    if 'time' not in dataset:
        return

    for cnt, swap_time in enumerate(swap_times):
        if isinstance(flies1, (list, tuple)) and isinstance(flies2, (list, tuple)):
            fly1, fly2 = flies1[cnt], flies2[cnt]
        else:
            fly1, fly2 = flies1, flies2

        nearest_time = dataset.time.sel(time=swap_time, method="nearest")

        index = int(np.where(dataset.time == nearest_time)[0])
        if 'pose_positions_allo' in dataset:
            dataset.pose_positions_allo.values[index:, [fly2, fly1], ...] = dataset.pose_positions_allo.values[index:, [fly1, fly2], ...]
        if 'pose_positions' in dataset:
            dataset.pose_positions.values[index:, [fly2, fly1], ...] = dataset.pose_positions.values[index:, [fly1, fly2], ...]
        if 'body_positions' in dataset:
            dataset.body_positions.values[index:, [fly2, fly1], ...] = dataset.body_positions.values[index:, [fly1, fly2], ...]

    return dataset


def load_raw_song(filepath_daq: str,
                  song_channels: Optional[Sequence[int]] = None,
                  return_nonsong_channels: bool = False,
                  lazy: bool = False):
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
        f = h5py.File(filepath_daq, mode='r', rdcc_w0=0, rdcc_nbytes=100 * (1024**2), rdcc_nslots=50000)
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


def load_timestamps(filepath_timestamps):
    with h5py.File(filepath_timestamps, 'r') as f:
        cam_stamps = f['timeStamps'][:]

    # time stamps at idx 0 can be a little wonky - so use the information embedded in the image
    if cam_stamps.shape[1] > 2:  # time stamps from flycapture (old point grey API)
        shutter_times = cam_stamps[:, 1] + cam_stamps[:, 2] / 1_000_000  # time of "Shutter OFF"
    elif cam_stamps.shape[1] == 2:  # time stamps from other camera drivers (spinnaker (new point grey/FLIR API) and ximea)
        # cut off empty time stamps
        last_frame_idx = np.argmax(cam_stamps[:, 1] == 0) - 1
        cam_stamps = cam_stamps[:last_frame_idx]

        # fix jumps from overflow in timestamp counter for ximea cameras
        frame_intervals = np.diff(cam_stamps[:, 1])
        frame_interval_median = np.median(frame_intervals)
        idxs = np.where(frame_intervals < -10 * frame_interval_median)[0]
        while len(idxs):
            idx = idxs[0]
            df_wrong = cam_stamps[idx + 1, 1] - cam_stamps[idx, 1]
            df_inferred = cam_stamps[idx + 1, 0] - cam_stamps[idx, 0]
            cam_stamps[idx + 1:, 1] = cam_stamps[idx + 1:, 1] - df_wrong + df_inferred

            frame_intervals = np.diff(cam_stamps[:, 1])
            idxs = np.where(frame_intervals < -10 * frame_interval_median)[0]

        shutter_times = cam_stamps[:, 1]

    last_frame_idx = np.argmax(shutter_times == 0) - 1
    shutter_times = shutter_times[:last_frame_idx]
    return shutter_times


def load_times(filepath_timestamps, filepath_daq):
    """Load daq and cam time stamps, create muxer"""

    shutter_times = load_timestamps(filepath_timestamps)

    # DAQ time stamps
    with h5py.File(filepath_daq, 'r') as f:
        daq_stamps = f['systemtime'][:]
        daq_sampleinterval = f['samplenumber'][:]

    # remove trailing zeros - may be left over if recording didn't finish properly
    if 0 in daq_stamps:
        last_valid_idx = np.argmax(daq_stamps == 0)
    else:
        last_valid_idx = len(daq_stamps) - 1  # in case there are no trailing zeros
    daq_samplenumber = np.cumsum(daq_sampleinterval)[:last_valid_idx, np.newaxis]
    last_sample = daq_samplenumber[-1, 0]
    # seconds - using mode here to be more robust
    nb_seconds_per_interval, _ = scipy.stats.mode(np.diff(daq_stamps[:last_valid_idx, 0]))
    nb_seconds_per_interval = nb_seconds_per_interval[0]
    nb_samples_per_interval = np.mean(np.diff(daq_samplenumber[:last_valid_idx, 0]))
    sampling_rate_Hz = np.around(nb_samples_per_interval / nb_seconds_per_interval, -3)  # round to 1000s of Hz

    ss = SampStamp(sample_times=daq_stamps[:last_valid_idx, 0],
                   frame_times=shutter_times,
                   sample_numbers=daq_samplenumber[:, 0],
                   auto_monotonize=False)
    # # different refs:
    #
    # # first sample is 0 seconds
    # s0 = ss.sample_time(0)
    # ss = SampStamp(sample_times=daq_stamps[:, 0] - s0, frame_times=cam_stamps[:, 0] - s0, sample_numbers=daq_samplenumber[:, 0])
    #
    # # first frame is 0 seconds - for no-resample-video-data
    # f0 = ss.frame_time(0)
    # ss = SampStamp(sample_times=daq_stamps[:, 0] - f0, frame_times=cam_stamps[:, 0] - f0, sample_numbers=daq_samplenumber[:, 0])

    return ss, last_sample, sampling_rate_Hz


def load_movietimes(filepath_timestamps, filepath_daq):
    """Load daq and cam time stamps, create muxer"""
    df = pd.read_csv(filepath_timestamps)

    # DAQ time stamps
    with h5py.File(filepath_daq, 'r') as f:
        daq_stamps = f['systemtime'][:]
        daq_sampleinterval = f['samplenumber'][:]

    # remove trailing zeros - may be left over if recording didn't finish properly
    if 0 in daq_stamps:
        last_valid_idx = np.argmax(daq_stamps == 0)
    else:
        last_valid_idx = len(daq_stamps) - 1  # in case there are no trailing zeros
    daq_samplenumber = np.cumsum(daq_sampleinterval)[:last_valid_idx, np.newaxis]
    last_sample = daq_samplenumber[-1, 0]

    # seconds - using mode here to be more robust
    nb_seconds_per_interval, _ = scipy.stats.mode(np.diff(daq_stamps[:last_valid_idx, 0]))
    nb_seconds_per_interval = nb_seconds_per_interval[0]
    nb_samples_per_interval = np.mean(np.diff(daq_samplenumber[:last_valid_idx, 0]))
    sampling_rate_Hz = np.around(nb_samples_per_interval / nb_seconds_per_interval, -3)  # round to 1000s of Hz

    # ss = SampStamp(sample_times=daq_stamps[:last_valid_idx, 0], frame_times=shutter_times, sample_numbers=daq_samplenumber[:, 0], auto_monotonize=False)
    ss = SampStamp(sample_times=daq_stamps[:last_valid_idx, 0],
                   sample_numbers=daq_samplenumber[:, 0],
                   frame_samples=df['sample'],
                   frame_numbers=df['movie_frame'],
                   auto_monotonize=False)
    # # different refs:
    #
    # # first sample is 0 seconds
    # s0 = ss.sample_time(0)
    # ss = SampStamp(sample_times=daq_stamps[:, 0] - s0, frame_times=cam_stamps[:, 0] - s0, sample_numbers=daq_samplenumber[:, 0])
    #
    # # first frame is 0 seconds - for no-resample-video-data
    # f0 = ss.frame_time(0)
    # ss = SampStamp(sample_times=daq_stamps[:, 0] - f0, frame_times=cam_stamps[:, 0] - f0, sample_numbers=daq_samplenumber[:, 0])

    return ss, last_sample, sampling_rate_Hz


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
