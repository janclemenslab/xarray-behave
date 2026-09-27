"""Load files created by various analysis programs created."""

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.ndimage
import scipy.signal
import scipy.stats
from scipy.ndimage import maximum_filter1d
from .io.samplestamps import SampStamp
from . import io


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
    b, a = scipy.signal.butter(6, (25, 1500), btype="bandpass", fs=sampling_rate)
    data = scipy.signal.filtfilt(b, a, data, axis=0, method="pad")
    # find loudest channel in 101-sample windows
    if filter_data:
        sng_max = maximum_filter1d(np.abs(data), size=101, axis=0)
        loudest_channel = np.argmax(sng_max, axis=-1)
    # get linear index and merge channels
    idx = np.ravel_multi_index((np.arange(sng_max.shape[0]), loudest_channel), data.shape)
    data_merged_max = data.ravel()[idx]
    data_merged_max = data_merged_max[:, np.newaxis]  # shape needs to be [nb_samples, 1]
    return data_merged_max


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
    if "time" not in dataset:
        return

    for cnt, swap_time in enumerate(swap_times):
        if isinstance(flies1, (list, tuple)) and isinstance(flies2, (list, tuple)):
            fly1, fly2 = flies1[cnt], flies2[cnt]
        else:
            fly1, fly2 = flies1, flies2

        nearest_time = dataset.time.sel(time=swap_time, method="nearest")

        index = int(np.where(dataset.time == nearest_time)[0])
        if "pose_positions_allo" in dataset:
            dataset.pose_positions_allo.values[index:, [fly2, fly1], ...] = dataset.pose_positions_allo.values[
                index:, [fly1, fly2], ...
            ]
        if "pose_positions" in dataset:
            dataset.pose_positions.values[index:, [fly2, fly1], ...] = dataset.pose_positions.values[index:, [fly1, fly2], ...]
        if "body_positions" in dataset:
            dataset.body_positions.values[index:, [fly2, fly1], ...] = dataset.body_positions.values[index:, [fly1, fly2], ...]

    return dataset


def load_times(filepath_timestamps, filepath_daq):
    """Load daq and cam time stamps, create muxer"""
    _, shutter_times = io.timestamps.CamStamps().load(filepath_timestamps)
    daq_samplenumber, daq_stamps = io.timestamps.DaqStamps().load(filepath_daq)

    last_sample = daq_samplenumber[-1]
    # seconds - using mode here to be more robust
    nb_seconds_per_interval, _ = scipy.stats.mode(np.diff(daq_stamps), keepdims=True)
    nb_seconds_per_interval = nb_seconds_per_interval[0]
    nb_samples_per_interval = np.mean(np.diff(daq_samplenumber))
    sampling_rate_Hz = np.around(nb_samples_per_interval / nb_seconds_per_interval, -3)  # round to 1000s of Hz

    ss = SampStamp(
        sample_times=daq_stamps,
        frame_times=shutter_times,
        sample_numbers=daq_samplenumber,
        auto_monotonize=False,
    )
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

    daq_samplenumber, daq_stamps = io.timestamps.DaqStamps().load(filepath_daq)
    last_sample = daq_samplenumber[-1]

    # seconds - using mode here to be more robust
    nb_seconds_per_interval, _ = scipy.stats.mode(np.diff(daq_stamps), keepdims=True)
    nb_seconds_per_interval = nb_seconds_per_interval[0]
    nb_samples_per_interval = np.mean(np.diff(daq_samplenumber))
    sampling_rate_Hz = np.around(nb_samples_per_interval / nb_seconds_per_interval, -3)  # round to 1000s of Hz

    ss = SampStamp(
        sample_times=daq_stamps,
        sample_numbers=daq_samplenumber,
        frame_samples=df["sample"],
        frame_numbers=df["movie_frame"],
        auto_monotonize=False,
    )

    return ss, last_sample, sampling_rate_Hz


def fix_keys(d):
    d_new = dict()
    # HACK zarr (or xarray) cuts off long string keys in event-types
    fix_dict = {"aggression_manu": "aggression_manual", "vibration_manua": "vibration_manual"}
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
