import numpy as np
import operator as op
from datetime import datetime
import scipy.interpolate


def ismonotonous(x, direction="increasing", strict=True):
    """Check if vector is monotonous.

    Args:
        x(np.ndarray)
        direction(str): 'increasing' or 'decreasing'
        strict(bool): defaults to True
    Returns:
        (bool)
    """
    allowed_directions = ["increasing", "decreasing"]
    if direction not in allowed_directions:
        raise ValueError(f'Direction "{direction}" must be in {allowed_directions}.')

    if direction == "decreasing":
        x = -x

    if strict:
        comp_op = op.gt  # >
    else:
        comp_op = op.ge  # >=

    return np.all(comp_op(x[1:], x[:-1]))


def monotonize(x, direction="increasing", strict=True):
    """Cut trailing non-monotonous values.

    Args:
        x
        direction - montonously 'increasing' (default) or 'decreasing'
        strict - strictly (default) or non-strictly monotonous
    Returns:
        truncated array
    """
    allowed_directions = ["increasing", "decreasing"]
    if direction not in allowed_directions:
        raise ValueError(f'Direction "{direction}" must be in {allowed_directions}.')

    if strict:
        comp_op = op.le  # >=
    else:
        comp_op = op.lt  # >

    if direction == "decreasing":
        comp = comp_op(x[:-1], x[1:])
    else:
        comp = comp_op(x[1:], x[:-1])

    if np.all(~comp):
        last_idx = len(comp) + 1
    else:
        last_idx = np.argmax(comp) + 1

    return x[:last_idx]


def interpolator(x, y, fill_value="extrapolate"):
    return scipy.interpolate.interp1d(x, y, fill_value=fill_value)


class SampleInterpolator:
    """For samples on a regular grid"""

    def __init__(self, sampling_rate: float):
        self.sampling_rate = float(sampling_rate)
        # dummy values for compatibililty with Interpolators from scipy.interpolate
        self.x = np.array([0.0, 1.0])
        self.y = self.x / self.sampling_rate

    def __call__(self, sample_numbers):
        return np.asarray(sample_numbers) / self.sampling_rate


def time_from_log(logfilename, line_number=1):
    """Parse time stamp from a specified lines in a log file.

    Args:
        logfilename(str)
        line_number(int): line in the log file from which parse the time stamp (defaults to 1 - will read the first line (not 0-indexed!))
    Returns:
        (datetime) time stamp
    """
    with open(logfilename, "r") as f:
        for _ in range(line_number):
            current_line = f.readline()

    current_line_parts = current_line.partition(" ")[0]
    return datetime.strptime(current_line_parts, "%Y-%m-%d,%H:%M:%S.%f")


def samplenumber_from_timestamps(target_time, timestamps, sample_at_timestamps=None):
    """Gets samplenumber from timestamps given time.

    Args:
        target_time (numpy.ndarrary): time of desired sample
        timestamps (numpy.ndarrary): list of timestamps
        sample_at_timestamps: can be provided for sparsely stamped data, the sample number for each timestamp
    Returns:
        samplenumber at target time (as an index, starts at 0, can be <0 if target is before first timestamps) (np.intp)
    """
    if not ismonotonous(timestamps, strict=True):
        raise ValueError(f"Timestamps must increase strictly monotonously.")

    if sample_at_timestamps is None:
        sample_at_timestamps = range(timestamps.shape[0])

    f = interpolator(timestamps, sample_at_timestamps)
    samplenumber = np.intp(np.round(f(target_time)))

    return samplenumber


def samplerange_from_timestamps(target_epoch, timestamps, sample_at_timestamps=None):
    """Gets range of samples from timestamps given a epoch defined by start and stop time.

    Args:
        target_epoch (numpy.ndarrary): start and stop time
        timestamps (numpy.ndarrary): list of timestamps
        sample_at_timestamps: can be provided for sparsely stamped data, the sample number for each timestamp
    Returns:
        range of samples spanning epoch (as an indices, starts at 0, can be <0 if targets extends to before first timestamps)
    """
    samplenumber_start = samplenumber_from_timestamps(target_epoch[0], timestamps, sample_at_timestamps)
    samplenumber_end = samplenumber_from_timestamps(target_epoch[1], timestamps, sample_at_timestamps)
    return range(samplenumber_start, samplenumber_end)


def timestamp_from_samplenumber(samplenumber, timestamps, sample_at_timestamps=None):
    """Gets samplenumber from timestamps given time.

    Args:
        samplenumber (numpy.ndarrary): sample number for which we want the time stamp
        timestamps (numpy.ndarrary): list of timestamps
        sample_at_timestamps: can be provided for sparsely stamped data, the sample number for each timestamp
    Returns:
        time stamp for that sample (float)
    """
    if not ismonotonous(timestamps, strict=True):
        raise ValueError(f"Timestamps must increase strictly monotonously.")

    if sample_at_timestamps is None:
        sample_at_timestamps = range(timestamps.shape[0])

    f = interpolator(sample_at_timestamps, timestamps)
    timestamp = f(samplenumber)

    return timestamp


def samples_from_samples(sample_in, in_stamps, in_samplenumber=None, out_stamps=None, out_samplenumber=None):
    """Convert between different sampling grids via a common clock.
    Args:
        sample_in: sample number in INPUT sampling grid
        in_stamps: time stamps of samples numbers `in_samplenumber` in OUTPUT sampling grid
        in_samplenumber: sample numbers for INPUT timestamps (defaults to None for densely stamped data)
        out_stamps: time stamps of samples numbers `out_samplenumber` in OUTPUT sampling grid
        out_samplenumber: sample numbers for INPUT timestamps (defaults to None for densely stamped data)
    Returns:
        sample number in the OUTPUT sampling grid corresponding to sample_in
    """
    time_in = timestamp_from_samplenumber(sample_in, in_stamps, in_samplenumber)
    sample_out = samplenumber_from_timestamps(time_in, out_stamps, out_samplenumber)
    return sample_out


def timestamp_from_cycles(cycleOffset, cycleSecs):
    times = cycleOffset + cycleSecs / 8000
    # correct counter overflows
    overflows = np.cumsum((np.diff(times) <= 0).astype(np.uint))
    overflows = np.insert(overflows, 0, 0)
    times = times + overflows * 128
    # offset such that first time is zero
    times = times - np.min(times)
    return times
