"""Tools for converting between samples and time stamps.

glossary:
    densely stamped - each sample is consecutively time stamped
    sparsely stamped - only selected samples are time stamped - need to provide sample numbers
"""

from . import utils


class SampStamp:
    """Converts between frames and samples."""

    def __init__(
        self,
        sample_times,
        frame_times=None,
        sample_numbers=None,
        frame_numbers=None,
        frame_samples=None,
        sample_times_offset=0,
        frame_times_offset=0,
        auto_monotonize=True,
    ):
        """Get converter.

        Args:
            sample_times(np.ndarray)
            frame_times(np.ndarray)
            sample_number(np.ndarray)
            frame_number(np.ndarray)
            frame_samples
            sample_times_offset(float)
            frame_times_offset(float)
            auto_monotonize(bool)
        """
        # we want:
        # samples -> frames
        # samples -> times
        # frames -> samples
        # frames -> times

        # we need:
        # (samples, times) from DAQ, (frames, times) from video timestamps
        # (samples, times) from DAQ, (samples, frames) from movieframes
        # (samples, frames) from movieframes, (frames, times) from video timestamps (ignore edge case)

        # generate dense x_number arrays
        if sample_numbers is None and sample_times is not None:
            sample_numbers = range(sample_times.shape[0])
        if frame_numbers is None and frame_times is not None:
            frame_numbers = range(frame_times.shape[0])

        # correct for offsets
        sample_times += sample_times_offset
        if frame_times is not None:
            frame_times += frame_times_offset

        if auto_monotonize:
            if sample_times is not None:
                sample_times = utils.monotonize(sample_times)
                sample_numbers = sample_numbers[: sample_times.shape[0]]
            if frame_times is not None:
                frame_times = utils.monotonize(frame_times)
                frame_numbers = frame_numbers[: frame_times.shape[0]]
            if frame_samples is not None:
                frame_samples = utils.monotonize(frame_samples)
                # frame_numbers = frame_numbers[:frame_times.shape[0]]

        # get all interpolators for re-use
        if sample_numbers is not None and sample_times is not None:
            self.samples2times = utils.interpolator(sample_numbers, sample_times)
        if frame_times is None and frame_samples is not None:
            frame_times = self.samples2times(frame_samples)
            # self.frame_times = self.samples2times(frame_samples)

        if frame_numbers is not None and frame_times is not None:
            self.frames2times = utils.interpolator(frame_numbers, frame_times)
        if frame_times is not None and sample_numbers is not None:
            self.times2samples = utils.interpolator(sample_times, sample_numbers)
        if frame_times is not None and frame_numbers is not None:
            self.times2frames = utils.interpolator(frame_times, frame_numbers)

        if frame_samples is not None:
            self.samples2frames = utils.interpolator(frame_samples, frame_numbers, fill_value="extrapolate")

    def frame(self, sample):
        """Get frame number from sample number."""
        return self.times2frames(self.sample_time(sample))

    def sample(self, frame):
        """Get sample number from frame number."""
        return self.times2samples(self.frame_time(frame))

    def frame_time(self, frame):
        """Get time of frame number."""
        return self.frames2times(frame)

    def sample_time(self, sample):
        """Get time of sample number."""
        return self.samples2times(sample)


class SimpleStamp(SampStamp):
    """If all you need is conversion between sampling rates."""

    def __init__(self, sampling_rate: float):

        self.sampling_rate = sampling_rate
        self.samples2times = utils.SampleInterpolator(self.sampling_rate)
        self.times2samples = utils.SampleInterpolator(1 / self.sampling_rate)

        self.frames2times = self.samples2times
        self.times2frames = self.times2samples
