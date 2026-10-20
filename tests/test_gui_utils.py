import numpy as np
import pytest

from xarray_behave.gui import utils
from xarray_behave.gui import views


def test_make_colors_returns_rgb_uint8_rows():
    colors = utils.make_colors(3)

    assert colors.shape == (3, 3)
    assert colors.dtype == np.uint8


def test_lookup_colormap_lut_falls_back_when_matplotlib_map_is_missing():
    lut = views._lookup_colormap_lut("turbo")

    assert lut.shape[1] in (3, 4)


@pytest.mark.parametrize("sample_index, expected", [(0, 0), (1, 0), (2, 1), (3, 2)])
def test_tracking_index_is_a_scalar_for_the_nearest_time(sample_index, expected):
    from types import SimpleNamespace
    import xarray as xr
    from xarray_behave.gui.app import PSV

    ds = xr.Dataset(coords={"sampletime": [0.0, 0.2, 0.7, 1.1], "time": [0.0, 0.5, 1.0]})
    window = SimpleNamespace(ds=ds, t0=sample_index, _sample_seconds=lambda sample: float(ds.sampletime.data[int(sample)]))

    index = PSV.index_other.fget(window)

    assert isinstance(index, int)
    assert index == expected


@pytest.mark.parametrize("times", [[10.0], [10.0, 10.001, 10.002], [10.0, 10.002, 10.009, 10.012], [10.0, 10.002, 10.002, 10.01]])
def test_audio_timestamp_conversion_matches_interpolation(times):
    from types import SimpleNamespace
    from xarray_behave.gui.app import PSV

    times = np.asarray(times)
    window = SimpleNamespace(_audio_time_values=lambda: times, fs_song=1000)
    indices = np.arange(len(times))
    for sample in [-1, 0, 0.25, 1, 1.5, len(times) - 1, len(times) + 1]:
        assert PSV._sample_seconds(window, sample) == pytest.approx(np.interp(sample, indices, times))
    for seconds in [9.0, 10.0, 10.001, 10.002, 10.005, times[-1], 11.0]:
        assert PSV._seconds_sample(window, seconds) == pytest.approx(np.interp(seconds, times, indices))


def test_audio_timestamp_conversion_without_coordinates():
    from types import SimpleNamespace
    from xarray_behave.gui.app import PSV

    window = SimpleNamespace(_audio_time_values=lambda: np.array([]), fs_song=1000)
    assert PSV._sample_seconds(window, 1.5) == 0.0015
    assert PSV._seconds_sample(window, 0.0015) == 1.5
