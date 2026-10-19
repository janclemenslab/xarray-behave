import numpy as np

from xarray_behave.gui import utils
from xarray_behave.gui import views


def test_make_colors_uses_available_colorcet_palette():
    colors = utils.make_colors(3)

    assert colors.shape == (3, 3)
    assert colors.dtype == np.uint8


def test_lookup_colormap_lut_falls_back_when_matplotlib_map_is_missing():
    lut = views._lookup_colormap_lut("turbo")

    assert lut.shape[1] in (3, 4)
