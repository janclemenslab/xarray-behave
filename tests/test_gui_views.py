from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from xarray_behave.gui.views import MovieView


@pytest.mark.parametrize("position_name", ["pose_positions_allo", "body_positions"])
@pytest.mark.parametrize("position, expected", [
    ([200, 650], ((600, 700), (150, 250))),
    ([5, 995], ((899, 999), (0, 100))),
])
def test_movie_crop_uses_xy_display_ranges(position_name, position, expected):
    ds = xr.Dataset({position_name: (("time", "flies", "parts", "coords"), np.array([[[position]]]))})
    model = SimpleNamespace(ds=ds, index_other=0, focal_fly=0, pose_center_index=0, track_center_index=0,
                            box_size=50, vr=SimpleNamespace(frame_height=600, frame_width=1_000))
    view = SimpleNamespace(m=model)

    assert MovieView.crop_frame(view, None) == expected
    np.testing.assert_array_equal(ds[position_name].data[0, 0, 0], position)


def test_movie_full_frame_uses_width_for_x_and_height_for_y():
    class Reader:
        frame_height = 6
        frame_width = 10

        def __getitem__(self, index):
            return np.zeros((6, 10, 3), dtype=np.uint8)

    ranges = []
    model = SimpleNamespace(vr=Reader(), framenumber=0, ds={}, frame_fliplr=False, frame_flipud=False,
                            maintain_custom_crop=False)
    view = SimpleNamespace(m=model, setImage=lambda *args, **kwargs: None,
                           viewBox=SimpleNamespace(setRange=lambda **kwargs: ranges.append(kwargs)))

    MovieView.update_frame(view)

    assert ranges == [{"xRange": [0, 10], "yRange": [0, 6], "padding": 0}]
