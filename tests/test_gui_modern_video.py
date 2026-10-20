import sys
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from xarray_behave.gui.modern_video import PyAVVideoReader


def test_video_reader_enables_frame_threading(tmp_path, monkeypatch):
    stream = SimpleNamespace(type="video", average_rate=60, width=4, height=3, frames=100, thread_type=None)
    container = SimpleNamespace(streams=[stream], close=lambda: None)
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: container))

    reader = PyAVVideoReader(tmp_path / "video.mp4")

    assert stream.thread_type == "AUTO"
    reader.close()


@pytest.mark.parametrize("frame_order, expected, midpoint", [
    (list(range(12)), [0, 1, 1, 2, 3, 4, 5, 6, 7], 1),
    ([0, 3, 2, 4, 1, 7, 6, 8, 5, 11, 10, 9], [0, 0, 0, 3, 3, 4, 4, 7, 7], 0),
])
def test_video_reader_reuses_decoder_and_seeks_after_navigation(tmp_path, monkeypatch, frame_order, expected, midpoint):
    rgb_frames = [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(12)]
    frames = [
        SimpleNamespace(pts=index, time_base=Fraction(1, 4),
                        to_ndarray=lambda format, index=index: rgb_frames[index])
        for index in frame_order
    ]
    seeks = []
    decodes = []
    stream = SimpleNamespace(type="video", average_rate=4, time_base=Fraction(1, 4),
                             width=2, height=2, frames=len(frames), codec_context=SimpleNamespace(flush_buffers=lambda: None))
    def decode(stream):
        decodes.append(True)
        return iter(frames)
    container = SimpleNamespace(streams=[stream], seek=lambda *args, **kwargs: seeks.append(args),
                                decode=decode, close=lambda: None)
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: container))
    reader = PyAVVideoReader(tmp_path / "video.mp4")
    try:
        for index, value in zip([0, 1, 1, 2, 3, 4, 5, 6, 7], expected):
            assert np.all(reader[index] == value)
        assert len(seeks) == len(decodes) == 1
        reader[7][:] = 99  # Drawing tracking overlays must not change cached decoded frames.
        assert np.all(reader[7] == expected[-1])
        assert np.all(reader[1] == expected[1])
        assert len(seeks) == 2
        assert np.all(reader[11] == 11)
        assert len(seeks) == 3
        assert np.all(reader[11] == 11)
        assert len(seeks) == 3
        assert np.all(reader.frame_at_seconds(3) == 11)
        assert np.all(reader.frame_at_seconds(3.1) == 11)
        assert np.all(reader.frame_at_seconds(0.125) == midpoint)
        assert np.all(reader.frame_at_seconds(0.13) == midpoint)
        assert np.all(reader.frame_at_seconds(0.3) == expected[1])
    finally:
        reader.close()
