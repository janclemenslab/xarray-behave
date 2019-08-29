import cv2
import numpy as np
import pyqtgraph as pg
from videoreader import VideoReader


def make_colors(nb_flies):
    colors = np.zeros((1, nb_flies, 3), np.uint8)
    colors[0, :, 1:] = 220  # set saturation and brightness to 220
    colors[0, :, 0] = np.arange(0, 180, 180.0 / nb_flies)  # set range of hues
    colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)[0].astype(np.uint8)[..., ::-1]
    return colors


def _autoupdate(func):
    def call_and_update(self):
        func(self)
        if self.STOP:  # avoid unecessary update if playback is running
            self.update()
    return call_and_update


class VideoReaderNP(VideoReader):
    """VideoReader posing as numpy array."""

    def __getitem__(self, index):
        return self.read(index)[1]

    @property
    def dtype(self):
        return np.uint8

    @property
    def shape(self):
        return (self.number_of_frames, *self.frame_shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.product(self.shape)

    def min(self):
        return 0

    def max(self):
        return 255

    def transpose(self, *args):
        return self


class ImageViewVR(pg.ImageView):

    def quickMinMax(self, data):
        """Dummy min/max for numpy videoreader. The original function tries to read the full video!"""
        return 0, 255
