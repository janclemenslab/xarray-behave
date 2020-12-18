from typing import Iterable
import cv2
import numpy as np
import h5py
import colorcet
from typing import Iterable

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets
import logging
from videoreader import VideoReader
from typing import Union


def make_colors(nb_colors: int) -> Iterable:
    colors = []
    if nb_colors > 0:
        cmap = colorcet.cm['glasbey_light']
        # cmap = colorcet.cm['glasbey_bw_minc_20_minl_50']
        cmap = colorcet.cm['glasbey_bw_minc_20_minl_30']
        # ignore first (red)
        colors = (cmap(np.arange(1, nb_colors+1)) * 255)[:, :3].astype(np.uint8)
    return colors

# def make_colors(nb_colors):
#     colors = np.zeros((1, nb_colors, 3), np.uint8)
#     if nb_colors > 0:
#         colors[0, :, 1:] = 220  # set saturation and brightness to 220
#         colors[0, :, 0] = np.arange(0, 180, 180.0 / nb_colors)  # set range of hues
#         colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)[0].astype(np.uint8)[..., ::-1]
#     return colors


def fast_plot(plot_widget, x, y, pen=None):
    """[summary]

    Args:
        plot_widget ([type]): [description]
        x ([type]): [description]
        y ([type]): [description]
        pen ([type], optional): [description]. Defaults to None.
    """
    # as per: https://stackoverflow.com/questions/17103698/plotting-large-arrays-in-pyqtgraph
    # not worth it when plotting all channels but may speed up drawing events
    conn = np.ones_like(x, dtype=np.bool)
    conn[:, -1] = False  # make sure plots are disconnected
    path = pg.arrayToQPath(x.flatten(), y.flatten(), conn.flatten())
    item = QtGui.QGraphicsPathItem(path)
    if pen is None:
        pen = pg.mkPen(color=(196, 128, 128))
    item.setPen(pen)
    plot_widget.addItem(item)
    return item


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


class FastImageWidget(pg.GraphicsLayoutWidget):

    def __init__(self, *args, useOpenGL=True, **kwargs):
        """[summary]

        Args:
            useOpenGL (bool, optional): [description]. Defaults to True.
        """
        super(FastImageWidget, self).__init__(*args, **kwargs)
        self.viewBox = self.addViewBox()
        self.useOpenGL(useOpenGL)
        self.pixmapItem = QtWidgets.QGraphicsPixmapItem()
        self.pixmapItem.setShapeMode(QtWidgets.QGraphicsPixmapItem.BoundingRectShape)
        self.viewBox.addItem(self.pixmapItem)
        self.viewBox.setAspectLocked(lock=True, ratio=1)
        self.viewBox.disableAutoRange()

    def registerMouseClickEvent(self, func):
        """[summary]

        Args:
            func ([type]): [description]
        """
        self.pixmapItem.mouseClickEvent = func

    def setImage(self, image, image_format=QtGui.QImage.Format_RGB888, auto_scale: bool = False):
        """[summary]

        Args:
            image ([type]): [description]
            image_format ([type], optional): [description]. Defaults to QtGui.QImage.Format_RGB888.
            auto_scale (bool, optional): [description]. Defaults to False.
        """
        qimg = QtGui.QImage(image, image.shape[1], image.shape[0], image_format)
        # qimg = QtGui.QImage(image.ctypes.data, image.shape[1], image.shape[0], image_format)

        qpix = QtGui.QPixmap(qimg)
        self.pixmapItem.setPixmap(qpix)
        if auto_scale:
            self.fitImage(image.shape[1], image.shape[0])

    def fitImage(self, width: int, height: int):
        """[summary]

        Args:
            width ([type]): [description]
            height ([type]): [description]
        """
        self.viewBox.setRange(xRange=(0, width), yRange=(0, height), padding=0)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def allkeys(obj, keys=[]):
  """Recursively find all keys"""
  # from https://stackoverflow.com/questions/59897093/get-all-keys-and-its-hierarchy-in-h5-file-using-python-library-h5py
  keys.append(obj.name)
  if isinstance(obj, h5py.Group):
    for item in obj:
      if isinstance(obj[item], h5py.Group):
        allkeys(obj[item], keys)
      else: # isinstance(obj[item], h5py.Dataset):
        keys.append(obj[item].name)
  return keys


def is_sorted(array):
    return np.all(np.diff(array) >= 0)

#`values` should be sorted
def find_nearest_idx(array: np.array, values: Union[int, float, np.array]):
    """Find nearest index of each value from values in array

    from https://stackoverflow.com/a/46184652

    Args:
        array (np.array): array to search in
        values (Union[int, float, np.array]): query

    Returns:
        [type]: indices of entries in array closest to each value in values
    """

    # scalar query
    if isinstance(values, float) or isinstance(values, int):
        return (np.abs(array - values)).argmin()

    # make sure array is a numpy array
    array = np.array(array)
    if not is_sorted(array):
        array = np.sort(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
    idxs[prev_idx_is_less] -= 1
    return idxs
