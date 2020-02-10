import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem

from videoreader import VideoReader


def make_colors(nb_colors):
    colors = np.zeros((1, nb_colors, 3), np.uint8)
    colors[0, :, 1:] = 220  # set saturation and brightness to 220
    colors[0, :, 0] = np.arange(0, 180, 180.0 / nb_colors)  # set range of hues
    colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)[0].astype(np.uint8)[..., ::-1]
    return colors


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
        self.pixmapItem = QGraphicsPixmapItem()
        self.pixmapItem.setShapeMode(QGraphicsPixmapItem.BoundingRectShape)
        self.viewBox.addItem(self.pixmapItem)
        self.viewBox.setAspectLocked(lock=True, ratio=1)
        self.viewBox.disableAutoRange()

    def registerMouseClickEvent(self, func):
        """[summary]
        
        Args:
            func ([type]): [description]
        """
        self.pixmapItem.mouseClickEvent = func

    def setImage(self, image, image_format=QImage.Format_RGB888, auto_scale: bool = False):
        """[summary]
        
        Args:
            image ([type]): [description]
            image_format ([type], optional): [description]. Defaults to QImage.Format_RGB888.
            auto_scale (bool, optional): [description]. Defaults to False.
        """
        qimg = QImage(image.ctypes.data, image.shape[1], image.shape[0], image_format)
        qpix = QPixmap(qimg)
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


def detect_events(ds):
    event_times = dict()
    event_names = ds.song_events.event_types.data
    event_categories = ds.song_events.event_categories.data
    for event_idx, (event_name, event_category) in enumerate(zip(event_names, event_categories)):
        if event_category == 'event':
            event_times[event_name] = ds.time.where(ds.song_events[:, event_idx], drop=True).data
        elif event_category == 'segment':
            onsets = ds.time.where(ds.song_events[:, event_idx].diff(dim='time') == 1, drop=True).data
            offsets = ds.time.where(ds.song_events[:, event_idx].diff(dim='time') == -1, drop=True).data
            if len(onsets) and len(offsets):
                # ensure onsets and offsets match            
                offsets = offsets[offsets>np.min(onsets)]
                onsets = onsets[onsets<np.max(offsets)]
            if len(onsets) != len(offsets):
                print('Inconsistent segment onsets or offsets - ignoring all on- and offsets.')
                onsets = []
                offsets = []
            event_times[event_name] = np.stack((onsets, offsets)).T
    return event_times

def eventtimes_to_traces(ds, event_times):
    event_names = ds.song_events.event_types.data
    event_categories = ds.song_events.event_categories.data
    for event_idx, (event_name, event_category) in enumerate(zip(event_names, event_categories)):
        ds.song_events.sel(event_types=event_name).data[:] = 0  # delete all events
        if event_category == 'event':
            ds.song_events.sel(time=event_times[event_name], event_types=event_name).data[:] = 1
        elif event_category == 'segment':
            for onset, offset in zip(event_times[event_name][:,0], event_times[event_name][:,1]):
                ds.song_events.sel(time=slice(onset, offset), event_types=event_name).data[:] = 1
    return ds

    

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]