import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets
import logging
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
        qimg = QtGui.QImage(image.ctypes.data, image.shape[1], image.shape[0], image_format)
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


def detect_events(ds):
    """Transform ds.song_events into dict of event (on/offset) times.
    Args:
        ds ([xarray.Dataset]): dataset with song_events

    Returns:
        dict: with event times or segment on/offsets.
    """
    event_times = dict()
    ds.song_events.data = ds.song_events.data.astype(np.float)  # make sure this is non-bool so diff works
    event_names = ds.song_events.event_types.data
    event_categories = ds.song_events.event_categories.data
    logging.info('Extracting event times from song_events:')
    for event_idx, (event_name, event_category) in enumerate(zip(event_names, event_categories)):
        logging.info(f'   {event_name}')
        if event_category == 'event':
            event_times[event_name] = ds.song_events.time.where(ds.song_events[:, event_idx] == 1, drop=True).data.ravel()
        elif event_category == 'segment':
            onsets = ds.song_events.time.where(ds.song_events[:, event_idx].diff(dim='time') == 1, drop=True).data
            offsets = ds.song_events.time.where(ds.song_events[:, event_idx].diff(dim='time') == -1, drop=True).data
            if len(onsets) and len(offsets):
                # ensure onsets and offsets match
                offsets = offsets[offsets>np.min(onsets)]
                onsets = onsets[onsets<np.max(offsets)]
            if len(onsets) != len(offsets):
                print('Inconsistent segment onsets or offsets - ignoring all on- and offsets.')
                onsets = []
                offsets = []

            if not len(onsets) and not len(offsets):
                event_times[event_name] = np.zeros((0,2))
            else:
                event_times[event_name] = np.stack((onsets, offsets)).T
    return event_times


def eventtimes_to_traces(ds, event_times):
    """Convert dict of event (on/offset) times into song_events.
    Args:
        ds ([xarray.Dataset]): dataset with song_events
        event_times ([dict]): event times or segment on/offsets.

    Returns:
        xarray.Dataset
    """
    event_names = ds.song_events.event_types.data
    event_categories = ds.song_events.event_categories.data
    logging.info('Updating song_events from event_times:')
    for event_idx, (event_name, event_category) in enumerate(zip(event_names, event_categories)):
        logging.info(f'   {event_name}')
        ds.song_events.sel(event_types=event_name).data[:] = 0  # delete all events
        if event_category == 'event':
            times = ds.song_events.time.sel(time=event_times[event_name].ravel(), method='nearest').data
            # this is sloooooow
            for time in times:
                idx = np.where(ds.time==time)[0]
                ds.song_events[idx, event_idx] = 1
        elif event_category == 'segment':
            if event_times[event_name].shape[0] > 0:
                for onset, offset in zip(event_times[event_name][:, 0], event_times[event_name][:, 1]):
                    ds.song_events.sel(time=slice(onset, offset), event_types=event_name).data[:] = 1
    logging.info(f'Done.')
    return ds


def eventtimes_delete(eventtimes, which):
    return eventtimes

def eventtimes_add(eventtimes, which, resort=False):
    return eventtimes

def eventtimes_replace(eventtimes, which_old, which_new, resort=False):
    eventtimes = eventtimes_delete(eventtimes, which_old)
    eventtimes = eventtimes_add(eventtimes, which_new)
    return eventtimes

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
