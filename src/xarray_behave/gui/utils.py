from typing import Iterable
import cv2
import numpy as np
import h5py
import colorcet
from typing import Iterable

import pyqtgraph as pg
from qtpy import QtGui, QtWidgets, QtCore
import logging
from videoreader import VideoReader
from typing import Union


def make_colors(nb_colors: int) -> Iterable:
    colors = []
    if nb_colors > 0:
        # cmap = colorcet.cm['glasbey_light']
        # cmap = colorcet.cm['glasbey_bw_minc_20_minl_50']
        cmap = colorcet.cm['glasbey_bw_minc_20_minl_30']
        # ignore first (red)
        colors = (cmap(np.arange(1, nb_colors + 1)) * 255)[:, :3].astype(np.uint8)
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
    item = QtWidgets.QGraphicsPathItem(path)
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


def allkeys(obj, keys=None):
    """Recursively find all keys"""
    # from https://stackoverflow.com/questions/59897093/get-all-keys-and-its-hierarchy-in-h5-file-using-python-library-h5py
    if keys is None:
        keys = []

    keys.append(obj.name)
    if isinstance(obj, h5py.Group):
        for item in obj:
            if isinstance(obj[item], h5py.Group):
                allkeys(obj[item], keys)
            else:  # isinstance(obj[item], h5py.Dataset):
                keys.append(obj[item].name)
    return keys


def is_sorted(array):
    return np.all(np.diff(array) >= 0)


def find_nearest_idx(array: np.array, values: Union[int, float, np.array]):
    """Find nearest index of each value from values in array

    from https://stackoverflow.com/a/46184652

    Args:
        array (np.array): array to search in
        values (Union[int, float, np.array]): query, should be sorted.

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
    prev_idx_is_less = (
        (idxs == len(array)) |
        (np.fabs(values - array[np.maximum(idxs - 1, 0)]) < np.fabs(values - array[np.minimum(idxs,
                                                                                              len(array) - 1)])))
    idxs[prev_idx_is_less] -= 1
    return idxs


class Worker(QtCore.QRunnable):
    """Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    Args:
    fn: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    args: Arguments to pass to the callback function
    kwargs: Keywords to pass to the callback function
    """

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @QtCore.Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        self.fn(*self.args, **self.kwargs)


class InvokeEvent(QtCore.QEvent):
    EVENT_TYPE = QtCore.QEvent.Type(QtCore.QEvent.registerEventType())

    def __init__(self, fn, *args, **kwargs):
        QtCore.QEvent.__init__(self, InvokeEvent.EVENT_TYPE)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


class Invoker(QtCore.QObject):

    def event(self, event):
        event.fn(*event.args, **event.kwargs)

        return True


_invoker = Invoker()


def invoke_in_main_thread(fn, *args, **kwargs):
    QtCore.QCoreApplication.postEvent(_invoker, InvokeEvent(fn, *args, **kwargs))


class CheckableComboBox(QtWidgets.QComboBox):
    # from https://gis.stackexchange.com/questions/350148/qcombobox-multiple-selection-pyqt5

    # Subclass Delegate to increase item height
    class Delegate(QtWidgets.QStyledItemDelegate):

        def sizeHint(self, option, index):
            size = super().sizeHint(option, index)
            size.setHeight(20)
            return size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make the combo editable to set a custom text, but readonly
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)

        # Use custom delegate
        self.setItemDelegate(CheckableComboBox.Delegate())

        # Update the text when an item is toggled
        self.model().dataChanged.connect(self.updateText)

        # Hide and show popup when clicking the line edit
        self.lineEdit().installEventFilter(self)
        self.closeOnLineEditClick = False

        # Prevent popup from closing when clicking on an item
        self.view().viewport().installEventFilter(self)

    def resizeEvent(self, event):
        # Recompute text to elide as needed
        self.updateText()
        super().resizeEvent(event)

    def eventFilter(self, object, event):

        if object == self.lineEdit():
            if event.type() == QtCore.QEvent.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return False

        if object == self.view().viewport():
            if event.type() == QtCore.QEvent.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                item = self.model().item(index.row())

                if item.checkState() == QtCore.Qt.Checked:
                    item.setCheckState(QtCore.Qt.Unchecked)
                else:
                    item.setCheckState(QtCore.Qt.Checked)
                return True
        return False

    def showPopup(self):
        super().showPopup()
        # When the popup is displayed, a click on the lineedit should close it
        self.closeOnLineEditClick = True

    def hidePopup(self):
        super().hidePopup()
        # Used to prevent immediate reopening when clicking on the lineEdit
        self.startTimer(100)
        # Refresh the display text when closing
        self.updateText()

    def timerEvent(self, event):
        # After timeout, kill timer, and reenable click on line edit
        self.killTimer(event.timerId())
        self.closeOnLineEditClick = False

    def updateText(self):
        texts = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == QtCore.Qt.Checked:
                texts.append(self.model().item(i).text())
        text = ", ".join(texts)

        # Compute elided text (with "...")
        metrics = QtGui.QFontMetrics(self.lineEdit().font())
        elidedText = metrics.elidedText(text, QtCore.Qt.ElideRight, self.lineEdit().width())
        self.lineEdit().setText(elidedText)

    def addItem(self, text, data=None):
        item = QtGui.QStandardItem()
        item.setText(text)
        if data is None:
            item.setData(text)
        else:
            item.setData(data)
        item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
        item.setData(QtCore.Qt.Unchecked, QtCore.Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts, datalist=None):
        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    def currentData(self):
        # Return the list of selected items data
        res = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == QtCore.Qt.Checked:
                res.append(self.model().item(i).data())
        return res
