from typing import Iterable
import numpy as np

import pyqtgraph as pg
from qtpy import QtGui, QtWidgets, QtCore


def make_colors(nb_colors: int) -> Iterable:
    if nb_colors <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.asarray(
        [pg.intColor(index, hues=nb_colors).getRgb()[:3] for index in range(nb_colors)],
        dtype=np.uint8,
    )


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
        qimg = QtGui.QImage(image, image.shape[1], image.shape[0], 3 * image.shape[1], image_format)

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


def find_nearest_idx(array, values):
    """Find nearest index for scalar or array values in a sorted array."""
    array = np.asarray(array)
    values = np.asarray(values)
    idxs = np.searchsorted(array, values, side="left")
    idxs = np.clip(idxs, 0, len(array) - 1)
    prev_idxs = np.clip(idxs - 1, 0, len(array) - 1)
    use_prev = np.abs(values - array[prev_idxs]) < np.abs(values - array[idxs])
    idxs = np.where(use_prev, prev_idxs, idxs)
    if values.ndim == 0:
        return int(idxs)
    return idxs


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
