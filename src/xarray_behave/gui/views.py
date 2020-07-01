try:
    import PySide2  # this will force pyqtgraph to use PySide instead of PyQt4/5
except ImportError:
    pass

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
# from PySide2.QtGui import QImage, QPixmap
import numpy as np
import skimage.draw
import scipy.signal
import time
import logging
from functools import partial
from typing import Tuple

from .. import xarray_behave as xb
from . import utils
from . import formbuilder
from . import app


# add io etc.
class Model():

    def __init__(self, ds):
        self.ds = ds

    def save(self, filename):
        xb.save(filename, self.ds)

    @classmethod
    def from_file(cls, filename):
        ds = xb.load(filename)
        return cls(ds)


class SegmentItem(pg.LinearRegionItem):

    def __init__(self, bounds: Tuple[float, float], event_index: int, xrange,
                 time_bounds: Tuple[float, float]=None, movable=True, pen=None, brush=None, **kwargs):
        """[summary]

        Args:
            bounds (Tuple[float, float]): (onset, offset) in seconds.
            event_index (int): for directly indexing event_types in ds.song_events
            xrange: prop from parent
            time_bounds (Tuple[float, float], optional): if axis coords is not seconds
            movable (bool, optional): [description]. Defaults to True.
            pen ([type], optional): [description]. Defaults to None.
            brush ([type], optional): [description]. Defaults to None.
            **kwargs passed to pg.LinearRegionItem
        """
        super().__init__(bounds, movable=movable, pen=pen, brush=brush, **kwargs)
        if time_bounds is None:
            self.bounds = bounds
        else:
            self.bounds = time_bounds
        self.event_index = event_index
        self.xrange = xrange
        # this corresponds to the undelrying only for TraveView, not for SpecView
        self._parent = self.getViewWidget


class EventItem(pg.InfiniteLine):

    def __init__(self, position: float, event_index: int, xrange,
                 time_pos=None, movable=True, pen=None, **kwargs):
        """[summary]

        Args:
            position (float): in axis coords).
            event_index (int): for directly indexing event_types in ds.song_events
            xrange: prop from parent
            time_pos (optional): if axis coords is not seconds
            movable (bool, optional): [description]. Defaults to True.
            pen ([type], optional): [description]. Defaults to None.
            brush ([type], optional): [description]. Defaults to None.
            **kwargs passed to pg.LinearRegionItem
        """
        super().__init__(pos=position, movable=movable, pen=pen, **kwargs)
        if time_pos is None:
            self.position = position
        else:
            self.position = time_pos
        self.event_index = event_index
        self.xrange = xrange
        # this corresponds to the undelrying only for TraveView, not for SpecView
        self._parent = self.getViewWidget


class Draggable(pg.GraphItem):
    """Draggable point cloud or graph."""
    def __init__(self, callback, acceptDrags=True):
        self.dragPoint = None
        self.dragOffset = None
        self.textItems = []
        self.acceptDrags = acceptDrags
        self.focalFly = None
        super().__init__()
        self.callback = callback

    def setData(self, **kwds):
        self.text = kwds.pop('text', [])
        self.data = kwds
        if 'pos' in self.data:
            if self.data['pos'] is None:
                self.data = {}
            else:
                npts = self.data['pos'].shape[0]
                self.data['data'] = np.empty(npts, dtype=[('index', int)])
                self.data['data']['index'] = np.arange(npts)
        self.setTexts(self.text)

        if 'acceptDrags' in kwds:
            self.acceptDrags = kwds['acceptDrags']

        if 'focalFly' in kwds:
            self.focalFly = kwds['focalFly']
        else:
            self.focalFly = None

        self.updateGraph()

    def setTexts(self, text):
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        for t in text:
            item = pg.TextItem(t)
            self.textItems.append(item)
            item.setParentItem(self)

    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        for i, item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])


    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton or not self.acceptDrags:
            ev.ignore()
            return

        if ev.isStart():
            # We are already one step into the drag.
            # Find the point(s) at the mouse cursor when the button was first
            # pressed:
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return

            if ev.modifiers() == QtCore.Qt.ControlModifier and self.focalFly is not None:
                inds = [pt.data()[0] for pt in pts]
                if self.focalFly not in inds:
                    logging.info(f'   Did not drag focal fly {self.focalFly} - dragged {inds}. Ignoring.')
                    ev.ignore()
                    return
                else:
                    logging.info(f'   Dragged focal fly {self.focalFly} - dragged {inds}. Moving.')
                    self.dragPoint = pts[inds.index(self.focalFly)]
                    ind = self.dragPoint.data()[0]
            else:
                self.dragPoint = pts[0]
                ind = self.dragPoint.data()[0]
            self.dragOffset = self.data['pos'][ind] - pos
        elif ev.isFinish():
            # when draggin is done, update the model
            ind = self.dragPoint.data()[0]
            drag_offset = self.dragOffset  # this should be the cumulative offset over the whole drag!!
            new_pos = ev.pos() + self.dragOffset
            self.callback(ind, new_pos, drag_offset)
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return
        # update data and fraph in view - that way we get faster visual feedback
        # w/o having to update the whole frame/view again
        ind = self.dragPoint.data()[0]
        self.data['pos'][ind] = ev.pos() + self.dragOffset
        self.updateGraph()
        ev.accept()


# the viewer is getting data from the Model (ds) but does not change it
# (so make ds-ref in View read-only via @property)
class TraceView(pg.PlotWidget):

    def __init__(self, model, callback):
        # additionally make names of trace and event arrays in ds args?
        super().__init__()
        self.setMouseEnabled(x=False, y=False)
        # this should be just a link/ref so changes in ds made by the controller will propagate
        # mabe make Model as thin wrapper around ds that also handles ion and use ref to Modle instance
        self.disableAutoRange()

        # leave enought space so axes are aligned aligned
        y_axis = self.getAxis('left')
        y_axis.setWidth(50)

        self._m = model
        self.callback = callback
        self.getPlotItem().mouseClickEvent = self._click

    @property
    def m(self):  # make read-only
        return self._m

    @property
    def trange(self):
        return None

    @property
    def xrange(self):
        return np.array(self.viewRange()[0])

    @property
    def yrange(self):
        return np.array(self.viewRange()[1])

    def update_trace(self):
        self.clear()
        if self.m.nb_channels is not None and self.m.show_all_channels and self.m.y_other is not None:
            for chan in range(self.m.nb_channels - 1):
                        self.addItem(pg.PlotCurveItem(self.m.x[::self.m.step],
                                                      self.m.y_other[::self.m.step, chan],
                                                      pen=pg.mkPen(color=[128, 128, 128])))
        # plot selected trace
        self.addItem(pg.PlotCurveItem(self.m.x[::self.m.step],
                                      np.array(self.m.y[::self.m.step])))
        self.autoRange(padding=0)
        # time of current frame in trace
        self.addItem(pg.InfiniteLine(movable=False, angle=90,
                                     pos=self.m.x[int(self.m.span / 2)],
                                     pen=pg.mkPen(color='r', width=1)))

    def add_segment(self, onset, offset, region_typeindex, brush=None, movable=True):
        region = SegmentItem((onset, offset), region_typeindex, self.xrange,
                             brush=brush, movable=movable)
        self.addItem(region)
        if movable:
            region.sigRegionChangeFinished.connect(self.m.on_region_change_finished)

    def add_event(self, xx, event_type, pen, movable=False):
        if not movable:
            xx = np.broadcast_to(xx[np.newaxis, :], (2, len(xx)))
            yy = np.zeros_like(xx) + self.yrange[:, np.newaxis]
            utils.fast_plot(self, xx.T, yy.T, pen)
        else:
            for x in xx:
                line = EventItem(x, event_type, self.xrange, movable=True, angle=90, pen=pen)
                line.sigPositionChangeFinished.connect(self.m.on_position_change_finished)
                self.addItem(line)

    def time_to_pos(self, time):
        return np.interp(time, self.m.trange, self.xrange)

    def pos_to_time(self, pos):
        return np.interp(pos, self.xrange, self.m.trange)

    def _click(self, event):
        event.accept()
        pos = event.pos()
        mouseT = self.getPlotItem().getViewBox().mapSceneToView(pos).x()
        self.callback(mouseT, event.button())


class SpecView(pg.ImageView):

    def __init__(self, model, callback, colormap=None):
        super().__init__(view=pg.PlotItem())
        self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.view.disableAutoRange()
        self.view.setMouseEnabled(x=False, y=False)

        # leave enought space so axes are aligned aligned
        y_axis = self.getView().getAxis('left')
        y_axis.setWidth(50)
        y_axis.setLabel('Frequency', units='Hz')
        y_axis.enableAutoSIPrefix()

        self._m = model
        self.callback = callback
        self.imageItem.mouseClickEvent = self._click

        if colormap is not None:
            self.imageItem.setLookupTable(colormap)  # apply the colormap
        self.old_items = []

    @property
    def m(self):  # read only access to the model
        return self._m

    @property
    def xrange(self):
        try:  # to prevent failure on init
            return np.array(self.view.viewRange()[0])
        except AttributeError:
            return None

    @property
    def yrange(self):
        try:  # to prevent failure on init
            return np.array(self.view.viewRange()[1])
        except AttributeError:
            return None

    def clear_annotations(self):
        [self.removeItem(item) for item in self.old_items]  # remove annotations

    def update_spec(self, x, y):
        # hash x to avoid re-calculation? only useful when annotating
        self.clear_annotations()

        S, f, t = self._calc_spec(y)
        self.setImage(S.T[:, ::-1])
        self.view.setLimits(xMin=0, xMax=S.shape[1], yMin=0, yMax=S.shape[0])

        f = f[::-1]
        y_axis = self.getView().getAxis('left')
        ticks = np.linspace(0, len(f)-1, 5, dtype=np.uintp)

        tick_label = []
        for ti in ticks:
            tick = f[ti]
            if tick > 10:
                tick = int(tick)
            else:
                tick = round(tick, 1)
            tick_label.append(str(tick))
        y_axis.setTicks([[(pos, label) for pos, label in zip(ticks, tick_label)]])

        x_axis = self.view.getAxis('bottom')
        ticks = np.linspace(0, len(t)-1, 10, dtype=np.uintp)
        t_offset = float(self.m.ds.sampletime[self.m.time0])
        x_axis.setTicks([[(ii, str(int((t_offset + t[ii])*self.m.fs_song)/self.m.fs_song)) for ii in ticks]])

    # @lru_cache(maxsize=2, typed=False)
    def _calc_spec(self, y):
        # signal.spectrogram will internally limit spec_win to len(y)
        # and will throw error since noverlap will then be too big
        self.spec_win = min(len(y), self.m.spec_win)
        f, t, psd = scipy.signal.spectrogram(y, self.m.fs_song, nperseg=self.spec_win,
                                             noverlap=self.spec_win // 2, nfft=self.spec_win * 4, mode='magnitude')
        self.spec_t = t

        if self.m.fmax is not None:
            f_idx1 = len(f) - 1 - np.argmax(f[::-1] <= self.m.fmax)
        else:
            f_idx1 = -1

        if self.m.fmin is not None:
            f_idx0 = np.argmax(f >= self.m.fmin)
        else:
            f_idx0 = 0

        S = np.log2(1 + psd[f_idx0:f_idx1, :])
        S = S / np.max(S) * 255  # normalize to 0...255
        return S, f[f_idx0:f_idx1], t

    def add_segment(self, onset, offset, region_typeindex, brush=None, movable=True):
        onset_spec, offset_spec = self.time_to_pos((onset, offset))
        region = SegmentItem((onset_spec, offset_spec), region_typeindex, self.xrange,
                             time_bounds=(onset, offset), brush=brush, movable=movable)
        self.addItem(region)
        self.old_items.append(region)
        if movable:
            region.sigRegionChangeFinished.connect(self.m.on_region_change_finished)

    def add_event(self, xx, event_type, pen, movable=False):
        xx0 = xx.copy()
        xx = self.time_to_pos(xx)
        if not movable:
            xx = np.broadcast_to(xx[np.newaxis, :], (2, len(xx)))
            yy = np.zeros_like(xx) + self.yrange[:, np.newaxis]
            self.old_items.append(utils.fast_plot(self, xx.T, yy.T, pen))
        else:
            for (x, x0) in zip(xx, xx0):
                line = EventItem(x, event_type, self.xrange, time_pos=x0, movable=True, angle=90, pen=pen)
                line.sigPositionChangeFinished.connect(self.m.on_position_change_finished)
                self.addItem(line)
                self.old_items.append(line)

    def time_to_pos(self, time):
        return np.interp(time, self.m.trange, self.xrange)

    def pos_to_time(self, pos):
        return np.interp(pos, self.xrange, self.m.trange)

    def _click(self, event):
        event.accept()
        pos = event.pos()[0]
        mouseT = self.pos_to_time(pos)
        self.callback(mouseT, event.button())


class MovieView(utils.FastImageWidget):

    def __init__(self, model, callback):
        super().__init__()
        self._m = model
        self.callback = callback
        self.registerMouseClickEvent(self._click)

        self.image_view_framenumber_text = pg.TextItem(color=(200, 0, 0), anchor=(-2, 1))
        self.viewBox.addItem(self.image_view_framenumber_text)

        self.fly_positions = Draggable(self.m.on_position_dragged, acceptDrags=not self.m.move_poses)
        self.viewBox.addItem(self.fly_positions)
        self.fly_poses = Draggable(self.m.on_poses_dragged, acceptDrags=self.m.move_poses)
        self.viewBox.addItem(self.fly_poses)

        self.brushes = []
        alpha = 96
        for dot_fly in range(self.m.nb_flies):
            self.brushes.append(pg.mkBrush(*self.m.fly_colors[dot_fly], alpha))

        self.pose_brushes = []
        for bodypart in range(self.m.nb_bodyparts):
            self.pose_brushes.append(pg.mkBrush(*self.m.bodypart_colors[bodypart], alpha))
        self.pose_brushes = self.pose_brushes * self.m.nb_flies

        self.fly_pens = []
        for dot_fly in range(self.m.nb_flies):
            for bodypart in range(self.m.nb_bodyparts):
                self.fly_pens.append(pg.mkBrush(*self.m.fly_colors[dot_fly], alpha))

        # build skeletons
        skeleton = np.array([[0, 1], [1, 8], [8, 11],  # body axis
                             [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7],  # legs
                             [8, 9], [8, 10]])  # wings
        self.skeletons = np.zeros((0, 2), dtype=np.uint)
        for dot_fly in range(self.m.nb_flies):
            self.skeletons = np.append(self.skeletons, self.m.nb_bodyparts * dot_fly + skeleton, axis=0)
        self.skeletons = self.skeletons.astype(np.uint)

    @property
    def m(self):  # read only access to the model
        return self._m

    def update_frame(self, ):
        frame = self.m.vr[self.m.framenumber]
        if frame is not None:  # frame is None when at end of video
            # # FIXME the annotations potentially waste time annotating outside of the cropped frame
            if 'pose_positions_allo' in self.m.ds:
                if self.m.show_poses:
                    frame = self.annotate_poses(frame)
                    self.fly_poses.setVisible(True)
                else:
                    self.fly_poses.setData(pos=None)
                    self.fly_poses.setVisible(False)

                if self.m.show_dot:
                    frame = self.annotate_dot(frame)
                else:
                    self.fly_positions.setData(pos=None)  # this deletes the graph

                if self.m.crop:
                    x_range, y_range = self.crop_frame(frame)
                else:
                    x_range, y_range = [0, self.m.vr.frame_width], [0, self.m.vr.frame_height]
            else:
                x_range, y_range = [0, self.m.vr.frame_width], [0, self.m.vr.frame_height]

            self.setImage(frame, auto_scale=True)
            self.viewBox.setRange(xRange=y_range, yRange=x_range)
            if self.m.show_framenumber:
                self.image_view_framenumber_text.setPlainText(f'frame {self.m.framenumber}')
            else:
                self.image_view_framenumber_text.setPlainText('')

    def annotate_dot(self, frame):
        # mark each fly with uniquely colored dots
        pos = np.array(self.m.ds.pose_positions_allo.data[self.m.index_other, :, self.m.thorax_index])
        self.fly_positions.setData(pos=pos[:,::-1],
                                   symbolBrush=self.brushes, pen=None, size=10,
                                   acceptDrags=not self.m.move_poses,
                                   focalFly=self.m.focal_fly)

        # mark *focal* and *other* fly with circle
        for this_fly, color in zip((self.m.focal_fly, self.m.other_fly), self.m.bodypart_colors[[2, 6]]):
            fly_pos = self.m.ds.pose_positions_allo.data[self.m.index_other,
                                                       this_fly,
                                                       self.m.thorax_index].astype(np.uintp)
            fly_pos = np.array(fly_pos)  # in case this is a dask.array
            # only plot circle if fly is within the frame (also prevents overflow errors
            # for tracking errors that lead to VERY large position values)
            if fly_pos[0] <= frame.shape[0] and fly_pos[1] <= frame.shape[1]:
                xx, yy = skimage.draw.circle_perimeter(fly_pos[0], fly_pos[1], self.m.circle_size, method='bresenham')
                frame[xx, yy, :] = color
        return frame

    def annotate_poses(self, frame):
        poses = np.array(self.m.ds.pose_positions_allo.data[self.m.index_other, :, :])
        poses = poses.reshape(-1, 2)  # flatten
        self.fly_poses.setData(pos=poses[:,::-1],
                               adj=self.skeletons,
                               symbolBrush=self.pose_brushes, size=6,
                               acceptDrags=self.m.move_poses)
        return frame

    def crop_frame(self, frame):
        fly_pos = self.m.ds.pose_positions_allo.data[self.m.index_other, self.m.focal_fly, self.m.thorax_index].astype(np.uintp)
        fly_pos = np.array(fly_pos)  # in case this is a dask.array
        # makes sure crop does not exceed frame bounds
        fly_pos[0] = np.clip(fly_pos[0], self.m.box_size, self.m.vr.frame_width - 1 - self.m.box_size)
        fly_pos[1] = np.clip(fly_pos[1], self.m.box_size, self.m.vr.frame_height - 1 - self.m.box_size)
        x_range = (int(fly_pos[0] - self.m.box_size), int(fly_pos[0] + self.m.box_size))
        y_range = (int(fly_pos[1] - self.m.box_size), int(fly_pos[1] + self.m.box_size))
        # frame = frame[slice(*x_range), slice(*y_range), :]  # now crop frame around the focal fly
        # return frame, x_range, y_range
        return x_range, y_range


    def _click(self, event):
        event.accept()
        pos = event.pos()
        mouseX, mouseY = int(pos.x()), int(pos.y())
        self.callback(mouseX, mouseY, event)
