try:
    import PySide2  # this will force pyqtgraph to use PySide instead of PyQt4/5
except ImportError:
    pass

import pyqtgraph as pg
from qtpy import QtGui, QtCore, QtWidgets
import numpy as np
import skimage.draw
import scipy.signal
import time
import logging
from functools import partial, lru_cache
from typing import Tuple

from .. import xarray_behave as xb
from . import utils


logger = logging.getLogger(__name__)


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
                 time_bounds: Tuple[float, float] = None, movable=True,
                 pen=None, brush=None, **kwargs):
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
        # this corresponds to the underlying view only for TraceView, not for SpecView
        self._parent = self.getViewWidget


class EventItem(pg.InfiniteLine):

    def __init__(self, position: float, event_index: int, xrange,
                 time_pos=None, movable=True,
                 pen=None, **kwargs):
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
        # this corresponds to the undelrying only for TraceView, not for SpecView
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
                    logger.info(f'   Did not drag focal fly {self.focalFly} - dragged {inds}. Ignoring.')
                    ev.ignore()
                    return
                else:
                    logger.info(f'   Dragged focal fly {self.focalFly} - dragged {inds}. Moving.')
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
        self.enableAutoRange(False, False)

        # leave enought space so axes are aligned aligned
        y_axis = self.getAxis('left')
        y_axis.setWidth(50)

        self._m = model
        self.callback = callback
        self.getPlotItem().mouseClickEvent = self._click

        self.threshold_line = pg.InfiniteLine(movable=True, angle=0,
                                pos=0,
                                pen=pg.mkPen(color='r', width=2, alpha=0.25),
                                bounds=[0, None],
                                label='Threshold',
                                labelOpts={'position': 0.9})

        self.annotation_items = []

    @property
    def m(self):  # read-only property
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

    @property
    def threshold(self):
        if self.threshold_line is not None:
            return self.threshold_line.value()

    def update_trace(self):
        self.clear()

        # draw background channels
        if self.m.nb_channels is not None and self.m.show_all_channels and self.m.y_other is not None:
            y_other = self.m.y_other[::int(self.m.step*2), :]
            x_other = self.m.x[::int(self.m.step * 2)]
            for chan in range(self.m.nb_channels - 1):
                item = pg.PlotCurveItem(x_other, y_other[:, chan], pen=pg.mkPen(color=[127, 127, 127]))
                self.addItem(item)

        # plot active channel
        x = self.m.x[::self.m.step]
        item = pg.PlotCurveItem(x=x, y=np.array(self.m.y[::self.m.step], dtype=np.float), pen=pg.mkPen(color=[220, 220, 220], width=1))
        self.addItem(item)

        self.autoRange(padding=0)

        # time of current frame in trace
        pos_line = pg.InfiniteLine(self.m.x[int(self.m.span / 2)], movable=False, angle=90,
                                   pen=pg.mkPen(color='r', width=1))
        self.addItem(pos_line)

        # draw actice envelope and threshold
        if self.m.threshold_mode:
            self.addItem(self.threshold_line)
            env_line = pg.PlotCurveItem(x=x, y=self.m.envelope[::self.m.step], pen=pg.mkPen(color=[196, 98, 98], width=1))
            self.addItem(env_line)


    def add_segment(self, onset, offset, region_typeindex, brush=None, pen=None, movable=True, text=None):
        region = SegmentItem((onset, offset), region_typeindex, self.xrange,
                             brush=brush, pen=pen, movable=movable)
        if text is not None:
            pg.InfLineLabel(region.lines[1], text, position=0.95, rotateAxis=(1,0), anchor=(1, 1))

        self.addItem(region)
        if movable:
            region.sigRegionChangeFinished.connect(self.m.on_region_change_finished)

    def add_event(self, xx, event_type, pen, movable=False, text=None):
        if not len(xx):
            return

        for x in xx:
            line = EventItem(x, event_type, self.xrange, movable=True, angle=90, pen=pen)
            if text is not None:
                pg.InfLineLabel(line, text, position=0.95, rotateAxis=(1,0), anchor=(1, 1))
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


class TrackView(TraceView):

    def update_trace(self):
        self.clear()

        # draw individ channels
        cols = self.m.bodypart_colors[self.m.track_sel_names]
        for ii in range(self.m.y_tracks.shape[1]):
            item = pg.PlotCurveItem(self.m.x_tracks, self.m.y_tracks[:, ii], pen=pg.mkPen(color=cols[ii]))
            self.addItem(item)
        self.autoRange(padding=0)

        # time of current frame in trace
        pos_line = pg.InfiniteLine(self.m.x_tracks[int(self.m.span / self.m.fs_ratio / 2)], movable=False, angle=90,
                                   pen=pg.mkPen(color='r', width=1))
        self.addItem(pos_line)


class SpecView(pg.ImageView):

    def __init__(self, model, callback, colormap='turbo'):
        super().__init__(view=pg.PlotItem())
        self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.view.disableAutoRange()
        self.view.setAspectLocked(False)
        self.view.getViewBox().invertY(False)
        self.view.setMouseEnabled(x=False, y=False)

        # leave enough space so axes are aligned aligned
        y_axis = self.getView().getAxis('left')
        y_axis.setWidth(50)
        y_axis.setLabel('Frequency', units='Hz')
        y_axis.enableAutoSIPrefix()

        self._m = model
        self.callback = callback
        self.imageItem.mouseClickEvent = self._click
        self.t_step = 1
        self.max_pix = 6_000

        self.pos_line = pg.InfiniteLine(pos=0.5, movable=False, angle=90,
                                        pen=pg.mkPen(color='r', width=1))
        self.addItem(self.pos_line)

        cmap = pg.colormap.getFromMatplotlib(colormap)
        lut = cmap.getLookupTable()
        self.imageItem.setLookupTable(lut)  # apply the colormap
        self.old_items = []

    @property
    def m(self):  # read only access to the model
        return self._m

    @property
    def xrange(self):
        try:  # to prevent failure on init
            return (0, self.S.shape[1]) #np.array(self.view.viewRange()[0])
        except AttributeError:
            return None

    @property
    def yrange(self):
        try:  # to prevent failure on init
            return np.array(self.view.viewRange()[1])
        except AttributeError:
            return None

    def clear_annotations(self):
        [self.removeItem(item) for item in self.old_items]  # remove annotations <- slowest part of update_spec!!!

    def update_spec(self, x, y):
        self.S, f, t = self._calc_spec(tuple(y), self.m.spec_win)  # tuple-ify for caching
        trange = (self.m.x[-1] - self.m.x[0])
        self.max_pix = 6_000
        self.t_step = max(1, self.S.shape[1] // self.max_pix)
        self.setImage(self.S.T[::self.t_step], autoRange=False, scale=[trange / len(t) * self.t_step, (f[-1] - f[0]) / len(f)], pos=[self.m.x[0], f[0]])
        self.view.setRange(xRange=self.m.x[[0, -1]], yRange=(f[0], f[-1]), padding=0)
        self.pos_line.setValue(self.m.x[int(self.m.span / 2)])

    @lru_cache(maxsize=4, typed=False)
    def _calc_spec(self, y, spec_win):
        y = np.array(y)
        # signal.spectrogram will internally limit spec_win to len(y)
        # and will throw error since noverlap will then be too big
        spec_win = min(len(y), spec_win)

        # ranges between 1 and 4 times spec_win - smaller when zoomed out
        # to accelerate calc
        nfft = spec_win * max(1, 4 // self.t_step)

        f, t, psd = scipy.signal.spectrogram(y, self.m.fs_song, nperseg=spec_win,
                                             noverlap=spec_win // 2, nfft=nfft, mode='magnitude')

        # select freq limits
        f_idx0 = 0
        if self.m.fmin is not None:
            f_idx0 = np.argmax(f >= self.m.fmin)
        f_idx1 = -1
        if self.m.fmax is not None:
            f_idx1 = len(f) - 1 - np.argmax(f[::-1] <= self.m.fmax)

        S = np.log2(1 + psd[f_idx0:f_idx1, :])
        S = S / np.max(S) * 255  # normalize to 0...255
        return S, f[f_idx0:f_idx1], t

    def add_segment(self, onset, offset, region_typeindex, brush=None, pen=None, movable=True, text=None):
        region = SegmentItem((onset, offset), region_typeindex, self.xrange,
                             time_bounds=(onset, offset), brush=brush, pen=pen, movable=movable)
        if text is not None:
            pg.InfLineLabel(region.lines[1], text, position=0.95, rotateAxis=(1,0), anchor=(1, 1))

        self.addItem(region)
        self.old_items.append(region)
        if movable:
            region.sigRegionChangeFinished.connect(self.m.on_region_change_finished)

    def add_event(self, xx, event_type, pen, movable=False, text=None):
        if not len(xx):
            return

        for x in xx:
            line = EventItem(x, event_type, self.xrange, time_pos=x, movable=True, angle=90, pen=pen)
            if text is not None:
                pg.InfLineLabel(line, text, position=0.95, rotateAxis=(1,0), anchor=(1, 1))
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
        mouseT = self.pos_to_time(pos) * self.t_step
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
        if 'pose_positions' in self._m.ds:
            skeleton = self._m.ds.pose_positions.attrs['skeleton']
        else:
            skeleton = np.zeros((0, 2), dtype=np.uint)
        # skeleton = np.array([[0, 1], [1, 8], [8, 11],  # body axis
        #                      [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7],  # legs
        #                      [8, 9], [8, 10]])  # wings
        self.skeletons = np.zeros((0, 2), dtype=np.uint)
        for dot_fly in range(self.m.nb_flies):
            self.skeletons = np.append(self.skeletons, self.m.nb_bodyparts * dot_fly + skeleton, axis=0)
        self.skeletons = self.skeletons.astype(np.uint)

    @property
    def m(self):  # read only access to the model
        return self._m

    def update_frame(self, ):
        frame = self.m.vr[self.m.framenumber]

        if self.m.frame_fliplr:
            frame = np.ascontiguousarray(frame[:, ::-1])

        if self.m.frame_flipud:
            frame = np.ascontiguousarray(frame[::-1, :])

        if frame is not None:  # frame is None when at end of video
            # # FIXME the annotations potentially waste time annotating outside of the cropped frame
            if 'pose_positions_allo' in self.m.ds:
                if self.m.show_poses:
                    frame = self.annotate_poses(frame)
                    self.fly_poses.setVisible(True)
                else:
                    self.fly_poses.setData(pos=None)
                    self.fly_poses.setVisible(False)

            if 'pose_positions_allo' in self.m.ds or 'body_positions' in self.m.ds:
                if self.m.show_dot:
                    frame = self.annotate_dot(frame)
                else:
                    self.fly_positions.setData(pos=None)  # this deletes the graph

                if self.m.crop:
                    x_range, y_range = self.crop_frame(frame)
                else:
                    x_range, y_range = [0, self.m.vr.frame_height], [0, self.m.vr.frame_width]
            else:
                x_range, y_range = [0, self.m.vr.frame_height], [0, self.m.vr.frame_width]

            self.setImage(frame, auto_scale=True)
            self.viewBox.setRange(xRange=y_range, yRange=x_range)

    def annotate_dot(self, frame):
        # mark each fly with uniquely colored dots
        if 'pose_positions_allo' in self.m.ds:
            pos = np.array(self.m.ds.pose_positions_allo.data[self.m.index_other, :, self.m.pose_center_index])
            cols = self.m.fly_colors
        elif 'body_positions' in self.m.ds:
            pos = np.array(self.m.ds.body_positions.data[self.m.index_other, :, self.m.track_center_index])
            cols = self.m.fly_colors

        self.fly_positions.setData(pos=pos[:,::-1],
                                   symbolBrush=self.brushes, pen=None, size=10,
                                   acceptDrags=not self.m.move_poses,
                                   focalFly=self.m.focal_fly)

        # mark *focal* and *other* fly with circle
        for this_fly, color in zip((self.m.focal_fly, self.m.other_fly), cols):
            if 'pose_positions_allo' in self.m.ds:
                fly_pos = self.m.ds.pose_positions_allo.data[self.m.index_other, this_fly, self.m.pose_center_index]
            elif 'body_positions' in self.m.ds:
                fly_pos = self.m.ds.body_positions.data[self.m.index_other, this_fly, self.m.track_center_index]

            fly_pos = np.array(fly_pos).astype(np.uintp)  # in case this is a dask.array
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
        if 'pose_positions_allo' in self.m.ds:
            fly_pos = self.m.ds.pose_positions_allo.data[self.m.index_other, self.m.focal_fly, self.m.pose_center_index]
        elif 'body_positions' in self.m.ds:
            fly_pos = self.m.ds.body_positions.data[self.m.index_other, self.m.focal_fly, self.m.track_center_index]
        fly_pos = np.array(fly_pos).astype(np.uintp)  # in case this is a dask.array
        # makes sure crop does not exceed frame bounds
        fly_pos[0] = np.clip(fly_pos[0], self.m.box_size, self.m.vr.frame_height - 1 - self.m.box_size)
        fly_pos[1] = np.clip(fly_pos[1], self.m.box_size, self.m.vr.frame_width - 1 - self.m.box_size)
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
