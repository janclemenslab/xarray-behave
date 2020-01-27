import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import skimage.draw
import scipy.signal
import time
import logging
from functools import partial

from .. import xarray_behave as xb
from .. import _ui_utils

# add io etc. 
class Model():

    def __init__(self, ds):
        self.m.ds = ds
       

    def save(self, filename):
        xb.save(filename, ds)
    
    @classmethod
    def from_file(cls, filename):
        ds = xb.load(filename)
        return cls(ds)

    
# the viewer is getting data from the Model (ds) but does not change it 
# (so make ds-ref in View read-only via @property)
class TraceView(pg.PlotWidget):

    def __init__(self, model, callback):
        # additionally make names of trace and event arrays in ds args?
        super().__init__(name="song")
        # this should be just a link/ref so changes in ds made by the controller will propagate
        # mabe make Model as thin wrapper around ds that also handles ion and use ref to Modle instance
        self.disableAutoRange()
        
        self._m = model
        self.callback = callback
        self.getPlotItem().mouseClickEvent = self._click

    @property
    def m(self):  # make read-only
        return self._m

    @property
    def xrange(self):
        return np.array(self.viewRange()[0])

    @property
    def yrange(self):
        return np.array(self.viewRange()[1])

    def update_trace(self):
        self.clear()
        if self.m.show_all_channels and self.m.y_other is not None:
            for chan in range(self.m.nb_channels - 1):
                        self.addItem(pg.PlotCurveItem(self.m.x[::self.m.step],
                                                      self.m.y_other[::self.m.step, chan]))
        # plot selected trace
        self.addItem(pg.PlotCurveItem(self.m.x[::self.m.step], 
                                      np.array(self.m.y[::self.m.step])))
        self.autoRange(padding=0)                    
        # time of current frame in tracec
        self.addItem(pg.InfiniteLine(movable=False, angle=90,
                                     pos=self.m.x[int(self.m.span / 2)],
                                     pen=pg.mkPen(color='r', width=1)))

    def add_segment(self, onset, offset, brush, movable=True):
        region = pg.LinearRegionItem(values=(onset, offset), movable=movable, brush=brush)
        region.original_region = (onset, offset)
        self.addItem(region)
        if movable:
            region.sigRegionChangeFinished.connect(partial(self.m.on_region_change_finished, self.xrange))
        
    def add_event(self, xx, event_type, pen, movable=False):        
        if not movable:
            xx = np.broadcast_to(xx[np.newaxis, :], (2, len(xx)))
            yy = np.zeros_like(xx) + self.yrange[:, np.newaxis]
            _ui_utils.fast_plot(self, xx.T, yy.T, pen)
        else:
            for x in xx:
                line = pg.InfiniteLine(movable=True, angle=90,
                                       pos=x, pen=pen)
                line.original_position = x
                line.event_type = event_type
                line.sigPositionChangeFinished.connect(partial(self.m.on_position_change_finished, self.xrange))
                self.addItem(line)

    def time_to_pos(self, time):
        return np.interp(time, self.m.trange, self.xrange) 

    def pos_to_time(self, pos):
        return np.interp(pos, self.xrange, self.m.trange)

    def _click(self, event):
        event.accept()
        pos = event.pos()
        mouseT = self.getPlotItem().getViewBox().mapSceneToView(pos).x()
        self.callback(mouseT)   


class SpecView(pg.ImageView):

    def __init__(self, model, callback, colormap=None):
        super().__init__(view=pg.PlotItem())
        self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.view.disableAutoRange()
        
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

        # FIXME: long yticklabels (e.g "5000") lead to misalignment between spec and trace plots
        f = f[::-1]
        # y_axis = self.getView().getAxis('left')
        # ticks = np.linspace(0, len(f)-1, 5, dtype=np.uintp)
        # y_axis.setTicks([[(ii, str(f[ii])) for ii in ticks]])

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
            f_idx1 = np.argmax(f > self.m.fmax)
        else:
            f_idx1 = -1
        
        if self.m.fmin is not None:
            f_idx0 = np.argmax(f > self.m.fmin)
        else:
            f_idx0 = 0

        S = np.log2(1 + psd[f_idx0:f_idx1, :])
        S = S / np.max(S) * 255  # normalize to 0...255
        return S, f[f_idx0:f_idx1], t

    def add_segment(self, onset, offset, brush, movable=True):
        onset_spec, offset_spec = self.time_to_pos((onset, offset)) 
        region = pg.LinearRegionItem(values=(onset_spec, offset_spec), movable=movable, brush=brush)
        region.original_region = (onset, offset)
        if movable:
            region.sigRegionChangeFinished.connect(partial(self.m.on_region_change_finished, self.xrange))
        self.addItem(region)
        self.old_items.append(region)

    def add_event(self, xx, event_type, pen, movable=False):
        xx0 = xx.copy()
        xx = self.time_to_pos(xx)
        if not movable:
            xx = np.broadcast_to(xx[np.newaxis, :], (2, len(xx)))
            yy = np.zeros_like(xx) + self.yrange[:, np.newaxis]
            self.old_items.append(_ui_utils.fast_plot(self, xx.T, yy.T, pen))
        else:
            for (x, x0) in zip(xx, xx0):
                line = pg.InfiniteLine(movable=True, angle=90,
                                       pos=x, pen=pen)
                line.original_position = x0
                line.event_type = event_type
                line.sigPositionChangeFinished.connect(partial(self.m.on_position_change_finished, self.xrange))
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
        self.callback(mouseT)


class MovieView(_ui_utils.FastImageWidget):

    def __init__(self, model, callback):
        super().__init__()
        self._m = model
        self.callback = callback
        self.registerMouseClickEvent(self._click)
        self.image_view_framenumber_text = pg.TextItem(color=(200, 0, 0), anchor=(-2, 1))
        self.viewBox.addItem(self.image_view_framenumber_text)
        
    @property
    def m(self):  # read only access to the model
        return self._m

    def update_frame(self, ):
        self.m.fn = self.m.ds.coords['nearest_frame'][self.m.index_other].data

        frame = self.m.vr[self.m.fn]
        if frame is not None:  # frame is None when at end of video
            # # FIXME the annotations potentially waste time annotating outside of the cropped frame
            if 'pose_positions_allo' in self.m.ds:
                if self.m.show_poses:
                    frame = self.annotate_poses(frame)
                if self.m.show_dot:
                    frame = self.annotate_dot(frame)
                if self.m.crop:
                    frame = np.ascontiguousarray(self.crop_frame(frame))

            self.setImage(frame, auto_scale=True)
            if self.m.show_framenumber:                    
                self.image_view_framenumber_text.setPlainText(f'frame {self.m.fn}')
            else:
                self.image_view_framenumber_text.setPlainText('')

    def annotate_dot(self, frame):
        # mark each fly with uniquely colored dots
        for dot_fly in range(self.m.nb_flies):
            fly_pos = self.m.ds.pose_positions_allo.data[self.m.index_other, dot_fly, self.m.thorax_index]
            x_dot = np.clip((fly_pos[0] - self.m.dot_size, fly_pos[0] + self.m.dot_size),
                             0, self.m.vr.frame_width - 1).astype(np.uintp)
            y_dot = np.clip((fly_pos[1] - self.m.dot_size, fly_pos[1] + self.m.dot_size),
                             0, self.m.vr.frame_height - 1).astype(np.uintp)
            frame[slice(*x_dot), slice(*y_dot), :] = self.m.fly_colors[dot_fly]  # set pixels around

        # mark *focal* and *other* fly with circle
        for this_fly, color in zip((self.m.focal_fly, self.m.other_fly), self.m.bodypart_colors[[2, 6]]):
            fly_pos = self.m.ds.pose_positions_allo.data[self.m.index_other,
                                                       this_fly,
                                                       self.m.thorax_index].astype(np.uintp)
            fly_pos = np.array(fly_pos)  # in case this is a dask.array
            xx, yy = skimage.draw.circle_perimeter(fly_pos[0], fly_pos[1], self.m.circle_size, method='bresenham')
            frame[xx, yy, :] = color
        return frame

    def annotate_poses(self, frame):
        for dot_fly in range(self.m.nb_flies):
            fly_pos = self.m.ds.pose_positions_allo.data[self.m.index_other, dot_fly, ...]
            x_dot = np.clip((fly_pos[..., 0] - self.m.dot_size, fly_pos[..., 0] + self.m.dot_size),
                            0, self.m.vr.frame_width - 1).astype(np.uintp)
            y_dot = np.clip((fly_pos[..., 1] - self.m.dot_size, fly_pos[..., 1] + self.m.dot_size),
                            0, self.m.vr.frame_height - 1).astype(np.uintp)
            for bodypart_color, x_pos, y_pos in zip(self.m.bodypart_colors, x_dot.T, y_dot.T):
                frame[slice(*x_pos), slice(*y_pos), :] = bodypart_color
        return frame

    def crop_frame(self, frame):
        fly_pos = self.m.ds.pose_positions_allo.data[self.m.index_other, self.m.focal_fly, self.m.thorax_index].astype(np.uintp)
        fly_pos = np.array(fly_pos)  # in case this is a dask.array
        # makes sure crop does not exceed frame bounds
        fly_pos[0] = np.clip(fly_pos[0], self.m.box_size, self.m.vr.frame_width - 1 - self.m.box_size)
        fly_pos[1] = np.clip(fly_pos[1], self.m.box_size, self.m.vr.frame_height - 1 - self.m.box_size)
        x_range = (int(fly_pos[0] - self.m.box_size), int(fly_pos[0] + self.m.box_size))
        y_range = (int(fly_pos[1] - self.m.box_size), int(fly_pos[1] + self.m.box_size))
        frame = frame[slice(*x_range), slice(*y_range), :]  # now crop frame around the focal fly
        return frame

    def _click(self, event):
        event.accept()
        pos = event.pos()
        mouseX, mouseY = int(pos.x()), int(pos.y())
        self.callback(mouseX, mouseY)
