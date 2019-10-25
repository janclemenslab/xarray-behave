"""PLOT SONG AND PLAY VIDEO IN SYNC

`python -m xarray_behave.ui datename root cuepoints`

datename: experiment name (e.g. localhost-20181120_144618)
root: defaults to the current directory - this will work if you're in #Common/chainingmic
cuepoints: string evaluating to a list (e.g. '[100, 50000, 100000]' - including the quotation ticks)
"""
# DONE make span and t0 properties with getter that checks bounds
# DONE fix sine annotation
# DONE check if loading song_raw really takes longer - if so, find way to lazy load raw song with h5py (via loader.load_raw_song)
# DONE toggle to only show manual events
# DONE implement toggle show all channels and overlay one active channel or just show merged
# DONE ensure this works if there is only one fly
# DONE implement saving annotations
# DONE remove autoupdate decorators
# FIXED current_event seems to be first item in list - should be nothing
# TODO allow moving sine song boundaries
# TODO allow selecting save filename with filename= QFileDialog.getSaveFileName((self, 'Dialog Title', '/path/to/default/directory', selectedFilter='*.txt')

import os
import sys
import logging
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from . import xarray_behave as xb
from . import loaders as ld
from . import _ui_utils
from . import colormaps
import time
import numpy as np
import scipy
from pathlib import Path
import defopt
import skimage.draw
from functools import partial

sys.setrecursionlimit(10**6)  # increase recursion limit to avoid errors when keeping key pressed for a long time


class PSV():

    BOX_SIZE = 200
    MAX_AUDIO_AMP = 3.0

    def __init__(self, ds, vr=None, cue_points=[], cmap_name: str = 'turbo'):
        pg.setConfigOptions(useOpenGL=False)   # appears to be faster that way
        self.ds = ds
        self.vr = vr
        self.cue_points = cue_points

        if 'song' in self.ds:
            self.tmax = self.ds.song.shape[0]
        elif 'song_raw' in self.ds:
            self.tmax = self.ds.song_raw.shape[0]
        elif 'body_positions' in self.ds:
            self.tmax = len(self.ds.body_positions) * 10  # TODO: get from  factor ds.attrs.body_positions.sampling_rate
        else:
            raise ValueError('No time stamp info in dataset.')

        self.crop = True
        self.thorax_index = 8        
        self.show_dot = True if 'body_positions' in self.ds else False
        self.old_show_dot_state = self.show_dot
        self.dot_size = 2
        self.show_poses = False
        self.circle_size = 8

        self.focal_fly = 0
        self.other_fly = 1
        self.cue_index = 0

        self.nb_flies = np.max(self.ds.flies).values + 1 if 'flies' in self.ds else 1
        self.fly_colors = _ui_utils.make_colors(self.nb_flies)
        self.nb_bodyparts = len(self.ds.poseparts) if 'poseparts' in self.ds else 1
        self.bodypart_colors = _ui_utils.make_colors(self.nb_bodyparts)
        self.nb_eventtypes = len(self.ds.event_types)
        self.eventype_colors = _ui_utils.make_colors(self.nb_eventtypes)

        self.STOP = True
        self.swap_events = []
        self.frame_interval = 100  # TODO: get from self.ds
        self.show_spec = True
        self.spec_win = 200        
        self.show_songevents = True
        self.show_manualonly = False
        self.show_all_channels = True
        self.sinet0 = None
        # FIXME: will fail for datasets w/o song
        if 'song' in self.ds:
            self.fs_song = self.ds.song.attrs['sampling_rate_Hz']
        elif 'song_raw' in self.ds:
            self.fs_song = self.ds.song_raw.attrs['sampling_rate_Hz']
        else:
            self.fs_song = 10_000
        self.fs_other = self.ds.song_events.attrs['sampling_rate_Hz']

        self.app = pg.QtGui.QApplication([])
        self.win = pg.QtGui.QMainWindow()

        # self.win.closeEvent = self.cleanup
        self.win.resize(1000, 800)
        self.win.setWindowTitle("psv")

        # build menu
        self.bar = self.win.menuBar()
        file = self.bar.addMenu("File")
        self.add_keyed_menuitem(file, "Save swap files", self.save_swaps)
        self.add_keyed_menuitem(file, "Save annotations", self.save_annotations)
        self.add_keyed_menuitem(file, "Save dataset", self.save_dataset)
        file.addSeparator()
        file.addAction("Exit")

        edit = self.bar.addMenu("Edit")
        self.add_keyed_menuitem(edit, "Swap flies", self.swap_flies, QtCore.Qt.Key_X)

        view_play = self.bar.addMenu("Playback")
        self.add_keyed_menuitem(view_play, "Play video", self.toggle_playvideo, QtCore.Qt.Key_Space,
                                checkable=True, checked=not self.STOP)
        view_play.addSeparator()
        self.add_keyed_menuitem(view_play, " < Reverse one frame", self.single_frame_reverse, QtCore.Qt.Key_Left),
        self.add_keyed_menuitem(view_play, "<< Reverse jump", self.jump_reverse, QtCore.Qt.Key_A)
        self.add_keyed_menuitem(view_play, ">> Forward jump", self.jump_forward, QtCore.Qt.Key_D)
        self.add_keyed_menuitem(view_play, " > Forward one frame", self.single_frame_advance, QtCore.Qt.Key_Right)
        view_play.addSeparator()
        self.add_keyed_menuitem(view_play, "Move to previous cue point", self.set_prev_cuepoint, QtCore.Qt.Key_K)
        self.add_keyed_menuitem(view_play, "Move to next cue point", self.set_next_cuepoint, QtCore.Qt.Key_L)
        view_play.addSeparator()
        self.add_keyed_menuitem(view_play, "Zoom in song", self.zoom_in_song, QtCore.Qt.Key_W)
        self.add_keyed_menuitem(view_play, "Zoom out song", self.zoom_out_song, QtCore.Qt.Key_S)

        view_video = self.bar.addMenu("Video")
        self.add_keyed_menuitem(view_video, "Crop frame", partial(self.toggle, 'crop'), QtCore.Qt.Key_C,
                                checkable=True, checked=self.crop)
        self.add_keyed_menuitem(view_video, "Change focal fly", self.change_focal_fly, QtCore.Qt.Key_F)
        view_video.addSeparator()
        self.add_keyed_menuitem(view_video, "Show fly position", partial(self.toggle, 'show_dot'), QtCore.Qt.Key_M,
                                checkable=True, checked=self.show_dot)
        self.add_keyed_menuitem(view_video, "Show poses", partial(self.toggle, 'show_poses'), QtCore.Qt.Key_P,
                                checkable=True, checked=self.show_poses)

        view_audio = self.bar.addMenu("Audio")
        self.add_keyed_menuitem(view_audio, "Play waveform as audio", self.play_audio, QtCore.Qt.Key_E)
        view_audio.addSeparator()
        self.add_keyed_menuitem(view_audio, "Show annotations", partial(self.toggle, 'show_songevents'), QtCore.Qt.Key_V,
                                checkable=True, checked=self.show_songevents)
        self.add_keyed_menuitem(view_audio, "Show manual annotations only", partial(self.toggle, 'show_manualonly'), QtCore.Qt.Key_I,
                                checkable=True, checked=self.show_manualonly)
        self.add_keyed_menuitem(view_audio, "Show all channels", partial(self.toggle, 'show_all_channels'), None,
                                checkable=True, checked=self.show_all_channels)
        view_audio.addSeparator()
        self.add_keyed_menuitem(view_audio, "Show spectrogram", partial(self.toggle, 'show_spec'), QtCore.Qt.Key_G,
                                checkable=True, checked=self.show_spec)
        self.add_keyed_menuitem(view_audio, "Increase frequency resolution", self.dec_freq_res, QtCore.Qt.Key_R)
        self.add_keyed_menuitem(view_audio, "Increase temporal resolution", self.inc_freq_res, QtCore.Qt.Key_T)
        view_audio.addSeparator()
        self.add_keyed_menuitem(view_audio, "Delete events of selected type in view",
                                self.delete_current_events, QtCore.Qt.Key_U)
        self.add_keyed_menuitem(view_audio, "Delete all events in view", self.delete_all_events, QtCore.Qt.Key_Y)
        
        self.bar.addMenu("View")

        self.hl = pg.QtGui.QHBoxLayout()

        # EVENT TYPE selector
        self.cb = pg.QtGui.QComboBox()
        self.cb.addItem("No annotation")
        self.eventList = [(cnt, evt) for cnt, evt in enumerate(self.ds.event_types.values) if 'manual' in evt]
        for event_type in self.eventList:
            self.cb.addItem("Add " + event_type[1])
        self.cb.currentIndexChanged.connect(self.update_xy)
        self.cb.setCurrentIndex(0)
        self.hl.addWidget(self.cb)

        # populate menu with event types so we can catch keys for them - allows changing event type for annotation using numeric keys
        view_audio.addSeparator()
        for ii in range(self.cb.count()):
            self.cb.itemText(ii)
            self.add_keyed_menuitem(view_audio,
                                    self.cb.itemText(ii),
                                    self.change_event_type,
                                    eval(f'QtCore.Qt.Key_{ii}'))


        # CHANNEL selector
        self.cb2 = pg.QtGui.QComboBox()
        if 'song' in self.ds:
            self.cb2.addItem("Merged channels")
        if 'song_raw' in self.ds:
            for chan in range(self.ds.song_raw.shape[1]):
                self.cb2.addItem("Channel " + str(chan))
        self.cb2.currentIndexChanged.connect(self.update_xy)
        self.cb2.setCurrentIndex(0)
        self.hl.addWidget(self.cb2)
        if self.vr is not None:
            self.image_view = _ui_utils.FastImageWidget()
            self.image_view.registerMouseClickEvent(self.click_video)

        self.spec_view = pg.ImageView(name="spec_view", view=pg.PlotItem())
        self.spec_view.view.disableAutoRange()
        self.spec_view.ui.histogram.hide()
        self.spec_view.ui.roiBtn.hide()
        self.spec_view.ui.menuBtn.hide()

        if cmap_name in colormaps.cmaps:
            colormap = colormaps.cmaps[cmap_name]
            colormap._init()
            lut = (colormap._lut * 255).view(np.ndarray)  # convert matplotlib colormap from 0-1 to 0 -255 for Qt
            self.spec_view.getImageItem().setLookupTable(lut)  # apply the colormap        


        self.slice_view = pg.PlotWidget(name="song")
        self.slice_view.getPlotItem().mouseClickEvent = self.click_song
        self.slice_view.disableAutoRange()

        self.cw = pg.GraphicsLayoutWidget()

        self.ly = pg.QtGui.QVBoxLayout()
        self.ly.addLayout(self.hl)
        if self.vr is not None:
            self.ly.addWidget(self.image_view, stretch=4)
        self.ly.addWidget(self.slice_view, stretch=1)
        self.ly.addWidget(self.spec_view, stretch=1)

        self.cw.setLayout(self.ly)

        self.win.setCentralWidget(self.cw)
        self.win.show()

        self._span = 100_000
        self._t0 = int(self.span / 2)
        self.span = 100_000
        self.t0 = int(self.span / 2)

        self.update_xy()
        self.update_frame()

    def cleanup(self, event):
        self.STOP = True
        time.sleep(0.1)
        self.app.processEvents()
        time.sleep(0.1)
        self.app.processEvents()
        time.sleep(0.1)
        self.app.processEvents()
        # self.update_frame()
        # self.update_xy()
        # self.play_video()
        # time.sleep(2)

        print('close')
        event.accept()
        return 0

    @property
    def fs_ratio(self):
        return self.fs_song / self.fs_other

    @property
    def time0(self):
        return int(int(max(0, self.t0 - self.span / 2) / self.fs_ratio) * self.fs_ratio)

    @property
    def time1(self):
        return int(int(max(0, self.t0 + self.span / 2) / self.fs_ratio) * self.fs_ratio)

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, val):
        self._t0 = np.clip(val, self.span / 2, self.tmax - self.span / 2)
        self.update_xy()
        self.update_frame()

    @property
    def span(self):
        return self._span

    @span.setter
    def span(self, val):
        # HACK fixes weird offset/jump error - then probably arise from self.fs_song / self.fs_other
        self._span = min(max(200, val), self.tmax)
        self.update_xy()

    @property
    def span_index(self):
        return self.span / (2 * self.fs_song / self.fs_other)

    @property
    def current_event_index(self):
        if self.cb.currentIndex() - 1 < 0:
            return None
        else:
            return self.eventList[self.cb.currentIndex() - 1][0]

    @property
    def current_event_name(self):
        if self.current_event_index is None:
            return None
        else:
            return str(self.ds.event_types[self.current_event_index].values)

    @property
    def current_channel_name(self):
        return self.cb2.currentText()

    @property
    def current_channel_index(self):
        if self.current_channel_name != 'Merged channels':
            return int(self.current_channel_name.split(' ')[-1])  # "Channel XX"
        else:
            return None

    @property
    def index_other(self):
        return int(self.t0 * self.fs_other / self.fs_song)

    def add_keyed_menuitem(self, parent, label: str, callback, qt_keycode=None, checkable=False, checked=True):
        """Add new action to menu and register key press."""
        menuitem = parent.addAction(label)
        menuitem.setCheckable(checkable)
        menuitem.setChecked(checked)
        if qt_keycode is not None:
            menuitem.setShortcut(qt_keycode)
        menuitem.triggered.connect(lambda: callback(qt_keycode))
        return menuitem

    def change_event_type(self, qt_keycode):
        """Select event to annotate using key presses (0-nb_events)."""
        key_pressed = QtGui.QKeySequence(qt_keycode).toString()  # numeric key code to actual char pressed
        try:
            self.cb.setCurrentIndex(int(key_pressed))
        except ValueError:  # if non-int pressed or int too large for index
            pass

    def save_swaps(self, qt_keycode):
        savefilename = Path(self.ds.attrs['root'], self.ds.attrs['res_path'],
                            self.ds.attrs['datename'], f"{self.ds.attrs['datename']}_idswaps_test.txt")
        savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(self.win, 'Save swaps to', str(savefilename),
                                                                filter="txt files (*.txt);;all files (*)")
        if len(savefilename):
            logging.info(f'   Saving list of swap indices to {savefilename}.')
            np.savetxt(savefilename, self.swap_events, fmt='%d', header='index fly1 fly2')
            logging.info(f'   Done.')

    def save_annotations(self, qt_keycode):
        savefilename = Path(self.ds.attrs['root'], self.ds.attrs['res_path'], self.ds.attrs['datename'],
                            f"{self.ds.attrs['datename']}_songmanual.zarr")
        savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(self.win, 'Save annotations to', str(savefilename),
                                                                filter="zarr files (*.zarr);;all files (*)")
        if len(savefilename):
            logging.info(f'   Saving annotations to {savefilename}.')
            # currently, can only save datasets as zarr - so convert song_events data array to dataset before saving
            xb.save(savefilename, self.ds.song_events.to_dataset())
            logging.info(f'   Done.')

    def save_dataset(self, qt_keycode):
        savefilename = Path(self.ds.attrs['root'], self.ds.attrs['dat_path'], self.ds.attrs['datename'],
                            f"{self.ds.attrs['datename']}.zarr")
        savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(self.win, 'Save dataset to', str(savefilename),
                                                                filter="zarr files (*.zarr);;all files (*)")
        if len(savefilename):
            logging.info(f'   Saving dataset to {savefilename}.')
            # currently, can only save datasets as zarr - so convert song_events data array to dataset before saving
            xb.save(savefilename, self.ds)
            logging.info(f'   Done.')

    def toggle(self, var_name, qt_keycode):
        self.__setattr__(var_name, not self.__getattribute__(var_name))
        if self.STOP:
            self.update_frame()
            self.update_xy()

    def delete_current_events(self, qt_keycode):
        if self.current_event_index is not None:
            self.ds.song_events.data[int(self.index_other - self.span_index):
                                     int(self.index_other + self.span_index), self.current_event_index] = False
            logging.info(f'   Deleted all {self.ds.event_types[self.current_event_index].values} in view.')
        else:
            logging.info(f'   No event type selected. Not deleting anything.')
        if self.STOP:
            self.update_xy()

    def delete_all_events(self, qt_keycode):
        self.ds.song_events.data[int(self.index_other - self.span_index):
                                 int(self.index_other + self.span_index), :] = False
        logging.info(f'   Deleted all events in view.')
        if self.STOP:
            self.update_xy()

    def inc_freq_res(self, qt_keycode):
        self.spec_win = int(self.spec_win * 2)
        if self.STOP:
            # need to update twice to fix axis limits for some reason
            self.update_xy()
            self.update_xy()

    def dec_freq_res(self, qt_keycode):
        self.spec_win = int(max(2, self.spec_win // 2))
        if self.STOP:
            # need to update twice to fix axis limits for some reason
            self.update_xy()
            self.update_xy()

    def toggle_playvideo(self, qt_keycode):
        self.STOP = not self.STOP
        if not self.STOP:
            self.play_video()

    def toggle_show_poses(self, qt_keycode):
        self.show_poses = not self.show_poses
        if self.show_poses:
            self.old_show_dot_state = self.show_dot
            self.show_dot = False
        else:
            self.show_dot = self.old_show_dot_state
        if self.STOP:
            self.update_frame()

    def change_focal_fly(self, qt_keycode):
        tmp = (self.focal_fly + 1) % self.nb_flies
        if tmp == self.other_fly:  # swap focal and other fly if same
            self.other_fly, self.focal_fly = self.focal_fly, self.other_fly
        else:
            self.focal_fly = tmp
        if self.STOP:
            self.update_frame()

    def set_prev_cuepoint(self, qt_keycode):
        if self.cue_points:
            self.cue_index = max(0, self.cue_index - 1)
            logging.debug(f'cue val at cue_index {self.cue_index} is {self.cue_points[self.cue_index]}')
            self.t0 = self.cue_points[self.cue_index]  # jump to PREV cue point

    def set_next_cuepoint(self, qt_keycode):
        if self.cue_points:
            self.cue_index = min(self.cue_index + 1, len(self.cue_points) - 1)
            logging.debug(f'cue val at cue_index {self.cue_index} is {self.cue_points[self.cue_index]}')
            self.t0 = self.cue_points[self.cue_index]  # jump to PREV cue point

    def zoom_in_song(self, qt_keycode):
        self.span /= 2

    def zoom_out_song(self, qt_keycode):
        self.span *= 2

    def single_frame_reverse(self, qt_keycode):
        self.t0 -= self.frame_interval

    def single_frame_advance(self, qt_keycode):
        self.t0 += self.frame_interval

    def jump_reverse(self, qt_keycode):
        self.t0 -= self.span / 2

    def jump_forward(self, qt_keycode):
        self.t0 += self.span / 2

    def update_xy(self):
        # TODO could atomize this further - currently, traces, spec and events are re-drawn
        #      but could e.g. only re-draw events if they have changed,
        #      or just the spec if spec resolution was changed
        self.x = self.ds.sampletime.data[self.time0:self.time1]
        step = int(max(1, np.ceil(len(self.x) / self.fs_song / 2)))  # make sure step is >= 1
        self.slice_view.clear()

        if 'song' in self.ds and self.current_channel_name == 'Merged channels':
            self.y = self.ds.song.data[self.time0:self.time1]
        else:
            # load song for current channel
            try:            
                self.y = self.ds.song_raw.data[self.time0:self.time1, self.current_channel_index].compute()
            except AttributeError:
                self.y = self.ds.song_raw.data[self.time0:self.time1, self.current_channel_index]

            if self.show_all_channels:
                nb_channels = self.ds.song_raw.shape[-1]
                channel_list = np.delete(np.arange(nb_channels), self.current_channel_index)
                self.y_other = self.ds.song_raw[self.time0:self.time1].values[:, channel_list]

                for chan in range(nb_channels - 1):
                    self.slice_view.addItem(pg.PlotCurveItem(self.x[::step], self.y_other[::step, chan]))

        # plot selected trace
        self.slice_view.addItem(pg.PlotCurveItem(self.x[::step], self.y[::step]))
        self.slice_view.autoRange(padding=0)

        if self.show_songevents:
            self.plot_song_events(self.x)

        # time of current frame in trace
        self.slice_view.addItem(pg.InfiniteLine(movable=False, angle=90,
                                                pos=self.x[int(self.span / 2)],
                                                pen=pg.mkPen(color='r', width=1)))

        if self.show_spec:
            self.plot_spec(self.x, self.y)
        else:
            self.spec_view.clear()

        self.app.processEvents()

    def update_frame(self):
        if self.vr is not None:

            # fn = self.ds.body_positions.nearest_frame[self.index_other]
            fn = self.ds.coords['nearest_frame'][self.index_other]
            
            self.frame = self.vr[fn]
            if self.frame is not None:  # frame is None when at end of video
                # FIXME the annotations potentially waste time annotating outside of the cropped frame
                if 'pose_positions_allo' in self.ds:
                    if self.show_poses:
                        self.frame = self.annotate_poses(self.frame)
                    if self.show_dot:
                        self.frame = self.annotate_dot(self.frame)
                    if self.crop:
                        self.frame = np.ascontiguousarray(self.crop_frame(self.frame))

                self.image_view.setImage(self.frame, auto_scale=True)

            self.app.processEvents()

    # TODO move the next three methods out of the class: self.vr.width/height to frame.shape,  box_size as args
    def annotate_poses(self, frame):
        for dot_fly in range(self.nb_flies):
            fly_pos = self.ds.pose_positions_allo.data[self.index_other, dot_fly, ...]
            x_dot = np.clip((fly_pos[..., 0] - self.dot_size, fly_pos[..., 0] + self.dot_size),
                            0, self.vr.frame_width - 1).astype(np.uintp)
            y_dot = np.clip((fly_pos[..., 1] - self.dot_size, fly_pos[..., 1] + self.dot_size),
                            0, self.vr.frame_height - 1).astype(np.uintp)
            for bodypart_color, x_pos, y_pos in zip(self.bodypart_colors, x_dot.T, y_dot.T):
                frame[slice(*x_pos), slice(*y_pos), :] = bodypart_color
        return frame

    def annotate_dot(self, frame):
        # mark each fly with uniquely colored dots
        for dot_fly in range(self.nb_flies):
            fly_pos = self.ds.pose_positions_allo.data[self.index_other, dot_fly, self.thorax_index]
            x_dot = np.clip((fly_pos[0] - self.dot_size, fly_pos[0] + self.dot_size),
                            0, self.vr.frame_width - 1).astype(np.uintp)
            y_dot = np.clip((fly_pos[1] - self.dot_size, fly_pos[1] + self.dot_size),
                            0, self.vr.frame_height - 1).astype(np.uintp)
            frame[slice(*x_dot), slice(*y_dot), :] = self.fly_colors[dot_fly]  # set pixels around

        # FIXME what if there is only one fly?
        # mark *focal* and *other* fly with circle
        for this_fly, color in zip((self.focal_fly, self.other_fly), self.bodypart_colors[[2, 6]]):
            fly_pos = self.ds.pose_positions_allo.data[self.index_other,
                                                       this_fly,
                                                       self.thorax_index].astype(np.uintp)
            xx, yy = skimage.draw.circle_perimeter(fly_pos[0], fly_pos[1], self.circle_size, method='bresenham')
            frame[xx, yy, :] = color
        return frame

    def crop_frame(self, frame):
        fly_pos = self.ds.pose_positions_allo.data[self.index_other, self.focal_fly, self.thorax_index].astype(np.uintp)
        # makes sure crop does not exceed frame bounds
        fly_pos[0] = np.clip(fly_pos[0], self.BOX_SIZE, self.vr.frame_width - 1 - self.BOX_SIZE)
        fly_pos[1] = np.clip(fly_pos[1], self.BOX_SIZE, self.vr.frame_height - 1 - self.BOX_SIZE)
        x_range = (int(fly_pos[0] - self.BOX_SIZE), int(fly_pos[0] + self.BOX_SIZE))
        y_range = (int(fly_pos[1] - self.BOX_SIZE), int(fly_pos[1] + self.BOX_SIZE))
        frame = frame[slice(*x_range), slice(*y_range), :]  # now crop frame around the focal fly
        return frame

    def plot_song_events(self, x):
        _, yrange = self.slice_view.getPlotItem().getViewBox().viewRange()
        yrange = np.array(yrange)
        for event_type in range(self.nb_eventtypes):
            event_pen = pg.mkPen(color=self.eventype_colors[event_type], width=1)
            if self.show_manualonly and 'manual' not in self.ds.event_types.values[event_type]:
                continue
            if 'sine' in self.ds.event_types.data[event_type]:
                event_trace = self.ds.song_events.data[int(self.index_other - self.span_index):
                                                       int(self.index_other + self.span_index),
                                                       event_type]
                d_event_trace = np.diff(event_trace.astype(np.int))
                event_onset_indices = x[np.where(d_event_trace > 0)[0] * int(1 / self.fs_other * self.fs_song)]
                event_offset_indices = x[np.where(d_event_trace < 0)[0] * int(1 / self.fs_other * self.fs_song)]

                # FIXME both will fail if there is only a single but incomplete sine song in the x-range since the respective other will have zero length
                # first offset outside of x-range
                if len(event_offset_indices) and len(event_onset_indices) and event_offset_indices[0] < event_onset_indices[0]:
                    event_onset_indices = np.pad(event_onset_indices, (1, 0), mode='constant',
                                                 constant_values=x[0])  # add additional onset at x[0] boundary
                # last offset outside of x-range
                if len(event_offset_indices) and len(event_onset_indices) and event_offset_indices[-1] < event_onset_indices[-1]:
                    # add additional offset at x[-1] boundary
                    event_offset_indices = np.pad(event_offset_indices, (0, 1), mode='constant', constant_values=x[-1])

                for onset, offset in zip(event_onset_indices, event_offset_indices):
                    self.slice_view.addItem(pg.LinearRegionItem(values=(onset, offset),
                                                                movable=False))
            else:
                event_indices = np.where(self.ds.song_events.data[int(self.index_other - self.span_index):
                                                                  int(self.index_other + self.span_index),
                                                                  event_type])[0] * int(1 / self.fs_other * self.fs_song)
                events = x[event_indices]
                if len(events):
                    xx = np.broadcast_to(events[np.newaxis, :], (2, len(events)))
                    yy = np.zeros_like(xx) + yrange[:, np.newaxis]
                    _ui_utils.fast_plot(self.slice_view, xx.T, yy.T, event_pen)

    def play_video(self, rate=100):  # TODO: get rate from ds (video fps attr)
        RUN = True
        cnt = 0
        dt0 = time.time()
        while RUN:
            self.t0 += rate
            cnt += 1
            if cnt % 10 == 0:
                # logging.debug(time.time() - dt0)
                dt0 = time.time()
            if self.STOP:
                RUN = False
                logging.debug('   Stopped playback.')

    def click_video(self, event):
        if self.vr is not None:
            event.accept()
            pos = event.pos()
            mouseX, mouseY = int(pos.x()), int(pos.y())
            fly_pos = self.ds.pose_positions_allo.data[self.index_other, :, self.thorax_index, :]
            if self.crop:  # transform fly pos to coordinates of the cropped box
                box_center = self.ds.pose_positions_allo.data[self.index_other,
                                                              self.focal_fly, self.thorax_index] + self.BOX_SIZE / 2
                fly_pos = fly_pos - box_center
            fly_dist = np.sum((fly_pos - np.array([mouseX, mouseY]))**2, axis=-1)
            fly_dist[self.focal_fly] = np.inf  # ensure that other_fly is not focal_fly
            self.other_fly = np.argmin(fly_dist)
            logging.debug(f"Selected {self.other_fly}.")
            self.update_frame()

    def click_song(self, event):
        event.accept()
        pos = event.pos()
        if self.current_event_index is not None:
            mouseT = self.slice_view.getPlotItem().getViewBox().mapSceneToView(pos).x()
            if 'sine' in self.current_event_name:
                if self.sinet0 is None:
                    self.sinet0 = mouseT
                else:
                    self.ds.song_events.sel(time=slice(
                        *sorted([self.sinet0, mouseT])))[:, self.current_event_index] = True
                    logging.info(f'  Added {self.current_event_name} at t=[{self.sinet0:1.4f}:{mouseT:1.4f}] seconds.')
                    self.sinet0 = None
            else:  # pulse-like event
                self.sinet0 = None
                self.ds.song_events.sel(time=mouseT, method='nearest')[self.current_event_index] = True
                logging.info(f'  Added {self.current_event_name} at t={mouseT:1.4f} seconds.')
            self.update_xy()
        else:
            self.sinet0 = None

    def play_audio(self, qt_keycode):
        """Play vector as audio using the simpleaudio package."""
        
        if 'song' in self.ds or 'song_raw' in self.ds:
            import simpleaudio
            if 'song' in self.ds and self.current_channel_name == 'Merged channels':
                y = self.ds.song.data[self.time0:self.time1]
            else:
                y = self.ds.song_raw.data[self.time0:self.time1, self.current_channel_index]            
            y = np.array(y)  # if y is a dask.array (lazy loaded)

            # normalize to 16-bit range and convert to 16-bit data
            max_amp = self.MAX_AUDIO_AMP
            if max_amp is None:
                max_amp = np.nanmax(np.abs(x))
            y = y * 32767 / max_amp
            y = y.astype(np.int16)
            # simpleaudio can only play at these rates - choose the one nearest to our rate
            allowed_sample_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000]  # Hz
            sample_rate = min(allowed_sample_rates, key=lambda x: abs(x - int(self.fs_song)))
            # FIXME resample audio to new sample_rate to preserve sound
            # start playback in background
            simpleaudio.play_buffer(y, num_channels=1, bytes_per_sample=2, sample_rate=sample_rate)
        else:
            logging.info(f'Could not play sound - no merged-channel sound data in the dataset.')

    def plot_spec(self, x, y):
        # hash x to avoid re-calculation? only useful when annotating
        S, f, t = self._calc_spec(y)
        self.spec_view.setImage(S.T[:, ::-1])
        self.spec_view.view.setLimits(xMin=0, xMax=S.shape[1], yMin=0, yMax=S.shape[0])
        y_axis = self.spec_view.getView().getAxis('left')
        f = f[::-1]
        ticks = np.linspace(0, len(f)-1, 5, dtype=np.uintp)
        y_axis.setTicks([[(ii, str(f[ii])) for ii in ticks]])

    # @lru_cache(maxsize=2, typed=False)
    def _calc_spec(self, y, fmax=1000):
        # signal.spectrogram will internally limit spec_win to len(y)
        # and will throw error since noverlap will then be too big
        self.spec_win = min(len(y), self.spec_win)
        f, t, psd = scipy.signal.spectrogram(y, self.fs_song, nperseg=self.spec_win,
                                             noverlap=self.spec_win // 2, nfft=self.spec_win * 4, mode='magnitude')
        f_idx = np.argmax(f > fmax)
        S = np.log2(1 + psd[:f_idx, :])
        S = S / np.max(S) * 255  # normalize to 0...255
        return S, f[:f_idx], t

    def swap_flies(self, qt_keycode):
        if self.vr is not None:
            # use swap_dims to swap flies in all dataarrays in the data set?
            # TODO make sure this does not fail for datasets w/o song
            logging.info(f'   Swapping flies 1 & 2 at time {self.t0}.')
            # if already in there remove - swapping a second time would negate first swap
            if [self.index_other, self.focal_fly, self.other_fly] in self.swap_events:
                self.swap_events.remove([self.index_other, self.focal_fly, self.other_fly])
            else:
                self.swap_events.append([self.index_other, self.focal_fly, self.other_fly])

            self.ds.pose_positions_allo.values[self.index_other:, [
                self.other_fly, self.focal_fly], ...] = self.ds.pose_positions_allo.values[self.index_other:, [self.focal_fly, self.other_fly], ...]
            self.ds.pose_positions.values[self.index_other:, [
                self.other_fly, self.focal_fly], ...] = self.ds.pose_positions.values[self.index_other:, [self.focal_fly, self.other_fly], ...]
            self.ds.body_positions.values[self.index_other:, [
                self.other_fly, self.focal_fly], ...] = self.ds.body_positions.values[self.index_other:, [self.focal_fly, self.other_fly], ...]


def main(datename: str = 'localhost-20181120_144618', root: str = '',
         cue_points: str = '[]',
         *,
         ignore_existing: bool = False,
         cmap_name: str = 'turbo'):
    """    
    Args:
        datename (str): Experiment id. Defaults to 'localhost-20181120_144618'.
        root (str): Path containing the `dat` and `res` folders for the experiment. Defaults to '' (current directory).
        cue_points (str): List of cue points (indices) for quickly jumping around time. Defaults to '[]'.
        ignore_existing (bool): Ignore existing song annotations. Defaults to False.
        cmap_name (str): Name of the colormap (one of ['magma', 'inferno', 'plasma', 'viridis', 'parula', 'turbo']). Defaults to 'turbo'.
    """

    if not ignore_existing and os.path.exists(datename + '.zarr'):
        logging.info(f'Loading ds from {datename}.zarr.')
        ds = xb.load(datename + '.zarr', lazy=True)

        logging.info(f'   Loading data from ds.')
        if 'song_events' in ds:
            ds.song_events.load()  # non-lazy load song events so we can edit them
        if 'song' in ds:
            ds.song.load()  # non-lazy load song for faster updates
        if 'pose_positions_allo' in ds:
            ds.pose_positions_allo.load()  # non-lazy load song for faster updates
        if 'sampletime' in ds:
            ds.sampletime.load()

        # this will take a long time:
        ds.song_raw.load()  # non-lazy load song for faster updates

    else:
        logging.info(f'Assembling dataset for {datename}.')
        ds = xb.assemble(datename, root=root, fix_fly_indices=False, keep_multi_channel=True)
        xb.save(datename + '.zarr', ds)

    ds = ld.initialize_manual_song_events(ds, from_segmentation=False, force_overwrite=False)

    logging.info(ds)
    filepath = ds.attrs['video_filename']
    vr = None
    try:
        try: 
            video_filename = filepath[:-3] + 'mp4'
            vr = _ui_utils.VideoReaderNP(video_filename)
        except:
            video_filename = filepath[:-3] + 'avi'
            vr = _ui_utils.VideoReaderNP(video_filename)
        print(vr)
    except FileNotFoundError:
        print(f'Video "{video_filename}" not found. Continuing without.')
    except:
        print(f'Something went wrong when loading the video. Continuing without.')

    cue_points = eval(cue_points)
    PSV(ds, vr, cue_points, cmap_name)

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    defopt.run(main)
