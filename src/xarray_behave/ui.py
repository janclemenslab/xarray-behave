"""PLOT SONG AND PLAY VIDEO IN SYNC

`python -m xarray_behave.ui datename root cuepoints`

datename: experiment name (e.g. localhost-20181120_144618)
root: defaults to the current directory - this will work if you're in #Common/chainingmic
cuepoints: string evaluating to a list (e.g. '[100, 50000, 100000]' - including the quotation ticks)
"""
# TODO make span and t0 properties with getter that checks bounds
# TODO fix sine annotation

import os
import sys
import logging
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from . import xarray_behave as xb
from . import _ui_utils
import numpy as np
from pathlib import Path
import defopt
import skimage.draw
from functools import partial


class PSV():

    BOX_SIZE = 200
    MAX_AUDIO_AMP = 3.0

    def __init__(self, ds, vr, cue_points=[]):
        pg.setConfigOptions(useOpenGL=False)   # appears to be faster that way
        self.ds = ds
        self.vr = vr
        self.cue_points = cue_points

        self.span = 100_000
        self.t0 = int(self.span/2)
        if hasattr(self.ds, 'song'):
            self.tmax = len(self.ds.song)
        else:
            self.tmax = len(self.ds.body_positions) * 10  # TODO: get factor from self.ds
        
        self.crop = False
        self.thorax_index = 8
        self.show_dot = True
        self.dot_size = 2
        self.circle_size = 8
        self.old_show_dot_state = self.show_dot
        self.show_poses = False
        self.index_other = 0
        self.focal_fly = 0
        self.other_fly = 1
        self.cue_index = 0
        
        self.nb_flies = np.max(self.ds.flies).values+1
        self.fly_colors = _ui_utils.make_colors(self.nb_flies)
        self.nb_bodyparts = len(self.ds.poseparts)
        self.bodypart_colors = _ui_utils.make_colors(self.nb_bodyparts)
        self.nb_eventtypes = len(self.ds.event_types)
        self.eventype_colors = _ui_utils.make_colors(self.nb_eventtypes)
        
        self.current_event_index = -1

        self.STOP = True
        self.swap_events = []
        self.mouseX, self.mouseY = None, None
        self.mouseT = None
        self.frame_interval = 100  # TODO: get from self.ds
        self.show_spec = True
        self.spec_win = 200
        self.show_songevents = True

        # FIXME: will fail for datasets w/o song        
        self.fs_song = self.ds.song.attrs['sampling_rate_Hz']
        self.fs_other = self.ds.song_events.attrs['sampling_rate_Hz']

        self.app = pg.QtGui.QApplication([])
        self.win = pg.QtGui.QMainWindow()
        self.win.resize(800, 800)
        self.win.setWindowTitle("psv")

        self.bar = self.win.menuBar()
        file = self.bar.addMenu("File")
        self.add_keyed_menuitem(file, "Save swap files", self.save_swaps)
        self.add_keyed_menuitem(file, "Save annotations", self.save_annotations)
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
        self.add_keyed_menuitem(view_video, "Crop frame", self.toggle_crop, QtCore.Qt.Key_C,
                                checkable=True, checked=self.crop)
        self.add_keyed_menuitem(view_video, "Change focal fly", self.change_focal_fly, QtCore.Qt.Key_F)
        view_video.addSeparator()
        self.add_keyed_menuitem(view_video, "Show fly position", self.toggle_show_dot, QtCore.Qt.Key_M,
                                checkable=True, checked=self.show_dot)
        self.add_keyed_menuitem(view_video, "Show poses", self.toggle_show_poses, QtCore.Qt.Key_P,
                                checkable=True, checked=self.show_poses)
        
        view_audio = self.bar.addMenu("Audio")
        self.add_keyed_menuitem(view_audio, "Play waveform as audio", self.play_audio, QtCore.Qt.Key_E)
        view_audio.addSeparator()
        self.add_keyed_menuitem(view_audio, "Show annotations", self.toggle_show_songevents, QtCore.Qt.Key_V, 
                                checkable=True, checked=self.show_songevents)
        view_audio.addSeparator()
        self.add_keyed_menuitem(view_audio, "Show spectrogram", self.toggle_show_spec, QtCore.Qt.Key_G,
                                checkable=True, checked=self.show_spec)
        self.add_keyed_menuitem(view_audio, "Increase frequency resolution", self.dec_freq_res, QtCore.Qt.Key_R)
        self.add_keyed_menuitem(view_audio, "Increase temporal resolution", self.inc_freq_res, QtCore.Qt.Key_T)
        
        view_annotate = self.bar.addMenu("Annotate")
        self.choices = []
        tmp = self.add_keyed_menuitem(view_annotate, "No annotation", partial(self.set_current_event, src=-1),
                                      eval('QtCore.Qt.Key_' + str(0)),
                                      checkable=True, checked=True)
        self.choices.append(tmp)

        # FIXME only allow *_manual events here?
        for cnt, event_type in enumerate(self.ds.event_types.values):
            tmp = self.add_keyed_menuitem(view_annotate, "add " + event_type, partial(self.set_current_event, src=cnt),
                                          eval('QtCore.Qt.Key_' + str(cnt+1)),
                                          checkable=True, checked=False)
            self.choices.append(tmp)
        view_annotate.addSeparator()
        self.add_keyed_menuitem(view_annotate, "Delete events of selected type in view", self.delete_current_events, QtCore.Qt.Key_U)
        self.add_keyed_menuitem(view_annotate, "Delete all events in view", self.delete_all_events, QtCore.Qt.Key_Y)
        
        self.bar.addMenu("View")
        
        self.image_view = _ui_utils.ImageViewVR(name="img_view")
        self.image_view.setImage(self.vr)
        self.image_view.getImageItem().mouseClickEvent = self.click_video

        self.spec_view = pg.ImageView(name="spec_view", view=pg.PlotItem())
        self.spec_view.view.enableAutoScale()
        
        self.slice_view = pg.PlotWidget(name="song")
        self.slice_view.getPlotItem().mouseClickEvent = self.click_song

        self.cw = pg.GraphicsLayoutWidget()
 
        self.ly = pg.QtGui.QVBoxLayout()
        self.ly.addWidget(self.image_view, stretch=4)
        self.ly.addWidget(self.slice_view, stretch=1)
        self.ly.addWidget(self.spec_view, stretch=1)
        
        self.cw.setLayout(self.ly)
        
        self.win.setCentralWidget(self.cw)
        self.win.show()
        
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        
        self.spec_view.ui.histogram.hide()
        self.spec_view.ui.roiBtn.hide()
        self.spec_view.ui.menuBtn.hide()
        
        self.update()

    @property
    def time0(self):
        return int(self.t0 - self.span / 2)

    @property
    def time1(self):
        return int(self.t0 + self.span / 2)

    def add_keyed_menuitem(self, parent, label: str, callback, qt_keycode=None, checkable=False, checked=True):
        """Add new action to menu and register key press."""
        menuitem = parent.addAction(label)
        menuitem.setCheckable(checkable)
        menuitem.setChecked(checked)
        if qt_keycode is not None:
            menuitem.setShortcut(qt_keycode)
        menuitem.triggered.connect(callback)
        return menuitem

    def save_swaps(self):
        savefilename = Path(self.ds.attrs['root'], self.ds.attrs['res_path'], self.ds.attrs['datename'], f"{self.ds.attrs['datename']}_idswaps_test.txt")
        np.savetxt(savefilename, self.swap_events, fmt='%d', header='index fly1 fly2')
        logging.info(f'   Saving list of swap indices to {savefilename}.')

    def save_annotations(self):
        logging.warning(f'   Not implemented yet.')
        # savefilename = Path(self.ds.attrs['root'], self.ds.attrs['res_path'], self.ds.attrs['datename'], f"{self.ds.attrs['datename']}_annotations.txt")
        # np.savetxt(savefilename, self.annotations, fmt='%d', header='index ' + ' '.join(self.ds.song_events))
        # logging.info(f'   Saving annotations to {savefilename}.')

    def set_current_event(self, evt, src):
        self.current_event_index = src
        if src == -1:
            logging.debug(f'  Turned off annotation.')
        else:
            logging.debug(f'  Annotating {self.ds.event_types[self.current_event_index].values}.')
            for cnt, choice in enumerate(self.choices):
                if cnt != src+1:
                    choice.setChecked(False)
                else:
                    choice.setChecked(True)
    
    @_ui_utils._autoupdate
    def delete_current_events(self):        
        self.ds.song_events[self.time0:self.time1, self.current_event_index] = False
        self.logging.inf(f'   Deleted all {self.ds.event_types[self.current_event_index].values} in view.')

    @_ui_utils._autoupdate
    def delete_all_events(self):
        self.ds.song_events[self.time0:self.time1, :] = False
        self.logging.inf(f'   Deleted all events in view.')

    @_ui_utils._autoupdate
    def toggle_show_songevents(self):
        self.show_songevents = not self.show_songevents
    
    @_ui_utils._autoupdate
    def toggle_show_spec(self):
        self.show_spec = not self.show_spec
    
    @_ui_utils._autoupdate
    def inc_freq_res(self):
        self.spec_win = int(self.spec_win * 2)
    
    @_ui_utils._autoupdate
    def dec_freq_res(self):
        self.spec_win = int(max(2, self.spec_win // 2))
    
    def toggle_playvideo(self):
        if self.STOP:
            logging.info(f'   Starting playback.')
            self.STOP = False
            self.play_video()
        else:
            logging.info(f'   Stopping playback.')
            self.STOP = True

    @_ui_utils._autoupdate
    def toggle_crop(self):
        self.crop = not self.crop
    
    @_ui_utils._autoupdate
    def toggle_show_dot(self):
        self.show_dot = not self.show_dot
    
    @_ui_utils._autoupdate
    def toggle_show_poses(self):
        self.show_poses = not self.show_poses
        if self.show_poses:
            self.old_show_dot_state = self.show_dot
            self.show_dot = False
        else:
            self.show_dot = self.old_show_dot_state
    
    @_ui_utils._autoupdate
    def change_focal_fly(self):
        tmp = (self.focal_fly+1) % self.nb_flies
        if tmp == self.other_fly:  # swap focal and other fly if same
            self.other_fly, self.focal_fly = self.focal_fly, self.other_fly
        else:
            self.focal_fly = tmp
    
    @_ui_utils._autoupdate
    def set_prev_cuepoint(self):
        if self.cue_points:
            self.cue_index = max(0, self.cue_index-1)
            logging.debug(f'cue val at cue_index {self.cue_index} is {self.cue_points[self.cue_index]}')
            self.t0 = self.cue_points[self.cue_index]  # jump to PREV cue point
    
    @_ui_utils._autoupdate
    def set_next_cuepoint(self):
        if self.cue_points:
            self.cue_index = min(self.cue_index+1, len(self.cue_points)-1)
            logging.debug(f'cue val at cue_index {self.cue_index} is {self.cue_points[self.cue_index]}')
            self.t0 = self.cue_points[self.cue_index]  # jump to PREV cue point
    
    @_ui_utils._autoupdate
    def zoom_in_song(self):
        self.span /= 2
    
    @_ui_utils._autoupdate
    def zoom_out_song(self):
        self.span *= 2
    
    @_ui_utils._autoupdate
    def single_frame_reverse(self):
        self.t0 -= self.frame_interval
    
    @_ui_utils._autoupdate
    def single_frame_advance(self):
        self.t0 += self.frame_interval
    
    @_ui_utils._autoupdate
    def jump_reverse(self):
        self.t0 -= self.span / 2
    
    @_ui_utils._autoupdate
    def jump_forward(self):
        self.t0 += self.span / 2
    
    def update(self):
        """Updates the view."""
        self.span = int(max(500, self.span))
        self.t0 = int(np.clip(self.t0, self.span/2, self.tmax - self.span/2))

        if 'song' in self.ds:
            self.index_other = int(self.t0 * self.fs_other / self.fs_song)

            # clear trace plot and update with new trace
            # time0 = int(self.t0 - self.span / 2)
            # time1 = int(self.t0 + self.span / 2)

            x = self.ds.sampletime[self.time0:self.time1].values
            y = self.ds.song[self.time0:self.time1].values
            step = int(np.ceil(len(x) / self.fs_song / 2))
            self.slice_view.clear()
            self.slice_view.plot(x[::step], y[::step])

            if self.show_songevents:
                self.plot_song_events(x)
            
            # time of current frame in trace
            self.slice_view.addItem(pg.InfiniteLine(movable=False, angle=90,
                                                    pos=x[int(self.span / 2)],
                                                    pen=pg.mkPen(color='r', width=1)))
            self.slice_view.autoRange(padding=0)
            if self.show_spec:
                self.plot_spec(x, y)
            else:
                self.spec_view.clear()
        
        else:
            self.index_other = int(self.t0 * self.fs_other / 10_000)

        fn = self.ds.body_positions.nearest_frame[self.index_other]
        frame = self.vr[fn]

        # FIXME the annotations potentially waste time annotating outside of the cropped frame
        if self.show_poses:
            frame = self.annotate_poses(frame)    
        if self.show_dot:
            frame = self.annotate_dot(frame)

        if self.crop:
            frame = self.crop_frame(frame)

        self.image_view.clear()
        self.image_view.setImage(frame)
        self.app.processEvents()

    def annotate_poses(self, frame):
        for dot_fly in range(self.nb_flies):
            fly_pos = self.ds.pose_positions_allo[self.index_other, dot_fly, ...].values
            x_dot = np.clip((fly_pos[..., 0]-self.dot_size, fly_pos[..., 0]+self.dot_size), 0, self.vr.frame_width-1).astype(np.uintp)
            y_dot = np.clip((fly_pos[..., 1]-self.dot_size, fly_pos[..., 1]+self.dot_size), 0, self.vr.frame_height-1).astype(np.uintp)
            for bodypart_color, x_pos, y_pos in zip(self.bodypart_colors, x_dot.T, y_dot.T):
                frame[slice(*x_pos), slice(*y_pos), :] = bodypart_color
        return frame

    def annotate_dot(self, frame):

        for dot_fly in range(self.nb_flies):
            fly_pos = self.ds.pose_positions_allo[self.index_other, dot_fly, self.thorax_index].values
            x_dot = np.clip((fly_pos[0]-self.dot_size, fly_pos[0]+self.dot_size), 0, self.vr.frame_width-1).astype(np.uintp)
            y_dot = np.clip((fly_pos[1]-self.dot_size, fly_pos[1]+self.dot_size), 0, self.vr.frame_height-1).astype(np.uintp)
            frame[slice(*x_dot), slice(*y_dot), :] = self.fly_colors[dot_fly]  # set pixels around
        
        # mark *focal* fly with circle
        fly_pos = self.ds.pose_positions_allo[self.index_other, self.focal_fly, self.thorax_index].values.astype(np.uintp)
        xx, yy = skimage.draw.circle_perimeter(fly_pos[0], fly_pos[1], self.circle_size, method='bresenham')
        frame[xx, yy, :] = self.bodypart_colors[2]

        # mark *other* fly with circle
        fly_pos = self.ds.pose_positions_allo[self.index_other, self.other_fly, self.thorax_index].values.astype(np.uintp)
        xx, yy = skimage.draw.circle_perimeter(fly_pos[0], fly_pos[1], self.circle_size, method='bresenham')
        frame[xx, yy, :] = self.bodypart_colors[6]
        return frame

    def crop_frame(self, frame):
        fly_pos = self.ds.pose_positions_allo[self.index_other, self.focal_fly, self.thorax_index].values
        # makes sure crop does not exceed frame bounds
        x_range = np.clip((fly_pos[0]-self.BOX_SIZE, fly_pos[0]+self.BOX_SIZE), 0, self.vr.frame_width-1).astype(np.uintp)
        y_range = np.clip((fly_pos[1]-self.BOX_SIZE, fly_pos[1]+self.BOX_SIZE), 0, self.vr.frame_height-1).astype(np.uintp)
        frame = frame[slice(*x_range), slice(*y_range), :]  # now crop frame around the focal fly        
        return frame

    def plot_song_events(self, x):
        for event_type in range(self.nb_eventtypes):
            event_pen = pg.mkPen(color=self.eventype_colors[event_type], width=1)
            if 'sine' in self.ds.event_types.values[event_type]:
                event_trace = self.ds.song_events[int(self.index_other - self.span / 20):int(self.index_other + self.span / 20), event_type].values
                d_event_trace = np.diff(event_trace.astype(np.int))
                event_onset_indices = x[np.where(d_event_trace > 0)[0] * int(1 / self.fs_other * self.fs_song)]
                event_offset_indices = x[np.where(d_event_trace < 0)[0] * int(1 / self.fs_other * self.fs_song)]

                # FIXME both will fail if there is only a single but incomplete sine song in the x-range since the respective other will have zero length
                if len(event_offset_indices) and len(event_onset_indices) and event_offset_indices[0] < event_onset_indices[0]:  # first offset outside of x-range
                    event_onset_indices = np.pad(event_onset_indices, (1, 0), mode='constant', constant_values=x[0])  # add additional onset at x[0] boundary
                if len(event_offset_indices) and len(event_onset_indices)  and event_offset_indices[-1] < event_onset_indices[-1]:  # last offset outside of x-range
                    event_offset_indices = np.pad(event_offset_indices, (0, 1), mode='constant', constant_values=x[-1])  # add additional offset at x[-1] boundary

                for onset, offset in zip(event_onset_indices, event_offset_indices):
                    self.slice_view.addItem(pg.LinearRegionItem(values=(onset, offset),
                                                                movable=False))
            else:
                event_indices = np.where(self.ds.song_events[int(self.index_other - self.span / 20):int(self.index_other + self.span / 20), event_type].values)[0] * int(1 / self.fs_other * self.fs_song)
                events = x[event_indices]
                for evt in events:
                    self.slice_view.addItem(pg.InfiniteLine(movable=False,
                                                            angle=90,
                                                            pos=evt,
                                                            pen=event_pen))

    def play_video(self, rate=100):  # TODO: get rate from ds (video fps attr)
        RUN = True
        while RUN:
            self.t0 += rate
            self.update()
            self.app.processEvents()
            if self.STOP:
                RUN = False

    def click_video(self, event):
        event.accept()
        pos = event.pos()
        self.mouseX, self.mouseY = int(pos.x()), int(pos.y())
        logging.debug(f'mouse clicked at x={self.mouseX}, y={self.mouseY}.')
        # find nearest fly
        fly_pos = self.ds.pose_positions_allo[self.index_other, :, self.thorax_index, :].values
        if self.crop:  # transform mouse pos to coordinates of the cropped box
            fly_pos = fly_pos - self.ds.pose_positions_allo[self.index_other, self.focal_fly, self.thorax_index].values + self.BOX_SIZE/2

        logging.debug(np.round(fly_pos))
        fly_dist = np.sum((fly_pos - np.array([self.mouseX, self.mouseY]))**2, axis=-1)
        fly_dist[self.focal_fly] = np.inf  # ensure that other_fly is not focal_fly
        self.other_fly = np.argmin(fly_dist)
        logging.debug(f"Selected {self.other_fly}.")
        self.update()

    def click_song(self, event):
        event.accept()
        if self.current_event_index == -1:
            logging.info(f'  No event type selected - see menu.')
        else:
            pos = event.pos()
            self.mouseT = self.slice_view.getPlotItem().getViewBox().mapSceneToView(pos).x()
            self.ds.song_events.sel(time=self.mouseT, method='nearest')[self.current_event_index] = True
            logging.info(f'  Added {self.ds.event_types[self.current_event_index].values} at t={self.mouseT:1.4f} seconds.')
            self.update()

    def play_audio(self):
        """Play vector as audio using the simpleaudio package."""
        if 'song' in self.ds:
            import simpleaudio
            # time0 = int(self.t0 - self.span / 2)
            # time1 = int(self.t0 + self.span / 2)
            x = self.ds.song[self.time0:self.time1].values
            # normalize to 16-bit range and convert to 16-bit data
            max_amp = self.MAX_AUDIO_AMP
            if max_amp is None:
                max_amp = np.nanmax(np.abs(x))
            x = x * 32767 / max_amp
            x = x.astype(np.int16)
            # simpleaudio can only play at these rates - choose the one nearest to our rate
            allowed_sample_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000]  # Hz
            sample_rate = min(allowed_sample_rates, key=lambda x:abs(x-int(self.fs_song)))
            # FIXME resample audio to new sample_rate to preserve sound
            # start playback in background
            simpleaudio.play_buffer(x, num_channels=1, bytes_per_sample=2, sample_rate=sample_rate)
        else:
            logging.info(f'Could not play sound - no merged-channel sound data in the dataset.')

    def plot_spec(self, x, y):
        # hash x to avoid re-calculation
        from scipy import signal
        f, t, psd = signal.spectrogram(y, self.fs_song, nperseg=self.spec_win, noverlap=self.spec_win/2, nfft=self.spec_win*4, mode='magnitude')
        f_idx = np.argmax(f > 1000)
        S = np.log2(1+psd[:f_idx, :])
        S /= np.max(S)/255
        self.spec_view.setImage(S.T[:, ::-1])
        self.spec_view.view.setXRange(0, len(t), padding=0)
        self.spec_view.view.setLimits(yMin=0, yMax=f_idx, minYRange=f_idx)

    def swap_flies(self):
        # use swap_dims to swap flies in all dataarrays in the data set?
        # TODO make sure this does not fail for datasets w/o song
        logging.info(f'   Swapping flies 1 & 2 at time {self.t0}.')
        if [self.index_other, self.focal_fly, self.other_fly] in self.swap_events:  # if already in there remove - swapping a second time would negate first swap
            self.swap_events.remove([self.index_other, self.focal_fly, self.other_fly])
        else:
            self.swap_events.append([self.index_other, self.focal_fly, self.other_fly])

        self.ds.pose_positions_allo.values[self.index_other:, [
            self.other_fly, self.focal_fly], ...] = self.ds.pose_positions_allo.values[self.index_other:, [self.focal_fly, self.other_fly], ...]
        self.ds.pose_positions.values[self.index_other:, [
            self.other_fly, self.focal_fly], ...] = self.ds.pose_positions.values[self.index_other:, [self.focal_fly, self.other_fly], ...]
        self.ds.body_positions.values[self.index_other:, [
            self.other_fly, self.focal_fly], ...] = self.ds.body_positions.values[self.index_other:, [self.focal_fly, self.other_fly], ...]
        

def main(datename: str = 'localhost-20181120_144618', root: str = '', cue_points: str = '[]'):
    """[summary]
    
    Args:
        datename (str): experiment id. Defaults to 'localhost-20181120_144618'.
        root (str): path containing the `dat` and `res` folders for the experiment. Defaults to ''.
        cue_points (str): Should evaluate to a list of indices. Defaults to '[]'.
    """
    if os.path.exists(datename + '.zarr'):
        logging.info(f'Loading ds from {datename}.zarr.')
        ds = xb.load(datename + '.zarr')
    else:
        logging.info(f'Assembling dataset for {datename}.')
        ds = xb.assemble(datename, root=root, fix_fly_indices=False)
    # logging.info(f'Saving self.ds to {datename}.zarr.')
    # xb.save(datename + '.zarr', self.ds)
    logging.info(ds)
    filepath = ds.attrs['video_filename']
    vr = _ui_utils.VideoReaderNP(filepath[:-3] + 'avi')

    cue_points = eval(cue_points)

    psv = PSV(ds, vr, cue_points)

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    defopt.run(main)
