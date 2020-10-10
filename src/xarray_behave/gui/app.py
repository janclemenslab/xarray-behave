"""PLOT SONG AND PLAY VIDEO IN SYNC

`python -m xarray_behave.ui datename root cuepoints`
"""
import os
import sys
import logging
import time
from pathlib import Path
from functools import partial
import warnings
import defopt

import numpy as np
import scipy
import skimage.draw
import scipy.interpolate
import scipy.signal as ss
import soundfile
from dataclasses import dataclass
from typing import Callable

try:
    import PySide2  # this will force pyqtgraph to use PySide instead of PyQt4/5
except ImportError:
    pass

from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

import xarray_behave
import xarray as xr
from .. import xarray_behave as xb
from .. import loaders as ld
from .. import event_utils

from . import colormaps
from . import utils
from . import views
from . import formbuilder
from .formbuilder import YamlDialog
from . import table
from . import deepss

sys.setrecursionlimit(10**6)  # increase recursion limit to avoid errors when keeping key pressed for a long time
package_dir = xarray_behave.__path__[0]


@dataclass
class DataSource:
    type: str
    name: str


class MainWindow(pg.QtGui.QMainWindow):

    def __init__(self, parent=None, title="xb.gui"):
        super().__init__(parent)

        self.windows = []
        self.parent = parent

        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtGui.QApplication([])

        self.resize(400, 400)
        self.setWindowTitle(title)

        # build menu
        self.bar = self.menuBar()
        file = self.bar.addMenu("File")
        self.add_keyed_menuitem(file, "New from wav", self.from_wav)
        self.add_keyed_menuitem(file, "New from data folder", self.from_dir)
        file.addSeparator()
        self.add_keyed_menuitem(file, "Load dataset", self.from_zarr)
        file.addSeparator()
        self.add_keyed_menuitem(file, "Save swap files", self.save_swaps)
        self.add_keyed_menuitem(file, "Save annotations", self.save_annotations)
        self.add_keyed_menuitem(file, "Export annotations to csv", self.export_annotations)
        self.add_keyed_menuitem(file, "Save dataset", self.save_dataset)
        file.addSeparator()
        file.addAction("Exit")

        # add initial buttons
        self.hb = pg.QtGui.QVBoxLayout()
        self.hb.addWidget(self.add_button("New dataset from wav", self.from_wav))
        self.hb.addWidget(self.add_button("New dataset from data folder", self.from_dir))
        self.hb.addWidget(self.add_button("Load dataset (zarr)", self.from_zarr))

        self.cb = pg.GraphicsLayoutWidget()
        self.cb.setLayout(self.hb)
        self.setCentralWidget(self.cb)

    def add_button(self, text: str, callback: Callable) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(self)
        button.setText(text)
        button.clicked.connect(callback)
        return button

    def add_keyed_menuitem(self, parent, label: str, callback, qt_keycode=None, checkable=False, checked=True):
        """Add new action to menu and register key press."""
        menuitem = parent.addAction(label)
        menuitem.setCheckable(checkable)
        menuitem.setChecked(checked)
        if qt_keycode is not None:
            menuitem.setShortcut(qt_keycode)
        menuitem.triggered.connect(lambda: callback(qt_keycode))
        return menuitem

    def save_swaps(self, qt_keycode=None):
        savefilename = Path(self.ds.attrs['root'], self.ds.attrs['res_path'],
                            self.ds.attrs['datename'], f"{self.ds.attrs['datename']}_idswaps_test.txt")
        savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save swaps to', str(savefilename),
                                                                filter="txt files (*.txt);;all files (*)")
        if len(savefilename):
            logging.info(f'   Saving list of swap indices to {savefilename}.')
            np.savetxt(savefilename, self.swap_events, fmt='%d', header='index fly1 fly2')
            logging.info(f'   Done.')

    def save_annotations(self, qt_keycode=None):
        if 'song_events' in self.ds:
            try:
                savefilename = Path(self.ds.attrs['root'], self.ds.attrs['res_path'], self.ds.attrs['datename'],
                                    f"{self.ds.attrs['datename']}_songmanual.zarr")
            except KeyError:
                savefilename = ''
            savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save annotations to', str(savefilename),
                                                                    filter="zarr files (*.zarr);;all files (*)")
            if len(savefilename):
                logging.info('   Updating song events')
                self.ds = event_utils.eventtimes_to_traces(self.ds, self.event_times)
                logging.info(f'   Saving annotations to {savefilename}.')
                # currently, can only save datasets as zarr - so convert song_events data array to dataset before saving
                xb.save(savefilename, self.ds.song_events.to_dataset())
                logging.info(f'   Done.')

    def export_annotations(self, qt_keycode=None):
        if 'song_events' in self.ds:
            try:
                import pandas as pd
            except ImportError:
                logging.warning('Need to install pandas for export to work: `conda install pandas`.')
                return

            try:
                savefilename = Path(self.ds.attrs['root'], self.ds.attrs['res_path'], self.ds.attrs['datename'],
                                    f"{self.ds.attrs['datename']}_annotations.csv")
            except KeyError:
                savefilename = ''
            savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save annotations to', str(savefilename),
                                                                    filter="csv files (*.csv);;all files (*)")
            if len(savefilename):
                logging.info('   Updating song events.')
                self.ds = event_utils.eventtimes_to_traces(self.ds, self.event_times)
                logging.info(f'   Formatting annotations for export.')

                df = pd.DataFrame()
                for event_name, event_data in self.event_times.items():
                    if event_data.ndim > 1 and event_data.shape[1] == 2:  # segments
                        df = pd.concat([df, pd.DataFrame({event_name + '_onsets': event_data[:, 0]})],
                                        axis=1)
                        df = pd.concat([df, pd.DataFrame({event_name + '_offsets': event_data[:, 1]})],
                                        axis=1)
                    elif event_data.ndim == 1:
                        df = pd.concat([df, pd.DataFrame({event_name: event_data})],
                                        axis=1)
                    else:
                        raise ValueError("Unexpected format - either shape should be (N,) for events and (N, 2) for segments.")

                logging.info(f'   Saving annotations to {savefilename}.')
                df.to_csv(savefilename, index=False)
                logging.info(f'   Done.')

    @classmethod
    def from_wav(cls, wav_filename=None, app=None, qt_keycode=None, events_string='',
                 spec_freq_min=None, spec_freq_max=None, target_samplingrate=None,
                 skip_dialog: bool = False):
        if not wav_filename:
            wav_filename, _ = QtWidgets.QFileDialog.getOpenFileName(parent=None,
                                                                    caption='Select audio file')
        # get samplerate
        if wav_filename:
            wave_fileinfo = soundfile.info(wav_filename)
            logging.info(wave_fileinfo)
            dialog = YamlDialog(yaml_file=package_dir + "/gui/forms/from_wav.yaml",
                                title=f'Make new dataset from wave file {wav_filename}')
            dialog.form['target_samplingrate'] = wave_fileinfo.samplerate
            dialog.form['spec_freq_max'] = wave_fileinfo.samplerate / 2


            # initialize form data with cli args
            if spec_freq_min is not None:
                dialog.form['spec_freq_min'] = spec_freq_min
            if spec_freq_max is not None:
                dialog.form['spec_freq_max'] = spec_freq_max
            if target_samplingrate is not None:
                dialog.form['target_samplingrate'] = target_samplingrate
            if len(events_string):
                dialog.form['init_annotations'] = True
                dialog.form['events_string'] = events_string

            if not skip_dialog:
                dialog.show()
                result = dialog.exec_()
            else:
                result = QtGui.QDialog.Accepted

            if result == QtGui.QDialog.Accepted:
                form_data = dialog.form.get_form_data()

                form_data = dialog.form.get_form_data()
                logging.info(f"Making new dataset from {wav_filename}.")
                if form_data['target_samplingrate'] < 1:
                    form_data['target_samplingrate'] = None

                event_names = []
                event_classes = []
                if form_data['init_annotations'] and len(form_data['events_string']):
                    for pair in form_data['events_string'].split(';'):
                        items = pair.strip().split(',')
                        if len(items)>0:
                            event_names.append(items[0].strip())
                        if len(items)>1:
                            event_classes.append(items[1].strip())
                        else:
                            event_classes.append('segment')

                ds = xb.from_wav(filepath=wav_filename,
                                    target_samplerate=form_data['target_samplingrate'],
                                    event_names=event_names,
                                    event_categories=event_classes)

                if form_data['filter_song'] == 'yes':
                    f_low = eval(form_data['f_low'])
                    f_high = eval(form_data['f_high'])
                    ds = cls.filter_song(ds, f_low, f_high)

                fmin = float(dialog.form['spec_freq_min']) if len(dialog.form['spec_freq_min']) else None
                fmax = float(dialog.form['spec_freq_max']) if len(dialog.form['spec_freq_max']) else None
                cue_points = []
                if form_data['load_cues']=='yes':
                    cue_points = cls.load_cuepoints(form_data['cues_file'])

                return PSV(ds, title=wav_filename, cue_points=cue_points,
                           fmin=fmin, fmax=fmax, data_source=DataSource('wav', wav_filename))

    @classmethod
    def from_dir(cls, dirname=None, app=None, qt_keycode=None, events_string='',
                 spec_freq_min=None, spec_freq_max=None, target_samplingrate=None,
                 box_size=None, skip_dialog: bool = False):
        if not dirname:
            dirname = QtWidgets.QFileDialog.getExistingDirectory(parent=None,
                                                                    caption='Select data directory')
        if dirname:
            dialog = YamlDialog(yaml_file=package_dir + "/gui/forms/from_dir.yaml",
                                title=f'Make new dataset from data directory {dirname}')

            # initialize form data with cli args
            if spec_freq_min is not None:
                dialog.form['spec_freq_min'] = spec_freq_min
            if spec_freq_max is not None:
                dialog.form['spec_freq_max'] = spec_freq_max
            if box_size is not None:
                dialog.form['box_size'] = box_size
            if target_samplingrate is not None:
                dialog.form['target_samplingrate'] = target_samplingrate
            if len(events_string):
                dialog.form['init_annotations'] = True
                dialog.form['events_string'] = events_string

            if not skip_dialog:
                dialog.show()
                result = dialog.exec_()
            else:
                result = QtGui.QDialog.Accepted

            if result == QtGui.QDialog.Accepted:
                form_data = dialog.form.get_form_data()
                logging.info(f"Making new dataset from directory {dirname}.")

                if form_data['target_samplingrate'] == 0:
                    resample_video_data = False
                else:
                    resample_video_data = True

                include_tracks = not form_data['ignore_tracks']
                include_poses = not form_data['ignore_tracks']

                base, datename = os.path.split(os.path.normpath(dirname))  # normpath removes trailing pathsep
                root, dat_path = os.path.split(base)
                ds = xb.assemble(datename, root, dat_path, res_path='res',
                                fix_fly_indices=False, include_song=~form_data['ignore_song'],
                                keep_multi_channel=True,
                                target_sampling_rate=form_data['target_samplingrate'],
                                resample_video_data=resample_video_data,
                                include_tracks=include_tracks, include_poses=include_poses)

                if form_data['filter_song'] == 'yes':
                    f_low = eval(form_data['f_low'])
                    f_high = eval(form_data['f_high'])
                    ds = cls.filter_song(ds, f_low, f_high)

                if form_data['init_annotations'] and len(form_data['events_string']):
                    # parse events_string
                    event_types = []
                    event_categories = []
                    for event in form_data['events_string'].split(';'):
                        event_types.append(event.split(',')[0])
                        event_categories.append(event.split(',')[1])
                    # add new event types
                    ds = ld.initialize_manual_song_events(ds, False, False,
                                                          event_types, event_categories)

                # add event categories if they are missing in the dataset
                if 'song_events' in ds and 'event_categories' not in ds:
                    event_categories = ['segment'
                                        if 'sine' in evt or 'syllable' in evt
                                        else 'event'
                                        for evt in ds.event_types.values]
                    ds = ds.assign_coords({'event_categories':
                                        (('event_types'),
                                        event_categories)})

                # add video file
                vr = None
                try:
                    try:
                        video_filename = os.path.join(dirname, datename + '.mp4')
                        vr = utils.VideoReaderNP(video_filename)
                    except:
                        video_filename = os.path.join(dirname, datename + '.avi')
                        vr = utils.VideoReaderNP(video_filename)
                    logging.info(vr)
                except FileNotFoundError:
                    logging.info(f'Video "{video_filename}" not found. Continuing without.')
                except:
                    logging.info(f'Something went wrong when loading the video. Continuing without.')

                fmin = float(dialog.form['spec_freq_min']) if len(dialog.form['spec_freq_min']) else None
                fmax = float(dialog.form['spec_freq_max']) if len(dialog.form['spec_freq_max']) else None
                box_size = float(dialog.form['box_size'])

                cue_points = []
                if form_data['load_cues']=='yes':
                    cue_points = cls.load_cuepoints(form_data['cues_file'])
                return PSV(ds, title=dirname, cue_points=cue_points, vr=vr,
                        fmin=fmin, fmax=fmax, box_size=box_size, data_source=DataSource('dir', dirname))

    @classmethod
    def from_zarr(cls, filename=None, app=None, qt_keycode=None,
                  spec_freq_min=None, spec_freq_max=None,
                  box_size=None, skip_dialog: bool = False):
        if not filename:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(parent=None, caption='Select dataset')
        if filename:
            dialog = YamlDialog(yaml_file=package_dir + "/gui/forms/from_zarr.yaml",
                                title=f'Load dataset from zarr file {filename}')

            # initialize form data with cli args
            if spec_freq_min is not None:
                dialog.form['spec_freq_min'] = spec_freq_min
            if spec_freq_max is not None:
                dialog.form['spec_freq_max'] = spec_freq_max
            if box_size is not None:
                dialog.form['box_size'] = box_size


            if not skip_dialog:
                dialog.show()
                result = dialog.exec_()
            else:
                result = QtGui.QDialog.Accepted

            if result == QtGui.QDialog.Accepted:
                form_data = dialog.form.get_form_data()
                logging.info(f'Loading {filename}.')
                ds = xb.load(filename, lazy=True, use_temp=True)
                if 'song_events' in ds:
                    ds.song_events.load()
                if not form_data['lazy']:
                    logging.info(f'   Loading data from ds.')
                    if 'song' in ds:
                        ds.song.load()  # non-lazy load song for faster updates
                    if 'pose_positions_allo' in ds:
                        ds.pose_positions_allo.load()  # non-lazy load song for faster updates
                    if 'sampletime' in ds:
                        ds.sampletime.load()
                    if 'song_raw' in ds:  # this will take a long time:
                        ds.song_raw.load()  # non-lazy load song for faster updates

                if form_data['filter_song'] == 'yes':
                    f_low = eval(form_data['f_low'])
                    f_high = eval(form_data['f_high'])
                    ds = cls.filter_song(ds, f_low, f_high)

                # add event categories if they are missing in the dataset
                if 'song_events' in ds and 'event_categories' not in ds:
                    event_categories = ['segment'
                                        if 'sine' in evt or 'syllable' in evt
                                        else 'event'
                                        for evt in ds.event_types.values]
                    ds = ds.assign_coords({'event_categories':
                                        (('event_types'),
                                        event_categories)})
                logging.info(ds)
                vr = None
                try:
                    video_filename = ds.attrs['video_filename']
                    vr = utils.VideoReaderNP(video_filename)
                    logging.info(vr)
                except FileNotFoundError:
                    logging.info(f'Video "{video_filename}" not found. Continuing without.')
                except:
                    logging.info(f'Something went wrong when loading the video. Continuing without.')

                fmin = float(dialog.form['spec_freq_min']) if len(dialog.form['spec_freq_min']) else None
                fmax = float(dialog.form['spec_freq_max']) if len(dialog.form['spec_freq_max']) else None
                box_size = float(dialog.form['box_size'])
                # load cues
                cue_points = []
                if form_data['load_cues']=='yes':
                    cue_points = cls.load_cuepoints(form_data['cues_file'])

                return PSV(ds, vr=vr, cue_points=cue_points, title=filename,
                        fmin=fmin, fmax=fmax, box_size=box_size, data_source=DataSource('zarr', filename))

    @classmethod
    def filter_song(cls, ds, f_low, f_high):
        if 'song_raw' in ds:  # this will take a long time:
            if f_high is None:
                f_high = ds.song_raw.attrs['sampling_rate_Hz'] / 2 - 1
            logging.info(f'Filtering `song_raw` between {f_low} and {f_high} Hz.')
            sos_bp = ss.butter(5, [f_low, f_high], 'bandpass', output='sos',
                                fs=ds.song_raw.attrs['sampling_rate_Hz'])
            ds.song_raw.data = ss.sosfiltfilt(sos_bp, ds.song_raw.data, axis=0)
        if 'song' in ds:
            if f_high is None:
                f_high = ds.song.attrs['sampling_rate_Hz'] / 2 - 1
            logging.info(f'Filtering `song` between {f_low} and {f_high} Hz.')
            sos_bp = ss.butter(5, [f_low, f_high], 'bandpass', output='sos',
                                fs=ds.song.attrs['sampling_rate_Hz'])
            ds.song.data = ss.sosfiltfilt(sos_bp, ds.song.data, axis=0)
        return ds

    @classmethod
    def from_npydir(cls, dirname=None, app=None, qt_keycode=None):
        logging.info('Not implemented yet')
        pass

    @staticmethod
    def load_cuepoints(filename=None, delimiter=','):
        if filename is None or not len(filename):
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(parent=None, caption='Select file with cue points')
        cues = []
        try:
            cues = np.loadtxt(fname=filename,
                              delimiter=delimiter)
        except FileNotFoundError:
            logging.warning(f"{filename} not found.")
        return cues

    def save_dataset(self, qt_keycode=None):
        try:
            savefilename = Path(self.ds.attrs['root'], self.ds.attrs['dat_path'], self.ds.attrs['datename'],
                                f"{self.ds.attrs['datename']}.zarr")
        except KeyError:
            savefilename = ""

        savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save dataset to', str(savefilename),
                                                                filter="zarr files (*.zarr);;all files (*)")

        if len(savefilename):
            file_exists = os.path.exists(savefilename)

            retval = QtWidgets.QMessageBox.Ignore
            if self.data_source.type=='zarr' and file_exists:
                retval = ZarrOverwriteWarning().exec_()

            if retval == QtWidgets.QMessageBox.Ignore:
                if 'song_events' in self.ds:
                    logging.info('   Updating song events')
                    self.ds = event_utils.eventtimes_to_traces(self.ds, self.event_times)
                logging.info(f'   Saving dataset to {savefilename}.')
                xb.save(savefilename, self.ds)
                logging.info(f'   Done.')
            else:
                logging.info(f'   Saving aborted.')


class ZarrOverwriteWarning(QtWidgets.QMessageBox):

    def __init__(self):
        super().__init__()
        self.setIcon(QtWidgets.QMessageBox.Warning)
        self.setText('Attempting to overwrite existing zarr file.')
        self.setInformativeText("This can corrupt the file and lead to data loss. \
                                 ABORT unless you know what you're doing.\
                                 Save to a file with a different name.")
        self.setStandardButtons(QtWidgets.QMessageBox.Ignore | QtWidgets.QMessageBox.Abort)
        self.setDefaultButton(QtWidgets.QMessageBox.Abort)
        self.setEscapeButton(QtWidgets.QMessageBox.Abort)


class PSV(MainWindow):

    MAX_AUDIO_AMP = 3.0

    def __init__(self, ds, vr=None, cue_points=[], title='xb.ui', cmap_name: str = 'turbo', box_size: int = 200,
                 fmin=None, fmax=None, data_source: DataSource=None):
        super().__init__(title=title)
        pg.setConfigOptions(useOpenGL=False)   # appears to be faster that way
        # build model:
        self.ds = ds
        self.data_source = data_source
        # self.m = views.Model(ds)
        self.vr = vr
        self.cue_points = cue_points
        # detect all event times and segment on/offsets
        if 'song_events' in ds:
            self.event_times = event_utils.detect_events(ds)
        else:
            self.event_times = dict()

        self.box_size = box_size
        self.fmin = fmin
        self.fmax = fmax

        if 'song' in self.ds:
            self.tmax = self.ds.song.shape[0]
        elif 'song_raw' in self.ds:
            self.tmax = self.ds.song_raw.shape[0]
        elif 'body_positions' in self.ds:
            # self.tmax = len(self.ds.body_positions) * 10  # TODO: get from  factor ds.attrs.body_positions.sampling_rate
            self.tmax = len(self.ds.body_positions) * ds.attrs['target_sampling_rate_Hz']
        else:
            raise ValueError('No time stamp info in dataset.')

        self.crop = True
        self.thorax_index = 8
        self.show_dot = True if 'body_positions' in self.ds else False
        self.old_show_dot_state = self.show_dot
        self.dot_size = 2
        self.show_poses = False
        self.move_poses = False
        self.circle_size = 8
        self.show_framenumber = False

        self.cue_index = -1  # set this to -1 not to 0 so that upon first increase we will jump to 0, not to 1

        self.nb_flies = np.max(self.ds.flies).values + 1 if 'flies' in self.ds.dims else 1
        self.focal_fly = 0
        self.other_fly = 1 if self.nb_flies > 1 else 0
        self.nb_bodyparts = len(self.ds.poseparts) if 'poseparts' in self.ds else 1

        self.fly_colors = utils.make_colors(self.nb_flies)
        self.bodypart_colors = utils.make_colors(self.nb_bodyparts)

        self.STOP = True
        self.swap_events = []
        self.show_spec = True
        self.spec_win = 200
        self.show_songevents = True
        self.movable_events = True
        self.move_only_current_events = True
        self.show_all_channels = True
        self.select_loudest_channel = False
        self.sinet0 = None

        if 'song_events' in self.ds:
            self.fs_other = self.ds.song_events.attrs['sampling_rate_Hz']
            self.nb_eventtypes = len(self.ds.event_types)
            self.eventype_colors = utils.make_colors(self.nb_eventtypes)
        else:
            self.fs_other = ds.attrs['target_sampling_rate_Hz']
            self.nb_eventtypes = 0
            self.eventype_colors = []

        self.nb_channels = None
        if 'song' in self.ds:
            self.fs_song = self.ds.song.attrs['sampling_rate_Hz']
        if 'song_raw' in self.ds:
            self.fs_song = self.ds.song_raw.attrs['sampling_rate_Hz']
            self.nb_channels = self.ds.song_raw.shape[1]
        else:
            self.fs_song = self.fs_other  # not sure this would work?

        if self.vr is not None:
            self.frame_interval = self.fs_song / self.vr.frame_rate  # song samples? TODO: get from self.ds
        else:
            self.frame_interval = self.fs_song / 1_000

        self._span = int(self.fs_song)
        self._t0 = int(self.span / 2)

        self.resize(1000, 800)

        # build UI/controller
        edit = self.bar.addMenu("Edit")
        self.add_keyed_menuitem(edit, "Swap flies", self.swap_flies, "X") #"X)

        view_play = self.bar.addMenu("Playback")
        self.add_keyed_menuitem(view_play, "Play video", self.toggle_playvideo, "Space",
                                checkable=True, checked=not self.STOP)
        view_play.addSeparator()
        self.add_keyed_menuitem(view_play, " < Reverse one frame", self.single_frame_reverse, "Left"),
        self.add_keyed_menuitem(view_play, "<< Reverse jump", self.jump_reverse, "A")
        self.add_keyed_menuitem(view_play, ">> Forward jump", self.jump_forward, "D")
        self.add_keyed_menuitem(view_play, " > Forward one frame", self.single_frame_advance, "Right")
        view_play.addSeparator()
        self.add_keyed_menuitem(view_play, "Load cue points", self.reload_cuepoints)  # text file with comma separated seconds or frames...
        self.add_keyed_menuitem(view_play, "Move to previous cue", self.set_prev_cuepoint, "K")
        self.add_keyed_menuitem(view_play, "Move to next cue", self.set_next_cuepoint, "L")
        view_play.addSeparator()
        self.add_keyed_menuitem(view_play, "Zoom in song", self.zoom_in_song, "W")
        self.add_keyed_menuitem(view_play, "Zoom out song", self.zoom_out_song, "S")
        view_play.addSeparator()
        self.add_keyed_menuitem(view_play, "Go to frame", self.go_to_frame)
        self.add_keyed_menuitem(view_play, "Go to time", self.go_to_time)

        view_video = self.bar.addMenu("Video")
        self.add_keyed_menuitem(view_video, "Crop frame", partial(self.toggle, 'crop'), "C",
                                checkable=True, checked=self.crop)
        self.add_keyed_menuitem(view_video, "Change focal fly", self.change_focal_fly, "F")
        view_video.addSeparator()
        self.add_keyed_menuitem(view_video, "Move poses", partial(self.toggle, 'move_poses'), "B",
                                checkable=True, checked=self.move_poses)
        view_video.addSeparator()
        self.add_keyed_menuitem(view_video, "Show fly position", partial(self.toggle, 'show_dot'), "O",
                                checkable=True, checked=self.show_dot)
        self.add_keyed_menuitem(view_video, "Show poses", partial(self.toggle, 'show_poses'), "P",
                                checkable=True, checked=self.show_poses)
        self.add_keyed_menuitem(view_video, "Show framenumber", partial(self.toggle, 'show_framenumber'), None,
                                checkable=True, checked=self.show_framenumber)

        view_audio = self.bar.addMenu("Audio")
        self.add_keyed_menuitem(view_audio, "Play waveform as audio", self.play_audio, "E")
        view_audio.addSeparator()
        self.add_keyed_menuitem(view_audio, "Initialize or edit annotation types", self.edit_annotation_types)
        view_audio.addSeparator()
        self.add_keyed_menuitem(view_audio, "Show annotations", partial(self.toggle, 'show_songevents'), "V",
                                checkable=True, checked=self.show_songevents)
        self.add_keyed_menuitem(view_audio, "Show all channels", partial(self.toggle, 'show_all_channels'), None,
                                checkable=True, checked=self.show_all_channels)
        self.add_keyed_menuitem(view_audio, "Auto-select loudest channel", partial(self.toggle, 'select_loudest_channel'), "Q",
                                checkable=True, checked=self.select_loudest_channel)
        self.add_keyed_menuitem(view_audio, "Select previous channel", self.set_next_channel, "Up")
        self.add_keyed_menuitem(view_audio, "Select next channel", self.set_prev_channel, "Down")
        view_audio.addSeparator()
        self.add_keyed_menuitem(view_audio, "Show spectrogram", partial(self.toggle, 'show_spec'), "G",
                                checkable=True, checked=self.show_spec)
        self.add_keyed_menuitem(view_audio, "Increase frequency resolution", self.dec_freq_res, "R")
        self.add_keyed_menuitem(view_audio, "Increase temporal resolution", self.inc_freq_res, "T")
        view_audio.addSeparator()
        self.add_keyed_menuitem(view_audio, "Move events", partial(self.toggle, 'movable_events'), "M",
                                checkable=True, checked=self.movable_events)
        self.add_keyed_menuitem(view_audio, "Move only selected events", partial(self.toggle, 'move_only_current_events'), None,
                                checkable=True, checked=self.move_only_current_events)
        self.add_keyed_menuitem(view_audio, "Delete events of selected type in view",
                                self.delete_current_events, "U")
        self.add_keyed_menuitem(view_audio, "Delete all events in view", self.delete_all_events, "Y")

        view_train = self.bar.addMenu("Training/Inference")
        self.add_keyed_menuitem(view_train, "Train", self.deepss_train, None)
        self.add_keyed_menuitem(view_train, "Predict", self.deepss_predict, None)

        self.bar.addMenu("View")

        self.hl = pg.QtGui.QHBoxLayout()

        # EVENT TYPE selector

        self.cb = pg.QtGui.QComboBox()
        self.cb.currentIndexChanged.connect(self.update_xy)
        self.cb.setCurrentIndex(0)
        self.hl.addWidget(self.cb)

        view_audio.addSeparator()
        self.view_audio = view_audio

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
            self.movie_view = views.MovieView(model=self, callback=self.on_video_clicked)

        if cmap_name in colormaps.cmaps:
            colormap = colormaps.cmaps[cmap_name]
            colormap._init()
            lut = (colormap._lut * 255).view(np.ndarray)  # convert matplotlib colormap from 0-1 to 0 -255 for Qt
        else:
            logging.warning(f'Unknown colormap "{cmap_name}"" provided. Using default (turbo).')

        self.slice_view = views.TraceView(model=self, callback=self.on_trace_clicked)
        self.spec_view = views.SpecView(model=self, callback=self.on_trace_clicked, colormap=lut)

        self.cw = pg.GraphicsLayoutWidget()

        self.ly = pg.QtGui.QVBoxLayout()
        self.ly.addLayout(self.hl)
        if self.vr is not None:
            self.ly.addWidget(self.movie_view, stretch=4)
        self.ly.addWidget(self.slice_view, stretch=1)
        self.ly.addWidget(self.spec_view, stretch=1)

        self.cw.setLayout(self.ly)

        self.setCentralWidget(self.cw)

        self.update_eventtype_selector()
        self.show()

        self.update_xy()
        self.update_frame()
        logging.info("xb gui initialized.")

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
    def trange(self):
        return np.array([self.time0, self.time1]) / self.fs_song

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, val):
        self._t0 = np.clip(val, self.span / 2, self.tmax - self.span / 2)  # ensure t0 stays within bounds
        self.update_xy()
        self.update_frame()

    @property
    def framenumber(self):
        return int(self.ds.coords['nearest_frame'][self.index_other].data)

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

    def toggle(self, var_name, qt_keycode):
        try:
            self.__dict__[var_name] = not self.__dict__[var_name]
            if self.STOP:
                self.update_frame()
                self.update_xy()
        except KeyError as e:
            logging.exception(e)

    def _delete_events_from_interval(self, event_name, t0, t1):
        """[summary]

        Args:
            event_name ([type]): [description]
            t0 ([type]): [description], seconds
            t1 ([type]): [description], seconds
        """
        this_event_times = self.event_times[event_name]
        if this_event_times.ndim == 1:  # event
            before = this_event_times<t0
            after = this_event_times>t1
        elif this_event_times.ndim == 2:  # segment
            # use any here to match both on and offsets (axis=1) - will leave partially displayed segments
            before = np.any(this_event_times < t0, axis=1)
            after = np.any(this_event_times > t1, axis=1)
        outside_of_view = np.logical_or(before, after)
        this_event_times = this_event_times[outside_of_view]
        self.event_times[event_name] = this_event_times


    def delete_current_events(self, qt_keycode):
        if self.current_event_index is not None:
            self._delete_events_from_interval(self.current_event_name,
                                              self.time0 / self.fs_song,
                                              self.time1 / self.fs_song)
            logging.info(f'   Deleted all {self.ds.event_types[self.current_event_index].values} in view.')
        else:
            logging.info(f'   No event type selected. Not deleting anything.')
        if self.STOP:
            self.update_xy()

    def delete_all_events(self, qt_keycode):
        # TODO currently not functional - needs to work on event_times (really??)
        for event_name in self.event_times.keys():
            self._delete_events_from_interval(event_name,
                                              self.time0 / self.fs_song,
                                              self.time1 / self.fs_song)
        logging.info(f'   Deleted all events in view.')
        if self.STOP:
            self.update_xy()

    def set_prev_channel(self, qt_keycode):
        idx = self.cb2.currentIndex()
        idx -= 1
        idx  = idx % self.cb2.count()

        old_status = self.select_loudest_channel
        self.select_loudest_channel = False
        self.cb2.setCurrentIndex(idx)
        self.select_loudest_channel = old_status

    def set_next_channel(self, qt_keycode):
        idx = self.cb2.currentIndex()
        idx += 1
        idx  = idx % self.cb2.count()

        old_status = self.select_loudest_channel
        self.select_loudest_channel = False
        self.cb2.setCurrentIndex(idx)
        self.select_loudest_channel = old_status

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
        if len(self.cue_points):
            self.cue_index = max(0, self.cue_index - 1)
            logging.debug(f'cue val at cue_index {self.cue_index} is {self.cue_points[self.cue_index]}')
            self.t0 = self.cue_points[self.cue_index]  * self.fs_song # jump to PREV cue point

    def set_next_cuepoint(self, qt_keycode):
        if len(self.cue_points):
            self.cue_index = min(self.cue_index + 1, len(self.cue_points) - 1)
            logging.debug(f'cue val at cue_index {self.cue_index} is {self.cue_points[self.cue_index]}')
            self.t0 = self.cue_points[self.cue_index] * self.fs_song # jump to PREV cue point

    def reload_cuepoints(self, qt_keycode):
        self.cue_points = PSV.load_cuepoints()
        self.cue_index = -1

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

    def go_to_frame(self, qt_keycode):
        fn, okPressed = QtGui.QInputDialog.getInt(self, "Enter frame number", "Frame number:",
                            value=self.framenumber, min=0, max=np.max(self.ds.nearest_frame.data), step=1)
        if okPressed:
            time_index = np.argmax(self.ds.nearest_frame.data>fn)
            self.t0 = int(time_index / self.fs_other * self.fs_song)

    def go_to_time(self, qt_keycode):
        time, okPressed = QtGui.QInputDialog.getDouble(self, "Enter time", "Seconds:",
                value=self.t0 / self.fs_song, min=0, max=self.tmax /self.fs_song)
        if okPressed:
            self.t0 = np.argmax(self.ds.sampletime.data>time)

    def update_xy(self):
        self.x = self.ds.sampletime.data[self.time0:self.time1]
        self.step = int(max(1, np.ceil(len(self.x) / self.fs_song / 2)))  # make sure step is >= 1
        self.y_other = None

        if 'song' in self.ds and self.current_channel_name == 'Merged channels':
            self.y = self.ds.song.data[self.time0:self.time1]
        elif 'song_raw' in self.ds:
            # load song for current channel
            try:
                y_all = self.ds.song_raw.data[self.time0:self.time1, :].compute()
            except AttributeError:
                y_all = self.ds.song_raw.data[self.time0:self.time1, :]

            self.y = y_all[:, self.current_channel_index]
            if self.show_all_channels:
                channel_list = np.delete(np.arange(self.nb_channels), self.current_channel_index)
                self.y_other = y_all[:, channel_list]

            # self.select_loudest_channel = True
            if self.select_loudest_channel:
                self.loudest_channel = np.argmax(np.max(y_all, axis=0))
                self.cb2.setCurrentIndex(self.loudest_channel)

        self.slice_view.update_trace()

        self.spec_view.clear_annotations()
        if self.show_spec:
            self.spec_view.update_spec(self.x, self.y)
        else:
            self.spec_view.clear()

        if self.show_songevents:
            self.plot_song_events(self.x)

    def update_frame(self):
        if self.vr is not None:
            self.movie_view.update_frame()

    def plot_song_events(self, x):
        for event_type in range(self.nb_eventtypes):
            movable = self.STOP and self.movable_events
            event_name = self.ds.event_types.values[event_type]
            if self.move_only_current_events:
                movable = movable and self.current_event_index==event_type

            event_pen = pg.mkPen(color=self.eventype_colors[event_type], width=1)
            event_brush = pg.mkBrush(color=[*self.eventype_colors[event_type], 25])

            this = self.event_times[event_name]
            if self.ds.event_categories.data[event_type] == 'segment':
                onsets_in_view = np.logical_and(this[:,0] >= x[0], this[:,0] < x[-1])
                offsets_in_view = np.logical_and(this[:,1] > x[0], this[:,1] <= x[-1])
                events_in_view = np.logical_or(onsets_in_view, offsets_in_view)
                event_onset_indices = np.sort(this[events_in_view, 0])
                event_offset_indices = np.sort(this[events_in_view, 1])
                if len(event_onset_indices) and len(event_offset_indices):
                    for onset, offset in zip(event_onset_indices, event_offset_indices):
                        self.slice_view.add_segment(onset, offset, event_type, event_brush, movable=movable)
                        if self.show_spec:
                            self.spec_view.add_segment(onset, offset, event_type, event_brush, movable=movable)
            elif len(this):
                events = this[np.logical_and(this>x[0], this<x[-1])]
                if len(events):
                    self.slice_view.add_event(events, event_type, event_pen, movable=movable)
                    if self.show_spec:
                        self.spec_view.add_event(events, event_type, event_pen, movable=movable)

    def play_video(self):  # TODO: get rate from ds (video fps attr)
        RUN = True
        cnt = 0
        dt0 = time.time()
        while RUN:
            self.t0 += self.frame_interval
            cnt += 1
            if cnt % 10 == 0:
                # logging.debug(time.time() - dt0)
                dt0 = time.time()
            if self.STOP:
                RUN = False
                self.update_xy()
                self.update_frame()
                logging.debug('   Stopped playback.')
            self.app.processEvents()

    def on_region_change_finished(self, region):
        """Called when dragging a segment-like song_event - will change its bounds."""
        if self.move_only_current_events and self.current_event_index != region.event_index:
            return

        # this could be done by the view:
        f = scipy.interpolate.interp1d(region.xrange, self.trange,
                                       bounds_error=False, fill_value='extrapolate')


        # find the moved segment in events
        this = self.event_times[self.current_event_name]
        event_idx = np.where(np.logical_and(this[:,0]==region.bounds[0],
                                            this[:,1]==region.bounds[1]))

        # get the new region bounds
        new_region = f(region.getRegion())

        # fixes editing segments that are only partially displayed
        # if the region bounds after editing match the bounds of the view,
        # keep the old region bounds
        if region.getRegion()[0] == region.xrange[0]:
            new_region[0] = this[event_idx, 0]
        if region.getRegion()[1] == region.xrange[1]:
            new_region[1] = this[event_idx, 1]

        # replace the move segment
        this[event_idx, :] = new_region

        logging.info(f'  Moved {self.current_event_name} from t=[{region.bounds[0]:1.4f}:{region.bounds[1]:1.4f}] to [{new_region[0]:1.4f}:{new_region[1]:1.4f}] seconds.')
        self.update_xy()

    def on_position_change_finished(self, position):
        """Called when dragging an event-like song_event - will change time."""
        if self.move_only_current_events and self.current_event_index != position.event_index:
            return

        # this should be done by the view:
        # convert position to time coordinates (important for spec_view since x-axis does not correspond to time)
        f = scipy.interpolate.interp1d(position.xrange, self.trange,
                                       bounds_error=False, fill_value='extrapolate')
        new_position = f(position.pos()[0])
        # set new position in ds
        this = self.event_times[self.current_event_name]
        this[np.where(this==position.position)] = new_position
        logging.info(f'  Moved {self.ds.event_types.values[position.event_index]} from t=[{position.position:1.4f} to {new_position:1.4f} seconds.')
        self.update_xy()

    def on_position_dragged(self, fly, pos, offset):
        """Called when dragging a fly body position - will change that pos."""
        pos0 = self.ds.pose_positions_allo.data[self.index_other, fly, self.thorax_index]
        try:
            pos1 = [pos.y(), pos.x()]
        except:
            pos1 = pos
        self.ds.pose_positions_allo.data[self.index_other, fly, :] += (pos1 - pos0)
        logging.info(f'   Moved fly from {pos0} to {pos1}.')
        self.update_frame()

    def on_poses_dragged(self, ind, pos, offset):
        """Called when dragging a fly body position - will change that pos."""
        fly, part = np.unravel_index(ind, (self.nb_flies, self.nb_bodyparts))
        pos0 = self.ds.pose_positions_allo.data[self.index_other, fly, part]
        try:
            pos1 = [pos.y(), pos.x()]
        except:
            pos1 = pos
        self.ds.pose_positions_allo.data[self.index_other, fly, part] += (pos1 - pos0)
        logging.info(f'   Moved {self.ds.poseparts[part].data} of fly {fly} from {pos0} to {pos1}.')
        self.update_frame()

    def on_video_clicked(self, mouseX, mouseY, event):
        """Called when clicking the video - will select the focal fly."""
        if event.modifiers() == QtCore.Qt.ControlModifier and self.focal_fly is not None:
            self.on_position_dragged(self.focal_fly, pos=[mouseY, mouseX], offset=None)
        else:
            fly_pos = self.ds.pose_positions_allo.data[self.index_other, :, self.thorax_index, :]
            fly_pos = np.array(fly_pos)  # in case this is a dask.array
            if self.crop:  # transform fly pos to coordinates of the cropped box
                box_center = self.ds.pose_positions_allo.data[self.index_other,
                                                            self.focal_fly,
                                                            self.thorax_index] + self.box_size / 2
                box_center = np.array(box_center)  # in case this is a dask.array
                fly_pos = fly_pos - box_center
            fly_dist = np.sum((fly_pos - np.array([mouseY, mouseX]))**2, axis=-1)
            fly_dist[self.focal_fly] = np.inf  # ensure that other_fly is not focal_fly
            self.other_fly = np.argmin(fly_dist)
            logging.debug(f"Selected {self.other_fly}.")
        self.update_frame()

    def on_trace_clicked(self, mouseT, mouseButton):
        """Called when traceview or specview have been clicked - will add new
        song event at click position.
        """
        if self.current_event_index is None:
            return
        if mouseButton == 1:  # add event
            if self.current_event_index is not None:
                if self.ds.event_categories.data[self.current_event_index] == 'segment':
                    if self.sinet0 is None:
                        self.spec_view.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
                        self.slice_view.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
                        self.sinet0 = mouseT
                    else:
                        self.spec_view.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
                        self.slice_view.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
                        self.event_times[self.current_event_name] = np.concatenate((self.event_times[self.current_event_name],
                                                                                    np.array(sorted([self.sinet0, mouseT]))[np.newaxis,:]),
                                                                                   axis=0)
                        logging.info(f'  Added {self.current_event_name} at t=[{self.sinet0:1.4f}:{mouseT:1.4f}] seconds.')
                        self.sinet0 = None
                else:  # pulse-like event
                    self.sinet0 = None
                    self.event_times[self.current_event_name] = np.append(self.event_times[self.current_event_name], mouseT)
                    logging.info(f'  Added {self.current_event_name} at t={mouseT:1.4f} seconds.')
                self.update_xy()
            else:
                self.sinet0 = None
        elif mouseButton == 2:  #delete nearest event
            self.sinet0 = None
            self.spec_view.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            self.slice_view.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            this = self.event_times[self.current_event_name]
            if self.ds.event_categories.data[self.current_event_index] == 'segment':
                nearest_onset = float(utils.find_nearest(this[:,0], mouseT))
                event_idx = np.where(this[:, 0]==nearest_onset)[0]
                matching_offset = float(this[event_idx, 1])
                event_at_mouseT = matching_offset > mouseT
                if event_at_mouseT:
                    self.event_times[self.current_event_name] = np.delete(this, event_idx, axis=0)
                    logging.info(f'  Deleted {self.current_event_name} from {nearest_onset:1.4f} to {matching_offset:1.4f} seconds.')
            elif self.ds.event_categories.data[self.current_event_index] == 'event':
                tol = 0.05
                nearest_event = utils.find_nearest(this, mouseT)
                event_at_mouseT = np.abs(mouseT - nearest_event) < tol
                if event_at_mouseT:
                    self.event_times[self.current_event_name] = np.delete(this, np.where(this==nearest_event)[0])
                    logging.info(f'  Deleted {self.current_event_name} at t={nearest_event:1.4f} seconds.')
            self.update_xy()

    def play_audio(self, qt_keycode):
        """Play vector as audio using the simpleaudio package."""

        if 'song' in self.ds or 'song_raw' in self.ds:
            has_sounddevice = False
            has_simpleaudio = False
            try:
                import sounddevice as sd
                has_sounddevice = True
            except (ImportError, ModuleNotFoundError):
                logging.info('Could not import python-sounddevice. Maybe you need to install it.\
                              See https://python-sounddevice.readthedocs.io/en/latest/installation.html for instructions.\
                              \
                              Trying to fall back to simpleaudio, which may be buggy.')

            if not has_sounddevice:
                try:
                    import simpleaudio
                    has_simpleaudio = True

                except (ImportError, ModuleNotFoundError):
                    logging.info('Could not import simpleaudio. Maybe you need to install it.\
                                See https://simpleaudio.readthedocs.io/en/latest/installation.html for instructions.')
                    return

            if has_sounddevice or has_simpleaudio:
                if 'song' in self.ds and self.current_channel_name == 'Merged channels':
                    y = self.ds.song.data[self.time0:self.time1]
                else:
                    y = self.ds.song_raw.data[self.time0:self.time1, self.current_channel_index]
                y = np.array(y)  # if y is a dask.array (lazy loaded)
            if has_sounddevice:
                sd.play(y, self.fs_song)
            elif has_simpleaudio:
                # normalize to 16-bit range and convert to 16-bit data
                max_amp = self.MAX_AUDIO_AMP
                if max_amp is None:
                    max_amp = np.nanmax(np.abs(y))
                y = y * 32767 / max_amp
                y = y.astype(np.int16)
                # simpleaudio can only play at these rates - choose the one nearest to our rate
                allowed_sample_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000]  # Hz
                sample_rate = min(allowed_sample_rates, key=lambda x: abs(x - int(self.fs_song)))
                # start playback in background
                simpleaudio.play_buffer(y, num_channels=1, bytes_per_sample=2, sample_rate=sample_rate)
            else:
                logging.info(f'No sound module installed - install python-sounddevice')
        else:
            logging.info(f'Could not play sound - no sound data in the dataset.')

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
            self.update_frame()

    def deepss_train(self, qt_keycode):

        def train(form_data):
            print(form_data)
            print('train')
            dialog.close()

        def save(arg):
            print(dialog.form.get_form_data())
            print('save')

        def load(arg):
            print(dialog.form.get_form_data())
            print('load')

        dialog = self._dialog_from_yaml(yaml_file="src/xarray_behave/gui/forms/train.yaml",
                               title='Train network',
                               main_callback=train,
                               callbacks={'save':save, 'load':load})
        dialog.show()

    def deepss_predict(self, qt_keycode):
        logging.info('DeepSS...')
        ret = deepss.predict(self.ds)
        if ret is not None:
            events, segments = ret

            # add to self.event_times
            # remove non-song keys
            samplerate_Hz = events['samplerate_Hz']
            del events['samplerate_Hz']
            del segments['samplerate_Hz']
            if 'noise' in segments:
                del segments['noise']

            if 'song_events' in self.ds:
                self.ds = event_utils.eventtimes_to_traces(self.ds, self.event_times)

            for event_name, event_data in events.items():
                self.event_times[event_name + '_dss'] = event_data['seconds']

            for segment_name, segment_data in segments.items():
                self.event_times[segment_name + '_dss'] = np.stack((segment_data['onsets_seconds'], segment_data['offsets_seconds']), axis=1)

            event_categories = {}
            for event_name, event_data in self.event_times.items():
                event_categories[event_name] = 'event' if event_data.ndim==1 else 'segment'

            self.ds = event_utils.update_traces(self.ds, self.event_times)

            self.fs_other = self.ds.song_events.attrs['sampling_rate_Hz']
            self.nb_eventtypes = len(self.ds.event_types)
            self.eventype_colors = utils.make_colors(self.nb_eventtypes)
            logging.info('  done.')
            self.update_eventtype_selector()


    def edit_annotation_types(self, qt_keycode):

        if 'event_types' in self.ds:
            types = self.ds.event_types.coords['event_types'].data
            cats = self.ds.event_types.coords['event_categories'].data
            table_data = [[typ, cat] for typ, cat in zip(types, cats)]
        else:
            table_data = []

        dialog = table.Table(table_data)
        dialog.show()
        result = dialog.exec_()
        if result == QtGui.QDialog.Accepted:
            data = dialog.get_table_data()
            # make sure event times are up to date
            if 'song_events' in self.ds:
                self.ds = event_utils.eventtimes_to_traces(self.ds, self.event_times)

            # now edit self.event_times
            event_names = []
            event_names_old = []
            event_categories = []
            for item in data:
                event_name, event_name_old = item[0]
                event_names.append(event_name)
                event_names_old.append(event_name_old)
                event_categories.append(item[1][0])

            # deletions: deletion in table will remove the entry from the list -
            # so it's name will not even be in event_names_old anymore
            event_names_current = list(self.event_times.keys())
            for event_name_current in event_names_current:
                if event_name_current not in event_names_old:
                    del self.event_times[event_name_current]

            # propagate existing and create new
            for event_name, event_category, event_name_old in zip(event_names, event_categories, event_names_old):
                if event_name_old in self.event_times and event_name != event_name_old:  # rename existing
                    self.event_times[event_name] = self.event_times.pop(event_name_old)
                    # event_categories[event_name] = event_categories.pop(event_name_old)
                elif event_name_old not in self.event_times:  # create new empty
                    if event_category=='segment':
                        self.event_times[event_name] = np.zeros((0, 2))
                    elif event_category=='event':
                        self.event_times[event_name] = np.zeros((0,))


            self.ds = event_utils.update_traces(self.ds, self.event_times)

            # update event-related attrs
            self.fs_other = self.ds.song_events.attrs['sampling_rate_Hz']
            self.nb_eventtypes = len(self.ds.event_types)
            self.eventype_colors = utils.make_colors(self.nb_eventtypes)

            self.update_eventtype_selector()

    def update_eventtype_selector(self):

        while self.cb.count() > 0:
            self.cb.removeItem(0)

        self.cb.addItem("No annotation")
        if 'event_types' in self.ds:
            self.eventList = [(cnt, evt) for cnt, evt in enumerate(self.ds.event_types.values)]
            self.eventList = sorted(self.eventList)
        else:
            self.eventList = []

        for event_type in self.eventList:
            self.cb.addItem("Add " + event_type[1])

        self.cb.currentIndexChanged.connect(self.update_xy)
        self.cb.setCurrentIndex(0)

        # update menus
        # remove associated menu items
        if not hasattr(self, 'event_items'):
            self.event_items = []
        else:
            for event_item in self.event_items:
                try:
                    self.view_audio.removeAction(event_item)
                except ValueError:
                    logging.warning('item not in actions')  # item not in actions

        # add new ones (make this function)
        self.event_items = []
        for ii in range(self.cb.count()):
            key = str(ii) if ii<10 else None
            key_label = f"({key})" if key is not None else ''
            self.cb.setItemText(ii, f"{self.cb.itemText(ii)} {key_label}")
            menu_item = self.add_keyed_menuitem(self.view_audio,
                                                self.cb.itemText(ii),
                                                self.change_event_type,
                                                key)
            self.event_items.append(menu_item)

        # color events in combobox as in slice_view and spec_view
        children = self.cb.children()
        itemList = children[0]
        for ii, col in zip(range(1, itemList.rowCount()), self.eventype_colors):
            itemList.item(ii).setForeground(QtGui.QColor(*col))


def main(source: str = '', *, events_string: str = '', target_samplingrate: float = None,
         spec_freq_min: float = None, spec_freq_max: float=None, box_size: int = 200,
         skip_dialog: bool = False):
    """
    Args:
        source (str): Data source to load.
            Optional - will open an empty file if omitted.
            Source can be the path to a wav audio file,
            to an xarray-behave dataset saved as a zarr file,
            or to a data folder (e.g. 'dat/localhost-xxx').
        events_string (str): "event_name,event_category;event_name,event_category".
                             Avoid spaces or trailing ';'.
                             Need to wrap the string in "..." in the terminal
                             "event_name" can be any string w/o space, ",", or ";"
                             "event_category" can by event (pulse) or segment (sine, syllable)
                             Only used if source is a data folder or a wav audio file.
        target_samplingrate (float): [description]. If 0, will use frame times. Defaults to None.
                                     Only used if source is a data folder or a wav audio file.
        spec_freq_min (float): [description].
        spec_freq_max (float): [description].
        box_size (int): desc.
                        Not used for wav audio files (no videos).
        skip_dialog (bool): If True, skips the opening dialog and goes straight to opening the data.
    """

    QtGui.QApplication([])
    mainwin = MainWindow()
    mainwin.show()
    if not len(source):
        pass
    elif source.endswith('.wav'):
        mainwin.windows.append(MainWindow.from_wav(wav_filename=source,
                                                   events_string=events_string,
                                                   target_samplingrate=target_samplingrate,
                                                   spec_freq_min=spec_freq_min, spec_freq_max=spec_freq_max,
                                                   skip_dialog=skip_dialog))
    elif source.endswith('.zarr'):
        mainwin.windows.append(MainWindow.from_zarr(filename=source,
                                                    box_size=box_size,
                                                    spec_freq_min=spec_freq_min, spec_freq_max=spec_freq_max,
                                                    skip_dialog=skip_dialog))
    elif os.path.isdir(source):
        mainwin.windows.append(MainWindow.from_dir(dirname=source,
                                                   events_string=events_string,
                                                   target_samplingrate=target_samplingrate, box_size=box_size,
                                                   spec_freq_min=spec_freq_min, spec_freq_max=spec_freq_max,
                                                   skip_dialog=skip_dialog))

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

def cli():
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)
    defopt.run(main)

if __name__ == '__main__':
    cli()
