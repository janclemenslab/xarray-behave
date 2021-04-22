"""PLOT SONG AND PLAY VIDEO IN SYNC

`python -m xarray_behave.ui datename root cuepoints`
"""
import os
os.environ['QT_MAC_WANTS_LAYER'] = '1'  # fixed app window not showing on macOS Big Sur

import sys
import logging
import time
from pathlib import Path
from functools import partial

import defopt
import yaml
import h5py
import functools

import numpy as np
import scipy.interpolate
import scipy.signal as ss
import pathlib
import peakutils
from typing import Callable, Optional, Dict, Any, List

try:
    import PySide2  # this will force pyqtgraph to use PySide instead of PyQt4/5
except ImportError:
    pass

from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.Qt as Qt


# # otherwise magicgui's qtpy defaults to pyqt and all hell breaks loose
# os.environ["QT_API"] = "pyside2"
# from magicgui import magicgui, magic_factory


import xarray_behave
from .. import (xarray_behave as xb,
                loaders as ld,
                annot,
                event_utils)

from . import (colormaps,
               utils,
               views,
               table)

try:
    from . import deepss
except Exception as e:
    logging.exception(e)
    logging.warning(f'Failed to import the DeepSS module.\nIgnore if you do not want to use DeepSS.\nOtherwise follow these instructions to install:\nhttps://janclemenslab.org/deepss/install.html')

from .formbuilder import YamlDialog


sys.setrecursionlimit(10**6)  # increase recursion limit to avoid errors when keeping key pressed for a long time
package_dir = xarray_behave.__path__[0]


# @dataclass
class DataSource:
    def __init__(self, type: str, name: str):
        self.type = type
        self.name = name


class MainWindow(pg.QtGui.QMainWindow):

    def __init__(self, parent=None, title="DeepSS"):
        super().__init__(parent)

        self.parent = parent

        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtGui.QApplication([])
        self.app.setWindowIcon(QtGui.QIcon(package_dir  + '/gui/icon.png'))

        self.resize(400, 200)
        self.setWindowTitle(title)
        self.setWindowIcon(QtGui.QIcon(package_dir  + '/gui/icon.png'))
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # build menu
        self.bar = self.menuBar()

        self.file_menu = self.bar.addMenu("File")
        self._add_keyed_menuitem(self.file_menu, "New from file", self.from_file)
        self._add_keyed_menuitem(self.file_menu, "New from ethodrome folder", self.from_dir)
        self.file_menu.addSeparator()
        self._add_keyed_menuitem(self.file_menu, "Load dataset", self.from_zarr)
        self.file_menu.addSeparator()
        self.file_menu.addAction("Exit")

        view_train = self.bar.addMenu("DeepSS")
        self._add_keyed_menuitem(view_train, "Make dataset for training", self.deepss_make, None)
        self._add_keyed_menuitem(view_train, "Train", self.deepss_train, None)

        # add initial buttons
        self.hb = pg.QtGui.QVBoxLayout()
        self.hb.addWidget(self.add_button("Load audio from file", self.from_file))
        self.hb.addWidget(self.add_button("Create dataset from ethodrome folder", self.from_dir))
        self.hb.addWidget(self.add_button("Load dataset (zarr)", self.from_zarr))

        self.cb = pg.GraphicsLayoutWidget()
        self.cb.setLayout(self.hb)
        self.setCentralWidget(self.cb)

    def closeEvent(self, event):
        # stuff_to_delete = ['ds', 'vr', 'x','y','y_other', 'event_times', 'slice_view', 'spec_view', 'movie_view', 'envelope']
        stuff_to_delete = list(self.__dict__.keys())
        for stuff in stuff_to_delete:
            try:
                del self.__dict__[stuff]
            except KeyError:
                pass
            except Exception as e:
                print(e)
        import gc
        gc.collect()
        event.accept()

    def add_button(self, text: str, callback: Callable) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(self)
        button.setText(text)
        button.clicked.connect(callback)
        return button

    def _add_keyed_menuitem(self, parent, label: str, callback, qt_keycode=None, checkable=False, checked=True):
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
                            self.ds.attrs['datename'], f"{self.ds.attrs['datename']}_idswaps.txt")
        savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save swaps to', str(savefilename),
                                                                filter="txt files (*.txt);;all files (*)")
        if len(savefilename):
            logging.info(f'   Saving list of swap indices to {savefilename}.')
            np.savetxt(savefilename, self.swap_events, fmt='%d', header='index fly1 fly2')
            logging.info(f'Done.')

    def save_annotations(self, qt_keycode=None):
        """Save annotations to csv.
        Each annotation is a row with name, start_seconds, stop_seconds.
        start_seconds = stop_seconds for events like pulses."""
        try:
            csv_filename = os.path.splitext(self.ds.attrs['datename'])[0] + '_annotations.csv'
            savefilename = Path(self.ds.attrs['root'], self.ds.attrs['res_path'], self.ds.attrs['datename'],
                                csv_filename)
        except KeyError:
            savefilename = ''
        savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save annotations to', str(savefilename),
                                                                filter="CSV files (*.csv);;all files (*)")
        if len(savefilename):
            logging.info(f'   Saving annotations to {savefilename}.')
            self.export_to_csv(savefilename)
            logging.info(f'Done.')


    def export_to_csv(self, savefilename: str = None,
                      start_seconds: float = 0, end_seconds: float = np.inf,
                      which_events: Optional[List[str]] = None,
                      match_to_samples: bool = False,
                      qt_keycode=None):
        """[summary]

        Args:
            savefilename (str, optional): [description]. Defaults to None.
            start_seconds (int, optional): [description]. Defaults to 0.
            end_seconds ([type], optional): [description]. Defaults to None.
            which_events ([type], optional): [description]. Defaults to None.
            match_to_samples (bool, optiona): Will adjust seconds so that seconds * samplerate yields the correct index.
                                              Otherwise, seconds will correspond to the correct time stamp of the event sample.
                                              Only relevant for xb.datasets with timestamp info.
                                              Defaults to False.
            qt_keycode ([type], optional): [description]. Defaults to None.
        """
        if which_events is None:
            which_events = self.event_times.names

        samplerate = self.fs_song
        event_times = annot.Events(self.event_times)
        for name in event_times.names:
            if name not in which_events:
                event_times.delete_name(name)
            else:
                if match_to_samples:
                    idx = utils.find_nearest_idx(self.ds.sampletime, event_times[name])
                    expected_time = idx / samplerate
                    error = expected_time - event_times[name]
                    times_correct = self.ds.sampletime.data[idx] + error
                    event_times[name] = times_correct
                event_times[name] = event_times.filter_range(name, start_seconds, end_seconds) - start_seconds

        df = event_times.to_df(preserve_empty=True)
        df = df.sort_values(by='start_seconds', ascending=True, ignore_index=True)
        df.to_csv(savefilename, index=False)


    def export_to_wav(self, savefilename: Optional[str] = None,
                      start_seconds: float = 0, end_seconds: Optional[float] = None,
                      scale: float = 1.0):
        """[summary]

        Args:
            savefilename (str, optional): [description]. Defaults to None.
            start_seconds (int, optional): [description]. Defaults to 0.
            end_seconds ([type], optional): [description]. Defaults to None.
            scale (float):
        """
        # transform to indices for cutting song data
        start_index = int(start_seconds * self.fs_song)
        end_index = None
        if end_seconds is not None:
            end_index = int(end_seconds * self.fs_song)

        # get clean data
        song = np.array(self.ds.song_raw.data[start_index:end_index])

        # scale if necessary while preserving the dtype
        original_dtype = song.dtype
        song = (song * scale).astype(original_dtype)

        # can only save float32 (or int) to WAV - other float formats will lead to corrupt files
        if song.dtype == 'float64' or song.dtype == 'float16':
            song = song.astype(np.float32)
        if song.dtype == 'int64':
            song = song.astype(np.int32)

        scipy.io.wavfile.write(savefilename, int(self.fs_song), song)


    def export_to_npz(self, savefilename: Optional[str] = None,
                      start_seconds: float = 0, end_seconds: Optional[float] = None,
                      scale: float = 1.0):
        """[summary]

        Args:
            savefilename (str, optional): [description]. Defaults to None.
            start_seconds (int, optional): [description]. Defaults to 0.
            end_seconds ([type], optional): [description]. Defaults to None.
            scale (float):
        """
        # transform to indices for cutting song data
        start_index = int(start_seconds * self.fs_song)
        end_index = None
        if end_seconds is not None:
            end_index = int(end_seconds * self.fs_song)

        # get clean data and save to WAV
        song = self.ds.song_raw.data[start_index:end_index]
        song = song * scale
        try:
            song = song.compute()
        except AttributeError:
            pass
        np.savez(savefilename, song=song, samplerate=self.fs_song)

    def export_for_deepss(self, qt_keycode=None):
        try:
            file_trunk = os.path.splitext(self.ds.attrs['datename'])[0]

            savefilename = Path(self.ds.attrs['root'], self.ds.attrs['res_path'], self.ds.attrs['datename'],
                                file_trunk)
        except KeyError:
            savefilename = ''

        savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export to', str(savefilename),
                                                                filter="all files (*)")
        if not savefilename:
            return

        savefilename_trunk = os.path.splitext(savefilename)[0]

        yaml_file=yaml_file=package_dir + "/gui/forms/export_for_deepss.yaml"
        with open(yaml_file, "r") as form_yaml:
            items_to_create = yaml.load(form_yaml, Loader=yaml.SafeLoader)

        for name in self.event_times.names:
            items_to_create['main'].insert(2, {'name': f'include_{name}', 'label': name, 'type':'bool', 'default':True})

        dialog = YamlDialog(yaml_file=items_to_create,
                            title=f'Export song and annotations for DeepSS')

        dialog.form.fields['start_seconds'].setRange(0, np.max(self.ds.sampletime))
        dialog.form.fields['end_seconds'].setRange(0, np.max(self.ds.sampletime))
        dialog.form.fields['end_seconds'].setValue(np.max(self.ds.sampletime))
        dialog.form.fields['end_seconds'].setToNone()

        dialog.show()
        result = dialog.exec_()

        if result == QtGui.QDialog.Accepted:
            form_data = dialog.form.get_form_data()

            which_events = []
            for key, val in form_data.items():
                if key.startswith('include_') and val:
                    which_events.append(key[len('include_'):])

            start_seconds = form_data['start_seconds']
            end_seconds = form_data['end_seconds']

            logging.info(f"Exporting for DeepSS:")
            if form_data['file_type'] == 'WAV':
                logging.info(f"   song to WAV: {savefilename_trunk + '.wav'}.")
                self.export_to_wav(savefilename_trunk + '.wav', start_seconds, end_seconds, form_data['scale_audio'])
            elif form_data['file_type'] == 'NPZ':
                logging.info(f"   song to NPZ: {savefilename_trunk + '.npz'}.")
                self.export_to_npz(savefilename_trunk + '.npz', start_seconds, end_seconds, form_data['scale_audio'])

            logging.info(f"   annotations to CSV: {savefilename_trunk + '.csv'}.")
            self.export_to_csv(savefilename_trunk + '_annotations.csv', start_seconds, end_seconds, which_events, match_to_samples=True)
            logging.info(f"Done.")

    def deepss_make(self, qt_keycode=None):
        data_folder = QtWidgets.QFileDialog.getExistingDirectory(parent=None,
                                                        caption='Select data folder')
        if not data_folder:
            return

        dialog = YamlDialog(yaml_file=package_dir + "/gui/forms/deepss_make.yaml",
                            title=f'Assemble dataset for training DeepSS')

        dialog.form.fields['data_folder'].setText(data_folder)
        dialog.form.fields['store_folder'].setText(data_folder + '.npy')

        dialog.show()
        result = dialog.exec_()

        if result == QtGui.QDialog.Accepted:
            form_data = dialog.form.get_form_data()

            # post process splits
            parts = ['train', 'val', 'test']
            file_splits = dict()
            data_splits = dict()
            for part in parts:
                if form_data[part + '_split']=='files':
                    file_splits[part] = form_data[part + '_split_fraction']
                else:
                    data_splits[part] = form_data[part +
                    '_split_fraction']

            deepss.make(data_folder=form_data['data_folder'],
                        store_folder=form_data['store_folder'],
                        file_splits=file_splits, data_splits=data_splits,
                        make_single_class_datasets=form_data['make_single_class_datasets'],
                        split_train_in_two=form_data['split_train_in_two'],
                        event_std_seconds=form_data['event_std_seconds'],
                        gap_seconds=form_data['gap_seconds'])
            logging.info('Done.')

    def deepss_train(self, qt_keycode):

        def _filter_form_data(form_data: Dict[str, Any], is_cli: bool = False) -> Dict[str, Any]:
            """[summary]

            Args:
                form_data (Dict[Any]): [description]
                cli (bool, optional): Process boolean flags. Defaults to False.

            Returns:
                Dict[Any]: [description]
            """

            form_data['model_name'] = 'tcn'
            if form_data['frontend'] == 'Yes':
                form_data['model_name'] += '_stft'
                form_data['pre_nb_conv'] = int(form_data['pre_nb_conv'])

            form_data['reduce_lr'] = form_data['reduce_lr_patience'] is not None
            form_data = {k: v for k, v in form_data.items() if v not in ['Yes', 'No', None]}

            if not len(form_data['y_suffix']):
                del form_data['y_suffix']

            for field in ['seed', 'fraction_data', 'reduce_lr_patience']:
                if field in form_data and form_data[field] is None:
                    del form_data[field]
            form_data['use_separable'] = [item.lower()=='true' for item in form_data['use_separable']]
            if is_cli:
                form_data['use_separable'] = ' '.join(form_data['use_separable'])
                for field in ['ignore_boundaries', 'reduce_lr', 'tensorboard']:
                    if field in form_data:
                        if form_data[field] is False:  # rename and pre-prend "no_"
                            del form_data[field]
                            form_data['no_' + field] = ''
                        else:  # empyt value
                            form_data[field] = ''

            return form_data

        def save(arg):
            savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save configuration to', '',
                                                        filter="yaml files (*.yaml);;all files (*)")
            if len(savefilename):
                data = dialog.form.get_form_data()
                logging.info(f"   Saving form fields to {savefilename}.")
                with open(savefilename, 'w') as stream:
                    yaml.safe_dump(data, stream)
                logging.info(f"Done.")

        def make_cli(arg):
            script_ext = 'cmd' if os.name == 'nt' else 'sh'
            savefilename = 'train.' + script_ext
            savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Script name', savefilename,
                                                        filter=f"script (*.{script_ext};;all files (*)")
            if len(savefilename):
                form_data = dialog.form.get_form_data()
                form_data = _filter_form_data(form_data, is_cli=True)

                cmd = 'python3 -m dss.train'
                # FIXME formatting
                for key, val in form_data.items():
                    cmd += f" --{key.replace('_','-')} {val}"
                with open(savefilename, 'w') as f:
                    f.write(cmd)

                # TODO: dialog to suggest editing the script (change paths, activate conda env, cluster specific stuff)
                logging.info(f"Done.")

        def load(arg):
            yaml_filename, _ = QtWidgets.QFileDialog.getOpenFileName(parent=None,
                                                                    caption='Select yaml file')
            if len(yaml_filename):
                logging.info(f"   Updating form fields with information from {yaml_filename}.")
                with open(yaml_filename, 'r') as stream:
                    data = yaml.safe_load(stream)
                dialog.form.set_form_data(data)  # update form
                logging.info(f"Done.")

        data_dir = QtWidgets.QFileDialog.getExistingDirectory(None, directory=None,
                                                              caption="Open Data Folder (*.npy)",
                                                              filter="npy folders (*.npy);;all files (*)")
        # TODO: check that this is a valid data_dir!
        dialog = YamlDialog(yaml_file=package_dir + "/gui/forms/dss_train.yaml",
                            title='Train network',
                            # main_callback=train,
                            callbacks={'save': save, 'load': load, 'make_cli': make_cli})
        dialog.form.fields['data_dir'].setText(data_dir)
        dialog.form.fields['save_dir'].setText(os.path.splitext(data_dir)[0] + '.res')
        y_suffices = list(set([p.stem.strip('y_') for p in pathlib.Path(data_dir).glob('**/y_*.npy')]))
        dialog.form.fields['y_suffix'].set_options(y_suffices)

        dialog.show()
        result = dialog.exec_()
        if result == QtGui.QDialog.Accepted:
            form_data = dialog.form.get_form_data()
            form_data = _filter_form_data(form_data)

            got_dss = False
            try:
                import dss.train
                got_dss = True
            except ImportError:
                logging.exception('Need to install dss. Alternatively, make scripts and run training elsewhere.')

            if got_dss:
                # start in independent process, otherwise GUI will freeze during training
                import multiprocessing as mp
                form_data['log_messages'] = True
                train_process = mp.Process(group=None, target=dss.train.train, kwargs=form_data)
                train_process.start()

    @classmethod
    def from_file(cls, filename=None, app=None, qt_keycode=None, events_string='',
                  spec_freq_min=None, spec_freq_max=None, target_samplingrate=None,
                  skip_dialog: bool = False, is_dss: bool = False):
        if not filename:
            # enable multiple filters: *.h5, *.npy, *.npz, *.wav, *.*
            file_filter = "Any file (*.*);;WAV files (*.wav);;HDF5 files (*.h5 *.hdf5);;NPY files (*.npy);;NPZ files (*.npz)"
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(parent=None,
                                                                caption='Select file',
                                                                filter=file_filter)
        if filename:
            # infer loader from file name and set default in form
            # infer samplerate (catch error) and set default in form
            samplerate = 10_000  # Hz
            data_loader = None
            datasets = ['']
            if filename.endswith('.npz'):
                try:
                    # TODO list variable for form
                    with np.load(filename) as file:
                        datasets = list(file.keys())
                        try:
                            samplerate = file['samplerate']
                        except KeyError:
                            samplerate = None
                    data_loader = 'npz'
                except KeyError:
                    logging.info(f'{filename} no sample rate info in NPZ file. Need to save "samplerate" variable with the audio data.')
            elif filename.endswith('.zarr'):
                pass  # lazy load and get samplerate attr
            elif filename.endswith('.npy'):
                data_loader = 'npy'
            elif filename.endswith('.h5') or filename.endswith('.hdfs') or filename.endswith('.hdf5') or filename.endswith('.mat'):
                # infer data set (for hdf5) and populate form
                try:
                    # list all data sets in file and add to list
                    with h5py.File(filename, 'r') as f:
                        datasets = utils.allkeys(f, keys=[])
                        datasets.remove('/')  # remove root
                    data_loader = 'h5'
                except:
                    pass
            elif filename.endswith('.wav'):
                try:  # to load as audio file
                    import scipy.io.wavfile
                    samplerate, _ = scipy.io.wavfile.read(filename, mmap=True)
                    data_loader = 'wav'
                except:
                    pass
            else:
                try:  # to load as audio file
                    import soundfile
                    fileinfo = soundfile.info(filename)
                    samplerate = fileinfo.samplerate
                    logging.info(fileinfo)
                    data_loader = 'audio'
                except:
                    pass


            if filename.endswith('.npy'):
                data_loader = 'npy'

            if data_loader is None:
                data_loader = 'audio'

            dialog = YamlDialog(yaml_file=package_dir + "/gui/forms/from_file.yaml",
                                title=f'Load {filename}')
            dialog.form['target_samplingrate'] = samplerate
            dialog.form['spec_freq_max'] = samplerate / 2

            dialog.form.fields['data_set'].set_options(datasets)  # add datasets
            dialog.form.fields['data_set'].setValue(datasets[0])  # select first

            data_loaders = list(xb.data_loaders.keys())
            dialog.form.fields['data_loader'].set_options(data_loaders)  # add datasets
            dialog.form.fields['data_loader'].setValue(data_loader)  # select first

            # set default based on data file
            annotation_path = os.path.splitext(filename)[0] + '_annotations.csv'
            dialog.form.fields['annotation_path'].setText(annotation_path)  # select first

            # initialize form data with cli args
            if spec_freq_min is not None:
                dialog.form['spec_freq_min'] = spec_freq_min
            if spec_freq_max is not None:
                dialog.form['spec_freq_max'] = spec_freq_max
            if target_samplingrate is not None:
                dialog.form['target_samplingrate'] = target_samplingrate
            if len(events_string):
                dialog.form['events_string'] = events_string


            if not skip_dialog:
                dialog.show()
                result = dialog.exec_()
            else:
                result = QtGui.QDialog.Accepted

            if result == QtGui.QDialog.Accepted:
                form_data = dialog.form.get_form_data()  # why call this twice

                form_data = dialog.form.get_form_data()
                logging.info(f"Making new dataset from {filename}.")
                if form_data['target_samplingrate'] < 1:
                    form_data['target_samplingrate'] = None

                event_names = []
                event_classes = []
                if len(form_data['events_string']):
                    for pair in form_data['events_string'].split(';'):
                        items = pair.strip().split(',')
                        if len(items)>0:
                            event_names.append(items[0].strip())
                        if len(items)>1:
                            event_classes.append(items[1].strip())
                        else:
                            event_classes.append('segment')

                ds = xb.from_file(filepath=filename,
                                 loader_name=form_data['data_loader'],
                                 samplerate=form_data['samplerate'],
                                 dataset=form_data['data_set'],
                                 target_samplerate=form_data['target_samplingrate'],
                                 event_names=event_names,
                                 event_categories=event_classes,
                                 annotation_path=form_data['annotation_path'])

                if form_data['filter_song'] == 'yes':
                    ds = cls.filter_song(ds, form_data['f_low'], form_data['f_high'])

                cue_points = []
                if form_data['load_cues']=='yes':
                    cue_points = cls.load_cuepoints(form_data['cues_file'])

                return PSV(ds, title=filename, cue_points=cue_points,
                           fmin=dialog.form['spec_freq_min'],
                           fmax=dialog.form['spec_freq_max'],
                           data_source=DataSource(form_data['data_loader'], filename))

    @classmethod
    def from_dir(cls, dirname=None, app=None, qt_keycode=None, events_string='',
                 spec_freq_min=None, spec_freq_max=None, target_samplingrate=None,
                 box_size=None, pixel_size_mm=None, skip_dialog: bool = False, is_dss: bool = False):

        if not dirname:
            dirname = QtWidgets.QFileDialog.getExistingDirectory(parent=None,
                                                                    caption='Select data directory')
        if dirname:
            dialog = YamlDialog(yaml_file=package_dir + "/gui/forms/from_dir.yaml",
                                title=f'Dataset from data directory {dirname}')

            # initialize form data with cli args
            dialog.form['pixel_size_mm'] = pixel_size_mm  # and un-disable
            dialog.form['spec_freq_min'] = spec_freq_min
            dialog.form['spec_freq_max'] = spec_freq_max
            if box_size is not None:
                dialog.form['box_size_px'] = box_size
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

                form_data['filter_song'] = form_data['filter_song'] == 'yes'

                include_tracks = not form_data['ignore_tracks']
                include_poses = not form_data['ignore_tracks']
                lazy_load_song = not form_data['filter_song']  # faster that way
                base, datename = os.path.split(os.path.normpath(dirname))  # normpath removes trailing pathsep
                root, dat_path = os.path.split(base)
                ds = xb.assemble(datename, root, dat_path, res_path='res',
                                 fix_fly_indices=form_data['fix_fly_indices'],
                                 include_song=~form_data['ignore_song'],
                                 target_sampling_rate=form_data['target_samplingrate'],
                                 resample_video_data=resample_video_data,
                                 pixel_size_mm=pixel_size_mm,
                                 lazy_load_song=lazy_load_song,
                                 include_tracks=include_tracks, include_poses=include_poses)

                if form_data['filter_song']:
                    ds = cls.filter_song(ds, form_data['f_low'], form_data['f_high'])

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

                # add event categories if they are missing in the dataset
                if 'song_events' in ds and 'event_categories' not in ds:
                    event_categories = ['segment'
                                        if 'sine' in evt or 'syllable' in evt
                                        else 'event'
                                        for evt in ds.event_types.values]
                    ds = ds.assign_coords({'event_categories':
                                        (('event_types'),
                                        event_categories)})

                # add missing song types
                if 'song_events' not in ds or len(ds.event_types) == 0:
                    cats = {event_name: event_class for event_name, event_class in zip(event_names, event_classes)}
                    ds.attrs['event_times'] = annot.Events(categories = cats)
                    # FIXME update ds.song_events!!

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

                cue_points = []
                if form_data['load_cues']=='yes':
                    cue_points = cls.load_cuepoints(form_data['cues_file'])
                return PSV(ds, title=dirname, cue_points=cue_points, vr=vr,
                        fmin=dialog.form['spec_freq_min'],
                        fmax=dialog.form['spec_freq_max'],
                        box_size=dialog.form['box_size_px'],
                        data_source=DataSource('dir', dirname))

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
                    ds = cls.filter_song(ds, form_data['f_low'], form_data['f_high'])


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

                # load cues
                cue_points = []
                if form_data['load_cues']=='yes':
                    cue_points = cls.load_cuepoints(form_data['cues_file'])

                return PSV(ds, vr=vr, cue_points=cue_points, title=filename,
                        fmin=dialog.form['spec_freq_min'],
                        fmax=dialog.form['spec_freq_max'],
                        box_size=dialog.form['box_size'],
                        data_source=DataSource('zarr', filename))

    @classmethod
    def filter_song(cls, ds, f_low, f_high):
        # TODO paralellize over channels
        if f_low is None:
            f_low = 1.0
        if 'song_raw' in ds:  # this will take a long time:
            if f_high is None:
                f_high = ds.song_raw.attrs['sampling_rate_Hz'] / 2 - 1
            sos_bp = ss.butter(5, [f_low, f_high], 'bandpass', output='sos',
                                fs=ds.song_raw.attrs['sampling_rate_Hz'])
            logging.info(f'Filtering `song_raw` between {f_low} and {f_high} Hz.')
            ds.song_raw.data = ss.sosfiltfilt(sos_bp, ds.song_raw.data, axis=0)
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
                    # TODO: replace with method in annot.Events
                    self.ds = event_utils.eventtimes_to_traces(self.ds, self.event_times)

                # scale tracks back to original units upon save
                logging.info(f'Converting spatial units back to {self.original_spatial_units} if required.')
                self.ds = xb.convert_spatial_units(self.ds, to_units=self.original_spatial_units)

                # update ds.event_times from event_times dict
                event_times = annot.Events(self.event_times)
                ds_event_times = event_times.to_dataset()
                if 'index' in self.ds.dims and 'event_time' in self.ds.dims:
                    self.ds = self.ds.drop_dims(['index', 'event_time'])
                    self.ds = self.ds.combine_first(ds_event_times)

                logging.info(f'   Saving dataset to {savefilename}.')
                xb.save(savefilename, self.ds)
                logging.info(f'Done.')
            else:
                logging.info(f'Saving aborted.')


class ZarrOverwriteWarning(QtWidgets.QMessageBox):

    def __init__(self):
        super().__init__()
        self.setIcon(QtWidgets.QMessageBox.Warning)
        self.setText('Attempting to overwrite existing zarr file.')
        self.setInformativeText("This can corrupt the file and lead to data loss. \
                                 ABORT unless you know what you're doing\
                                 or save to a file with a different name.")
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

        # TODO allow vr to be a string with the video file name
        self.vr = vr
        self.cue_points = cue_points

        # detect all event times and segment on/offsets
        if 'event_times' in ds and 'event_names' in ds and len(ds['event_names']) > 0:
            self.event_times = annot.Events.from_dataset(ds)
        elif 'event_times' in ds.attrs:
            self.event_times = ds.attrs['event_times'].copy()
        elif 'song_events' in ds:
            self.event_times = event_utils.detect_events(ds)  # detect events from ds.song_events traces
        else:
            self.event_times = dict()
        self.event_times = annot.Events(self.event_times)

        self.nb_eventtypes = len(self.event_times.names)
        self.eventype_colors = utils.make_colors(self.nb_eventtypes)

        self.box_size = box_size
        self.fmin = fmin
        self.fmax = fmax

        if 'song' in self.ds:
            self.tmax = self.ds.song.shape[0]
        elif 'song_raw' in self.ds:
            self.tmax = self.ds.song_raw.shape[0]
        elif 'body_positions' in self.ds:
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

        self.cue_index = -1  # set this to -1 not to 0 so that upon first increase we will jump to 0, not to 1

        self.nb_flies = np.max(self.ds.flies).values + 1 if 'flies' in self.ds.dims else 1
        self.focal_fly = 0
        self.other_fly = 1 if self.nb_flies > 1 else 0
        if 'poseparts' in self.ds:
            self.nb_bodyparts = len(self.ds.poseparts)
        elif 'bodyparts' in self.ds:
            self.nb_bodyparts = len(self.ds.bodyparts)
        else:
            self.nb_bodyparts = 1

        self.fly_colors = utils.make_colors(self.nb_flies)
        self.bodypart_colors = utils.make_colors(self.nb_bodyparts)

        # scale tracks back to px - required for display purposes
        names = ['body_positions', 'pose_positions', 'pose_positions_allo']
        # save original spatial units so we can convert back upon save
        for name in names:
            if name in self.ds:
                self.original_spatial_units = self.ds[name].attrs['spatial_units']
        self.ds = xb.convert_spatial_units(self.ds, to_units='pixels')

        self.STOP = True
        self.swap_events = []
        self.show_spec = True
        self.spec_win = 200
        self.show_songevents = True
        self.movable_events = True
        self.move_only_current_events = True
        self.show_all_channels = True
        self.select_loudest_channel = False
        self.threshold_mode = False
        self.sinet0 = None

        self.thres_min_dist = 0.025  # seconds
        self.thres_env_std = 0.004  # seconds

        if 'song_events' in self.ds:
            self.fs_other = self.ds.song_events.attrs['sampling_rate_Hz']
        else:
            self.fs_other = ds.attrs['target_sampling_rate_Hz']

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
        self.step = 1

        self.resize(1000, 800)

        # build UI/controller
        # MENU
        self.file_menu.clear()
        self._add_keyed_menuitem(self.file_menu, "New from file", self.from_file)
        self._add_keyed_menuitem(self.file_menu, "New from ethodrome folder", self.from_dir)
        self.file_menu.addSeparator()
        self._add_keyed_menuitem(self.file_menu, "Load dataset", self.from_zarr)
        self.file_menu.addSeparator()
        self._add_keyed_menuitem(self.file_menu, "Save swap files", self.save_swaps)
        self._add_keyed_menuitem(self.file_menu, "Save annotations", self.save_annotations)
        self._add_keyed_menuitem(self.file_menu, "Export for DeepSS", self.export_for_deepss)
        self.file_menu.addSeparator()
        self._add_keyed_menuitem(self.file_menu, "Save dataset", self.save_dataset)
        self.file_menu.addSeparator()
        self.file_menu.addAction("Exit")

        view_play = self.bar.addMenu("Playback")
        self._add_keyed_menuitem(view_play, "Play video", self.toggle_playvideo, "Space",
                                checkable=True, checked=not self.STOP)
        view_play.addSeparator()
        self._add_keyed_menuitem(view_play, " < Reverse one frame", self.single_frame_reverse, "Left"),
        self._add_keyed_menuitem(view_play, "<< Reverse jump", self.jump_reverse, "A")
        self._add_keyed_menuitem(view_play, ">> Forward jump", self.jump_forward, "D")
        self._add_keyed_menuitem(view_play, " > Forward one frame", self.single_frame_advance, "Right")
        view_play.addSeparator()
        self._add_keyed_menuitem(view_play, "Load cue points", self.reload_cuepoints)  # text file with comma separated seconds or frames...
        self._add_keyed_menuitem(view_play, "Move to previous cue", self.set_prev_cuepoint, "K")
        self._add_keyed_menuitem(view_play, "Move to next cue", self.set_next_cuepoint, "L")
        view_play.addSeparator()
        self._add_keyed_menuitem(view_play, "Zoom in song", self.zoom_in_song, "W")
        self._add_keyed_menuitem(view_play, "Zoom out song", self.zoom_out_song, "S")

        view_video = self.bar.addMenu("Video")
        self._add_keyed_menuitem(view_video, "Crop frame", partial(self.toggle, 'crop'), "C",
                                checkable=True, checked=self.crop)
        self._add_keyed_menuitem(view_video, "Change focal fly", self.change_focal_fly, "F")
        self._add_keyed_menuitem(view_video, "Swap flies", self.swap_flies, "X")
        view_video.addSeparator()
        self._add_keyed_menuitem(view_video, "Move poses", partial(self.toggle, 'move_poses'), "B",
                                checkable=True, checked=self.move_poses)
        view_video.addSeparator()
        self._add_keyed_menuitem(view_video, "Show fly position", partial(self.toggle, 'show_dot'), "O",
                                checkable=True, checked=self.show_dot)
        self._add_keyed_menuitem(view_video, "Show poses", partial(self.toggle, 'show_poses'), "P",
                                checkable=True, checked=self.show_poses)

        view_audio = self.bar.addMenu("Audio")
        self._add_keyed_menuitem(view_audio, "Play waveform through speakers", self.play_audio, "E")
        view_audio.addSeparator()
        self._add_keyed_menuitem(view_audio, "Show all channels", partial(self.toggle, 'show_all_channels'), None,
                                checkable=True, checked=self.show_all_channels)
        self._add_keyed_menuitem(view_audio, "Auto-select loudest channel", partial(self.toggle, 'select_loudest_channel'), "Q",
                                checkable=True, checked=self.select_loudest_channel)
        self._add_keyed_menuitem(view_audio, "Select previous channel", self.set_next_channel, "Up")
        self._add_keyed_menuitem(view_audio, "Select next channel", self.set_prev_channel, "Down")
        view_audio.addSeparator()
        self._add_keyed_menuitem(view_audio, "Show spectrogram", partial(self.toggle, 'show_spec'), None,
                                checkable=True, checked=self.show_spec)
        self._add_keyed_menuitem(view_audio, "Set display frequency limits", self.set_spec_freq)
        self._add_keyed_menuitem(view_audio, "Increase frequency resolution", self.dec_freq_res, "R")
        self._add_keyed_menuitem(view_audio, "Increase temporal resolution", self.inc_freq_res, "T")
        view_audio.addSeparator()

        view_annotations = self.bar.addMenu("Annotations")
        self._add_keyed_menuitem(view_annotations, "Add or edit song types", self.edit_annotation_types)
        self._add_keyed_menuitem(view_annotations, "Show annotations", partial(self.toggle, 'show_songevents'), "V",
                                checkable=True, checked=self.show_songevents)
        view_annotations.addSeparator()
        self._add_keyed_menuitem(view_annotations, "Allow moving annotations", partial(self.toggle, 'movable_events'), "M",
                                checkable=True, checked=self.movable_events)
        self._add_keyed_menuitem(view_annotations, "Only move active song type", partial(self.toggle, 'move_only_current_events'), None,
                                checkable=True, checked=self.move_only_current_events)
        view_annotations.addSeparator()
        self._add_keyed_menuitem(view_annotations, "Delete active song type in view",
                                self.delete_current_events, "U")
        self._add_keyed_menuitem(view_annotations, "Delete all song types in view", self.delete_all_events, "Y")
        view_annotations.addSeparator()
        self._add_keyed_menuitem(view_annotations, "Toggle thresholding mode", partial(self.toggle, 'threshold_mode'),
                                checkable=True, checked=self.threshold_mode)
        self._add_keyed_menuitem(view_annotations, "Generate proposal by envelope thresholding", self.threshold, "I")
        self._add_keyed_menuitem(view_annotations, "Adjust thresholding mode", self.set_envelope_computation)
        view_annotations.addSeparator()
        self._add_keyed_menuitem(view_annotations, "Approve proposals for active song type in view", self.approve_active_proposals, "G")
        self._add_keyed_menuitem(view_annotations, "Approve proposals for all song types in view", self.approve_all_proposals, "H")

        view_train = self.bar.addMenu("DeepSS")
        self._add_keyed_menuitem(view_train, "Make dataset for training", self.deepss_make, None)
        self._add_keyed_menuitem(view_train, "Train", self.deepss_train, None)
        self._add_keyed_menuitem(view_train, "Predict", self.deepss_predict, None)

        self.bar.addMenu("View")

        self.hl = pg.QtGui.QHBoxLayout()

        # EVENT TYPE selector
        self.cb = pg.QtGui.QComboBox()
        self.cb.currentIndexChanged.connect(self.update_xy)
        self.cb.setCurrentIndex(0)
        event_sel_label = pg.Qt.QtWidgets.QLabel('Song types:')
        event_sel_label.setStyleSheet("QLabel { background-color : black; color : gray; }");
        self.hl.addWidget(event_sel_label, stretch=1)
        self.hl.addWidget(self.cb, stretch=4)


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
        channel_sel_label = pg.Qt.QtWidgets.QLabel('Audio channels:')
        channel_sel_label.setStyleSheet("QLabel { background-color : black; color : gray; }");
        self.hl.addWidget(channel_sel_label, stretch=1)
        self.hl.addWidget(self.cb2, stretch=4)

        if self.vr is not None:
            self.movie_view = views.MovieView(model=self, callback=self.on_video_clicked)

        if cmap_name in colormaps.cmaps:
            colormap = colormaps.cmaps[cmap_name]
            colormap._init()
            lut = (colormap._lut * 255).view(np.ndarray)  # convert matplotlib colormap from 0-1 to 0 -255 for Qt
        else:
            logging.warning(f'Unknown colormap "{cmap_name}"" provided. Using default (turbo).')
            lut = None

        self.slice_view = views.TraceView(model=self, callback=self.on_trace_clicked)
        self.spec_view = views.SpecView(model=self, callback=self.on_trace_clicked, colormap=lut)

        self.cw = pg.GraphicsLayoutWidget()

        self.ly = pg.QtGui.QVBoxLayout()
        self.ly.addLayout(self.hl)
        splitter = QtWidgets.QSplitter(pg.QtCore.Qt.Vertical)


        if self.vr is not None:
            splitter.addWidget(self.movie_view)

        splitter.addWidget(self.slice_view)
        splitter.addWidget(self.spec_view)
        splitter.setSizes([400, 10, 100, 100])

        self.ly.addWidget(splitter)

        def edit_time_finished(source=None):
            try:
                self.t0 = float(source.text()) * self.fs_song
            except Exception as e:
                print(e)

        def edit_frame_finished(source=None):
            try:
                frame_number = float(source.text())
                # get sampletime from framenumber
                idx = np.argmax(self.ds.nearest_frame.data>=frame_number)
                self.t0 = self.ds.nearest_frame[idx].time.values * self.fs_song
            except Exception as e:
                print(e)

        def on_slider_moved(value):
            self.t0 = value

        def on_play_pressed(source):
            self.toggle_playvideo()

        self.scrollbar = pg.Qt.QtWidgets.QScrollBar(pg.QtCore.Qt.Horizontal)
        self.tmin = np.min(self.ds.sampletime.values)
        self.scrollbar.setMinimum(self.tmin)
        self.scrollbar.setMaximum(self.tmax)
        self.scrollbar.setPageStep(max((self.tmax-self.tmin)/100, self._span / self.fs_song))
        self.scrollbar.valueChanged.connect(on_slider_moved)
        scrollbar_layout = pg.QtGui.QHBoxLayout()

        self.playButton = pg.Qt.QtWidgets.QPushButton()
        self.playButton.setIcon(self.style().standardIcon(pg.Qt.QtWidgets.QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(functools.partial(on_play_pressed, source=self.playButton))

        scrollbar_layout.addWidget(self.playButton, stretch= 0.5)
        scrollbar_layout.addWidget(self.scrollbar, stretch=10)

        self.edit_time = pg.Qt.QtWidgets.QLineEdit()
        self.edit_time.editingFinished.connect(functools.partial(edit_time_finished, source=self.edit_time))
        scrollbar_layout.addWidget(self.edit_time, stretch=1)
        edit_time_label = pg.Qt.QtWidgets.QLabel('seconds')
        edit_time_label.setStyleSheet("QLabel { background-color : black; color : gray; }");
        scrollbar_layout.addWidget(edit_time_label, stretch=1)

        if self.vr is not None:
            self.edit_frame = pg.Qt.QtWidgets.QLineEdit()
            self.edit_frame.editingFinished.connect(functools.partial(edit_frame_finished, source=self.edit_frame))
            scrollbar_layout.addWidget(self.edit_frame, stretch=1)
            edit_frame_label = pg.Qt.QtWidgets.QLabel('frame')
            edit_frame_label.setStyleSheet("QLabel { background-color : black; color : gray; }");
            scrollbar_layout.addWidget(edit_frame_label, stretch=1)

        self.ly.addLayout(scrollbar_layout)

        self.cw.setLayout(self.ly)

        self.setCentralWidget(self.cw)

        self.update_eventtype_selector()

        self.show()
        self.update_xy()
        self.update_frame()
        self.t0 = self.t0 + 0.0000000001
        self.span = self.span + 0.0000000001
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
    def t0(self, val: float):
        old_t0 = self._t0
        self._t0 = np.clip(val, self.span / 2, self.tmax - self.span / 2)  # ensure t0 stays within bounds
        if self._t0 != old_t0:
            self.scrollbar.setValue(self.t0)
            self.edit_time.setText(str(self.t0 / self.fs_song))
            if self.vr is not None:
                self.edit_frame.setText(str(self.framenumber))

            self.update_xy()
            self.update_frame()

    @property
    def framenumber(self):
        if 'nearest_frame' in self.ds.coords:
            return int(self.ds.coords['nearest_frame'][self.index_other].data)
        else:
            return None

    @property
    def span(self):
        return self._span

    @span.setter
    def span(self, val):
        # HACK fixes weird offset/jump error - probably arises from self.fs_song / self.fs_other
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
            return self.event_times.names[self.current_event_index]

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

    def _add_keyed_menuitem(self, parent, label: str, callback, qt_keycode=None, checkable=False, checked=True):
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

    def delete_current_events(self, qt_keycode):
        if self.current_event_index is not None:
            deleted_events = 0
            deleted_events += self.event_times.delete_range(self.current_event_name,
                                                            self.time0 / self.fs_song,
                                                            self.time1 / self.fs_song)
            logging.info(f'   Deleted all  {deleted_events} of {self.current_event_name} in view.')
        else:
            logging.info(f'   No event type selected. Not deleting anything.')

    def threshold(self, qt_keycode):
        if self.STOP and self.current_event_name is not None:
            if self.event_times.categories[self.current_event_name] != 'event':
                return

            indexes = peakutils.indexes(self.envelope,
                                        thres=self.slice_view.threshold,
                                        min_dist=self.thres_min_dist * self.fs_song,
                                        thres_abs=True)

            # add events to current song type
            for t in self.x[indexes]:
                self.event_times.add_time(self.current_event_name, t)
                logging.info(f"   Added {self.current_event_name} at t={t:1.4f} seconds.")
            # TODO ensure we did not add duplicates - maybe call np.unique at the end?
            old_len = self.event_times[self.current_event_name].shape[0]
            self.event_times[self.current_event_name] = np.unique(self.event_times[self.current_event_name], axis=0)
            new_len = self.event_times[self.current_event_name].shape[0]
            if new_len != old_len:
                logging.info(f"   Removed {old_len - new_len} duplicates in {self.current_event_name}.")
            self.update_xy()

    def get_envelope(self):
        import scipy.signal.windows
        std = self.thres_env_std * self.fs_song
        win = scipy.signal.windows.gaussian(int(std * 6), std)
        win /= np.sum(win)
        env = np.sqrt(np.convolve(self.y**2, win, mode='same'))
        return env

    def delete_all_events(self, qt_keycode):
        deleted_events = 0
        for event_name in self.event_times.names:
            deleted_events += self.event_times.delete_range(event_name,
                                                            self.time0 / self.fs_song,
                                                            self.time1 / self.fs_song)
        logging.info(f'   Deleted all {deleted_events} events in view.')
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

    def dec_freq_res(self, qt_keycode):
        self.spec_win = int(max(2, self.spec_win // 2))
        if self.STOP:
            # need to update twice to fix axis limits for some reason
            self.update_xy()

    def toggle_playvideo(self, qt_keycode=None):
        self.STOP = not self.STOP
        if not self.STOP:
            self.playButton.setIcon(self.style().standardIcon(pg.Qt.QtWidgets.QStyle.SP_MediaPause))
            self.play_video()
        else:
            self.playButton.setIcon(self.style().standardIcon(pg.Qt.QtWidgets.QStyle.SP_MediaPlay))

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

    def set_spec_freq(self, qt_keycode):
        dialog = YamlDialog(yaml_file=package_dir + "/gui/forms/spec_freq.yaml",
                            title=f'Set frequency limits in display')
        dialog.form['spec_freq_min'] = self.fmin
        dialog.form['spec_freq_max'] = self.fmax
        dialog.show()

        result = dialog.exec_()

        if result == QtGui.QDialog.Accepted:
            form_data = dialog.form.get_form_data()  # why call this twice
            self.fmin = form_data['spec_freq_min']
            self.fmax = form_data['spec_freq_max']
            logging.info(f'Set frequency range for spectrogram display to {self.fmin}:{self.fmax} Hz.')
            self.update_xy()

    def set_envelope_computation(self, qt_keycode):
        dialog = YamlDialog(yaml_file=package_dir + "/gui/forms/envelope_computation.yaml",
                            title=f'Set options for envelope computation')

        dialog.form['thres_min_dist'] = self.thres_min_dist
        dialog.form['thres_env_std'] = self.thres_env_std

        dialog.show()
        result = dialog.exec_()

        if result == QtGui.QDialog.Accepted:
            form_data = dialog.form.get_form_data()  # why call this twice
            self.thres_min_dist = form_data['thres_min_dist']
            self.thres_env_std = form_data['thres_env_std']
            self.update_xy()

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

            if self.select_loudest_channel:
                self.loudest_channel = np.argmax(np.max(y_all, axis=0))
                self.cb2.setCurrentIndex(self.loudest_channel)

        if self.threshold_mode:
            self.envelope = self.get_envelope()

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
        for event_index in range(self.nb_eventtypes):
            movable = self.STOP and self.movable_events
            event_name = self.event_times.names[event_index]
            if self.move_only_current_events:
                movable = movable and self.current_event_index==event_index

            event_pen = pg.mkPen(color=self.eventype_colors[event_index], width=1.5)
            event_brush = pg.mkBrush(color=[*self.eventype_colors[event_index], 25])

            events_in_view = self.event_times.filter_range(event_name, x[0], x[-1], strict=False)

            if self.event_times.categories[event_name] == 'segment':
                for onset, offset in zip(events_in_view[:, 0], events_in_view[:, 1]):
                    self.slice_view.add_segment(onset, offset, event_index, event_brush, movable=movable)
                    if self.show_spec:
                        self.spec_view.add_segment(onset, offset, event_index, event_brush, movable=movable)
            elif self.event_times.categories[event_name] == 'event':
                    self.slice_view.add_event(events_in_view[:, 0], event_index, event_pen, movable=movable)
                    if self.show_spec:
                        self.spec_view.add_event(events_in_view[:, 0], event_index, event_pen, movable=movable)

    def play_video(self):  # TODO: get rate from ds (video fps attr)
        RUN = True
        cnt = 0
        dt0 = time.time()
        while RUN:
            self.t0 += self.frame_interval
            cnt += 1
            if cnt % 10 == 0:
                # logging.info(time.time() - dt0)
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

        event_name_to_move = self.current_event_name
        if self.current_event_index != region.event_index:
            event_name_to_move = self.event_times.names[region.event_index]

        new_region = region.getRegion()
        # need to figure out event_name of the moved one if moving non-selected event
        self.event_times.move_time(event_name_to_move, region.bounds, new_region)
        logging.info(f'  Moved {event_name_to_move} from t=[{region.bounds[0]:1.4f}:{region.bounds[1]:1.4f}] to [{new_region[0]:1.4f}:{new_region[1]:1.4f}] seconds.')
        self.update_xy()

    def on_position_change_finished(self, position):
        """Called when dragging an event-like song_event - will change time."""
        if self.move_only_current_events and self.current_event_index != position.event_index:
            return
        event_name_to_move = self.current_event_name
        if self.current_event_index != position.event_index:
            event_name_to_move = self.event_times.names[position.event_index]

        new_position = position.pos()[0]
        self.event_times.move_time(event_name_to_move, position.position, new_position)
        logging.info(f'  Moved {event_name_to_move} from t={position.position:1.4f} to {new_position:1.4f} seconds.')
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
                if self.event_times.categories[self.current_event_name] == 'segment':
                    if self.sinet0 is None:
                        self.spec_view.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
                        self.slice_view.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
                        self.sinet0 = mouseT
                    else:
                        self.spec_view.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
                        self.slice_view.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
                        self.event_times.add_time(self.current_event_name, self.sinet0, mouseT)
                        logging.info(f'  Added {self.current_event_name} at t=[{self.sinet0:1.4f}:{mouseT:1.4f}] seconds.')
                        self.sinet0 = None
                if self.event_times.categories[self.current_event_name] == 'event':
                    self.sinet0 = None
                    self.event_times.add_time(self.current_event_name, mouseT)
                    logging.info(f'  Added {self.current_event_name} at t={mouseT:1.4f} seconds.')
                self.update_xy()
            else:
                self.sinet0 = None
        elif mouseButton == 2:  #delete nearest event
            self.spec_view.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            self.slice_view.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            self.sinet0 = None
            deleted_time = self.event_times.delete_time(self.current_event_name, mouseT, tol=0.05)
            if len(deleted_time):
                logging.info(f'  Deleted {self.current_event_name} at t={deleted_time[0]:1.4f}:{deleted_time[1]:1.4f} seconds.')
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

    def deepss_predict(self, qt_keycode):
        logging.info('Predicting song using DeepSS:')

        if 'song_raw' not in self.ds:
            logging.error('   Missing `song_raw`. skipping.')

        try:
            import dss.predict
            import dss.utils
        except ImportError as e:
            logging.exception(e)
            logging.info('   Failed to import DeepSS. Install via `pip install deepss`.')
            return

        dialog = YamlDialog(yaml_file=package_dir + "/gui/forms/dss_predict.yaml",
                            title='Predict labels using DeepSS')

        dialog.show()
        result = dialog.exec_()

        if result == QtGui.QDialog.Accepted:
            form_data = dialog.form.get_form_data()
            model_path = form_data['model_path']
            model_path = model_path.rsplit('_',1)[0]  # split off suffix

            start_seconds = form_data['start_seconds']
            end_seconds = form_data['end_seconds']

            start_index = utils.find_nearest_idx(self.ds.sampletime.values, start_seconds)
            # start_index = int(start_seconds * self.fs_song)
            end_index = end_seconds
            if end_seconds is not None:
                end_index = utils.find_nearest_idx(self.ds.sampletime.values, end_seconds)
                # end_index = int(end_seconds * self.fs_song)

            params = dss.utils.load_params(model_path)

            audio = self.ds.song_raw.data[start_index:end_index]
            if audio.shape[0] < params['nb_hist']:
                logging.info(f"   Aborting. Audio has fewer samples ({audio.shape[0]}) shorter than network chunk size ({params['nb_hist']}). Fix by select longer audio.")
                return

            # batch size so that at least 10 batches are run - minimizes loss of annotations from batch size "quantiziation" errors
            batch_size = 96
            nb_batches = lambda batch_size: int(np.floor((audio.shape[0] - (((batch_size-1)) + params['nb_hist'])) / (params['stride'] * (batch_size))))
            while nb_batches(batch_size) < 10 and batch_size > 1:
                batch_size -= 1

            logging.info(f'   Running inference on audio.')
            logging.info(f'   Model from {model_path}.')

            events, segments, _, _ = dss.predict.predict(audio, model_path, verbose=1, batch_size=batch_size,
                                                    event_thres=form_data['event_thres'], event_dist=form_data['event_dist'],
                                                    event_dist_min=form_data['event_dist_min'], event_dist_max=form_data['event_dist_max'],
                                                    segment_thres=form_data['event_thres'], segment_fillgap=form_data['segment_fillgap'],
                                                    segment_minlen=form_data['segment_minlen'],
                                                    )
            # remove non-song keys
            if events:
                for key in events.keys():
                    samplerate_Hz = events['samplerate_Hz']
            elif segments:
                samplerate_Hz = segments['samplerate_Hz']
            else:
                logging.warning("no predictions")
                return

            suffix = ''
            if form_data['proof_reading_mode']:
                suffix = '_proposals'

            # events['seconds'] is in samples/fs - translate to time stamps via sample time
            if 'sequence' in events:
                detected_event_names = np.unique(events['sequence'])
            else:
                detected_event_names = []
            if len(detected_event_names) > 0 and detected_event_names[0] is not None:
                logging.info(f"   found {len(events['seconds'])} instances of events '{detected_event_names}'.")
                event_samples = (np.array(events['seconds']) * self.fs_song  + start_index).astype(np.uintp)
                event_seconds = self.ds.sampletime[event_samples]
                for name, seconds in zip(events['sequence'], event_seconds):
                    self.event_times.add_time(str(name) + str(suffix), seconds, seconds, category='event')

            if 'sequence' in segments:
                detected_segment_names = np.unique(segments['sequence'])
            else:
                detected_segment_names = []
            if len(detected_segment_names) > 0 and detected_segment_names[0] is not None:
                logging.info(f"   found {len(segments['onsets_seconds'])} instances of segments '{detected_segment_names}'.")
                onsets_samples = (np.array(segments['onsets_seconds']) * self.fs_song + start_index).astype(np.uintp)
                offsets_samples = (np.array(segments['offsets_seconds']) * self.fs_song + start_index).astype(np.uintp)

                onsets_seconds = self.ds.sampletime[onsets_samples]
                offsets_seconds = self.ds.sampletime[offsets_samples]
                for name, onset_seconds, offset_seconds in zip(segments['sequence'], onsets_seconds, offsets_seconds):
                    self.event_times.add_time(str(name) + str(suffix), onset_seconds, offset_seconds, category='segment')

            self.nb_eventtypes = len(self.event_times)
            self.eventype_colors = utils.make_colors(self.nb_eventtypes)
            self.update_eventtype_selector()
        logging.info('Done.')

    def approve_active_proposals(self, qt_keycode):
        self.approve_proposals(appprove_only_active_event=True)

    def approve_all_proposals(self, qt_keycode):
        self.approve_proposals(appprove_only_active_event=False)

    def approve_proposals(self, appprove_only_active_event: bool = False):
        t0 = self.ds.sampletime.data[self.time0]
        t1 = self.ds.sampletime.data[self.time1]

        proposal_suffix = '_proposals'
        logging.info("Approving:")
        for name in self.event_times.names:
            if appprove_only_active_event and name != self.current_event_name:
                continue

            if name.endswith(proposal_suffix):
                # get event times within range
                within_range_times = self.event_times.filter_range(name, t0, t1)
                # delete from `songtype_proposals`, add to `songtype`
                self.event_times.add_name(name=name[:-len(proposal_suffix)],
                              category=self.event_times.categories[name],
                              times=within_range_times,
                              append=True,
                              overwrite=False)
                self.event_times.delete_range(name, t0, t1)
                logging.info(f"   {len(within_range_times)} events of {name} to {name[:-len(proposal_suffix)]}")
        # update event selector in case the event did not exist yet
        self.nb_eventtypes = len(self.event_times)
        self.eventype_colors = utils.make_colors(self.nb_eventtypes)
        self.update_eventtype_selector()

        logging.info("Done.")
        self.update_xy()

    def edit_annotation_types(self, qt_keycode):
        if hasattr(self, 'event_times'):
            types = self.event_times.names
            cats = list(self.event_times.categories.values())
            table_data = [[typ, cat] for typ, cat in zip(types, cats)]
        else:
            table_data = []

        dialog = table.Table(table_data)
        dialog.show()
        result = dialog.exec_()
        if result == QtGui.QDialog.Accepted:
            data = dialog.get_table_data()

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
            event_names_current = self.event_times.names
            for event_name_current in event_names_current:
                if event_name_current not in event_names_old:
                    del self.event_times[event_name_current]
                    del self.event_times.categories[event_name_current]

            # propagate existing and create new
            for event_name, event_category, event_name_old in zip(event_names, event_categories, event_names_old):
                if event_name_old in self.event_times and event_name != event_name_old:  # rename existing
                    self.event_times[event_name] = self.event_times.pop(event_name_old)
                    self.event_times.categories[event_name] = self.event_times.categories.pop(event_name_old)
                elif event_name_old not in self.event_times:  # create new empty
                    self.event_times.add_name(event_name, event_category)

            # update event-related attrs
            if 'song_events' in self.ds:
                self.fs_other = self.ds.song_events.attrs['sampling_rate_Hz']
            elif 'target_sampling_rate_Hz' in self.ds.attrs:
                self.fs_other = self.ds.attrs['target_sampling_rate_Hz']
            else:
                self.fs_other = self.fs_song

            self.nb_eventtypes = len(self.event_times)
            self.eventype_colors = utils.make_colors(self.nb_eventtypes)

            self.update_eventtype_selector()

    def update_eventtype_selector(self):

        currentIndex = self.cb.currentIndex()
        # delete all existing entries
        while self.cb.count() > 0:
            self.cb.removeItem(0)

        self.cb.addItem("No annotation")
        if hasattr(self, 'event_times'):
            self.eventList = [(cnt, evt) for cnt, evt in enumerate(self.event_times.names)]
            self.eventList = sorted(self.eventList)
        else:
            self.eventList = []

        for event_type in self.eventList:
            self.cb.addItem("Add " + event_type[1])

        # set current index again - reset after deleting all items above
        self.cb.setCurrentIndex(min(currentIndex, len(self.eventList)))

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
            menu_item = self._add_keyed_menuitem(self.view_audio,
                                                self.cb.itemText(ii),
                                                self.change_event_type,
                                                key)
            self.event_items.append(menu_item)

        # color events in combobox as in slice_view and spec_view
        children = self.cb.children()
        itemList = children[0]
        for ii, col in zip(range(1, itemList.rowCount()), self.eventype_colors):
            itemList.item(ii).setForeground(QtGui.QColor(*col))


def main(source: str = '', *, events_string: str = '',
         target_samplingrate: Optional[float] = None,
         spec_freq_min: Optional[float] = None, spec_freq_max: Optional[float] = None,
         box_size: int = 200, pixel_size_mm: Optional[float] = None,
         skip_dialog: bool = False, is_dss: bool = False):
    """
    Args:
        source (str): Data source to load.
            Optional - will open an empty GUI if omitted.
            Source can be the path to:
            - an audio file,
            - a numpy file (npy or npz),
            - an h5 file
            - an xarray-behave dataset constructed from an ethodrome data folder saved as a zarr file,
            - an ethodrome data folder (e.g. 'dat/localhost-xxx').
        song_types_string (str): Initialize song types for annotations.
                             String of the form "song_name,song_category;song_name,song_category".
                             Avoid spaces or trailing ';'.
                             Need to wrap the string in "..." in the terminal
                             "song_name" can be any string w/o space, ",", or ";"
                             "song_category" can be "event" (e.g. pulse) or "segment" (sine, syllable
        target_samplingrate (Optional[float]): [description]. If 0, will use frame times. Defaults to None.
                                     Only used if source is a data folder or a wav audio file.
        spec_freq_min (Optional[float]): Smallest frequency displayed in the spectrogram view. Defaults to 0 Hz.
        spec_freq_max (Optional[float]): Largest frequency displayed in the spectrogram view. Defaults to samplerate/2.
        box_size (int): Crop size around tracked fly. Not used for wav audio files (no videos).
        pixel_size_mm (Optional[float]): Size of a pixel (in mm) in the video. Used to convert tracking data to mm.
        skip_dialog (bool): If True, skips the loading dialog and goes straight to the data view.
        is_dss (bool): reduced GUI for audio only data
    """
    QtGui.QApplication([])
    mainwin = MainWindow()
    mainwin.show()
    if not len(source):
        pass
    elif not os.path.exists(source):
        logging.info(f'{source} does not exist - skipping.')
    elif source.endswith('.wav') or source.endswith('.npz') or source.endswith('.h5') or source.endswith('.hdf5') or source.endswith('.mat'):
        MainWindow.from_file(filename=source,
                             events_string=events_string,
                             target_samplingrate=target_samplingrate,
                             spec_freq_min=spec_freq_min, spec_freq_max=spec_freq_max,
                             skip_dialog=skip_dialog,
                             is_dss=is_dss)
    elif source.endswith('.zarr'):
        MainWindow.from_zarr(filename=source,
                             box_size=box_size,
                             spec_freq_min=spec_freq_min, spec_freq_max=spec_freq_max,
                             skip_dialog=skip_dialog,
                             is_dss=is_dss)
    elif os.path.isdir(source):
        MainWindow.from_dir(dirname=source,
                            events_string=events_string,
                            target_samplingrate=target_samplingrate, box_size=box_size,
                            spec_freq_min=spec_freq_min, spec_freq_max=spec_freq_max,
                            pixel_size_mm=pixel_size_mm,
                            skip_dialog=skip_dialog,
                            is_dss=is_dss)

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


def main_dss(source: str = '', *, song_types_string: str = '',
         spec_freq_min: Optional[float] = None, spec_freq_max: Optional[float] = None,
         skip_dialog: bool = False):
    """GUI for annotating song and training and using DeepSS networks.

    Args:
        source (str): Data source to load.
            Optional - will open an empty GUI if omitted.
            Source can be the path to:
            - an audio file,
            - a numpy file (npy or npz),
            - an h5 file
            - an xarray-behave dataset constructed from an ethodrome data folder saved as a zarr file,
            - an ethodrome data folder (e.g. 'dat/localhost-xxx').
        song_types_string (str): Initialize song types for annotations.
                             String of the form "song_name,song_category;song_name,song_category".
                             Avoid spaces or trailing ';'.
                             Need to wrap the string in "..." in the terminal
                             "song_name" can be any string w/o space, ",", or ";"
                             "song_category" can be "event" (e.g. pulse) or "segment" (sine, syllable)
        spec_freq_min (Optional[float]): Smallest frequency displayed in the spectrogram view. Defaults to 0 Hz.
        spec_freq_max (Optional[float]): Largest frequency displayed in the spectrogram view. Defaults to samplerate/2.
        skip_dialog (bool): If True, skips the loading dialog and goes straight to the data view.
    """
    main(source,
         events_string=song_types_string,
         spec_freq_min=spec_freq_min, spec_freq_max=spec_freq_max,
         skip_dialog=skip_dialog, is_dss=True)


def cli():
    import warnings
    warnings.filterwarnings("ignore")

    # enforce log level
    try:  # py38+
        logging.basicConfig(level=logging.INFO, force=True)
    except ValueError: # <py38
        logging.getLogger().setLevel(logging.INFO)

    defopt.run(main, show_defaults=False)
