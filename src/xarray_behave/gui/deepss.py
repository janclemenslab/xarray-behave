import sys
import logging
import collections
from glob import glob
import os.path

import zarr
import numpy as np

import xarray_behave as xb
from .formbuilder import YamlFormWidget, YamlDialog

import dss.make_dataset as dsm
import dss.npy_dir

import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile

from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from .. import annot

try:
    import dss.make_dataset
    import dss.npy_dir
    import dss.train
    import dss.predict
except ImportError as e:
    print(e)
    print('you may need to install DeepSS: link_to_pypi')

package_dir = xb.__path__[0]


def predict(ds):
    dialog = YamlDialog(yaml_file=package_dir + "/gui/forms/dss_predict.yaml",
                        title='Predict labels using DeepSS')
    dialog.show()
    result = dialog.exec_()

    if result == QtGui.QDialog.Accepted:
        form = dialog.form.get_form_data()
        model_path = form['model_path']
        model_path = model_path.rsplit('_',1)[0]  # split off suffix

        logging.info('   running inference.')
        events, segments, _ = dss.predict.predict(ds.song_raw.compute(), model_path, verbose=1, batch_size=96,
                                                  event_thres=form['event_thres'], event_dist=form['event_dist'],
                                                  event_dist_min=form['event_dist_min'], event_dist_max=form['event_dist_max'],
                                                  segment_thres=form['event_thres'], segment_fillgap=form['segment_fillgap'],
                                                  segment_minlen=form['segment_minlen'],
                                                  )
        return events, segments
    else:
        logging.info('   aborting.')
        return None

# def update_predictions(ds):
#     from xarray_behave.gui.formbuilder import YamlFormWidget

#     import sys
#     from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
#     import pyqtgraph as pg

#     def save_as():
#         print('saved')

#     app = pg.QtGui.QApplication([])
#     win = pg.QtGui.QMainWindow()

#     win.setWindowTitle("Configure training")
#     widget = YamlFormWidget(yaml_file="/Users/janc/Dropbox/code.py/xarray_behave/src/xarray_behave/gui/forms/update_labels.yaml")
#     widget.mainAction.connect(save_as)
#     win.setCentralWidget(widget)
#     win.show()



def data_loader_wav(filename):
    # load the recording
    fs, x = scipy.io.wavfile.read(filename)
    x = x[:, np.newaxis] if x.ndim==1 else x  # adds singleton dim for single-channel wavs
    return fs, x

def data_loader_npz(filename):
    # load the recording
    file = np.load(filename)
    fs = file['samplerate']
    x = file['song']
    x = x[:, np.newaxis] if x.ndim==1 else x  # adds singleton dim for single-channel wavs
    return fs, x

# auto register with decorator
data_loaders = {'npz': data_loader_npz, 'wav': data_loader_wav, None: None}


def load_data(filename):
    extension = os.path.splitext(filename)[1][1:]  # [1:] strips '.'
    data_loader = data_loaders[extension]
    if data_loader is not None:
        fs, x = data_loader(filename)
        return fs, x
    else:
        return None


def make(data_folder, store_folder,
         file_splits, data_splits,
         make_single_class_datasets: bool = False,
         split_train_in_two: bool = True,
         event_std_seconds: float = 0,
         gap_seconds: float = 0):

    annotation_loader = pd.read_csv
    files_annotation = glob(data_folder + '/*.csv')


    # go through all annotation files and collect info on classes
    class_names = []
    class_types = []
    for file_annotation in files_annotation:
        print(file_annotation)
        df = annotation_loader(file_annotation)
        event_times = annot.Events.from_df(df)
        class_names.extend(event_times.names)
        class_types.extend(event_times.categories.values())

    class_names, first_indices = np.unique(class_names, return_index=True)
    class_types = list(np.array(class_types)[first_indices])
    class_names = list(class_names)

    for class_name, class_type in zip(class_names, class_types):
        print(f'found {class_name} of type {class_type}')

    class_names.insert(0, 'noise')
    class_types.insert(0, 'segment')

    file_bases = [f.rpartition('.')[0] for f in files_annotation]
    data_files = []
    for file_base in file_bases:
        if os.path.exists(file_base + '.npz'):
            data_files.append(file_base + '.npz')
        elif os.path.exists(file_base + '.wav'):
            data_files.append(file_base + '.wav')
        else:
            data_files.append(None)

    # read first data file to infer number of audio channels
    # file_base = files_annotation[0].rpartition('.csv')[0]
    # load the recording

    # get valid file
    dfs = [data_file for data_file in data_files if data_file is not None]
    if not len(dfs):
        logging.exception('No valid data files found.')
        raise ValueError('No valid data files found.')

    fs, x = load_data(dfs[0])
    nb_channels = x.shape[1]

    store = dsm.init_store(nb_channels=nb_channels,
                        nb_classes=len(class_names),
                        samplerate=fs,
                        class_names=class_names,
                        class_types=class_types,
                        make_single_class_datasets=make_single_class_datasets,
                        store_type=zarr.DictStore, store_name='store.zarr', chunk_len=1_000_000)
    store['train']['x'].shape, store['val']['x'].shape, store['test']['x'].shape

    # first split files into a train-test and val
    file_split_dict = collections.defaultdict(lambda: None)
    parts = ['train', 'val', 'test']

    if len(file_splits):
        if sum(file_splits.values())>1.0:
            raise ValueError(f'Sum of file splits > 1.0!')

        if len(file_splits)<3:
            file_splits['remainder'] = max(0, 1 - sum(file_splits.values()))

        file_splits = dsm.generate_file_splits(file_bases,
                                            splits=list(file_splits.values()),
                                            split_names=list(file_splits.keys()))


        for file_base in file_bases:
            for part in parts:
                if part in file_splits and file_base in file_splits[part]:
                    file_split_dict[file_base] = part

    data_split_targets = []
    if len(data_splits):
        fractions = np.array(list(data_splits.values()))
        probs = fractions / np.sum(fractions)
        data_splits = {key: val for key, val in zip(data_splits.keys(), probs)}
        data_split_targets = list(data_splits.keys())
        data_splits = list(data_splits.values())
        if split_train_in_two and 'train' in data_split_targets:
            train_idx = data_split_targets.index('train')
            data_splits[train_idx] /= 2
            data_splits.append(data_splits[train_idx])
            data_split_targets.append('train')

    for file_base, data_file in zip(file_bases, data_files):
        if data_file is None:
            logging.warning(f'Unknown data file for {file_base} - skipping.')
            continue
        logging.info(f'  {file_base}')

        # load the recording
        if data_file.endswith('.npz'):
            data_loader = data_loader_npz
        elif data_file.endswith('.wav'):
            data_loader = data_loader_wav
        else:
            logging.warning(f'Unknown data file {file_base} - skipping.')
            continue

        fs, x = data_loader(data_file)
        nb_samples = x.shape[0]

        # load annotations
        df = annotation_loader(file_base + '.csv')

        # make initial annotation matrix
        y = dsm.make_annotation_matrix(df, nb_samples, fs, class_names)

        # blur events
        # OPTIONAL but highly recommended
        if event_std_seconds > 0:
            for class_index, class_type in enumerate(class_types):
                if class_type == 'event':
                    y[:, class_index] = dsm.blur_events(y[:, class_index],
                                                        event_std_seconds=event_std_seconds,
                                                        samplerate=fs)

        # introduce small gaps between adjacent segments
        # OPTIONAL but makes boundary detection easier
        if gap_seconds > 0:
            segment_idx = np.where(np.array(class_types)=='segment')[0]
            if len(segment_idx):
                y[:, segment_idx] = dsm.make_gaps(y[:, segment_idx],
                                                  gap_seconds=gap_seconds,
                                                  samplerate=fs)

        # all validation files
        if file_split_dict[file_base] is not None:
            name = file_split_dict[file_base]
            logging.info(f'    {name} data.')
            store[name]['x'].append(x)
            store[name]['y'].append(dsm.normalize_probabilities(y))
            # make prediction targets for individual song types [OPTIONAL]
            for cnt, class_name in enumerate(class_names[1:]):
                store[name][f'y_{class_name}'].append(dsm.normalize_probabilities(y[:, [0, cnt+1]]))
        else:
            # split data from each remaining file into train and test chunks according to `splits`
            split_arrays = dsm.generate_data_splits({'x': x, 'y': y}, data_splits, data_split_targets)
            logging.info(f'    splitting {data_splits} into {data_split_targets}.')
            for name in data_split_targets:
                store[name]['x'].append(split_arrays['x'][name])
                store[name]['y'].append(dsm.normalize_probabilities(split_arrays['y'][name]))
                # make prediction targets for individual song types [OPTIONAL]
                for cnt, class_name in enumerate(class_names[1:]):
                    store[name][f'y_{class_name}'].append(dsm.normalize_probabilities(split_arrays['y'][name][:, [0, cnt+1]]))

    # report
    logging.info(f"  Got {store['train']['x'].shape}, {store['val']['x'].shape}, {store['test']['x'].shape} train/test/val samples.")
    # save as npy_dir
    logging.info(f'  Saving to {store_folder}.')
    dss.npy_dir.save(store_folder, store)

    # delete intermediate store
    pass
