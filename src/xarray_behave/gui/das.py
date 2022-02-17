import sys
import logging
import collections
from glob import glob
import os.path
from typing import Optional, Dict

import zarr
import numpy as np

import das.make_dataset as dsm
import das.npy_dir
import das.annot

import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile

from .. import annot

import das.make_dataset
import das.npy_dir
import das.train
import das.predict

import xarray_behave


package_dir = xarray_behave.__path__[0]


# these should be xb.loaders!!
def data_loader_wav(filename):
    # load the recording
    fs, x = scipy.io.wavfile.read(filename)
    x = x[:, np.newaxis] if x.ndim==1 else x  # adds singleton dim for single-channel wavs
    return fs, x

def data_loader_npz(filename):
    # load the recording
    file = np.load(filename)
    fs = file['samplerate']
    x = file['data']
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


def make(data_folder: str, store_folder: str,
         file_splits: Dict[str, float], data_splits: Dict[str, float],
         make_single_class_datasets: bool = False,
         split_in_two: bool = False,
         block_stratify: bool = False,
         block_size: int = 5,
         event_std_seconds: float = 0,
         gap_seconds: float = 0,
         delete_intermediate_store: bool = True,
         make_onset_offset_events: float = False,
         seed: Optional[float] = None):

    annotation_loader = pd.read_csv
    files_annotation = glob(data_folder + '/*_annotations.csv')

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

    if make_onset_offset_events:
        class_names.extend(['syllable_onset', 'syllable_offset'])
        class_types.extend(['event', 'event'])

    file_bases = [f[:-len('_annotations.csv')] for f in files_annotation]
    data_files = []
    for file_base in file_bases:
        if os.path.exists(file_base + '.npz'):
            data_files.append(file_base + '.npz')
        elif os.path.exists(file_base + '.wav'):
            data_files.append(file_base + '.wav')
        else:
            data_files.append(None)

    # read first data file to infer number of audio channels
    # get valid file
    dfs = [data_file for data_file in data_files if data_file is not None]
    if not len(dfs):
        logging.exception('No valid data files found.')
        raise ValueError('No valid data files found.')

    fs, x = load_data(dfs[0])
    fs = float(fs)
    nb_channels = x.shape[1]

    store = dsm.init_store(nb_channels=nb_channels,
                        nb_classes=len(class_names),
                        samplerate=fs,
                        class_names=class_names,
                        class_types=class_types,
                        make_single_class_datasets=make_single_class_datasets,
                        store_type=zarr.TempStore, store_name='store.zarr', chunk_len=1_000_000)

    # save args for reproducibility
    store.attrs['event_std_seconds'] = event_std_seconds
    store.attrs['gap_seconds'] = gap_seconds
    store.attrs['data_folder'] = data_folder
    store.attrs['store_folder'] = store_folder
    store.attrs['file_splits'] = file_splits
    store.attrs['data_splits'] = data_splits
    store.attrs['make_single_class_datasets'] = make_single_class_datasets
    store.attrs['split_in_two'] = split_in_two
    store.attrs['delete_intermediate_store'] = delete_intermediate_store

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
                                            split_names=list(file_splits.keys()),
                                            seed=seed)

        for file_base in file_bases:
            for part in parts:
                if part in file_splits and file_base in file_splits[part]:
                    file_split_dict[file_base] = part
    store.attrs['file_split_dict'] = dict(file_split_dict)

    data_split_targets = []
    if len(data_splits):
        fractions = np.array(list(data_splits.values()))
        probs = fractions / np.sum(fractions)
        data_splits = {key: val for key, val in zip(data_splits.keys(), probs)}
        data_split_targets = list(data_splits.keys())
        data_splits = list(data_splits.values())
        if split_in_two:
            if 'train' in data_split_targets:
                train_idx = data_split_targets.index('train')
                data_splits[train_idx] /= 2
                data_splits.append(data_splits[train_idx])
                data_split_targets.append('train')

            if 'val' in data_split_targets:
                val_idx = data_split_targets.index('val')
                data_splits[val_idx] /= 2
                data_splits.append(data_splits[val_idx])
                data_split_targets.append('val')

            if 'test' in data_split_targets:
                test_idx = data_split_targets.index('test')
                data_splits[test_idx] /= 2
                data_splits.append(data_splits[test_idx])
                data_split_targets.append('test')

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
        df = annotation_loader(file_base + '_annotations.csv')
        df = df.dropna()

        # make new events from syllable on- and offsets
        if make_onset_offset_events:
            adf = das.annot.Events.from_df(df)
            start_seconds = []
            stop_seconds = []
            for nam, cat in adf.categories.items():
                if cat == 'segment':
                    start_seconds.extend(adf.start_seconds(nam))
                    stop_seconds.extend(adf.stop_seconds(nam))

            logging.info(f'   Making syllable {len(start_seconds) + len(stop_seconds)} onset/offset events.')
            adf.add_name(name='syllable_onset', category='event')
            adf.add_name(name='syllable_offset', category='event')
            for start, stop in zip(start_seconds, stop_seconds):
                adf.add_time(name='syllable_onset', start_seconds=start)
                adf.add_time(name='syllable_offset', start_seconds=stop)

            df = adf.to_df()

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
                                                  samplerate=fs,
                                                  start_seconds=df['start_seconds'],
                                                  stop_seconds=df['start_seconds'])
        # all validation files
        if file_split_dict[file_base] is not None:
            name = file_split_dict[file_base]
            logging.info(f'    {name} data.')
            store[name]['x'].append(x)
            store[name]['y'].append(dsm.normalize_probabilities(y))
            # make prediction targets for individual song types [OPTIONAL]
            if make_single_class_datasets:
                for cnt, class_name in enumerate(class_names[1:]):
                    store[name][f'y_{class_name}'].append(dsm.normalize_probabilities(y[:, [0, cnt+1]]))
        else:
            # split data from each remaining file into train and test chunks according to `splits`
            if block_stratify:
                stratify = y
            else:
                stratify = None

            # convert block_size from seconds to samples
            if block_size is not None:
                block_size = int(block_size * fs)

            # FIXME: lenght of split does not match with fractions
            split_arrays = dsm.generate_data_splits({'x': x, 'y': y}, data_splits, data_split_targets, seed=seed,
                                                     block_stratify=stratify, block_size=block_size)
            logging.info(f'    splitting {data_splits} into {data_split_targets}.')
            for name in set(data_split_targets):
                store[name]['x'].append(split_arrays['x'][name])
                store[name]['y'].append(dsm.normalize_probabilities(split_arrays['y'][name]))
                # make prediction targets for individual song types [OPTIONAL]
                if make_single_class_datasets:
                    for cnt, class_name in enumerate(class_names[1:]):
                        store[name][f'y_{class_name}'].append(dsm.normalize_probabilities(split_arrays['y'][name][:, [0, cnt+1]]))

    # report
    logging.info(f"  Got {store['train']['x'].shape}, {store['val']['x'].shape}, {store['test']['x'].shape} train/val/test samples.")
    # save as npy_dir
    logging.info(f'  Saving to {store_folder}.')
    das.npy_dir.save(store_folder, store)
    if delete_intermediate_store:
        pass  # TODO delete intermediate store
