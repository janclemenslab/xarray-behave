import sys
import logging
import collections
from glob import glob
import os.path
from typing import Optional, Dict, List

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
import das.block_stratify

import xarray_behave


package_dir = xarray_behave.__path__[0]
logger = logging.getLogger(__name__)


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


# TODO: auto register with decorator or, even better, use io.audio
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
         block_size: int = 10,
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
    logger.info('Collecting song types from annotation and definition files:')
    for file_annotation in files_annotation:
        # TODO load definition files
        logger.info(f"   {file_annotation}")
        df = annotation_loader(file_annotation)
        event_times = annot.Events.from_df(df)
        class_names.extend(event_times.names)
        class_types.extend(event_times.categories.values())

        # this only works in py39!!
        # file_definition = file_annotation.removesuffix('_annotations.csv') + '_definitions.csv'
        suffix = '_annotations.csv'
        file_definition = file_annotation
        if file_definition.endswith(suffix):
            file_definition = file_definition[:-len(suffix)] + '_definitions.csv'

        if os.path.exists(file_definition):
            data = np.loadtxt(file_definition, dtype=str, delimiter=",")
            class_names.extend(list(data[:, 0]))
            class_types.extend(list(data[:, 1]))

    logger.info('Done.')

    class_names, first_indices = np.unique(class_names, return_index=True)
    class_types = list(np.array(class_types)[first_indices])
    class_names = list(class_names)

    logger.info('Identifying song types:')
    for class_name, class_type in zip(class_names, class_types):
        logger.info(f'   Found {class_name} of type {class_type}')
    logger.info('Done.')

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
        logger.exception('No valid data files found.')
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
    # parts = ['train', 'val', 'test']

    if len(file_splits):
        logger.info(f"Splitting data by files: {file_splits}.")
        if sum(file_splits.values()) > 1.0:
            raise ValueError('Sum of file splits > 1.0!')

        # remove zero-size sets
        file_splits = {name: fraction for name, fraction in file_splits.items() if fraction > 0}

        if len(file_splits) < 3:
            file_splits['remainder'] = max(0, 1 - sum(file_splits.values()))

        if block_stratify:
            logger.info("   Block stratification.")
            annotation_files = [file_base + '_annotations.csv' for file_base in file_bases]
            block_stats_map = das.block_stratify.blockstats_from_files(annotation_files)
            block_stats = block_stats_map.values()
        else:
            block_stats = None

        file_split_dict = das.block_stratify.block(block_names=file_bases,
                                                   group_sizes=list(file_splits.values()),
                                                   group_names=list(file_splits.keys()),
                                                   seed=seed,
                                                   block_stats=block_stats)
        logger.info("Done.")
    store.attrs['file_split_dict'] = dict(file_split_dict)

    data_split_names = []
    if len(data_splits):
        # remove zero-size sets
        data_splits = {name: fraction for name, fraction in data_splits.items() if fraction > 0}

        logger.info(f"Splitting individual data files: {data_splits}.")
        # TODO: Keep this as a dict!! SIMPLIFY THIS!!
        fractions = np.array(list(data_splits.values()))
        probs = fractions / np.sum(fractions)
        data_splits = {key: val for key, val in zip(data_splits.keys(), probs)}
        data_split_names = list(data_splits.keys())
        data_split_sizes = list(data_splits.values())
        if split_in_two:
            logger.info(f"   Splitting in two.")
            data_split_names_old = data_split_names.copy()
            for target in data_split_names_old:
                idx = data_split_names_old.index(target)
                data_split_sizes[idx] /= 2
                data_split_sizes.append(data_split_sizes[idx])
                data_split_names.append(target)
        logger.info("Done.")

    logger.info(f"Creating dataset:")
    for file_base, data_file in zip(file_bases, data_files):
        if data_file is None:
            logger.warning(f'Unknown data file for {file_base} - skipping.')
            continue
        logger.info(f'   {file_base}')

        # load the recording
        if data_file.endswith('.npz'):
            data_loader = data_loader_npz
        elif data_file.endswith('.wav'):
            data_loader = data_loader_wav
        else:
            logger.warning(f'Unknown data file {file_base} - skipping.')
            continue

        fs, x = data_loader(data_file)
        nb_samples = x.shape[0]

        # load annotations
        df = annotation_loader(file_base + '_annotations.csv')
        df = df.dropna()

        # make new events from syllable on- and offsets
        if make_onset_offset_events:
            df = onset_offset_events(df)
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

        if file_split_dict[file_base] != 'remainder' and file_split_dict[file_base] is not None:
            blocks_x = [x]
            blocks_y = [y]
            block_names = [file_split_dict[file_base]]
            logger.info(f'      File added to {file_split_dict[file_base]} data.')
        elif block_stratify:  # split data from each remaining file into train and test chunks according to `splits`
            block_size = int(block_size * fs)
            block_stats_map = das.block_stratify.blockstats_from_data(data=y, block_size=block_size)
            block_stats = list(block_stats_map.values())
            split_points = list(block_stats_map.keys())
            blocks_x = das.block_stratify.blocks_from_split_points(x, split_points)
            blocks_y = das.block_stratify.blocks_from_split_points(y, split_points)
            # get stats from blocks here
            block_dict = das.block_stratify.block(block_names=split_points,
                                                  group_sizes=data_split_sizes,
                                                  group_names=data_split_names,
                                                  seed=seed,
                                                  block_stats=block_stats)
            block_names = list(block_dict.values())
            logger.info(f'      Stratified {data_split_sizes} split into {data_split_names} data.')
        elif not block_stratify:
            # shuffle
            order = np.random.permutation(len(data_split_sizes))
            data_split_sizes = np.array(data_split_sizes)[order]
            data_split_names = np.array(data_split_names)[order]

            logger.info(f'      Random {data_split_sizes} split into {data_split_names} data.')
            blocks_x = das.block_stratify.group_splits(x, group_sizes=data_split_sizes)
            blocks_y = das.block_stratify.group_splits(y, group_sizes=data_split_sizes)
            block_names = data_split_names
        else:
            logger.info('      Skipping.')
            continue

        for x, y, block_name in zip(blocks_x, blocks_y, block_names):
            store[block_name]['x'].append(x)

            y_norm = dsm.normalize_probabilities(y)
            store[block_name]['y'].append(y_norm)

            if make_single_class_datasets:
                for cnt, class_name in enumerate(class_names[1:]):
                    y_norm = dsm.normalize_probabilities(y[:, [0, cnt+1]])
                    store[block_name][f'y_{class_name}'].append(y_norm)

    logger.info('Done.')
    # report
    logger.info(f"  Got {store['train']['x'].shape}, {store['val']['x'].shape}, {store['test']['x'].shape} train/val/test samples.")
    # save as npy_dir
    logger.info(f'  Saving to {store_folder}.')
    das.npy_dir.save(store_folder, store)
    if delete_intermediate_store:
        pass
    logger.info('The dataset has been made.')


def onset_offset_events(df, ):
    adf = das.annot.Events.from_df(df)
    start_seconds = []
    stop_seconds = []
    for nam, cat in adf.categories.items():
        if cat == 'segment':
            start_seconds.extend(adf.start_seconds(nam))
            stop_seconds.extend(adf.stop_seconds(nam))

    logger.info(f'      Making onset and offset events for {len(start_seconds)} syllables.')
    adf.add_name(name='syllable_onset', category='event')
    adf.add_name(name='syllable_offset', category='event')
    for start, stop in zip(start_seconds, stop_seconds):
        adf.add_time(name='syllable_onset', start_seconds=start)
        adf.add_time(name='syllable_offset', start_seconds=stop)

    df = adf.to_df()
    return df
