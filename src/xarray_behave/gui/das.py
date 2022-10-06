import logging
import collections
from glob import glob
import os.path
from typing import Optional, Dict, List, Any, Union

import zarr
import numpy as np
import pandas as pd
import librosa

import das.make_dataset as dsm
import das.npy_dir
import das.annot

from .. import annot

import das.make_dataset
import das.npy_dir
import das.train
import das.predict
import das.block_stratify

import xarray_behave

package_dir = xarray_behave.__path__[0]
logger = logging.getLogger(__name__)


def data_loader_wav(filename: str, fs: Optional[float] = None, duration: Optional[float] = None):
    x, fs = librosa.load(filename, sr=fs, mono=False, duration=duration)
    x = x.T  # librosa returns mono data as (samples,) and multi as (channels, samples)
    if x.ndim == 1:
        x = x[:, np.newaxis]  # adds singleton dim for single-channel wavs
    return fs, x


def data_loader_npz(filename: str, fs: Optional[float] = None, duration: Optional[float] = None):
    file = np.load(filename)
    fs_file = float(file['samplerate'])
    if duration is None:
        dur = -1
    else:
        dur = min(file['data'].shape[0], int(duration * fs_file))
    x = file['data'][:dur]
    x = x[:, np.newaxis] if x.ndim == 1 else x  # adds singleton dim for single-channel wavs
    if fs is not None and fs != fs_file:
        x = librosa.resample(x, orig_sr=fs_file, target_sr=fs, res_type='kaiser_best')
    else:
        fs = fs_file
    return fs, x


# FIXME auto register with decorator or, even better, use io.audio
data_loaders = {'npz': data_loader_npz, 'wav': data_loader_wav, None: None}


def load_data(filename: str, **kwargs):
    extension = os.path.splitext(filename)[1][1:]  # [1:] strips '.'
    data_loader = data_loaders[extension]
    if data_loader is not None:
        fs, x = data_loader(filename, **kwargs)
        return fs, x
    else:
        return None


def make(data_folder: str,
         store_folder: str,
         file_splits: Dict[str, float],
         data_splits: Dict[str, float],
         make_single_class_datasets: bool = False,
         split_in_two: bool = False,
         block_stratify: bool = False,
         block_size: float = 10,
         event_std_seconds: float = 0,
         gap_seconds: float = 0,
         delete_intermediate_store: bool = True,
         make_onset_offset_events: float = False,
         seed: Optional[float] = None):
    """_summary_

    Args:
        data_folder (str): _description_
        store_folder (str): _description_
        file_splits (Dict[str, float]): _description_
        data_splits (Dict[str, float]): _description_
        make_single_class_datasets (bool, optional): _description_. Defaults to False.
        split_in_two (bool, optional): _description_. Defaults to False.
        block_stratify (bool, optional): _description_. Defaults to False.
        block_size (float, optional): _description_. Defaults to 10.
        event_std_seconds (float, optional): _description_. Defaults to 0.
        gap_seconds (float, optional): _description_. Defaults to 0.
        delete_intermediate_store (bool, optional): _description_. Defaults to True.
        make_onset_offset_events (float, optional): _description_. Defaults to False.
        seed (Optional[float], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_
    """
    # FIXME this function should be in DAS
    annotation_loader = pd.read_csv
    files_annotation = glob(data_folder + '/*_annotations.csv')

    # go through all annotation files and collect info on classes
    class_names = []
    class_types = []
    logger.info('Collecting info on song types from annotation and definition files:')
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

    logger.info('   Identifying song types:')
    for class_name, class_type in zip(class_names, class_types):
        logger.info(f'      Found {class_name} of type {class_type}')
    logger.info('Done.')

    class_names.insert(0, 'noise')
    class_types.insert(0, 'segment')

    if make_onset_offset_events:
        class_names.extend(['syllable_onset', 'syllable_offset'])
        class_types.extend(['event', 'event'])

    file_bases = [f[:-len('_annotations.csv')] for f in files_annotation]
    data_files: List[Union[str, None]] = []
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

    logger.info('Collecting info on sample rates and channels from audio files:')
    # load small snippet from each file to get fs and channels
    fss = set()
    nb_channelss = set()
    for df in dfs:
        fs, xx = load_data(df, duration=0.01)
        fss.add(fs)
        nb_channelss.add(xx.shape[1])

    if len(fss) > 1:
        fs = float(max(fss))
        logger.info(f"   Found multiple samplerates: {fss}.")
        logger.warning(f"   Will resample all files to the max rate {fs} Hz.")
    else:
        logger.info(f"   Sample rate is {fs} Hz.")

    if len(nb_channelss) > 1:
        logger.exception(f"   Stopping. Files have different numbers of channels. Found these: {nb_channelss}.")
        raise ValueError(f"Stopping. Files have different numbers of channels. Found these: {nb_channelss}.")
    nb_channels = int(list(nb_channelss)[0])
    logger.info(f"   Audio has {nb_channels} channels.")

    store = dsm.init_store(nb_channels=nb_channels,
                           nb_classes=len(class_names),
                           samplerate=fs,
                           class_names=class_names,
                           class_types=class_types,
                           make_single_class_datasets=make_single_class_datasets,
                           store_type=zarr.TempStore,
                           store_name='store.zarr',
                           chunk_len=1_000_000)

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
    file_split_dict: Dict = collections.defaultdict(lambda: None)
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
            logger.info("   Splitting in two.")
            data_split_names_old = data_split_names.copy()
            for target in data_split_names_old:
                idx = data_split_names_old.index(target)
                data_split_sizes[idx] /= 2
                data_split_sizes.append(data_split_sizes[idx])
                data_split_names.append(target)
        logger.info("Done.")

    logger.info("Creating dataset:")
    data_split_dict: Dict[str, Any] = {}
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

        fs, x = data_loader(data_file, fs=fs)
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
                    y[:, class_index] = dsm.blur_events(y[:, class_index], event_std_seconds=event_std_seconds, samplerate=fs)

        # introduce small gaps between adjacent segments
        # OPTIONAL but makes boundary detection easier
        if gap_seconds > 0:
            segment_idx = np.where(np.array(class_types) == 'segment')[0]
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
            split_points = []
            logger.info(f'      File added to {file_split_dict[file_base]} data.')
        elif block_stratify:  # split data from each remaining file into train and test chunks according to `splits`
            block_stats_map = das.block_stratify.blockstats_from_data(data=y, block_size=int(block_size * fs))
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
            blocks_x, split_points = das.block_stratify.group_splits(x, group_sizes=data_split_sizes)
            blocks_y, _ = das.block_stratify.group_splits(y, group_sizes=data_split_sizes)
            block_names = data_split_names
        else:
            logger.info('      Skipping.')
            split_points = []
            continue

        data_split_dict[file_base] = {name: [] for name in data_split_names}
        for name, start_end in zip(block_names, split_points):
            data_split_dict[file_base][name].append([int(item) for item in start_end])

        for x, y, block_name in zip(blocks_x, blocks_y, block_names):
            store[block_name]['x'].append(x)

            y_norm = dsm.normalize_probabilities(y)
            store[block_name]['y'].append(y_norm)

            if make_single_class_datasets:
                for cnt, class_name in enumerate(class_names[1:]):
                    y_norm = dsm.normalize_probabilities(y[:, [0, cnt + 1]])
                    store[block_name][f'y_{class_name}'].append(y_norm)

    store.attrs['data_splits'] = data_split_dict
    logger.info('Done.')
    # report
    logger.info(
        f"  Got {store['train']['x'].shape}, {store['val']['x'].shape}, {store['test']['x'].shape} train/val/test samples.")
    # save as npy_dir
    logger.info(f'  Saving to {store_folder}.')
    das.npy_dir.save(store_folder, store)
    if delete_intermediate_store:
        pass
    logger.info('The dataset has been made.')


def onset_offset_events(df,):
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
