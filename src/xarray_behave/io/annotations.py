import h5py
import flammkuchen
import numpy as np
import pandas as pd
import scipy.io
import logging
from typing import Optional
from .. import (io,
                annot)


def fill_gaps(sine_pred, gap_dur=100):
    onsets = np.where(np.diff(sine_pred.astype(np.int))==1)[0]
    offsets = np.where(np.diff(sine_pred.astype(np.int))==-1)[0]
    if len(onsets) and len(offsets):
        onsets = onsets[onsets<offsets[-1]]
        offsets = offsets[offsets>onsets[0]]
        durations = offsets - onsets
        for idx, (onset, offset, duration) in enumerate(zip(onsets, offsets, durations)):
            if idx>0 and offsets[idx-1]>onsets[idx]-gap_dur:
                sine_pred[offsets[idx-1]:onsets[idx]+1] = 1
    return sine_pred


def remove_short(sine_pred, min_len=100):
    # remove too short sine songs
    onsets = np.where(np.diff(sine_pred.astype(np.int))==1)[0]
    offsets = np.where(np.diff(sine_pred.astype(np.int))==-1)[0]
    if len(onsets) and len(offsets):
        onsets = onsets[onsets<offsets[-1]]
        offsets = offsets[offsets>onsets[0]]
        durations = offsets - onsets
        for cnt, (onset, offset, duration) in enumerate(zip(onsets, offsets, durations)):
            if duration<min_len:
                sine_pred[onset:offset+1] = 0
    return sine_pred


@io.register_provider
class Deepss(io.BaseProvider):

    KIND = 'annotations'
    NAME = 'deepss'
    SUFFIXES = ['_song.h5', '_vibration.h5', '_pulse.h5', '_sine.h5', '_dss.h5']

    def load(self, filename: Optional[str] = None):
        """Load output produced by DeepSongSegmenter wrapper."""
        if filename is None:
            filename = self.path

        onsets = []
        offsets = []
        names = []

        res = flammkuchen.load(filename)

        if 'event_indices' in res or 'segment_labels' in res:  # load old style format
            logging.info('   Converting legacy DSS format...')
            res['event_seconds'] = np.zeros((0,))
            res['event_sequence'] = []
            for event_name, event_idx in zip(res['event_names'], res['event_indices']):
                res['event_seconds'] = np.append(res['event_seconds'], np.array(event_idx) / res['samplerate_Hz'])
                res['event_sequence'] = np.append(res['event_sequence'], [event_name for _ in range(len(event_idx))])

            res['segment_onsets_seconds'] = np.zeros((0,))
            res['segment_offsets_seconds'] = np.zeros((0,))
            res['segment_sequence'] = []
            for segment_name, segment_labels in zip(res['segment_names'], res['segment_labels']):
                if 'sine' in segment_name:
                    logging.info(f'   Postprocessing {segment_name}')
                    segment_labels = fill_gaps(segment_labels, gap_dur=0.02 * res['samplerate_Hz'])
                    segment_labels = remove_short(segment_labels, min_len=0.02 * res['samplerate_Hz'])

                # detect on and offset from binary labels
                segment_onset_idx = np.where(np.diff(segment_labels.astype(np.float), prepend=0)==1)[0].astype(np.float)
                segment_offset_idx = np.where(np.diff(segment_labels.astype(np.float), append=0)==-1)[0].astype(np.float)

                res['segment_onsets_seconds'] = np.append(res['segment_onsets_seconds'], segment_onset_idx / res['samplerate_Hz'])
                res['segment_offsets_seconds'] = np.append(res['segment_offsets_seconds'], segment_offset_idx / res['samplerate_Hz'])
                res['segment_sequence'] = np.append(res['segment_sequence'], [segment_name for _ in range(len(segment_onset_idx))])

        for event_seconds, event_names in zip(res['event_seconds'], res['event_sequence']):
            onsets.append(event_seconds)
            offsets.append(event_seconds)
            names.append(event_names)

        if 'event_names' in res:  # ensure empty event types are initialized
            for name in res['event_names']:
                if name not in names:
                    names.append(name)
                    onsets.append(np.nan)
                    offsets.append(np.nan)

        for segment_onsets, segment_offsets, segment_names in zip(res['segment_onsets_seconds'], res['segment_offsets_seconds'], res['segment_sequence']):
            onsets.append(segment_onsets)
            offsets.append(segment_offsets)
            names.append(segment_names)

        if 'segment_names' in res:  # ensure empty segment types are initialized
            for name in res['segment_names']:
                if name not in names:
                    names.append(name)
                    onsets.append(np.nan)
                    offsets.append(0)

        et = annot.Events.from_lists(names=names, start_seconds=onsets, stop_seconds=offsets)

        return et, et.categories


@io.register_provider
class FlySongSegmenter(io.BaseProvider):

    KIND = 'annotations'
    NAME = 'FlySongSegmenter'
    SUFFIXES = ['_song.mat']

    def load(self, filename: Optional[str] = None):
        """Load output produced by FlySongSegmenter."""
        if filename is None:
            filename = self.path

        res = dict()
        try:
            d = scipy.io.loadmat(filename)
            res['pulse_times_samples'] = d['pInf'][0, 0]['wc']
            res['pulse_labels'] = d['pInf'][0, 0]['pulseLabels']
            res['song_mask'] = d['bInf'][0, 0]['Mask'].T  # 0 - silence, 1 - pulse, 2 - sine
        except NotImplementedError:
            with h5py.File(filename, 'r') as f:
                res['pulse_times_samples'] = f['pInf/wc'][:].T
                res['pulse_labels'] = f['pInf/pulseLabels'][:].T
                res['song_mask'] = f['bInf/Mask'][:]

        sine_song = res['song_mask'] == 2
        fs = 10_000  # Hz

        res['event_names'] = ['song_pulse_any_fss', 'song_pulse_slow_fss', 'song_pulse_fast_fss', 'sine_fss']
        res['event_categories'] = ['event', 'event', 'event', 'segment']
        res['event_indices'] = [res['pulse_times_samples'],
                                res['pulse_times_samples'][res['pulse_labels'] == 1],
                                res['pulse_times_samples'][res['pulse_labels'] == 0],
                                np.where(sine_song == 2)[0]]
        # extract event_seconds
        event_seconds = {'song_pulse_any_fss': res['pulse_times_samples'] / fs,
                        'song_pulse_slow_fss': res['pulse_times_samples'][res['pulse_labels'] == 1] / fs,
                        'song_pulse_fast_fss': res['pulse_times_samples'][res['pulse_labels'] == 0] / fs,
                        'sine_fss': np.where(sine_song == 2)[0] / fs}
        # event_categories
        event_categories = {}
        for cat, typ in zip(event_seconds.keys(), ['event', 'event', 'event', 'segment']):
            event_categories[cat] = typ
        return event_seconds, event_categories