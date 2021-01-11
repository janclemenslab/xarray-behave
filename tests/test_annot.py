import pytest

import numpy as np
import xarray as xr
import pandas as pd
from xarray_behave.annot import *


@pytest.fixture()
def test_data():
    names = []
    times = []
    for ii in range(10):
        times.append([ii, ii])
        names.append('pulse')
    for ii in range(10):
        times.append([10+ii, 10.1+ii])
        names.append('sine')

    return names, times


@pytest.fixture()
def test_lists():
    names = []
    start_seconds = []
    stop_seconds = []
    for ii in range(10):
        start_seconds.append(ii)
        stop_seconds.append(ii)
        names.append('pulse')
    for ii in range(10):
        start_seconds.append(10+ii)
        stop_seconds.append(10.1+ii)
        names.append('sine')
    names.extend(['empty_event', 'empty_segment'])
    start_seconds.extend([np.nan, np.nan])
    stop_seconds.extend([np.nan, 0])
    return names, start_seconds, stop_seconds


@pytest.fixture()
def test_events(test_lists):
    names, start_seconds, stop_seconds = test_lists
    et = Events.from_lists(names, start_seconds, stop_seconds)
    return et


def test_from_ds(test_data):
    names, times = test_data
    names = xr.DataArray(name='event_names', data=np.array(names, dtype='U128'), dims=['index',])
    names.attrs['allowed_names'] = list(set(names.data))
    times = xr.DataArray(name='event_times', data=times, dims=['index','event_time'], coords={'event_time': ['start_seconds', 'stop_seconds']})
    names, times
    ds = xr.Dataset({da.name: da for da in [names, times]})
    ds.attrs['pulse'] = 'event'
    ds.attrs['sine'] = 'segment'
    ds

    et = Events.from_dataset(ds)
    print(et)


def test_from_lists(test_lists):
    names, start_seconds, stop_seconds = test_lists
    et = Events.from_lists(names, start_seconds, stop_seconds)
    assert et.categories['pulse'] == 'event'
    assert et.categories['sine'] == 'segment'
    assert et.categories['empty_event'] == 'event'
    assert et.categories['empty_segment'] == 'segment'


def test_to_df(test_events):
    df = test_events.to_df()
    assert df.shape==(22, 3)


def test_to_df_noempty(test_events):
    df = test_events.to_df(preserve_empty=False)
    assert df.shape==(20, 3)
    print(df.columns)
    assert tuple(df.columns) == ('name', 'start_seconds', 'stop_seconds')

def test_to_lists(test_events):
    names, starts, stops = test_events.to_lists()
    assert len(names)==22
    assert len(starts)==22
    assert len(stops)==22


def test_update():
    evt_main = Events(data={'sine': [[1,2],[3,4]], 'pulse': [[5,5],[6,6]], 'vibration': [25, 25]})
    evt_new = Events(data={'sine': [[1,2],[3,4]], 'vibration': [[7,7],[8,8]], 'whatnew': [[10,10],[18,18]]})
    evt_main.update(evt_new)
    assert len(evt_main.names) == 4
    assert evt_main['vibration'].shape[0] == 2

def test_filter(test_events):
    evt_list = test_events.filter_range('pulse', 5, 100)
    assert evt_list.shape==(4, 2)
    evt_list = test_events.filter_range('sine', 5, 100)
    assert evt_list.shape==(10, 2)


def test_delete(test_events):
    test_events.delete_range('pulse', 5, 100)
    assert test_events['pulse'].shape==(6, 2)
    test_events.delete_range('sine', 5, 100)
    assert test_events['sine'].shape==(0, 2)

def test_make_empty():

    # evt = Events(categories={'sine': 'segment', 'pulse': 'event', 'vibration': 'event'})
    evt = Events(data={'sine': [], 'pulse': [], 'vibration': []})
    print(evt)
    df = evt.to_df()
    print(df)
    df.to_csv('test.txt')

    df2 = pd.read_csv('test.txt')
    print(df2)
    evt2 = Events.from_df(df2)
    print(evt2)
