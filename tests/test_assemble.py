import xarray_behave as xb
import logging

logging.getLogger().setLevel(logging.INFO)

recs = {
    'rpi9-20210409_093149': {
        'target_sampling_rate': 100,  # from default of 1_000 to speed things up
        'root': 'tests/data',
    },
    'localhost-20210617_113024': {
        'root': 'tests/data',
    },
    'localhost-20181120_144618': {
        'root': 'tests/data',
    },
    'localhost-20210624_104612': {
        'root': 'tests/data',
    },
    'localhost-20210628_145223': {
        'dat_path': 'dat',
        'res_path': 'res',
        'root': 'tests/data'
    },
    'localhost-20210629_171532': {
        'dat_path': 'dat',
        'res_path': 'res',
        'root': 'tests/data'
    },
    'Dmel_male': {
        'dat_path': 'dat',
        'res_path': 'dat',
        'filepath_daq': 'tests/data/dat/Dmel_male.wav',
        'filepath_annotations': 'dat/Dmel_male_annotations.csv'
    },
    'Dmel_male2': {
        'dat_path': 'dat',
        'res_path': 'dat',
        'filepath_daq': 'tests/data/dat/Dmel_male.npz',
        'filepath_annotations': 'tests/data/dat/Dmel_male_annotations.csv'
    }
}


# TODO: add asserts to ensure results conform to expected ds structure
def test_assemble0():
    datenames = list(recs.keys())
    ii = 0
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)


def test_assemble1():
    datenames = list(recs.keys())
    ii = 1
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)


def test_assemble2():
    datenames = list(recs.keys())
    ii = 2
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)


def test_assemble3():
    datenames = list(recs.keys())
    ii = 3
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)


def test_assemble4():
    datenames = list(recs.keys())
    ii = 4
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)


def test_assemble5():
    datenames = list(recs.keys())
    ii = 5
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)


def test_assemble6():
    datenames = list(recs.keys())
    ii = 6
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)


def test_assemble7():
    datenames = list(recs.keys())
    ii = 7
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)


if __name__ == '__main__':
    test_assemble0()
    test_assemble1()
    test_assemble2()
    test_assemble3()
    test_assemble4()
    test_assemble5()
    test_assemble6()
    test_assemble7()
