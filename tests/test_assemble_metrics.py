import xarray_behave as xb
import logging

logging.getLogger().setLevel(logging.INFO)

recs = {'rpi9-20210409_093149': {},
        'localhost-20210617_113024': {},
        'localhost-20181120_144618': {},
        'localhost-20210624_104612': {},
        'localhost-20210628_145223': {'dat_path': 'dat', 'res_path': 'res', 'root': 'tests/data'},
        'localhost-20210629_171532': {'dat_path': 'dat', 'res_path': 'res', 'root': 'tests/data'},
        'Dmel_male': {'dat_path': 'dat', 'res_path': 'dat', 'root': 'tests/data', 'filepath_daq': 'dat/Dmel_male.wav', 'filepath_annotations': 'dat/Dmel_male_annotations.csv'},
        'Dmel_male2': {'dat_path': 'dat', 'res_path': 'dat', 'root': 'tests/data', 'filepath_daq': 'dat/Dmel_male.npz', 'filepath_annotations': 'dat/Dmel_male_annotations.csv'}}


# TODO: add asserts to ensure results conform to expected ds structure
def test_assemble2():
    datenames = list(recs.keys())
    ii = 2
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)
    dm = xb.assemble_metrics(ds)


if __name__ == '__main__':
    test_assemble2()
