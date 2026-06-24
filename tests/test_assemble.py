import xarray_behave as xb
import logging
import numpy as np
import pandas as pd

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


def test_video_only_assembly_uses_pyav_reader(monkeypatch, tmp_path):
    from xarray_behave.gui import modern_video

    calls = []

    class FakeVideoReader:
        def __init__(self, filename):
            calls.append(str(filename))
            self.number_of_frames = 100
            self.frame_rate = 20.0

    monkeypatch.setattr(modern_video, "PyAVVideoReader", FakeVideoReader)

    datename = "video_only"
    data_dir = tmp_path / "dat" / datename
    data_dir.mkdir(parents=True)
    video_path = data_dir / f"{datename}.mp4"
    video_path.write_bytes(b"")

    ds = xb.assemble(
        datename,
        root=str(tmp_path),
        target_sampling_rate=0,
        include_song=False,
        include_tracks=False,
        include_poses=False,
        include_balltracker=False,
        include_movieparams=False,
    )

    assert calls == [str(video_path)]
    assert np.isclose(ds.attrs["target_sampling_rate_Hz"], 20.0)
    assert np.isclose(ds.attrs["sampling_rate_Hz"], 200.0)
    assert "nearest_frame" in ds.coords


def test_assemble_loads_custom_raven_annotation_column(tmp_path):
    annotation_path = tmp_path / "Dmel_male.Table.1.selections.txt"
    pd.DataFrame(
        {
            "Selection": [1],
            "Channel": [1],
            "Begin Time (s)": [0.1],
            "End Time (s)": [0.2],
            "Species": ["Dmel"],
        }
    ).to_csv(annotation_path, sep="\t", index=False)

    ds = xb.assemble(
        "Dmel_male",
        filepath_daq="tests/data/dat/Dmel_male.wav",
        filepath_annotations=str(annotation_path),
        annotation_column="Species",
        target_sampling_rate=100,
        include_tracks=False,
        include_poses=False,
        include_balltracker=False,
        include_movieparams=False,
    )

    assert ds.event_names.data.tolist() == ["Dmel"]
    np.testing.assert_allclose(ds.event_times.data, np.array([[0.1, 0.2, 0]]))


if __name__ == '__main__':
    test_assemble0()
    test_assemble1()
    test_assemble2()
    test_assemble3()
    test_assemble4()
    test_assemble5()
    test_assemble6()
    test_assemble7()
