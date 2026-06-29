import numpy as np
import pandas as pd
import pytest
from xarray_behave import io, loaders


def test_call():
    io


def test_fix_keys_renames_legacy_keys_without_mutating_iteration():
    fixed = loaders.fix_keys({"aggression_manu": [1], "vibration_manua": [2]})

    assert fixed == {"aggression_manual": [1], "vibration_manual": [2]}


def test_audiofile_lazy_loads_multichannel_wav_slices(tmp_path):
    import dask.array as daskarray
    import soundfile as sf

    filename = tmp_path / "multi.wav"
    data = np.arange(30, dtype=np.float32).reshape(10, 3)
    sf.write(filename, data, samplerate=10_000, subtype="FLOAT")

    song, non_song, sampling_rate = io.audio.AudioFile(str(filename)).load(str(filename), lazy=True)

    assert sampling_rate == 10_000
    assert non_song is None
    assert isinstance(song, daskarray.Array)
    assert song.shape == data.shape
    np.testing.assert_allclose(song[2:5, :].compute(), data[2:5, :])


def test_raven_pro_loader_discovers_selection_table_and_deduplicates_views(tmp_path):
    filename = tmp_path / "recording.Table.1.selections.txt"
    pd.DataFrame(
        {
            "Selection": [1, 1, 2, 3],
            "View": ["Spectrogram 1", "Waveform 1", "Spectrogram 1", "Spectrogram 1"],
            "Channel": [1, 1, 2, 2],
            "Begin Time (s)": [0.1, 0.1, 0.3, 0.3],
            "End Time (s)": [0.2, 0.2, 0.3, 0.3],
            "Annotation": [" call ", "call", "pulse", "pulse"],
        }
    ).to_csv(filename, sep="\t", index=False)

    loader = io.get_loader(kind="annotations_manual", basename=str(tmp_path / "recording"))
    events, categories = loader.load()

    assert isinstance(loader, io.annotations_manual.RavenPro)
    np.testing.assert_allclose(events["call"], np.array([[0.1, 0.2, 0]]))
    np.testing.assert_allclose(events["pulse"], np.array([[0.3, 0.3, 1], [0.3, 0.3, 1]]))
    assert categories == {"call": "event", "pulse": "event"}


def test_raven_pro_loader_supports_custom_annotation_column_without_channels(tmp_path):
    filename = tmp_path / "recording_raven.txt"
    pd.DataFrame(
        {
            "Begin Time (s)": [1.0],
            "End Time (s)": [1.5],
            "Species": ["Dmel"],
        }
    ).to_csv(filename, sep="\t", index=False)

    loader = io.get_loader(kind="annotations_manual", basename=str(tmp_path / "recording"))
    events, _ = loader.load(annotation_column="Species")

    assert isinstance(loader, io.annotations_manual.RavenPro)
    np.testing.assert_allclose(events["Dmel"], np.array([[1.0, 1.5, -1]]))


def test_raven_pro_loader_supports_empty_tables(tmp_path):
    filename = tmp_path / "empty_raven.txt"
    pd.DataFrame(columns=["Begin Time (s)", "End Time (s)", "Annotation"]).to_csv(filename, sep="\t", index=False)

    events, categories = io.annotations_manual.RavenPro(str(filename)).load()

    assert not events
    assert categories == {}


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ({"Begin Time (s)": [0.1], "End Time (s)": [0.2]}, "missing required columns"),
        ({"Begin Time (s)": [0.1], "End Time (s)": [0.2], "Annotation": [" "]}, "blank 'Annotation'"),
        ({"Begin Time (s)": ["bad"], "End Time (s)": [0.2], "Annotation": ["call"]}, "finite numeric"),
        (
            {"Begin Time (s)": [0.2], "End Time (s)": [0.1], "Annotation": ["call"]},
            "0 <= begin <= end",
        ),
        (
            {"Begin Time (s)": [0.1], "End Time (s)": [0.2], "Annotation": ["call"], "Channel": [0]},
            "positive integers",
        ),
        (
            {"Selection": ["bad"], "Begin Time (s)": [0.1], "End Time (s)": [0.2], "Annotation": ["call"]},
            "'Selection' must contain finite numeric values",
        ),
    ],
)
def test_raven_pro_loader_rejects_malformed_tables(tmp_path, data, error):
    filename = tmp_path / "bad_raven.txt"
    pd.DataFrame(data).to_csv(filename, sep="\t", index=False)

    with pytest.raises(ValueError, match=error):
        io.annotations_manual.RavenPro(str(filename)).load()
