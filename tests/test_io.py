import numpy as np
import xarray as xr
import pandas as pd
from xarray_behave import io


def test_call():
    io


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
