from pathlib import Path
import os
import runpy

import h5py
import numpy as np
import pytest


fix_timestamps = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "fix_camera_timestamps.py"))["fix_timestamps"]


@pytest.mark.parametrize("camera_start", [60_000.0, 1_788_340_000.0])
def test_fix_timestamps_preserves_backup_and_other_data(tmp_path, camera_start):
    path = tmp_path / "timestamps.h5"
    elapsed = np.arange(2000) / 80
    expected = 1_788_340_000 + 0.003 + elapsed * 1.00005
    jitter = np.random.default_rng(0).uniform(-0.0002, 0.0002, len(elapsed))
    stamps = np.vstack([np.column_stack([expected + jitter, camera_start + elapsed]), np.zeros((3, 2))])
    with h5py.File(path, "w") as file:
        file.attrs["experiment"] = "keep"
        file.create_dataset("timeStamps", data=stamps, compression="gzip").attrs["units"] = "seconds"
        file["other"] = [1, 2, 3]
    original = path.read_bytes()

    fix_timestamps(path)

    assert path.with_name(path.name + ".bak").read_bytes() == original
    with h5py.File(path, "r") as file:
        np.testing.assert_allclose(file["timeStamps"][:-3, 1], expected, atol=0.00005, rtol=0)
        np.testing.assert_array_equal(file["timeStamps"][:, 0], stamps[:, 0])
        np.testing.assert_array_equal(file["timeStamps"][-3:], 0)
        np.testing.assert_array_equal(file["other"][:], [1, 2, 3])
        assert file.attrs["experiment"] == "keep"
        assert file["timeStamps"].attrs["units"] == "seconds"
        assert file["timeStamps"].compression == "gzip"
    fixed = path.read_bytes()
    with pytest.raises(FileExistsError):
        fix_timestamps(path)
    assert path.read_bytes() == fixed


def test_failed_replacement_leaves_original_intact(tmp_path, monkeypatch):
    path = tmp_path / "timestamps.h5"
    with h5py.File(path, "w") as file:
        file["timeStamps"] = np.array([[1000.0, 1], [1001, 2], [1002, 3]])
    original = path.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        fix_timestamps(path)
    assert path.read_bytes() == original
    assert path.with_name(path.name + ".bak").read_bytes() == original
    assert not list(tmp_path.glob(".timestamps.h5.*"))
