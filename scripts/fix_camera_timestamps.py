"""Back up an etho timestamps HDF5 file and map its camera clock to system time.

Usage: python scripts/fix_camera_timestamps.py /path/to/recording_timestamps.h5
Requires numpy and h5py. Close the file in other applications before running.
The fit corrects offset and linear drift, but includes image-retrieval latency.
"""

import argparse
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np


def fix_timestamps(path):
    path = Path(path).resolve()
    backup = path.with_name(path.name + ".bak")
    if backup.exists():
        raise FileExistsError(f"Backup already exists; refusing to overwrite it: {backup}")

    with h5py.File(path, "r") as file:
        stamps = file["timeStamps"][:]
    if stamps.ndim != 2 or stamps.shape[1] != 2 or stamps.dtype.kind != "f" or stamps.dtype.itemsize < 8:
        raise ValueError("Expected float64 timeStamps with columns [system_time, camera_time].")

    # Keep unused, zero-padded rows unchanged.
    used = np.any(stamps != 0, axis=1)
    count = int(used.sum())
    if count < 3 or not used[:count].all():
        raise ValueError("Need at least three timestamp pairs, with any empty rows at the end.")
    system, camera = stamps[:count].T
    if not np.isfinite(stamps[:count]).all() or np.any(stamps[:count] <= 0):
        raise ValueError("Timestamp pairs must be finite and positive.")
    if np.any(np.diff(camera) <= 0) or np.any(np.diff(system) < 0):
        raise ValueError("Camera times must increase strictly; system times must not go backwards.")

    # Fit relative times to avoid loss of precision with large Unix timestamps.
    elapsed = camera - camera[0]
    scale, intercept = np.polyfit(elapsed, system - system[0], 1)
    corrected = system[0] + intercept + scale * elapsed
    if not np.isfinite(corrected).all() or np.any(np.diff(corrected) <= 0):
        raise ValueError("The fitted camera timestamps are not finite and strictly increasing.")

    with path.open("rb") as source, backup.open("xb") as destination:
        shutil.copyfileobj(source, destination)
    shutil.copystat(path, backup)

    # Replace only after the corrected copy has been written and closed successfully.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        temporary = Path(file.name)
    try:
        shutil.copy2(backup, temporary)
        with h5py.File(temporary, "r+") as file:
            file["timeStamps"][:count, 1] = corrected
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Fixed {count} frames: {path}")
    print(f"Backup: {backup}")
    print(f"Camera correction at first frame: {corrected[0] - camera[0]:.6f} s")
    print(f"Clock rate correction: {(scale - 1) * 1e6:.3f} ppm")
    print(f"95th percentile |system - corrected camera|: {np.percentile(np.abs(system - corrected), 95) * 1000:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timestamps_file", type=Path)
    fix_timestamps(parser.parse_args().timestamps_file)
