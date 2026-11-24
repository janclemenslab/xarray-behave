"""On-demand zero-phase filtering for the DAQ audio array."""

from functools import lru_cache
from uuid import uuid4

import dask.array as da
import numpy as np
from scipy import signal


def filtered_blocks(source, sos, sampling_rate, *, cache_bytes=256_000_000):
    """Expose filtered, five-second blocks as one sliceable (sample, channel) array."""
    samples, channels = source.shape
    dtype = np.result_type(source.dtype, sos.dtype)
    block_samples = max(1, round(5 * sampling_rate))
    block_bytes = block_samples * channels * dtype.itemsize

    # Let the slowest pole decay to 1e-12 before retaining the interior.
    # At the true recording ends, sosfiltfilt uses its usual endpoint padding.
    _, poles, _ = signal.sos2zpk(sos)
    radius = max(abs(poles))
    margin = samples if radius >= 1 else int(np.ceil(np.log(1e-12) / np.log(radius)))
    margin = max(margin, 3 * (2 * len(sos) + 1))

    @lru_cache(maxsize=cache_bytes // block_bytes)
    def read(start, stop):
        left = max(0, start - margin)
        right = min(samples, stop + margin)
        window = source[left:right, :]
        if isinstance(window, da.Array):
            # The outer Dask graph already schedules independent output blocks.
            window = window.compute(scheduler="synchronous")
        filtered = signal.sosfiltfilt(sos, window, axis=0)
        # Own just the retained samples, so the cache budget excludes discarded margins.
        return filtered[start - left : stop - left].copy()

    def block(block_info):
        start, stop = block_info[None]["array-location"][0]
        return read(start, stop)

    chunks = da.core.normalize_chunks((block_samples, -1), shape=source.shape, dtype=dtype)
    return da.map_blocks(
        block, chunks=chunks, dtype=dtype, meta=np.empty((0, 0), dtype=dtype),
        name=f"filtered-daq-{uuid4().hex}",
    )
