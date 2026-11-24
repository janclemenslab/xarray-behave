"""Prepare array-backed audio pages for a continuous Qt audio stream."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


def visible_audio_range(start, span, length, ratio):
    start = min(max(0, start), max(0, length - span))
    start = int(int(start / ratio) * ratio)
    stop = min(length, start + span)
    return start, int(int(stop / ratio) * ratio)


def audio_pcm(y, *, copy=False):
    pcm = y.astype(np.float32, order="C", copy=copy)
    peak = np.nanmax(np.abs(pcm)) if pcm.size else 0.0
    if np.isfinite(peak) and peak > 1.0:
        pcm *= 0.95 / peak
    return np.nan_to_num(pcm, copy=False)


@dataclass
class AudioPage:
    stop: int
    visible_range: tuple[int, int]
    visible_audio: np.ndarray
    pcm: bytes
    written: int = 0


def prepare_audio_page(start, stop, *, source, span, length, ratio, all_channels, auto_channel, channel):
    visible_start, visible_stop = visible_audio_range(start, span, length, ratio)
    read_start = min(start, visible_start)
    data = source[read_start : max(stop, visible_stop)]
    if hasattr(data, "compute"):
        data = data.compute()
    data = np.asarray(data)
    visible_audio = data[visible_start - read_start : visible_stop - read_start]
    audio = data[start - read_start : stop - read_start]
    if audio.ndim == 2 and not all_channels:
        if auto_channel and visible_audio.size:
            channel = np.argmax(np.max(visible_audio, axis=0))
        audio = audio[:, channel]
    pcm = audio_pcm(audio, copy=True).tobytes()
    return AudioPage(stop, (visible_start, visible_stop), visible_audio, pcm)


class ArrayAudioStream:
    """Keep the current page and one upcoming page ready for the audio device."""

    def __init__(self, sink, output, first_page, prepare, *, page_size, length):
        self._sink = sink
        self._output = output
        self._pages = deque([first_page])
        self._prepare = prepare
        self._page_size = page_size
        self._length = length
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xb-audio")
        self._pending = None

    def fill(self, sample):
        if self._pending is not None and self._pending.done():
            self._pages.append(self._pending.result())
            self._pending = None
        while len(self._pages) > 1 and self._pages[0].stop <= sample:
            self._pages.popleft()
        next_start = self._pages[-1].stop
        if len(self._pages) == 1 and self._pending is None and next_start < self._length:
            next_stop = min(self._length, next_start + self._page_size)
            self._pending = self._worker.submit(self._prepare, next_start, next_stop)

        space = self._sink.bytesFree()
        for page in self._pages:
            while space > 0 and page.written < len(page.pcm):
                chunk = page.pcm[page.written : page.written + space]
                written = self._output.write(chunk)
                if written < 0:
                    raise OSError("Qt audio output write failed")
                if written == 0:
                    return
                page.written += written
                space -= written

    def visible_audio(self, start, stop):
        for page in self._pages:
            if page.visible_range == (start, stop):
                return page.visible_audio
        return None

    def close(self):
        self._worker.shutdown(wait=False, cancel_futures=True)
