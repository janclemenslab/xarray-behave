from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


class PyAVVideoReader:
    """Small PyAV-backed reader exposing the attributes used by MovieView."""

    def __init__(self, filename: str | Path):
        try:
            import av  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency checked at runtime
            raise RuntimeError("PyAV is required for video display") from exc

        self.filename = str(Path(filename).expanduser().resolve())
        self._av = av
        self._container = av.open(self.filename)
        self._stream = next((stream for stream in self._container.streams if stream.type == "video"), None)
        if self._stream is None:
            self.close()
            raise RuntimeError(f"No video stream in {self.filename}")

        self.frame_rate = self._rate_hz(self._stream)
        self.frame_width = int(getattr(self._stream, "width", 0) or 0)
        self.frame_height = int(getattr(self._stream, "height", 0) or 0)
        self.frame_shape = (self.frame_height, self.frame_width, 3)
        self.number_of_frames = int(getattr(self._stream, "frames", 0) or 0)
        if self.number_of_frames <= 0:
            duration = self._duration_seconds()
            self.number_of_frames = int(np.ceil(max(0.0, duration) * self.frame_rate))

    def __getitem__(self, index: int):
        return self.read(index)[1]

    @property
    def dtype(self):
        return np.uint8

    @property
    def shape(self):
        return (self.number_of_frames, *self.frame_shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def read(self, index: int):
        frame_index = max(0, min(int(index), max(0, self.number_of_frames - 1)))
        timestamp_seconds = frame_index / self.frame_rate if self.frame_rate > 0 else 0.0
        frame = self.frame_at_seconds(timestamp_seconds)
        return frame_index, frame

    def frame_at_seconds(self, seconds: float):
        target_seconds = max(0.0, float(seconds))
        target_pts = self._seconds_to_pts(target_seconds)
        try:
            self._container.seek(target_pts, stream=self._stream, backward=True, any_frame=False)
        except Exception:
            self._container.seek(int(target_seconds * 1_000_000), backward=True)
        try:
            self._stream.codec_context.flush_buffers()
        except Exception:
            pass

        best_frame = None
        best_distance = None
        for frame in self._container.decode(self._stream):
            frame_seconds = self._frame_seconds(frame)
            distance = abs(frame_seconds - target_seconds)
            if best_distance is None or distance <= best_distance:
                best_distance = distance
                best_frame = frame
            if frame_seconds >= target_seconds and best_frame is not None:
                break

        if best_frame is None:
            return None
        return np.ascontiguousarray(best_frame.to_ndarray(format="rgb24"))

    def close(self):
        container = getattr(self, "_container", None)
        self._container = None
        if container is not None:
            try:
                container.close()
            except Exception:
                pass

    def _rate_hz(self, stream: Any) -> float:
        for attr in ("average_rate", "base_rate", "guessed_rate"):
            value = getattr(stream, attr, None)
            if value:
                return float(value)
        return 30.0

    def _duration_seconds(self) -> float:
        if getattr(self._stream, "duration", None) is not None and getattr(self._stream, "time_base", None) is not None:
            return float(Fraction(self._stream.duration) * self._stream.time_base)
        if getattr(self._container, "duration", None):
            return float(self._container.duration) / 1_000_000.0
        return 0.0

    def _seconds_to_pts(self, seconds: float) -> int:
        time_base = getattr(self._stream, "time_base", None)
        if time_base is None:
            return int(seconds * 1_000_000)
        return int(Fraction(seconds).limit_denominator(1_000_000) / time_base)

    def _frame_seconds(self, frame: Any) -> float:
        if getattr(frame, "pts", None) is not None and getattr(frame, "time_base", None) is not None:
            return float(Fraction(frame.pts) * frame.time_base)
        if getattr(frame, "time", None) is not None:
            return float(frame.time)
        return 0.0

    def __del__(self):  # pragma: no cover - destructor behavior is runtime dependent
        self.close()
