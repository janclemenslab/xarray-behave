"""Small WAV playback debugger.

Examples:
    python -m xarray_behave.gui.debug_play_wav scratch/dat/Dmel_male.wav --backend qmedia
    python -m xarray_behave.gui.debug_play_wav scratch/dat/Dmel_male.wav --backend sounddevice
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

import xarray_behave  # noqa: F401 - sets QT_API before qtpy imports


def _print_audio_stats(data: np.ndarray, samplerate: int) -> None:
    data = np.asarray(data)
    print(f"samplerate: {samplerate}")
    print(f"shape: {data.shape}")
    print(f"dtype: {data.dtype}")
    print(f"min/max: {np.nanmin(data):.6g} / {np.nanmax(data):.6g}")
    print(f"peak abs: {np.nanmax(np.abs(data)):.6g}")


def _play_sounddevice(path: Path, seconds: float | None, start_seconds: float, normalize: bool) -> int:
    import sounddevice as sd
    import soundfile as sf

    print("backend: sounddevice")
    print(f"default device: {sd.default.device}")
    try:
        print(f"default output: {sd.query_devices(kind='output')}")
    except Exception as exc:
        print(f"default output query failed: {type(exc).__name__}: {exc}")

    info = sf.info(path)
    start = max(0, int(round(start_seconds * info.samplerate)))
    stop = -1
    if seconds is not None:
        stop = min(info.frames, start + max(1, int(round(seconds * info.samplerate))))

    data, samplerate = sf.read(path, start=start, stop=stop, always_2d=False)
    _print_audio_stats(data, samplerate)

    if normalize:
        peak = float(np.nanmax(np.abs(data)))
        if np.isfinite(peak) and peak > 0:
            data = data / peak * 0.4
        print("normalized for sounddevice playback: yes")
    else:
        print("normalized for sounddevice playback: no")

    print("starting sounddevice playback")
    sd.play(data, samplerate)
    sd.wait()
    print("sounddevice playback finished")
    return 0


def _play_qmedia(path: Path, seconds: float | None, start_seconds: float, volume: float) -> int:
    from PySide6 import QtCore, QtWidgets
    from PySide6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer

    print("backend: QMediaPlayer")
    print(f"QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM', '<unset>')}")
    print(f"Qt version: {QtCore.qVersion()}")

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    try:
        devices = QMediaDevices.audioOutputs()
        default = QMediaDevices.defaultAudioOutput()
        default_name = default.description() if not default.isNull() else "NULL"
        print(f"QMedia audio outputs: {len(devices)}")
        print(f"QMedia default output: {default_name}")
        for index, device in enumerate(devices):
            print(f"  [{index}] {device.description()}")
    except Exception as exc:
        print(f"QMedia device query failed: {type(exc).__name__}: {exc}")

    audio_output = QAudioOutput()
    audio_output.setMuted(False)
    audio_output.setVolume(float(volume))
    player = QMediaPlayer()
    player.setAudioOutput(audio_output)

    last_position_bucket = {"value": -1}

    def on_status(status) -> None:
        print(f"mediaStatusChanged: {status}")

    def on_state(state) -> None:
        print(f"playbackStateChanged: {state}")

    def on_duration(duration_ms: int) -> None:
        print(f"durationChanged: {duration_ms} ms")

    def on_position(position_ms: int) -> None:
        bucket = int(position_ms // 250)
        if bucket != last_position_bucket["value"]:
            last_position_bucket["value"] = bucket
            print(f"positionChanged: {position_ms} ms")

    def on_error(error, error_string: str = "") -> None:
        text = error_string or player.errorString()
        print(f"errorOccurred: {error} {text}")

    player.mediaStatusChanged.connect(on_status)
    player.playbackStateChanged.connect(on_state)
    player.durationChanged.connect(on_duration)
    player.positionChanged.connect(on_position)
    try:
        player.errorOccurred.connect(on_error)
    except TypeError:
        player.errorOccurred.connect(lambda error: on_error(error))

    url = QtCore.QUrl.fromLocalFile(str(path))
    print(f"source: {url.toString()}")
    player.setSource(url)

    start_ms = max(0, int(round(start_seconds * 1000)))

    def start_playback() -> None:
        print(f"pre-play status: {player.mediaStatus()}")
        print(f"pre-play error: {player.error()} {player.errorString()}")
        print(f"setting position: {start_ms} ms")
        player.setPosition(start_ms)
        print(f"volume: {audio_output.volume()}")
        print(f"muted: {audio_output.isMuted()}")
        print("calling play()")
        player.play()

    QtCore.QTimer.singleShot(500, start_playback)
    if seconds is not None:
        QtCore.QTimer.singleShot(max(250, int(round(seconds * 1000)) + 1000), app.quit)

    if seconds is None:
        player.playbackStateChanged.connect(
            lambda state: app.quit() if state == QMediaPlayer.PlaybackState.StoppedState and player.position() > 0 else None
        )

    app.exec()

    print(f"final state: {player.playbackState()}")
    print(f"final status: {player.mediaStatus()}")
    print(f"final position: {player.position()} ms")
    print(f"final duration: {player.duration()} ms")
    print(f"final error: {player.error()} {player.errorString()}")
    return 0 if player.error() == QMediaPlayer.Error.NoError else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Debug WAV playback with QMediaPlayer or sounddevice.")
    parser.add_argument("path", nargs="?", default="scratch/dat/Dmel_male.wav", help="WAV/audio file to play.")
    parser.add_argument("--backend", choices=("qmedia", "sounddevice"), default="qmedia")
    parser.add_argument("--seconds", type=float, default=5.0, help="Playback duration. Use <=0 for full file.")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--volume", type=float, default=1.0, help="QMediaPlayer volume from 0.0 to 1.0.")
    parser.add_argument("--raw", action="store_true", help="Do not normalize sounddevice playback.")
    args = parser.parse_args(argv)

    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        print(f"file does not exist: {path}", file=sys.stderr)
        return 2

    seconds = None if args.seconds <= 0 else float(args.seconds)
    print(f"path: {path}")

    if args.backend == "sounddevice":
        return _play_sounddevice(path, seconds, args.start, normalize=not args.raw)
    return _play_qmedia(path, seconds, args.start, args.volume)


if __name__ == "__main__":
    raise SystemExit(main())
