import numpy as np
import logging
import scipy.signal

logger = logging.getLogger(__name__)


class AudioDevice:
    MAX_AUDIO_AMP = 3.0

    def __init__(self):
        self._player = None

    def test(self):
        test_sound = np.zeros((1000,))
        try:
            if self.play(test_sound, fs=44100) > 0:
                return True
        except:
            return False

    @property
    def is_working(self):
        return self._player is not None

    def play(self, y, fs):
        pass


class SoundDevice(AudioDevice):
    NAME = "sounddevice"

    def __init__(self):
        super().__init__()
        try:
            import sounddevice

            self._player = sounddevice
            if not self.test():
                self._player = None
        except (ImportError, ModuleNotFoundError):
            logger.info(
                "Could not import python-sounddevice. Maybe you need to install it. See https://python-sounddevice.readthedocs.io/en/latest/installation.html for instructions."
            )

    def play(self, y, fs):
        if self._player is None:
            return 0
        # scale sound so we do not blow out the speakers
        try:
            y = y.astype(float) / np.iinfo(y.dtype).max * self.MAX_AUDIO_AMP
        except ValueError:
            y = y / (np.nanmax(y) + 1e-10) / 10 * self.MAX_AUDIO_AMP
        self._player.play(y, fs)
        return 1


class SimpleAudio(AudioDevice):
    ALLOWED_FS = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000]  # Hz
    NAME = "simpleaudio"

    def __init__(self):
        super().__init__()
        try:
            import simpleaudio

            self._player = simpleaudio
            if not self.test():
                self._player = None
        except (ImportError, ModuleNotFoundError):
            logger.info(
                "Could not import simpleaudio. Maybe you need to install it. See https://simpleaudio.readthedocs.io/en/latest/installation.html for instructions."
            )

    def play(self, y, fs):
        if not self.is_working:
            return 0
        # convert to 16bit int
        y = y / (np.nanmax(y) + 1e-10)
        y = y * 32767 / self.MAX_AUDIO_AMP
        y = y.astype(np.int16)

        # simpleaudio can only play at these rates - choose the one nearest to our rate
        fs_nearest = min(self.ALLOWED_FS, key=lambda x: abs(x - int(fs)))
        y = scipy.signal.resample_poly(y, fs_nearest, fs)

        # start playback in background
        self._player.play_buffer(y, num_channels=1, bytes_per_sample=2, sample_rate=fs_nearest)
        return 1


class AudioPlayer:
    DEVICES = [SoundDevice, SimpleAudio]

    def __init__(self):
        self.player = None

        for device in self.DEVICES:
            logging.debug(device.NAME)
            self.player = device()
            if self.player.is_working:
                self.play = self.player.play
                break
            else:
                self.player = None

    def __str__(self):
        return str(self.player)
