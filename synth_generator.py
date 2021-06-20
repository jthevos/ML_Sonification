# Cell 1
from abc import ABC, abstractmethod
from scipy.io import wavfile
import math, pyaudio
import numpy as np

class Oscillator(ABC):
    """
    The property ._freq represents the fundamental frequency
    of the oscillator, this doesn’t change, and the property ._f
    represents the altered frequency which is the frequency of the
    returned wave, obtained on calling __next__.

    The idea is that when a key is pressed __iter__ is called once,
    and the __next__ is called as long as the key is held.
    """

    def __init__(self, freq=440, phase=0, amp=1, \
                 sample_rate=44_100, wave_range=(-1, 1)):
        self._freq = freq
        self._amp = amp
        self._phase = phase
        self._sample_rate = sample_rate
        self._wave_range = wave_range

        # Properties that will be changed
        self._f = freq
        self._a = amp
        self._p = phase

    @property
    def init_freq(self):
        return self._fre

    @property
    def init_amp(self):
        return self._am

    @property
    def init_phase(self):
        return self._phas

    @property
    def freq(self):
        return self._

    @freq.setter
    def freq(self, value):
        self._f = value
        self._post_freq_set()

    @property
    def amp(self):
        return self._

    @amp.setter
    def amp(self, value):
        self._a = value
        self._post_amp_set()

    @property
    def phase(self):
        return self._

    @phase.setter
    def phase(self, value):
        self._p = value
        self._post_phase_set()

    def _post_freq_set(self):
        pass

    def _post_amp_set(self):
        pass

    def _post_phase_set(self):
        pass

    @abstractmethod
    def _initialize_osc(self):
        pass

    @staticmethod
    def squish_val(val, min_val=0, max_val=1):
        """
        This ensures output is in the correct range.
        """
        return ((((val + 1) / 2 ) * (max_val - min_val)) + min_val).astype(np.float32)

    @abstractmethod
    def __next__(self):
        return None

    def __iter__(self):
        self.freq = self._freq
        self.phase = self._phase
        self.amp = self._amp
        self._initialize_osc()
        return self


class SineOscillator(Oscillator):
    """
    Now, we extend our Astract Base Class and add in all the stuff we
    had in our original sine wave generator.
    """

    def _post_freq_set(self):
        self._step = (2 * math.pi * self._f) / self._sample_rate

    def _post_phase_set(self):
        self._p = (self._p / 360) * 2 * math.pi

    def _initialize_osc(self):
        self._i = 0

    def __next__(self):
        val = math.sin(self._i + self._p)
        self._i = self._i + self._step
        if self._wave_range is not (-1, 1):
            val = self.squish_val(val, *self._wave_range)
        return val * self._a


class SquareOscillator(SineOscillator):
    """
    This extends sine because we're modifying the sine wave as the base wave
    """
    def __init__(self, freq=440, phase=0, amp=1, \
                 sample_rate=44_100, wave_range=(-1, 1), threshold=0):
        super().__init__(freq, phase, amp, sample_rate, wave_range)
        self.threshold = threshold

    def __next__(self):
        val = math.sin(self._i + self._p)
        self._i = self._i + self._step
        if val < self.threshold:
            val = self._wave_range[0]
        else:
            val = self._wave_range[1]
        return val * self._a

class SawtoothOscillator(Oscillator):
    """
    The sawtooth is not derived from the sine wave, so we extend the ABC
    """
    def _post_freq_set(self):
        self._period = self._sample_rate / self._f
        self._post_phase_set

    def _post_phase_set(self):
        self._p = ((self._p + 90)/ 360) * self._period

    def _initialize_osc(self):
        self._i = 0

    def __next__(self):
        div = (self._i + self._p )/self._period
        val = 2 * (div - math.floor(0.5 + div))
        self._i = self._i + 1
        if self._wave_range is not (-1, 1):
            val = self.squish_val(val, *self._wave_range)
        return val * self._a

class TriangleOscillator(SawtoothOscillator):
    """
    This is essentially taking the absolute value of the sawtooth.
    """
    def __next__(self):
        div = (self._i + self._p)/self._period
        val = 2 * (div - math.floor(0.5 + div))
        val = (abs(val) - 0.5) * 2
        self._i = self._i + 1
        if self._wave_range is not (-1, 1):
            val = self.squish_val(val, *self._wave_range)
        return val * self._a

class WaveAdder:
    """
    This is a utility class that allows additive synthesis.
    """
    def __init__(self, *oscillators):
        self.oscillators = oscillators
        self.n = len(oscillators)

    def __iter__(self):
        [iter(osc) for osc in self.oscillators]
        return self

    def __next__(self):
        return sum(next(osc) for osc in self.oscillators) / self.n

def wave_to_file(wav, wav2=None, fname="temp.wav", amp=0.1, sample_rate=44100):
    wav = np.array(wav)
    wav = np.int16(wav * amp * (2**15 - 1))

    if wav2 is not None:
        wav2 = np.array(wav2)
        wav2 = np.int16(wav2 * amp * (2 ** 15 - 1))
        wav = np.stack([wav, wav2]).T

    wavfile.write(fname, sample_rate, wav)

def make_stream_compatible(list):
    """
    We have created our generator in an agnostic way. I.e., we are not assuming
    what we want to with the lists of numbers it generates. But, if we want
    to live stream the sounds, we need to convert to np.float32 and represent
    the numbers as a byte stream. This function does just this.
    """
    return np.array(list).astype(np.float32).tobytes()

some_generator = WaveAdder(
    SineOscillator(freq=440.0, phase=0.2, amp=0.5),
    TriangleOscillator(freq=220.0, phase=-0.2, amp=0.7),
    SawtoothOscillator(freq=220.0*1.5, phase=0, amp=0.9),
    SquareOscillator(freq=110.0, phase=0, amp=1.0)
)

# Cell 2
iter(some_generator)
wav = [next(some_generator) for _ in range(44100 * 4)] # 4 Seconds


sample = make_stream_compatible(wav)

fs = 44100       # sampling rate, Hz, must be integer

volume = 0.5     # range [0.0, 1.0]
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

# play. May repeat with different volume values (if done interactively)
stream.write(sample)

stream.stop_stream()
stream.close()

p.terminate()

#wave_to_file(wav, fname="prelude_two.wav")
