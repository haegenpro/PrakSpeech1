"""Shared, deterministic audio preprocessing and MFCC extraction."""
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from python_speech_features import delta, mfcc
from scipy.signal import resample_poly


@dataclass(frozen=True)
class PreprocessingConfig:
    """Implementation choices shared by every dataset partition."""
    target_sample_rate: int = 16000
    frame_length: float = 0.025
    frame_step: float = 0.01
    nfilt: int = 26
    nfft: int = 512
    preemphasis: float = 0.97
    delta_window: int = 2
    trim_silence: bool = False
    trim_threshold: float = 0.01
    normalize: bool = False

    def __post_init__(self) -> None:
        numeric = (self.target_sample_rate, self.frame_length, self.frame_step,
                   self.nfilt, self.nfft, self.preemphasis, self.delta_window,
                   self.trim_threshold)
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("Configuration values must be finite")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in
               (self.target_sample_rate, self.nfilt, self.nfft, self.delta_window)):
            raise ValueError("Rate, filter count, FFT size and delta window must be integers")
        if (self.target_sample_rate <= 0 or self.frame_length <= 0
                or self.frame_step <= 0 or self.nfilt < 13
                or self.frame_length * self.target_sample_rate < 1
                or self.frame_step * self.target_sample_rate < 1
                or self.delta_window < 1
                or self.nfft < round(self.target_sample_rate * self.frame_length)
                or not 0 <= self.preemphasis < 1
                or not 0 < self.trim_threshold < 1):
            raise ValueError("Invalid preprocessing configuration")


DEFAULT_CONFIG = PreprocessingConfig()


def _validate_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio must be a nonempty mono waveform")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinity")
    if not np.any(audio):
        raise ValueError("Audio is silent (all samples are zero)")
    return audio


def load_audio(file_path: str | Path) -> tuple[np.ndarray, int]:
    """Decode WAV, average channels, and return float32 mono at its original rate."""
    path = Path(file_path)
    try:
        info = sf.info(path)
        if info.format not in {"WAV", "WAVEX", "RF64"}:
            raise ValueError("Only WAV audio is supported")
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, OSError) as exc:
        raise ValueError(f"Cannot decode WAV {path}: {exc}") from exc
    if not np.isfinite(audio).all():
        raise ValueError(f"Audio contains NaN or infinity: {path}")
    return _validate_audio(audio.mean(axis=1)), int(rate)


def preprocess_audio(audio: np.ndarray, sample_rate: int,
                     config: PreprocessingConfig = DEFAULT_CONFIG) -> tuple[np.ndarray, int]:
    """Resample, optionally trim low-amplitude edges and normalize peak amplitude."""
    audio = _validate_audio(audio)
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if sample_rate != config.target_sample_rate:
        divisor = gcd(sample_rate, config.target_sample_rate)
        audio = resample_poly(audio, config.target_sample_rate // divisor,
                              sample_rate // divisor).astype(np.float32)
    if config.trim_silence:
        active = np.flatnonzero(np.abs(audio) >= config.trim_threshold * np.max(np.abs(audio)))
        audio = audio[active[0]:active[-1] + 1]
    if config.normalize:
        audio = audio / np.max(np.abs(audio))
    return _validate_audio(audio), config.target_sample_rate


def validate_features(features: np.ndarray) -> np.ndarray:
    """Require a finite, nonempty frame sequence with exactly 39 columns."""
    if (features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != 39
            or not np.issubdtype(features.dtype, np.number)
            or np.iscomplexobj(features) or not np.isfinite(features).all()):
        raise ValueError("Features must be a finite numeric array shaped (T, 39), T > 0")
    return features


def extract_mfcc39(audio: np.ndarray, sample_rate: int,
                   config: PreprocessingConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Extract 13 static + 13 delta + 13 delta-delta features in temporal order.

    Input must already be preprocessed at the configured sample rate. Short
    inputs are zero-padded to one frame; delta() replicates boundary frames.
    """
    audio = _validate_audio(audio)
    if sample_rate != config.target_sample_rate:
        raise ValueError("Call preprocess_audio before MFCC extraction at the target rate")
    frame_samples = int(np.floor(config.frame_length * sample_rate + 0.5))
    if len(audio) < frame_samples:
        audio = np.pad(audio, (0, frame_samples - len(audio)))
    static = mfcc(audio, samplerate=sample_rate, numcep=13,
                  winlen=config.frame_length, winstep=config.frame_step,
                  nfilt=config.nfilt, nfft=config.nfft, preemph=config.preemphasis,
                  winfunc=np.hamming, appendEnergy=True)
    first = delta(static, config.delta_window)
    second = delta(first, config.delta_window)
    return validate_features(np.concatenate([static, first, second], axis=1).astype(np.float32))


def process_audio_file(file_path: str | Path,
                       config: PreprocessingConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Load and preprocess one WAV, returning a variable-length (T, 39) array."""
    audio, rate = load_audio(file_path)
    audio, rate = preprocess_audio(audio, rate, config)
    return extract_mfcc39(audio, rate, config)


def load_feature_file(file_path: str | Path) -> np.ndarray:
    """Load a validated NumPy sequence without allowing pickled objects."""
    return validate_features(np.load(file_path, allow_pickle=False))
