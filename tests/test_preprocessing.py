"""Engineering checks use synthetic waveforms only in temporary directories."""
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
from python_speech_features import delta, mfcc

from src.preprocessing import (DEFAULT_CONFIG, extract_mfcc39, load_audio,
                               load_feature_file, preprocess_audio, process_audio_file)


@pytest.fixture
def tone():
    return np.sin(2 * np.pi * 440 * np.arange(8000) / 16000).astype(np.float32) * 0.3


def test_load_audio(tmp_path, tone):
    path = tmp_path / "audio.wav"
    sf.write(path, tone, 16000, subtype="FLOAT")
    audio, rate = load_audio(path)
    assert rate == 16000 and audio.ndim == 1 and audio.size == 8000
    assert audio.dtype == np.float32
    np.testing.assert_array_equal(audio, tone)


def test_stereo(tmp_path, tone):
    path = tmp_path / "stereo.wav"
    sf.write(path, np.column_stack([tone, tone * 0.5]), 16000, subtype="FLOAT")
    audio, _ = load_audio(path)
    np.testing.assert_allclose(audio, tone * 0.75, atol=1e-7)


def test_resampling():
    rate = 44100
    audio = 0.3 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
    processed, output_rate = preprocess_audio(audio, rate)
    assert output_rate == 16000 and len(processed) == 16000
    frequencies = np.fft.rfftfreq(len(processed), 1 / output_rate)
    assert frequencies[np.argmax(np.abs(np.fft.rfft(processed))) ] == pytest.approx(440, abs=1)


def test_mfcc_dimensions_finite_and_deterministic(tone):
    features = extract_mfcc39(tone, 16000)
    assert features.ndim == 2 and features.shape == (49, 39)
    assert features.dtype == np.float32 and np.isfinite(features).all()
    np.testing.assert_array_equal(features, extract_mfcc39(tone, 16000))


def test_coefficient_order_and_single_preemphasis(tone):
    static = mfcc(tone, samplerate=16000, numcep=13, winlen=0.025,
                  winstep=0.01, nfilt=26, nfft=512, preemph=0.97,
                  winfunc=np.hamming, appendEnergy=True)
    first = delta(static, 2)
    actual = extract_mfcc39(tone, 16000)
    for block, expected in [(actual[:, :13], static), (actual[:, 13:26], first),
                            (actual[:, 26:], delta(first, 2))]:
        np.testing.assert_allclose(block, expected, rtol=1e-6, atol=1e-6)


def test_persistence(tmp_path, tone):
    features = extract_mfcc39(tone, 16000)
    path = tmp_path / "features.npy"
    np.save(path, features)
    np.testing.assert_array_equal(features, load_feature_file(path))


def test_variable_length_and_process_file(tmp_path, tone):
    paths = [tmp_path / "short.wav", tmp_path / "long.wav"]
    sf.write(paths[0], tone[:1600], 16000)
    sf.write(paths[1], tone, 16000)
    short, long = map(process_audio_file, paths)
    assert short.shape[1] == long.shape[1] == 39
    assert len(short) < len(long)


def test_short_recording():
    features = extract_mfcc39(np.array([0.1], dtype=np.float32), 16000)
    assert features.shape == (1, 39) and np.isfinite(features).all()
    np.testing.assert_allclose(features[:, 13:], 0, atol=1e-6)


@pytest.mark.parametrize("audio", [np.array([]), np.zeros(100),
                                  np.array([np.nan]), np.array([np.inf])])
def test_invalid_waveform(audio):
    with pytest.raises(ValueError):
        extract_mfcc39(audio, 16000)


def test_corrupted_and_empty_wav(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not a wav")
    with pytest.raises(ValueError, match="decode"):
        load_audio(path)
    sf.write(path, np.array([], dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="nonempty"):
        load_audio(path)


def test_nonfinite_wav(tmp_path):
    path = tmp_path / "nan.wav"
    sf.write(path, np.array([0.1, np.nan]), 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="NaN"):
        load_audio(path)


def test_optional_preprocessing(tone):
    audio = np.concatenate([np.zeros(100), tone, np.zeros(100)])
    processed, _ = preprocess_audio(audio, 16000,
                                    replace(DEFAULT_CONFIG, trim_silence=True, normalize=True))
    assert len(processed) < len(audio)
    assert np.max(np.abs(processed)) == pytest.approx(1)


def test_invalid_config_and_feature_file(tmp_path):
    with pytest.raises(ValueError):
        replace(DEFAULT_CONFIG, nfft=128)
    with pytest.raises(ValueError):
        replace(DEFAULT_CONFIG, frame_step=float("nan"))
    with pytest.raises(ValueError):
        replace(DEFAULT_CONFIG, delta_window=2.5)
    path = tmp_path / "invalid.npy"
    np.save(path, np.zeros((3, 13)))
    with pytest.raises(ValueError, match="39"):
        load_feature_file(path)
