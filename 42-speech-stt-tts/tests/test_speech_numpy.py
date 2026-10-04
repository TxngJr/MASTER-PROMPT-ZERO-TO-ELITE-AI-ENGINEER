from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "speech_numpy.py"
SPEC = importlib.util.spec_from_file_location("speech_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_frame_signal_shape() -> None:
    x = np.arange(10, dtype=float)
    frames = mod.frame_signal(
        x,
        frame_length=4,
        hop_length=2,
    )
    assert frames.shape == (4, 4)
    np.testing.assert_array_equal(frames[1], [2, 3, 4, 5])


def test_stft_detects_sine_frequency_region() -> None:
    sample_rate = 8000
    frequency = 1000.0
    t = np.arange(sample_rate) / sample_rate
    waveform = np.sin(2 * np.pi * frequency * t)

    spectrum = mod.stft(
        waveform,
        n_fft=256,
        hop_length=128,
    )
    mean_magnitude = np.abs(spectrum).mean(axis=1)
    peak_bin = int(np.argmax(mean_magnitude))
    peak_hz = peak_bin * sample_rate / 256

    assert abs(peak_hz - frequency) < 50.0


def test_mel_round_trip() -> None:
    hz = np.array([0.0, 100.0, 1000.0, 4000.0])
    recovered = mod.mel_to_hz(mod.hz_to_mel(hz))
    np.testing.assert_allclose(recovered, hz)


def test_log_mel_shape_and_finite() -> None:
    rng = np.random.default_rng(3)
    waveform = rng.normal(size=16000)

    features = mod.log_mel_spectrogram(
        waveform,
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=32,
    )

    assert features.shape[0] == 32
    assert np.all(np.isfinite(features))


def test_ctc_collapse_repeats_then_blank() -> None:
    path = [0, 1, 1, 0, 1, 2, 2, 0]
    assert mod.ctc_greedy_decode(path, blank_id=0) == [1, 1, 2]


def test_word_error_rate() -> None:
    wer = mod.word_error_rate(
        "the cat sat",
        "the dog sat",
    )
    np.testing.assert_allclose(wer, 1 / 3)
