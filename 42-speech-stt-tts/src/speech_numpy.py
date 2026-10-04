"""Educational speech/audio DSP and CTC utilities implemented with NumPy."""

from __future__ import annotations

import math
import numpy as np


def frame_signal(
    waveform: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    x = np.asarray(waveform, dtype=float).reshape(-1)

    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame/hop lengths must be positive")
    if len(x) < frame_length:
        raise ValueError("waveform shorter than one frame")

    frame_count = 1 + (len(x) - frame_length) // hop_length
    return np.stack(
        [
            x[start : start + frame_length]
            for start in range(
                0,
                frame_count * hop_length,
                hop_length,
            )
        ],
        axis=0,
    )


def stft(
    waveform: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int | None = None,
) -> np.ndarray:
    x = np.asarray(waveform, dtype=float).reshape(-1)

    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if win_length is None:
        win_length = n_fft
    if win_length <= 0 or win_length > n_fft:
        raise ValueError("win_length must lie in (0,n_fft]")

    frames = frame_signal(x, win_length, hop_length)
    window = np.hanning(win_length)
    windowed = frames * window[None, :]

    if win_length < n_fft:
        pad = n_fft - win_length
        windowed = np.pad(
            windowed,
            ((0, 0), (0, pad)),
        )

    # Return frequency x time, mirroring common STFT layouts.
    return np.fft.rfft(windowed, n=n_fft, axis=1).T


def hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray:
    f = np.asarray(frequency_hz, dtype=float)
    if np.any(f < 0):
        raise ValueError("frequency cannot be negative")
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    m = np.asarray(mel, dtype=float)
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    if sample_rate <= 0 or n_fft <= 0 or n_mels <= 0:
        raise ValueError("dimensions must be positive")

    nyquist = sample_rate / 2.0
    if f_max is None:
        f_max = nyquist

    if not 0.0 <= f_min < f_max <= nyquist:
        raise ValueError("invalid frequency range")

    mel_points = np.linspace(
        hz_to_mel(f_min),
        hz_to_mel(f_max),
        n_mels + 2,
    )
    hz_points = mel_to_hz(mel_points)

    fft_freqs = np.linspace(
        0.0,
        nyquist,
        n_fft // 2 + 1,
    )

    bank = np.zeros(
        (n_mels, len(fft_freqs)),
        dtype=float,
    )

    for index in range(n_mels):
        left, center, right = hz_points[index : index + 3]

        left_mask = (fft_freqs >= left) & (fft_freqs <= center)
        right_mask = (fft_freqs >= center) & (fft_freqs <= right)

        if center > left:
            bank[index, left_mask] = (
                fft_freqs[left_mask] - left
            ) / (center - left)

        if right > center:
            bank[index, right_mask] = (
                right - fft_freqs[right_mask]
            ) / (right - center)

    return bank


def log_mel_spectrogram(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int | None = None,
    n_mels: int = 40,
    eps: float = 1e-10,
) -> np.ndarray:
    spectrum = stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    power = np.abs(spectrum) ** 2
    bank = mel_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
    )
    mel_power = bank @ power
    return np.log(np.maximum(mel_power, eps))


def ctc_greedy_decode(
    class_ids: list[int] | np.ndarray,
    *,
    blank_id: int = 0,
) -> list[int]:
    path = np.asarray(class_ids, dtype=np.int64).reshape(-1)

    output: list[int] = []
    previous: int | None = None

    for token in path:
        token_id = int(token)

        # Merge adjacent repeats first, then remove blanks.
        if token_id != previous and token_id != blank_id:
            output.append(token_id)

        previous = token_id

    return output


def _levenshtein(
    reference: list[str],
    hypothesis: list[str],
) -> int:
    previous = list(range(len(hypothesis) + 1))

    for i, ref_item in enumerate(reference, start=1):
        current = [i]

        for j, hyp_item in enumerate(hypothesis, start=1):
            substitution = (
                previous[j - 1]
                + (0 if ref_item == hyp_item else 1)
            )
            deletion = previous[j] + 1
            insertion = current[j - 1] + 1
            current.append(
                min(substitution, deletion, insertion)
            )

        previous = current

    return previous[-1]


def word_error_rate(
    reference: str,
    hypothesis: str,
) -> float:
    ref = reference.split()
    hyp = hypothesis.split()

    if not ref:
        raise ValueError("reference must contain at least one word")

    return float(_levenshtein(ref, hyp) / len(ref))
