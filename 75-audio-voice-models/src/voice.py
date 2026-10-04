"""Streaming audio, VAD, codec, and latency utilities."""

from __future__ import annotations

import math
import numpy as np


def sample_count(
    duration_seconds: float,
    *,
    sample_rate: int,
) -> int:
    if duration_seconds < 0:
        raise ValueError("duration_seconds must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    return int(round(duration_seconds * sample_rate))


def frame_ranges(
    num_samples: int,
    *,
    frame_size: int,
    hop_size: int,
) -> list[tuple[int, int]]:
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size/hop_size must be positive")

    if num_samples == 0:
        return []

    ranges = []
    start = 0

    while start < num_samples:
        end = min(start + frame_size, num_samples)
        ranges.append((start, end))

        if end == num_samples:
            break

        start += hop_size

    return ranges


def rms_energy(samples: np.ndarray) -> float:
    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x**2)))


def frame_rms(
    samples: np.ndarray,
    *,
    frame_size: int,
    hop_size: int,
) -> np.ndarray:
    x = np.asarray(samples, dtype=float).reshape(-1)
    ranges = frame_ranges(
        len(x),
        frame_size=frame_size,
        hop_size=hop_size,
    )

    return np.asarray(
        [
            rms_energy(x[start:end])
            for start, end in ranges
        ],
        dtype=float,
    )


def vad_flags(
    energies: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    values = np.asarray(energies, dtype=float)
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if np.any(values < 0):
        raise ValueError("energies must be non-negative")

    return values >= threshold


def endpoint_after_silence(
    speech_flags: np.ndarray,
    *,
    required_silent_frames: int,
) -> int | None:
    flags = np.asarray(speech_flags, dtype=bool).reshape(-1)

    if required_silent_frames <= 0:
        raise ValueError("required_silent_frames must be positive")

    silent_run = 0

    for index, is_speech in enumerate(flags):
        if is_speech:
            silent_run = 0
        else:
            silent_run += 1
            if silent_run >= required_silent_frames:
                return int(index)

    return None


def real_time_factor(
    *,
    processing_seconds: float,
    audio_seconds: float,
) -> float:
    if processing_seconds < 0:
        raise ValueError("processing_seconds must be non-negative")
    if audio_seconds <= 0:
        raise ValueError("audio_seconds must be positive")

    return float(processing_seconds / audio_seconds)


def codec_bitrate_kbps(
    *,
    codebooks: int,
    codebook_size: int,
    frames_per_second: float,
) -> float:
    if codebooks <= 0:
        raise ValueError("codebooks must be positive")
    if codebook_size <= 1:
        raise ValueError("codebook_size must exceed 1")
    if frames_per_second <= 0:
        raise ValueError("frames_per_second must be positive")

    bits_per_code = math.ceil(math.log2(codebook_size))
    bits_per_second = (
        codebooks
        * bits_per_code
        * frames_per_second
    )

    return float(bits_per_second / 1000.0)


def audio_tokens_per_second(
    *,
    total_tokens: int,
    duration_seconds: float,
) -> float:
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    return float(total_tokens / duration_seconds)


def streaming_first_response_latency(
    *,
    capture_chunk_ms: float,
    endpoint_or_partial_ms: float,
    model_first_token_ms: float,
    tts_first_audio_ms: float,
    network_ms: float = 0.0,
) -> float:
    values = [
        capture_chunk_ms,
        endpoint_or_partial_ms,
        model_first_token_ms,
        tts_first_audio_ms,
        network_ms,
    ]
    if any(value < 0 for value in values):
        raise ValueError("latencies must be non-negative")

    return float(sum(values))


def buffered_audio_ms(
    *,
    chunks: int,
    samples_per_chunk: int,
    sample_rate: int,
) -> float:
    if chunks < 0 or samples_per_chunk < 0:
        raise ValueError("counts must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    samples = chunks * samples_per_chunk
    return float(samples / sample_rate * 1000.0)
