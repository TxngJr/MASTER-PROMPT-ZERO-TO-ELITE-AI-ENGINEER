from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "voice.py"
SPEC = importlib.util.spec_from_file_location("voice_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_sample_count() -> None:
    assert mod.sample_count(
        1.5,
        sample_rate=16000,
    ) == 24000


def test_frame_ranges_include_partial_tail() -> None:
    ranges = mod.frame_ranges(
        10,
        frame_size=4,
        hop_size=3,
    )
    assert ranges == [
        (0, 4),
        (3, 7),
        (6, 10),
    ]


def test_rms_energy() -> None:
    value = mod.rms_energy(
        np.array([1.0, -1.0, 1.0, -1.0])
    )
    np.testing.assert_allclose(value, 1.0)


def test_vad_and_endpoint() -> None:
    energies = np.array([0.0, 0.2, 0.3, 0.0, 0.0, 0.0])
    flags = mod.vad_flags(
        energies,
        threshold=0.1,
    )
    assert flags.tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
    ]

    endpoint = mod.endpoint_after_silence(
        flags,
        required_silent_frames=2,
    )
    assert endpoint == 4


def test_real_time_factor() -> None:
    assert mod.real_time_factor(
        processing_seconds=0.5,
        audio_seconds=2.0,
    ) == 0.25


def test_codec_bitrate() -> None:
    bitrate = mod.codec_bitrate_kbps(
        codebooks=8,
        codebook_size=1024,
        frames_per_second=50,
    )
    assert bitrate == 4.0


def test_streaming_latency() -> None:
    latency = mod.streaming_first_response_latency(
        capture_chunk_ms=20,
        endpoint_or_partial_ms=100,
        model_first_token_ms=150,
        tts_first_audio_ms=80,
        network_ms=20,
    )
    assert latency == 370.0
