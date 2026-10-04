"""Batch 25 integration: reasoning, VLM, and real-time voice planning."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _load_module(relative_path: str, name: str):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_reasoning() -> dict[str, Any]:
    reasoning = _load_module(
        "73-reasoning-models/src/reasoning.py",
        "batch25_reasoning",
    )

    answers = ["42", "42", "41", "42", "43"]
    verifier_scores = [0.82, 0.94, 0.20, 0.91, 0.15]

    majority, count = reasoning.majority_vote(answers)
    best, best_score, best_index = reasoning.best_of_n(
        answers,
        verifier_scores,
    )
    allocations = reasoning.allocate_sample_budget(
        total_token_budget=2400,
        samples=5,
        reserve_tokens=400,
    )

    return {
        "mode": "reasoning",
        "majority_answer": majority,
        "majority_count": count,
        "self_consistency": reasoning.self_consistency_confidence(
            answers
        ),
        "best_of_n_answer": best,
        "best_of_n_score": best_score,
        "best_of_n_index": best_index,
        "pass_at_3": reasoning.pass_at_k(
            total_samples=5,
            correct_samples=3,
            k=3,
        ),
        "sample_token_allocations": allocations,
        "search_efficiency": reasoning.search_efficiency(
            solved=8,
            attempted=10,
            generated_tokens=4000,
        ),
    }


def run_vlm() -> dict[str, Any]:
    vlm = _load_module(
        "74-vision-language-models/src/vlm.py",
        "batch25_vlm",
    )

    image_tokens = [
        vlm.image_token_count(
            224,
            224,
            patch_height=14,
            extra_tokens=1,
        ),
        vlm.image_token_count(
            336,
            336,
            patch_height=14,
            extra_tokens=1,
        ),
    ]

    remaining = vlm.multimodal_context_remaining(
        max_context_tokens=8192,
        text_tokens=1800,
        image_tokens=image_tokens,
        reserved_output_tokens=1024,
    )

    predicted = [
        (0.1, 0.1, 0.4, 0.4),
        (0.6, 0.6, 0.8, 0.8),
    ]
    targets = [
        (0.1, 0.1, 0.4, 0.4),
    ]
    grounding = vlm.grounding_precision_recall(
        predicted,
        targets,
        iou_threshold=0.5,
    )

    return {
        "mode": "vlm",
        "image_tokens": image_tokens,
        "remaining_context_tokens": remaining,
        "grounding": grounding,
        "first_box_iou": vlm.bbox_iou(
            predicted[0],
            targets[0],
        ),
    }


def run_voice() -> dict[str, Any]:
    voice = _load_module(
        "75-audio-voice-models/src/voice.py",
        "batch25_voice",
    )

    energies = np.array(
        [0.01, 0.15, 0.22, 0.18, 0.02, 0.01, 0.01]
    )
    flags = voice.vad_flags(
        energies,
        threshold=0.1,
    )
    endpoint = voice.endpoint_after_silence(
        flags,
        required_silent_frames=2,
    )

    return {
        "mode": "voice",
        "vad_flags": flags.tolist(),
        "endpoint_frame": endpoint,
        "codec_bitrate_kbps": voice.codec_bitrate_kbps(
            codebooks=8,
            codebook_size=1024,
            frames_per_second=50,
        ),
        "rtf": voice.real_time_factor(
            processing_seconds=0.75,
            audio_seconds=3.0,
        ),
        "first_response_ms": voice.streaming_first_response_latency(
            capture_chunk_ms=20,
            endpoint_or_partial_ms=120,
            model_first_token_ms=180,
            tts_first_audio_ms=90,
            network_ms=30,
        ),
    }


def run_full() -> dict[str, Any]:
    reasoning = run_reasoning()
    vlm = run_vlm()
    voice = run_voice()

    reasoning_ok = (
        reasoning["majority_answer"] == "42"
        and reasoning["best_of_n_answer"] == "42"
        and sum(reasoning["sample_token_allocations"]) == 2000
    )
    vlm_ok = (
        vlm["remaining_context_tokens"] > 0
        and vlm["first_box_iou"] == 1.0
        and vlm["grounding"]["recall"] == 1.0
    )
    voice_ok = (
        voice["endpoint_frame"] is not None
        and voice["rtf"] < 1.0
        and voice["first_response_ms"] < 1000
    )

    return {
        "mode": "full",
        "system_ready": reasoning_ok and vlm_ok and voice_ok,
        "reasoning": reasoning,
        "vlm": vlm,
        "voice": voice,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 25 reasoning/VLM/voice lab."
    )
    parser.add_argument(
        "--mode",
        choices=["reasoning", "vlm", "voice", "full"],
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch25"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "reasoning":
        report = run_reasoning()
    elif args.mode == "vlm":
        report = run_vlm()
    elif args.mode == "voice":
        report = run_voice()
    else:
        report = run_full()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.mode}.json"
    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"mode: {report['mode']}")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
