from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "serving_metrics.py"
SPEC = importlib.util.spec_from_file_location("serving_metrics_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_request_latency_components() -> None:
    assert mod.request_latency(
        queue_seconds=0.1,
        prefill_seconds=0.2,
        decode_seconds=0.5,
        network_seconds=0.05,
    ) == 0.85


def test_percentile_tail_is_not_average() -> None:
    values = [1.0, 1.0, 1.0, 10.0]
    p99 = mod.percentile(values, 99)
    assert p99 > np.mean(values)


def test_little_law() -> None:
    concurrency = mod.little_law_concurrency(
        arrival_rate_per_second=10.0,
        average_latency_seconds=0.5,
    )
    assert concurrency == 5.0


def test_required_replicas_with_headroom() -> None:
    replicas = mod.required_replicas(
        offered_requests_per_second=70,
        sustainable_requests_per_second_per_replica=50,
        target_utilization=0.7,
    )
    assert replicas == 2


def test_success_rate() -> None:
    assert mod.success_rate(
        successes=99,
        total_requests=100,
    ) == 0.99


def test_canary_split() -> None:
    stable, canary = mod.canary_split(
        1000,
        canary_fraction=0.05,
    )
    assert stable == 950
    assert canary == 50
