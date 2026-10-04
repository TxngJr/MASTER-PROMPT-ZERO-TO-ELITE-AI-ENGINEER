from pathlib import Path
import importlib.util
import sys


MODULE_PATH = Path(__file__).parents[1] / "src" / "distributed_systems.py"
SPEC = importlib.util.spec_from_file_location("distributed_systems_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_backoff_caps() -> None:
    assert mod.exponential_backoff(
        0,
        base_seconds=1.0,
        cap_seconds=5.0,
    ) == 1.0
    assert mod.exponential_backoff(
        10,
        base_seconds=1.0,
        cap_seconds=5.0,
    ) == 5.0


def test_queue_utilization() -> None:
    assert mod.queue_utilization(
        arrival_rate=8,
        service_rate_per_worker=5,
        workers=2,
    ) == 0.8


def test_bounded_admission() -> None:
    admitted, rejected = mod.bounded_admission(
        queue_size=8,
        queue_capacity=10,
        incoming=5,
    )
    assert admitted == 2
    assert rejected == 3


def test_rendezvous_is_deterministic_and_unique() -> None:
    first = mod.rendezvous_nodes(
        "user-1",
        ["a", "b", "c"],
        replicas=2,
    )
    second = mod.rendezvous_nodes(
        "user-1",
        ["a", "b", "c"],
        replicas=2,
    )
    assert first == second
    assert len(first) == 2
    assert len(set(first)) == 2


def test_quorum_overlap() -> None:
    assert mod.quorum_overlap(
        replicas=3,
        read_quorum=2,
        write_quorum=2,
    )
    assert not mod.quorum_overlap(
        replicas=3,
        read_quorum=1,
        write_quorum=2,
    )


def test_straggler_flags() -> None:
    flags = mod.straggler_flags(
        [1.0, 1.1, 0.9, 3.0],
        median_multiplier=2.0,
    )
    assert flags == [False, False, False, True]


def test_circuit_breaker_opens_and_recovers() -> None:
    state = mod.CircuitBreakerState()
    state = mod.circuit_breaker_record(
        state,
        success=False,
        failure_threshold=2,
    )
    assert state.state == "closed"

    state = mod.circuit_breaker_record(
        state,
        success=False,
        failure_threshold=2,
    )
    assert state.state == "open"

    state = mod.circuit_breaker_record(
        state,
        success=True,
        failure_threshold=2,
        half_open_probe=True,
    )
    assert state.state == "closed"
