from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).parents[1] / "src" / "deployment_math.py"
SPEC = importlib.util.spec_from_file_location("deployment_math_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_hpa_style_replica_math() -> None:
    assert mod.desired_replicas_from_metric(
        current_replicas=4,
        current_metric=80,
        target_metric=50,
        min_replicas=2,
        max_replicas=10,
    ) == 7


def test_rollout_bounds() -> None:
    report = mod.rolling_update_bounds(
        desired_replicas=4,
        max_surge=1,
        max_unavailable=1,
    )
    assert report["max_total_pods"] == 5
    assert report["min_available_pods"] == 3


def test_canary_counts() -> None:
    stable, canary = mod.canary_request_counts(
        1000,
        canary_fraction=0.05,
    )
    assert stable == 950
    assert canary == 50


def test_capacity_headroom() -> None:
    assert mod.capacity_with_headroom(
        replicas=3,
        sustainable_rps_per_replica=10,
        target_utilization=0.7,
    ) == 21.0


def test_gpu_pool_capacity() -> None:
    assert mod.gpu_pool_capacity(
        nodes=3,
        gpus_per_node=4,
        gpus_per_replica=1,
        reserved_gpus=2,
    ) == 10
