from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "distributed_cuda_precision_lab.py"
SPEC = importlib.util.spec_from_file_location("batch18_lab_common", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_distributed_plan_is_consistent() -> None:
    report = lab.run_distributed_plan()

    assert report["unique_sample_ids"] == report["sample_count"]
    assert sum(report["shard_sizes"]) == report["sample_count"]
    np.testing.assert_allclose(
        report["all_reduce_mean"],
        [2.5, 5.0],
    )
    assert report["effective_global_batch"] == 128

    memory = report["zero_style_state_bytes"]
    assert (
        memory["stage_0"]
        > memory["stage_1"]
        > memory["stage_2"]
        > memory["stage_3"]
    )
    assert 0.0 < report["pipeline_efficiency"] <= 1.0
