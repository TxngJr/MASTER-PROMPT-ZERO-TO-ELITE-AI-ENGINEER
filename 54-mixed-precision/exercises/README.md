# Chapter 54 Exercises — Mixed Precision

Complete all 20. Keep configs, commands, seeds, profiler/timing notes, and concise conclusions.

## Level 1 — Recall
1. Define **FP64/FP32/TF32**.
2. Define **FP16/BF16** and its primary resource trade-off.
3. Define **FP8 and low-precision training concepts** and one correctness risk.
4. Define **loss scaling and accumulation precision** and one operational concern.

## Level 2 — Understanding
5. Trace one batch/shard/kernel/model step end to end and identify ownership/dtype/device at each stage.
6. State assumptions behind FP16/BF16 and construct one violation.
7. Derive a central memory/communication/precision/global-batch formula and verify a numeric example.
8. Compare FP8 and low-precision training concepts and loss scaling and accumulation precision in memory, compute, communication/numerics, and operational complexity.

## Level 3 — Coding
9. Implement or simulate one central system/precision/data calculation using low-level Python/NumPy/PyTorch primitives.
10. Add validation for ranks/shapes/dtypes/ranges/shard boundaries/checkpoint metadata.
11. Build a deterministic tiny smoke test that can run on CPU and, when available, GPU.
12. Add invariant tests for sample ownership, byte/count calculations, numerical tolerance, or checkpoint round-trip.

## Level 4 — Debugging
13. Create a wrong global-batch/shard/dtype/cast-order bug that still runs; diagnose and regression-test it.
14. Create duplicated/missing samples, incomplete checkpoint state, or train/eval mismatch; demonstrate and fix.
15. Trigger OOM/overflow/underflow/precision drift or communication imbalance and add a principled mitigation.
16. Benchmark correctly with warmup/synchronization as needed; identify and optimize the actual bottleneck.

## Level 5 — Challenge
17. Compare two configurations with matched model/data/tokens across at least three runs; report quality plus resource metrics.
18. Ablate one sharding/precision/data/fine-tuning component and explain why the result changes.
19. Write an operational contract covering topology/device/dtype/data/checkpoint versions, restart, monitoring and rollback.
20. Write a research note with claim, matched baseline, protocol, hardware/resource accounting, results, failure analysis, limitation and next experiment.
