# Completion Pack — Chapter 22: PyTorch

This pack completes the standardized teaching contract around the main README and source code.

## Prerequisites
Complete the earlier neural-network/math chapters, run all tests, and record Python/framework versions, seed, dataset identity, and hardware.

## Mental Model
~~~text
input tensor → representation → forward computation → objective → gradient/update → diagnostics → evaluation
~~~
Core concepts: **tensor semantics, autograd, nn.Module and optimizers, DataLoader/checkpoint/eval workflows**.

## Core Theory Map
1. **tensor semantics** — know the mathematical definition, tensor shapes, implementation path, numerical issues, and one failure mode.
2. **autograd** — know the mathematical definition, tensor shapes, implementation path, numerical issues, and one failure mode.
3. **nn.Module and optimizers** — know the mathematical definition, tensor shapes, implementation path, numerical issues, and one failure mode.
4. **DataLoader/checkpoint/eval workflows** — know the mathematical definition, tensor shapes, implementation path, numerical issues, and one failure mode.

## Mathematics and Invariants
Derive the chapter's central transforms/objectives. Annotate shapes and reduction axes. Verify gradients with a tiny finite-difference or trusted-reference check where feasible. Useful invariants: finite outputs, expected range/probability sum, shape preservation, deterministic eval under fixed state, and decreasing toy loss when optimization is correctly wired.

## Code Walkthrough
Trace data batch → tensor creation → forward graph → loss → backward/gradient → optimizer/update → eval. Identify stateful components, train/eval differences, device movement, and serialization boundaries.

## Visualization Lab
Visualize inputs/features/state, learning curves or gradient statistics, and one failure case such as saturation, exploding gradient, overfitting, or bad padding/masking.

## Experiment Design
Run baseline, one-factor ablation, and stress test. Record parameter count, batch size, seed, dataset size, dtype/device, metric, and memory behavior.

## Failure Cases and Debugging
Check tensor shapes and axes first, then dtype/device, train/eval state, loss reduction, gradient flow, optimizer state, numerical stability, and data leakage.

## Performance Perspective
Profile compute vs data loading vs memory transfers. Avoid assuming GPU is faster for tiny workloads. Measure warm and steady-state behavior separately.

## Hardware-Aware Guidance
~~~yaml
Expected hardware:
  CPU: Ryzen 7 5825U-class or similar
  RAM: 16 GB recommended
  GPU: optional for tiny examples; NVIDIA laptop GPU useful for framework/CNN/RNN labs
  VRAM: inspect with nvidia-smi; do not hard-code a capacity
  Dataset size: start with small synthetic/sklearn/torchvision-style subsets
  Batch size: start 8–32; reduce for OOM or large sequence/image shapes
~~~

## Production Perspective
Freeze model code/config, preprocessing, dtype/device assumptions, checkpoint format, class/token mapping, and eval mode behavior. Add input limits, monitoring, rollback, and safe serialization.

## Research Perspective
Use matched parameter/data/compute budgets, multiple seeds, ablations, uncertainty, and failure analyses. Report hardware/dtype because numerical paths can affect results.

## Common Mistakes
- wrong reduction axis;
- mixing logits and probabilities;
- missing train/eval transition;
- stale gradients;
- detached tensors;
- silent dtype/device mismatch;
- comparing runs with different data or compute budgets.

## Interview Questions
1. Explain tensor semantics mathematically.
2. What numerical issue appears in autograd?
3. How is nn.Module and optimizers represented in code/state?
4. Compare nn.Module and optimizers and DataLoader/checkpoint/eval workflows.
5. Which tensor invariant would you assert?
6. How would you gradient-check a tiny case?
7. What dominates memory?
8. Give a train/eval bug.
9. Design an ablation.
10. What belongs in a safe checkpoint/deployment contract?

## Summary
Mastery requires correct math, shape reasoning, implementation, gradient/state debugging, profiling, controlled experimentation, and reproducible deployment.
