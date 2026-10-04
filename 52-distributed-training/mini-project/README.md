# Mini Project — Distributed Training Planner

Given:
- model parameter count
- parameter/gradient/optimizer dtypes
- activation memory estimate
- GPU count
- link bandwidth
- local batch
- sequence length

Build a planner that estimates:
- replicated DDP memory
- ZeRO/FSDP idealized state memory
- global tokens/update
- ring all-reduce payload/time
- pipeline bubble efficiency

Then recommend a candidate parallelism layout and clearly state assumptions.
