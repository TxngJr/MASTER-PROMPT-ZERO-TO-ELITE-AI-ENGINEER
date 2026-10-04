# Mini Project — Inference Capacity Planner

Given a model and server workload, estimate:
- weight memory
- KV memory/request
- max theoretical concurrent sequences
- paged block waste
- prompt-prefix reuse
- padding efficiency
- memory-bandwidth lower bound

Then benchmark or simulate:
- static batches
- continuous batching
- prefix caching
- speculative decoding

Report TTFT, TPOT and throughput separately.
