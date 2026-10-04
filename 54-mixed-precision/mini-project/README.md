# Mini Project — Precision Stability Lab

Train the same tiny network under:
- FP32
- CPU BF16 autocast
- CUDA FP16 autocast + GradScaler when CUDA is available
- CUDA BF16 autocast when hardware supports it

Track:
- loss curve
- gradient norm
- non-finite events
- scale value
- peak memory
- step time
- final validation metric

Explain any numerical differences rather than assuming lower precision is incorrect.
