# Mini Project — Quantization Quality Lab

Quantize the same toy model weights using:
- symmetric INT8
- symmetric INT4
- per-channel INT8
- grouped INT4

Report:
- raw weight bytes
- metadata estimate
- MSE
- output error
- downstream metric
- wall-clock inference where meaningful

Then explain why the lowest tensor MSE may not produce the best application metric.
