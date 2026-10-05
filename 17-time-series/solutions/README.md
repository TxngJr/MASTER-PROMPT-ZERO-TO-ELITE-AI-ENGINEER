# Time Series — Full Solutions / Expected Checks

## Level 1 — Recall
1. **trend:** define its purpose, input, output, and the problem it addresses.
2. **seasonality:** give an operational definition and an explicit example.
3. **stationarity:** explain where it appears in the algorithm and what quantity it changes.
4. **autocorrelation:** state what it controls and one important condition that must remain valid.

## Level 2 — Understanding
5. Compare trend and seasonality by role, assumptions, and computational effect.
6. Trace autocorrelation from its setting to an intermediate computation and then to the final metric.
7. For ARIMA, name the limitation, the observable symptom, and an appropriate response.
8. For walk-forward validation, compare two settings and state what statistical or computational property changes.

## Level 3 — Coding
9. Keep the trend calculation visible, validate inputs, and use deterministic tiny data.
10. Include one hand-checkable normal case and one edge case.
11. Check all important intermediate values and the core property of stationarity.
12. Fix the seed, record data/configuration/metric, and state one limitation of the experiment.

## Level 4 — Analysis
13. Compare expected and actual shapes or data types stage by stage; the earliest mismatch identifies the correction.
14. Use a stable mathematical formulation and verify that outputs remain finite on extreme input values.
15. Fit learned preprocessing on training data only and keep validation/test data separate until evaluation.
16. Report measured runtime/memory, identify the dominant operation, and verify that an optimization preserves numerical output.

## Level 5 — Challenge
17. Trace input → trend → stationarity → autocorrelation → output and identify dominant time/memory complexity.
18. Match seed, data, preprocessing, parameters, and reduction conventions before comparing results.
19. Change only one factor, keep other conditions fixed, and report the measured difference across more than one run when randomness matters.
20. A complete answer includes input checks, quality/resource monitoring, versioned code/data/configuration, reproducibility metadata, and a falsifiable question involving ARIMA or walk-forward validation.
