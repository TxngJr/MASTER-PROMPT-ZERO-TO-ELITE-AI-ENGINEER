# Mini Project — Anomaly Detector Benchmark

Create mostly-normal 2D/5D data and inject rare anomalies.

Compare:
- robust z-score
- Mahalanobis
- IsolationForest
- LOF
- OneClassSVM

Protocol:
1. training = mostly normal
2. validation contains labeled synthetic anomalies
3. choose thresholds/hyperparameters on validation
4. final test once
5. report precision/recall/F1/PR-AUC
6. visualize score distributions
7. repeat anomaly types

Do not delete anomalies automatically; discuss whether they are errors, risks, or valuable rare events.
