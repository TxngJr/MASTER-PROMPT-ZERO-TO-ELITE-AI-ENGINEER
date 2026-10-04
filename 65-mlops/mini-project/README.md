# Mini Project — Model Promotion Pipeline

Build a local pipeline that:
1. records a training-run fingerprint
2. stores evaluation artifacts
3. registers candidate metadata
4. checks quality/latency gates
5. promotes candidate to champion only if all gates pass
6. monitors simulated production drift/error rate
7. rolls champion alias back after a forced regression

Optional: reproduce registry operations using MLflow.
