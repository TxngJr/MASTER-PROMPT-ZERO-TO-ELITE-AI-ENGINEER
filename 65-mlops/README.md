# Chapter 65 — MLOps: Tracking, Registry, Monitoring & Lifecycle

## 1. What MLOps Solves

MLOps makes experiments, models, data and deployments reproducible and governable.

~~~text
data + code + config
↓
training run
↓
metrics + artifacts
↓
model registry
↓
validation / promotion
↓
deployment
↓
monitoring
↓
retrain / rollback
~~~

## 2. Learning Objectives

- track parameters, metrics and artifacts
- fingerprint experiments
- version model/data/code together
- understand model registries
- design aliases/tags and promotion gates
- distinguish offline and online metrics
- detect data and prediction drift
- monitor data quality
- design retraining triggers
- understand CI/CD/CT
- implement rollback
- preserve lineage
- avoid training-serving skew

## 3. Experiment Tracking

Every run should record at minimum:
- code revision
- dataset version
- hyperparameters
- seed
- environment/dependencies
- metrics
- artifacts
- start/end timestamps

An experiment dashboard is useful only if runs are comparable.

## 4. Run Identity

A reproducible run can be fingerprinted from canonicalized inputs:

~~~text
hash(
  code_revision
  dataset_revision
  config
  environment
)
~~~

Do not rely only on a human-written run name.

## 5. Artifacts

Artifacts can include:
- model weights
- tokenizer
- plots
- evaluation JSON
- preprocessing objects
- manifests

Store content digests so corrupted or replaced artifacts can be detected.

## 6. MLflow Tracking

Current MLflow Tracking records run parameters, code versions, metrics and output artifacts and exposes API/UI workflows for inspecting runs.

Use MLflow as one concrete implementation of the broader tracking concepts.

## 7. Model Registry

A registry provides a controlled model namespace with versions and metadata.

Current MLflow Model Registry supports model versions, aliases, tags, descriptions and lineage back to runs.

Examples of aliases:
- candidate
- champion
- rollback

Alias names should express operational intent rather than being treated as immutable history.

## 8. Version vs Alias

Version:
~~~text
immutable historical identity
~~~

Alias:
~~~text
movable pointer to a version
~~~

Deploy by immutable version/digest underneath even when humans use aliases operationally.

## 9. Promotion Gate

A candidate should pass explicit requirements:
- validation quality
- slice quality
- latency/memory
- robustness
- security/safety checks where relevant
- reproducibility

Promotion is a policy decision, not simply highest validation score.

## 10. CI / CD / CT

### CI
test code, data contracts and training components.

### CD
package/promote/deploy a validated artifact.

### CT
continuous training or retraining when policy says new training is needed.

Continuous training does not mean retrain on every new row.

## 11. Offline vs Online Metrics

Offline:
- validation loss
- F1
- ranking metrics
- benchmark quality

Online:
- request errors
- latency
- business/task outcomes
- human feedback

A model can improve offline while hurting production outcomes.

## 12. Data Drift

Input distribution changes:

~~~text
P_train(X) != P_prod(X)
~~~

Drift is a warning signal, not proof that model quality has degraded.

## 13. Concept Drift

The relationship between inputs and targets changes:

~~~text
P_train(Y|X) != P_prod(Y|X)
~~~

This can directly invalidate learned decision rules.

## 14. Prediction Drift

Monitor changes in model output distributions when labels are delayed/unavailable.

Again, output drift does not automatically mean failure.

## 15. Population Stability Index

For reference/current bin proportions p_i and q_i:

~~~text
PSI = sum((q_i - p_i) * ln(q_i/p_i))
~~~

Use smoothing for zero bins.

Thresholds are domain-dependent; do not treat one internet rule as universal.

## 16. Data Quality Monitoring

Track:
- missing rate
- invalid categories
- schema violations
- duplicate rate
- out-of-range values
- freshness
- volume

Data failures can look like model drift.

## 17. Training-Serving Skew

Occurs when feature/preprocessing logic differs between training and production.

Mitigations:
- shared transformation code
- versioned features
- contract tests
- replay production samples through training pipeline

## 18. Delayed Labels

Many production labels arrive hours/days/weeks later.

Separate:
- immediate proxy monitoring
- delayed ground-truth evaluation

## 19. Retraining Trigger

Possible triggers:
- scheduled
- enough new labeled data
- quality degradation
- drift + validated impact
- domain/business change

Retrain only when policy and evaluation justify it.

## 20. Champion / Challenger

Champion is currently serving.

Challenger is a candidate evaluated against champion.

Use paired/consistent datasets and production shadow/canary evidence where possible.

## 21. Rollback

Rollback requires:
- previous immutable model version
- previous preprocessing/config
- compatible runtime
- deployment manifest

Keeping only model weights is not enough.

## 22. Lineage

Lineage connects:

~~~text
source data
↓
dataset snapshot
↓
training run
↓
model version
↓
deployment
↓
predictions / metrics
~~~

This enables debugging and auditability.

## 23. Reproducibility

Exact numerical determinism is not always possible on all accelerators.

Still record enough metadata to reproduce the process and explain expected nondeterminism.

## 24. Monitoring Windows

Use windows appropriate to traffic volume:
- fixed time
- rolling count
- event-time windows

Very small windows create noisy alerts.

## 25. Alert Design

Alerts should be:
- actionable
- rate limited
- severity classified
- linked to runbooks

Alerting on every small metric movement creates fatigue.

## 26. From Scratch

src/mlops_utils.py implements:

- canonical_json
- experiment_fingerprint
- artifact_sha256
- population_stability_index
- missing_rate
- promotion_gate
- registry_alias_update
- rolling_error_rate

## 27. Optional MLflow Lab

Install:

~~~bash
python -m pip install -r requirements-batch22-mlops.txt
~~~

Then reproduce run tracking and model registration with MLflow.

## 28. Common Mistakes

1. run names used as unique identity
2. data revision not recorded
3. model registered without evaluation artifact
4. alias confused with immutable version
5. drift threshold copied without validation
6. missing-data outage labeled concept drift
7. retraining automatically triggered by any drift
8. training-serving preprocessing differs
9. no rollback artifact bundle
10. monitoring has no actionable runbook

## 29. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 30. Checklist

- [ ] tracking
- [ ] artifact hashes
- [ ] model registry
- [ ] aliases/tags
- [ ] promotion gate
- [ ] CI/CD/CT
- [ ] data/prediction drift
- [ ] quality monitoring
- [ ] training-serving skew
- [ ] lineage
- [ ] rollback

## 31. What's Next

Chapter 66 builds the data pipelines and contracts that feed reproducible training and production systems.