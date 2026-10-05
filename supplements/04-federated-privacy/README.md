# Supplement 04 — Federated and Privacy-Preserving ML

Federated learning moves model updates toward data owners rather than centralizing all raw records. Differential privacy (DP) limits how much an output can depend on one individual's record under a formal randomized definition.

## Learning objectives

- explain cross-device/cross-silo federated learning;
- implement weighted FedAvg;
- understand client sampling and non-IID data;
- distinguish secure aggregation concepts from differential privacy;
- understand per-client/per-example clipping concepts;
- understand Gaussian-noise mechanisms conceptually;
- reason about privacy budget parameters epsilon/delta;
- understand privacy accounting as composition over repeated steps;
- evaluate utility, fairness, communication and privacy together.

## Federated averaging

For client updates/weights (w_k) with example counts (n_k):

~~~text
w_global = sum_k (n_k / sum_j n_j) * w_k
~~~

Weight by the intended data contribution; a plain unweighted mean is a different algorithm when client dataset sizes differ.

## Non-IID data

Clients may differ by geography, device, language, behavior or label distribution. Local training can drift toward client-specific optima. Measure per-client and aggregate performance, not only one global average.

## Secure aggregation concept

Secure aggregation is a cryptographic systems concept that lets a coordinator learn an aggregate of client updates without directly observing each individual update. It protects a different threat surface than DP and does not by itself provide a DP guarantee.

## Differential privacy concept

A randomized mechanism M is approximately differentially private when neighboring datasets produce similar output distributions, controlled by `epsilon` and `delta`.

In DP-SGD-style training the high-level pattern is:

1. compute bounded contributions;
2. clip contribution norm to C;
3. aggregate;
4. add calibrated random noise;
5. account for privacy loss across steps.

This supplement teaches the mechanics and vocabulary, not a production privacy accountant. Use a vetted DP library/accountant for real claims.

## Clipping

For update vector (g):

~~~text
g_clipped = g * min(1, C / max(||g||_2, tiny))
~~~

Clipping bounds sensitivity but can bias large updates.

## Evaluation dimensions

- global utility;
- worst/client-group utility;
- convergence;
- communication bytes/round;
- client participation/dropout;
- clipping fraction;
- noise scale;
- documented privacy-accounting result when using a vetted accountant.

## Common mistakes

- calling federated learning "private" automatically;
- confusing encryption/secure aggregation with DP;
- claiming epsilon without a correct accountant;
- ignoring non-IID client distributions;
- averaging clients equally when the intended objective is example-weighted;
- logging raw sensitive client updates while claiming privacy.
