# Chapter 16 Solutions — Anomaly Detection

Equivalent implementations are acceptable if they satisfy the same mathematical and engineering invariants.

## Level 1
### 1
Define **statistical outliers** precisely, state the problem it addresses, and identify its observable effect.
### 2
Define **Isolation Forest**, its inputs/outputs or state, and one small example.
### 3
Explain **one-class methods** and connect its main control to optimization, capacity, stability, or compute.
### 4
Explain **threshold calibration** and give a limitation caused by assumptions, finite data, approximation, or resources.

## Level 2
### 5
Trace the mechanism step by step from input/state through **statistical outliers** to the output; name a diagnostic that verifies each critical transition.
### 6
List the assumptions behind **Isolation Forest**, build a violating example, and predict the failure before running it.
### 7
Write and derive the central formula/update/recurrence, define symbols/shapes, calculate a toy case, and state one invariant that the numerical implementation must satisfy.
### 8
Compare **one-class methods** and **threshold calibration** under a matched protocol, discussing stability, bias/variance or optimization behavior, time/memory, data sensitivity, and interpretability.

## Level 3
### 9
Use low-level Python/NumPy operations, keep the implementation small, and make it match a hand-computed case. Avoid hiding the key mechanism in a high-level library call.
### 10
Reject incompatible shapes/dtypes, non-finite values, invalid ranges, empty inputs, and illegal hyperparameters with clear errors.
### 11
Fix all randomness, create a toy dataset/problem with known expected behavior, record config/seed, and verify reruns reproduce the result within tolerance.
### 12
One test covers normal behavior; one targets an edge/invariant. If a trusted implementation exists, compare on a tiny deterministic input without making that library the implementation itself.

## Level 4
### 13
A good bug is plausible but violates a mathematical invariant. The fix should repair the smallest cause and add a regression test.
### 14
Demonstrate the incorrect state/split/eval behavior, then isolate learned state to training and apply evaluation correctly. Report the before/after metric and why it changed.
### 15
Use stable forms such as max-subtraction, clipping only when mathematically justified, epsilon guards, normalized scales, or higher precision as appropriate. Add a regression test for the extreme.
### 16
Measure before/after on the same input. Optimize the actual hotspot (loop, allocation, matrix op, data transfer), then verify numerical equivalence and note any tolerance change.

## Level 5
### 17
Keep split/budget/metric matched. Report every run plus mean and standard deviation/interval; do not cherry-pick.
### 18
Change exactly one factor, keep all others fixed, and explain the change using the algorithm's mechanism.
### 19
Include schema/range validation, frozen data/preprocessing/model identifiers, seed/config, latency/memory envelope, quality/drift metrics, alert/rollback criteria, and security/privacy constraints where relevant.
### 20
The report needs a falsifiable claim, fair baseline, reproducible protocol, all relevant results, failure evidence, limitations, and a next experiment that reduces uncertainty.
