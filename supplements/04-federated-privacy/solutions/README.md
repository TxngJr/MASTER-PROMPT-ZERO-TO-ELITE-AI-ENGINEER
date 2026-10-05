# Federated and Privacy Solutions

1. Federated learning trains/updates across data owners while keeping raw data local under the chosen system design.
2. FedAvg aggregates client model/update vectors, commonly weighted by local example count.
3. DP is a randomized stability guarantee limiting how much one individual's inclusion changes the output distribution.
4. Clipping caps contribution norm to bound influence/sensitivity.
5. Example-weighted averaging optimizes proportional to examples; client-weighted gives every participating client equal weight.
6. Client updates can leak information and servers/logging can still expose data; federated architecture alone is not a formal privacy guarantee.
7. Non-IID means client distributions differ, affecting convergence and fairness.
8. Secure aggregation hides individual updates from the aggregator under its cryptographic model; DP limits information about individual records in released outputs.
9–11. See `src/federated.py`; validate client shapes/counts, norm cap and deterministic lab seed.
12. Multiply parameter count by bytes per transmitted element plus any protocol metadata; distinguish upload and download and multiply by participating clients/rounds.
13. Use highly unequal counts; compare plain mean against the weighted formula and identify the objective change.
14. Give clients different label/feature distributions and compare local/global loss or parameter directions.
15. Sweep C; report fraction clipped, update norm statistics and task metric.
16. Treat this only as a utility/noise experiment. A noise standard deviation alone is not epsilon; a valid privacy accountant and sampling/composition assumptions are required.
17. Include global, median, worst/group slices, participation, communication, convergence and uncertainty.
18. State adversary, protected values, collusion assumptions and what metadata remains visible.
19. Specify neighboring-dataset definition, clipping mechanism, sampling, noise, number of steps, accountant/library/version and resulting epsilon/delta.
20. Document local data handling, client sampling, telemetry, aggregation, privacy mechanism/accounting, security assumptions, fairness/utility and limitations.
