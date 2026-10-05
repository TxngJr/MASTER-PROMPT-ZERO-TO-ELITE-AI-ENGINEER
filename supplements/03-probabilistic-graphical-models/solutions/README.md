# PGM Solutions

1. A Bayesian network is a DAG whose local conditional distributions factor a joint distribution.
2. A factor is a non-negative function over a subset of variables used in a product representation.
3. A hidden state is an unobserved variable whose transitions generate/condition observations.
4. A marginal sums/integrates unwanted variables out of a joint distribution.
5. For A->B->C: P(A,B,C)=P(A)P(B|A)P(C|B).
6. Chain/fork paths are blocked by conditioning on the middle variable; a collider path is normally blocked but can open when conditioning on the collider or descendants.
7. Multiply the predicted state mass `alpha_(t-1) @ A` elementwise by the emission likelihood at t.
8. Intermediate factor scope depends on elimination order; large scopes cause exponential tables.
9–11. See `src/pgm.py`, with validation of non-negativity, normalization and aligned dimensions.
12. Enumerate all assignments, multiply local CPT terms, sum assignments matching the query condition, then normalize if computing a conditional.
13. Use per-step scaling factors or log-space log-sum-exp; verify the recovered log-likelihood.
14. Assert each transition row is a probability vector within tolerance.
15. List intermediate variable scopes/cardinalities and compare maximum factor size.
16. On a tree, each message is the product of incoming factors/messages summed over the sender variable as required.
17. Enumeration grows over full state sequences; forward DP reuses prefixes and costs roughly O(T K^2).
18. Define latent state, sensor observations, conditional independence and calibrated likelihoods before choosing inference.
19. Loops cause messages to reuse information; convergence/exactness guarantees from trees no longer apply generally.
20. Include graph/factorization assumptions, numeric parameters, inference algorithm, elimination/message schedule, complexity, stability and validation tests.
