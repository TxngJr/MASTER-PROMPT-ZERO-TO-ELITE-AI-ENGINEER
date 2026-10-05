# Supplement 02 — Causal Inference

Predictive association asks what tends to occur together. Causal inference asks what would change under an intervention.

## Learning objectives

- distinguish association from causation;
- reason with DAGs, confounders, colliders and mediators;
- understand interventions and potential outcomes;
- estimate average treatment effects under explicit assumptions;
- reason about backdoor adjustment;
- understand propensity scores and inverse-probability weighting;
- diagnose overlap and covariate imbalance;
- report causal assumptions separately from statistical estimation.

## Potential outcomes

For unit i:

~~~text
Y_i(1) = outcome if treated
Y_i(0) = outcome if untreated
ATE = E[Y(1) - Y(0)]
~~~

Only one potential outcome is observed per unit. Random assignment helps identify an average effect because treatment assignment is independent of potential outcomes in expectation.

## DAG vocabulary

- confounder: common cause of treatment and outcome;
- mediator: lies on a causal path from treatment to outcome;
- collider: common effect of two variables.

Conditioning on a confounder can block a backdoor path. Conditioning on a collider can open a spurious association.

## Identification assumptions

A common observational identification setup uses:

~~~text
(Y(1), Y(0)) independent of T given X
~~~

together with positivity/overlap and consistency. These are substantive assumptions, not facts learned automatically from a dataset.

## Propensity score

~~~text
e(X) = P(T=1 | X)
~~~

Conceptual inverse-probability weights:

~~~text
treated: 1/e(X)
control: 1/(1-e(X))
~~~

Extreme probabilities produce unstable weights and indicate poor overlap.

## Diagnostics

- standardized mean difference before and after adjustment;
- propensity overlap;
- sensitivity to trimming/stabilization;
- temporal order and measurement review;
- negative controls where scientifically justified.

## Research warning

A predictive model with excellent accuracy does not validate a causal claim. State the causal question, graph/assumptions, estimand, estimator, diagnostics, uncertainty and sensitivity analysis separately.
