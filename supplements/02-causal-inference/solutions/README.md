# Causal Inference Solutions

1. A confounder is a common cause of treatment/exposure and outcome.
2. A collider is a common effect of two variables; conditioning on it can open a path.
3. ATE is the population average of Y(1)-Y(0).
4. The propensity score is the probability of treatment conditional on observed covariates.
5. Association describes an observed distribution; intervention describes a changed data-generating process.
6. Randomization breaks systematic treatment–potential-outcome dependence in expectation.
7. Positivity requires each compared treatment option to have nonzero probability in the relevant covariate strata.
8. Conditioning on a collider can make its otherwise independent causes statistically dependent.
9–11. Use the reference functions, verify group membership and document any propensity clipping.
12. A minimal confounding DAG is X -> T -> Y together with X -> Y.
13. Simulate X affecting both treatment and outcome; the naive difference then mixes treatment effect with X imbalance.
14. Propensities near zero or one produce very large inverse weights and high-variance estimates.
15. Generate two independent causes of C and condition/select on C; the causes become associated in the selected data.
16. Compare standardized mean differences before/after adjustment; improved balance does not prove absence of unmeasured confounding.
17. Include consistency, conditional exchangeability, positivity, interference/measurement assumptions as applicable and justify each.
18. Choose adjustment variables from the causal graph/data-generating process rather than predictive feature importance alone.
19. Vary the assumed strength/prevalence of an unmeasured confounder or use another domain-appropriate sensitivity method and report conclusion changes.
20. Separate causal question, graph/assumptions, estimand, estimator, diagnostics, uncertainty, sensitivity and limitations.
