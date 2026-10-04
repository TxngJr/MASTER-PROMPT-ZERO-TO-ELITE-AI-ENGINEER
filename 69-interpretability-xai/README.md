# Chapter 69 — Interpretability & Explainable AI (XAI)

## 1. What Are We Trying to Explain?

Interpretability methods answer different questions:

- Which input features matter for this prediction?
- Which features matter globally?
- What changes would change the output?
- Which internal representations encode information?
- Which components causally affect a behavior?

Do not treat one heatmap as answering all of them.

## 2. Learning Objectives

- distinguish global and local explanations
- distinguish intrinsic and post-hoc interpretability
- implement permutation importance
- understand gradient saliency
- derive Integrated Gradients
- understand baseline choice
- test the completeness property
- understand Shapley values
- compute exact Shapley values for small feature sets
- use occlusion/perturbation explanations
- understand correlated-feature problems
- distinguish attention visualization from explanation
- understand probes and activation patching conceptually
- test explanation stability and faithfulness
- understand current Captum/SHAP APIs

## 3. Global vs Local

Global explanation:
~~~text
What generally drives the model across a dataset?
~~~

Local explanation:
~~~text
Why was this individual prediction high/low?
~~~

Permutation importance is commonly global; Integrated Gradients is naturally local to an input/baseline pair.

## 4. Intrinsic vs Post-Hoc

Intrinsic models are designed to be interpretable, such as a small linear model or shallow tree.

Post-hoc methods explain a trained black-box model after the fact.

Post-hoc explanations approximate/model some aspect of behavior; they are not automatically the model's literal reasoning process.

## 5. Feature Importance in Linear Models

For standardized independent-ish features, coefficient magnitude can be informative.

But coefficients depend on:
- feature scaling
- collinearity
- interactions
- regularization

Never compare raw coefficient magnitude across arbitrary units without care.

## 6. Tree Importance

Impurity-based tree importance can be biased toward features with many split opportunities.

Permutation importance offers a model-agnostic alternative but has its own issues with correlated features.

## 7. Permutation Importance

Procedure:
1. compute baseline metric
2. shuffle one feature
3. recompute metric
4. importance = performance drop

~~~text
importance_j
=
baseline_score - permuted_score_j
~~~

Repeat permutations to estimate variability.

## 8. Correlated Features

If two features carry the same information, permuting one may not hurt performance because the other substitutes for it.

Therefore low permutation importance does not always mean the information is unimportant.

## 9. Gradient Saliency

For differentiable model output F(x):

~~~text
saliency_i = ∂F(x) / ∂x_i
~~~

Large magnitude means a small local change can affect the output strongly.

Gradient saturation can make raw saliency misleading.

## 10. Integrated Gradients

Choose baseline x' and input x.

~~~text
IG_i(x)
=
(x_i - x'_i)
×
integral from alpha=0 to 1
∂F(x' + alpha(x-x')) / ∂x_i
d alpha
~~~

Approximate the integral numerically.

## 11. Baseline Choice

Baseline represents a reference input.

Examples:
- zero vector when meaningful
- blank image
- neutral embedding/reference

A poor baseline can make attributions hard to interpret.

Always report it.

## 12. Completeness

For Integrated Gradients under the method's assumptions:

~~~text
sum_i IG_i
≈
F(x) - F(x')
~~~

Approximation error falls as integration becomes more accurate.

Captum can return a convergence delta related to this property.

## 13. Current Captum

Captum currently provides a generic IntegratedGradients implementation for PyTorch models and supports Riemann-sum / Gauss-Legendre numerical integration.

Captum also provides many perturbation and gradient-based attribution methods plus metrics such as sensitivity/infidelity.

## 14. Shapley Values

From cooperative game theory, a feature's Shapley value is its average marginal contribution across all subsets/orders of the other features.

For feature i:

~~~text
phi_i
=
sum over S not containing i
|S|!(M-|S|-1)! / M!
× [v(S ∪ {i}) - v(S)]
~~~

Exact calculation is exponential in feature count.

## 15. SHAP

SHAP is a framework/library built around Shapley-value explanations and related algorithms.

Current SHAP exposes a generic `shap.Explainer` plus specialized explainers such as TreeExplainer and LinearExplainer.

Algorithm choice, masker/background distribution and feature dependence assumptions matter.

## 16. Base Value

Additive explanation often looks like:

~~~text
prediction
≈
base_value
+ sum(feature contributions)
~~~

Know which output space is explained:
- probability
- logit
- raw score

## 17. Occlusion

Replace/remove one feature or region and measure output change.

~~~text
importance_i
=
F(x) - F(x with feature i replaced by baseline)
~~~

Simple but potentially expensive and sensitive to unrealistic perturbed inputs.

## 18. Counterfactual Explanations

Ask:

~~~text
What feasible minimal change would alter the prediction?
~~~

Constraints matter:
- immutable attributes
- actionability
- domain validity

Counterfactual explanations should not imply causal guarantees unless the model/data support causal claims.

## 19. PDP / ICE

Partial Dependence Plot averages predictions as one feature is varied.

ICE shows one curve per sample.

They can be misleading when varied feature combinations fall outside realistic joint data distributions.

## 20. Explanation Stability

Small input/model perturbations should not cause wildly different explanations unless the model behavior itself changes substantially.

Measure similarity across:
- random seeds
- nearby examples
- retrained models
- explanation hyperparameters

## 21. Faithfulness

A visually appealing explanation may be unfaithful.

Test by perturbation/intervention:
- remove top-attributed features
- compare output change
- compare against random features

## 22. Sanity Checks

Useful tests:
- randomize model parameters
- randomize labels/retrain
- compare explanation before/after

If explanation barely changes after model randomization, it may reflect the input visualization more than learned behavior.

## 23. Attention Is Not Automatically Explanation

Attention weights show one internal routing/weighting mechanism.

They do not automatically prove:
- feature causality
- token importance
- human-interpretable reasoning

Use interventions/ablation when making causal claims.

## 24. LLM Token Attribution

Possible methods:
- gradient attribution to embeddings
- Layer Integrated Gradients
- perturb/occlude input tokens
- logit difference attribution

Current Captum's LLM examples use LayerIntegratedGradients for gradient-based token attribution.

## 25. Probing

Train a simple probe on hidden activations to test whether information is decodable.

Important limitation:

> decodable information does not prove the model actually uses that information for the behavior.

## 26. Activation Patching

Intervene on internal activations:

~~~text
clean run activation
↓ replace activation in corrupted run
↓
measure behavior recovery
~~~

This moves toward causal mechanistic analysis.

## 27. Ablation

Disable/remove a component and observe effect:
- attention head
- neuron/channel
- layer
- feature

Ablation can reveal causal contribution but interventions may push the model off-distribution.

## 28. Mechanistic Interpretability

Goal: understand internal algorithms/circuits rather than only feature attribution.

Tools/concepts:
- activation inspection
- probes
- ablations
- activation patching
- feature dictionaries/sparse representations
- circuit hypotheses

Treat mechanistic conclusions as hypotheses requiring interventions and replication.

## 29. Explanation vs Causality

Predictive association:
~~~text
feature helps predict output
~~~

Causal effect:
~~~text
intervening on feature changes outcome under a causal model
~~~

Most XAI methods are not causal inference.

## 30. From Scratch

src/xai.py implements:

- finite_difference_gradient
- integrated_gradients
- completeness_gap
- occlusion_importance
- exact_shapley_values
- permutation_importance
- cosine_attribution_similarity

## 31. Common Mistakes

1. explanation treated as ground-truth reasoning
2. raw coefficients compared across incompatible scales
3. correlated features ignored
4. IG baseline not reported
5. probability/logit explanation space mixed
6. attention heatmap called causal importance
7. probe decodability called causal use
8. explanation stability never tested
9. only one explanation method used
10. counterfactual interpreted as real-world causal advice

## 32. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 33. Checklist

- [ ] global/local
- [ ] permutation importance
- [ ] saliency
- [ ] Integrated Gradients
- [ ] baseline/completeness
- [ ] Shapley/SHAP
- [ ] occlusion
- [ ] counterfactual concepts
- [ ] sanity/stability tests
- [ ] attention limitations
- [ ] probing/activation patching
- [ ] causality distinction

## 34. What's Next

Batch 24 covers model compression, Mixture-of-Experts and long-context/memory systems.