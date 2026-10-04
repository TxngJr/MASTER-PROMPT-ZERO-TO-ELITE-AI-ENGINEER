# Chapter 60 — LLM Evaluation

## 1. Why Evaluation Is a System

A single benchmark score does not describe an LLM.

A rigorous evaluation records:

~~~text
model checkpoint
+
tokenizer/chat template
+
prompt
+
few-shot examples
+
decoding settings
+
dataset version
+
metric implementation
+
random seed
=
evaluation result
~~~

Change any component and comparability can break.

## 2. Learning Objectives

You should be able to:

- design reproducible evaluation protocols
- distinguish intrinsic and task evaluations
- compute exact match and token F1
- evaluate multiple-choice log-probabilities
- use pairwise preference evaluation
- understand human and LLM-judge evaluation
- detect position/order bias
- measure calibration
- compute bootstrap confidence intervals
- perform paired model comparisons
- reason about statistical significance
- audit contamination
- evaluate grounding/citations
- evaluate instruction following
- record latency/cost/memory separately from quality
- use evaluation-harness concepts

## 3. Evaluation Dimensions

Possible dimensions:
- factual/task correctness
- instruction following
- reasoning/task completion
- formatting
- calibration
- grounding/citation quality
- preference quality
- robustness
- safety-policy adherence
- latency
- throughput
- memory
- cost

Do not collapse everything into one number without justification.

## 4. Evaluation Unit

Define what one sample means:
- question
- conversation
- document
- prompt family
- user session

This matters for confidence intervals and leakage.

## 5. Deterministic Tasks

Examples:
- classification
- exact numeric answer
- JSON schema
- multiple choice

Metrics can be deterministic after normalization.

## 6. Exact Match

~~~text
EM =
mean(
normalize(prediction)
==
normalize(reference)
)
~~~

Normalization policy must be versioned.

Over-aggressive normalization can hide real formatting errors.

## 7. Token F1

Treat normalized answer tokens as bags/multisets.

~~~text
precision =
overlap / predicted_tokens

recall =
overlap / reference_tokens

F1 =
2PR / (P+R)
~~~

Useful when partial lexical overlap matters.

## 8. Multiple-Choice Evaluation

For choices c_i:

~~~text
score(c_i)
=
log pi(c_i | prompt)
~~~

Predict highest score.

Important protocol choices:
- completion normalization
- leading-space/token handling
- length normalization when specified
- chat template
- few-shot formatting

## 9. Perplexity

~~~text
PPL =
exp(mean token CE)
~~~

Useful for language modeling, but:
- tokenizer dependent
- corpus dependent
- not direct instruction-following quality

Do not compare perplexity across different tokenizers naively.

## 10. Pairwise Evaluation

Given model A and B responses, evaluator selects:
- A better
- B better
- tie

Metrics:
- win rate
- loss rate
- tie rate

Pairwise can be easier than assigning absolute scores.

## 11. Position Bias

A judge may prefer:
- first answer
- second answer

Mitigation:
- randomize order
- evaluate both orders
- report consistency

## 12. LLM-as-Judge

Useful for open-ended tasks, but judges have:
- position bias
- verbosity bias
- self-preference
- domain gaps
- prompt sensitivity

Validate against human/reference labels on a subset.

## 13. Human Evaluation

Define:
- rubric
- annotator instructions
- tie policy
- blind model identity
- random response ordering

Track inter-annotator agreement.

## 14. Calibration

A model can be accurate but overconfident.

For binary prediction with confidence p and outcome y:

~~~text
Brier =
mean((p-y)^2)
~~~

Lower is better.

## 15. Expected Calibration Error

Partition predictions into confidence bins.

For bin b:

~~~text
ECE =
Σ_b
(n_b/N)
*
|accuracy_b - confidence_b|
~~~

ECE depends on binning and should not be treated as a universal scalar truth.

## 16. Abstention

Some applications should allow:
- answer
- abstain / insufficient evidence

Evaluate:
- coverage
- accuracy at coverage
- selective risk

Forcing an answer can inflate hallucinations.

## 17. Grounded / RAG Evaluation

Separate:
- retrieval correctness
- answer correctness
- faithfulness to context
- citation correctness
- citation completeness

Correct answer with wrong citation is a citation failure.

## 18. Instruction-Following Evaluation

Check:
- requested task completed
- constraints followed
- output format valid
- language/style requirements satisfied

A semantically correct response can still fail explicit format instructions.

## 19. Robustness

Create controlled perturbations:
- wording changes
- order changes
- irrelevant context
- formatting changes

Measure performance degradation.

## 20. Contamination

Evaluation data may appear in:
- pretraining
- fine-tuning
- preference data
- synthetic data

Audits:
- exact hashes
- normalized exact matches
- n-gram/minhash near duplicates
- source overlap
- benchmark-family overlap

No single contamination detector is complete.

## 21. Exact Contamination Hash

Normalize sample text and hash it.

Then compare evaluation hashes against training-data hashes.

This catches exact normalized duplicates only.

## 22. Confidence Intervals

A point estimate:

~~~text
accuracy = 0.73
~~~

is incomplete without uncertainty.

Bootstrap:
1. sample evaluation rows with replacement
2. recompute metric
3. repeat
4. report percentile interval

## 23. Paired Comparison

When A and B are evaluated on the same samples, analyze sample-level differences:

~~~text
d_i =
score_A_i - score_B_i
~~~

Bootstrap d_i rather than treating model scores as independent.

This usually provides a more sensitive comparison.

## 24. Statistical vs Practical Significance

A tiny effect can be statistically detectable but operationally irrelevant.

Report:
- effect size
- confidence interval
- sample count
- practical threshold

## 25. Multiple Comparisons

If many models/tasks/hyperparameters are tested, lucky wins become more likely.

Avoid selecting only the best-looking slice after seeing results.

## 26. Slice Evaluation

Break results down by:
- domain
- language
- difficulty
- length
- prompt type

Aggregate scores can hide severe subgroup failures.

Use enough samples per slice to avoid overinterpreting noise.

## 27. Generation Settings

Record:
- temperature
- top-p
- top-k
- max tokens
- stop strings/tokens
- seed when available

Greedy and sampled evaluations answer different questions.

## 28. Latency / Throughput

Measure quality separately from systems metrics.

Useful:
- time to first token
- inter-token latency
- tokens/sec
- request throughput
- peak memory

Warm up the system and define concurrency/batch settings.

## 29. Reproducibility Manifest

Store:
- model revision
- tokenizer revision
- adapter revision
- dataset version
- task config
- prompt/chat template
- few-shot examples/seed
- decoding config
- metric version
- code commit

## 30. Evaluation Harness

A harness should:
1. load task config
2. render prompt
3. run model
4. postprocess
5. compute metrics
6. aggregate
7. save per-sample results
8. save manifest

Per-sample outputs are essential for debugging and paired analysis.

## 31. Current lm-evaluation-harness Concepts

EleutherAI's current harness supports:
- many standard benchmarks
- Hugging Face / vLLM / API backends
- YAML evaluation configuration
- programmatic simple_evaluate()
- custom task definitions

Current CLI uses commands such as:

~~~text
lm-eval run
lm-eval ls
lm-eval validate
~~~

Use a pinned environment/config for reproducibility.

## 32. From Scratch

src/evaluation_metrics.py implements:

- normalize_answer
- exact_match
- token_f1
- binary_brier_score
- expected_calibration_error
- pairwise_rates
- bootstrap_mean_interval
- paired_bootstrap_difference
- normalized_text_hashes
- exact_contamination_rate

## 33. Common Mistakes

1. benchmark score without protocol/version
2. test set repeatedly used for tuning
3. different chat templates across models
4. random generation compared without repeated seeds
5. pairwise judge order not randomized
6. no confidence intervals
7. independent test used instead of paired comparison
8. contamination ignored
9. aggregate metric hides weak slices
10. per-sample outputs discarded

## 34. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 35. Checklist

- [ ] protocol manifest
- [ ] exact/F1
- [ ] multiple choice
- [ ] pairwise
- [ ] calibration
- [ ] grounding
- [ ] instruction following
- [ ] confidence intervals
- [ ] paired significance
- [ ] contamination
- [ ] slices
- [ ] systems metrics

## 36. What's Next

Batch 21 moves into inference efficiency: quantization, KV-cache/batching/attention optimization and production serving.
