# Chapter 32 Solutions — NLP Fundamentals

Equivalent implementations are acceptable when token semantics, math, masks, and reproducibility invariants match.

## Level 1
### 1
Define **text normalization and tokenization**, its input/output representation, and why it is needed.
### 2
Define **n-grams and language modeling**, identify where it appears in training/inference, and state the observable effect.
### 3
Explain **subword tokenization families** and at least one assumption about data, context, geometry, or supervision.
### 4
Explain **NLP evaluation** and the associated compute/memory/quality trade-off.

## Level 2
### 5
Show each transformation from raw input through token/patch/embedding/contextual representation to output, with dimensions and special/mask state.
### 6
Explain the learning signal introduced by **n-grams and language modeling** and what information is or is not available at each position/example.
### 7
Derive the central formula, define symbols, calculate a tiny case, and verify range/normalization/shape invariants.
### 8
Compare the two concepts with matched data: representational bias, objective, compute/memory, data sensitivity, and appropriate evaluation.

## Level 3
### 9
Use primitive operations. The implementation must expose the core math rather than calling a prebuilt model. Match a manual or trusted tiny reference.
### 10
Validate token/index ranges, vocab/config agreement, masks, sequence dimensions, finite values, and legal special-token settings.
### 11
Fix seeds and splits, use a tiny corpus/image subset with known structure, and verify expected learning or deterministic behavior. Record all identities/configs.
### 12
Test target shift, mask semantics, normalization/probability sums where applicable, output shape, and exact token/patch round-trip properties where relevant.

## Level 4
### 13
Use a 2–4 token/patch toy input so allowed dependencies are obvious. Repair mask/ID/shift semantics and add a regression test.
### 14
Demonstrate contamination/leakage such as fitting tokenizer/statistics on held-out data or reusing evaluation examples during tuning; then isolate training-only fitting and report corrected metrics.
### 15
Bound sequence/output sizes, use stable softmax/log-sum-exp, reduce width/batch, or choose an appropriate approximation based on the measured failure. Do not silently truncate without recording it.
### 16
Benchmark tokenization/data loading and model separately. Optimize the actual hotspot and verify identical token IDs/masks or numerically equivalent outputs.

## Level 5
### 17
Match split, tokens/examples, model budget, evaluation and decoding. Report all runs plus variability and resource use.
### 18
Change one component only, keep the rest fixed, and connect the result to representation/objective/context mechanics.
### 19
Version tokenizer/vocab/special tokens/template/model/config together; enforce length/batch/concurrency limits; monitor latency, memory, error slices and quality; retain rollback artifacts.
### 20
Use a falsifiable claim, fair baseline, frozen protocol, contamination controls, complete results including failures, limitations, and a targeted next experiment.
