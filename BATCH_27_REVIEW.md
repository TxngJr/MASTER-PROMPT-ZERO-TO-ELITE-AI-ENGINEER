# Batch 27 Review — Chapters 79–80 + Final Integration Audit

## Chapters

- 79 — Research Paper Engineering
- 80 — Build Your Own LLM From Scratch
- Final Course Integration Audit (not a numbered chapter)

## Concepts Learned

Research:
- claim/evidence/assumption
- reproduction contracts
- exact/close/concept reproduction
- baseline fairness
- multiple-seed uncertainty
- result gaps/improvement recovery
- ablations
- leakage/contamination audit
- compute accounting
- experiment fingerprints
- reproducibility metadata

Final LLM:
- corpus/provenance
- split-before-tokenizer fitting
- byte-BPE reuse
- causal windows
- random-init decoder-only Transformer
- causal SDPA
- CE/perplexity
- parameter/Adam/KV memory estimates
- pretraining/validation/checkpoint
- SFT
- DPO-style preference optimization
- INT8 reconstruction
- bounded generation
- API/deployment/monitoring contracts

## Mathematical Tools

- mean/std/standard error
- approximate confidence interval
- absolute/relative errors
- normalized improvement recovery
- causal-LM NLL/perplexity
- DPO log-ratio objective
- quantization scale
- parameter/KV memory accounting

## Implemented

Chapter 79:
- relative_error
- result_within_tolerance
- normalized_improvement_recovery
- ablation_effect
- seed_statistics
- accelerator_hours
- experiment_fingerprint
- audit_reproducibility_metadata

Chapter 80:
- causal_windows
- memory estimators
- INT8 quantize/dequantize
- DPO scalar loss
- fingerprints
- tiny decoder LM
- pretraining/validation
- SFT/preference smoke
- checkpoint/generation

## Common Weak Points

- exact vs concept reproduction confusion
- tuning before baseline works
- reporting best seed
- tokenizer fit on held-out data
- target-shift/mask mistakes
- perplexity across tokenizers
- optimizer/KV memory underestimation
- preference sign errors
- quantization without re-evaluation
- serving without limits/monitoring

## Mastery Gate

Learner must be able to:

### Explain
teach the concept without hiding behind APIs

### Derive
derive important equations

### Implement
write representative algorithms and an end-to-end tiny LLM

### Debug
data, shape, gradient, loss, memory, evaluation and serving failures

### Modify
architecture, tokenizer, training and alignment recipes

### Compare
run fair controlled evaluations

### Apply
solve a new problem

### Research
read/reproduce a paper claim

### Ship
deploy and monitor a bounded AI system

## Coverage Audit

See [FINAL_CURRICULUM_AUDIT.md](FINAL_CURRICULUM_AUDIT.md).

## Exit Gate

1. core CI passes
2. PyTorch capstone smoke passes
3. Chapter 79 tests pass
4. Chapter 80 tests pass
5. final integration gate passes
6. no pretrained weights required
7. README/Course Map marks Batch 27 complete
8. Final Audit records remaining non-core gaps honestly
