# Mini Project — Reproducible LLM Evaluation Harness

Build a local evaluator that saves:
- manifest.json
- per_sample.jsonl
- aggregate.json

Include:
- exact match
- token F1
- pairwise outcomes
- calibration
- bootstrap confidence intervals
- paired model differences
- contamination hashes
- task/language/length slices

Then reproduce one result from a clean environment using only the saved manifest/config.
