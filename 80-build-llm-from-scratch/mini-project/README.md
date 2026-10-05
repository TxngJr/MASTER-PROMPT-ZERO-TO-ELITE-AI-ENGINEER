# Final Mini Project — End-to-End Tiny LLM

## Required Pipeline

~~~text
raw text
→ clean/dedup
→ split
→ tokenizer
→ encode/pack
→ random-init decoder
→ pretrain
→ validate
→ checkpoint
→ generate
→ SFT
→ preference optimization
→ quantization evaluation
→ API/container
→ monitoring
~~~

## Required Artifacts

~~~text
data/manifest.json
tokenizer/merges.json
configs/model.json
configs/train.json
checkpoints/best.pt
reports/pretrain.json
reports/validation.json
reports/alignment.json
reports/quantization.json
reports/final_report.md
~~~

อย่า commit datasets/checkpoints ขนาดใหญ่โดยไม่ตรวจ policy

## Acceptance Gates

- tokenizer UTF-8 round-trip
- no train/validation hash overlap
- forward shape
- target shift
- tiny-batch overfit
- finite loss
- checkpoint reload deterministic logits in eval mode
- SFT stage runs
- preference stage runs
- quantization error measured
- generation bounded
- hardware/versions/seed recorded

## Final Report

Scope → Hardware → Dataset → Tokenizer → Model → Memory → Training → Evaluation → SFT → Preference → Quantization → Serving → Monitoring/Safety → Limitations → Next experiments
