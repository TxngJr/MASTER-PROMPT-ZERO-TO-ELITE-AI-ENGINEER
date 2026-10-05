# Chapter 79 — Research Paper Engineering

## 1. Why This Matters

AI Engineer ระดับสูงต้องทำได้มากกว่าเรียก library: ต้องอ่าน paper แล้วแปลง **claim → implementation → controlled experiment → evidence** ได้

บทนี้สอนให้ตอบว่า paper เสนออะไรใหม่, สมการทำงานอย่างไร, baseline fair หรือไม่, ทำไม reproduction ต่างจาก original และข้อสรุปยังอยู่หรือไม่เมื่อควบคุม seed/compute/ablation

## 2. Prerequisites

ควรผ่าน Chapter 02, 03, 05, 19, 22, 29–30, 50, 60 และ 65

ต้องคุ้นเคยกับ Python, NumPy, PyTorch, Git, train/validation/test, random seed, mean/variance และ experiment tracking

## 3. Learning Objectives

By the end of this chapter you can:

- อ่าน abstract → method → equations → experiments → limitations อย่างเป็นระบบ
- แยก claim, evidence, assumption และ implementation detail
- แปลงสมการเป็น executable specification
- สร้าง reproduction contract ก่อนเขียน code
- reproduce baseline ก่อน method ใหม่
- ทำ multi-seed evaluation และ uncertainty summary
- ทำ ablation study
- audit data leakage / benchmark contamination
- เปรียบเทียบ compute budget อย่างยุติธรรม
- สร้าง experiment fingerprint
- audit reproducibility metadata
- เขียน technical reproduction report ที่ระบุข้อจำกัดตรงไปตรงมา

## 4. Mental Model

~~~text
paper
 ├─ claim
 ├─ equations
 ├─ architecture
 ├─ data protocol
 ├─ training protocol
 ├─ evaluation protocol
 └─ ablations/limitations
        ↓
reproduction contract
        ↓
implementation + tests
        ↓
baseline
        ↓
proposed method
        ↓
multi-seed + ablation
        ↓
comparison + uncertainty
        ↓
technical report
~~~

อย่าเริ่มจาก “วาด architecture ตามรูป” ให้เริ่มจาก “claim ไหนต้องพิสูจน์ซ้ำ และ evidence อะไรจึงเพียงพอ”

## 5. Read Papers in Passes

### Pass 1 — Problem / Claim / Evidence

อ่าน title, abstract, introduction, conclusion แล้วเขียน:

~~~text
Problem:
Claim:
Evidence:
~~~

### Pass 2 — Mechanism

อ่าน method, equations, algorithm box และ architecture แล้วตอบ input/output, learnable parameters, objective, optimizer update และ training/inference differences

### Pass 3 — Evidence Quality

ตรวจ dataset split, preprocessing, parameter/compute budget, tuning effort, metrics, seeds และ uncertainty

## 6. Claim → Testable Contract

~~~text
Claim:
Method A improves validation accuracy over baseline B.

Contract:
same dataset
same split
same preprocessing
same metric
comparable training budget
multiple seeds
paired comparison where possible
~~~

ถ้าเปลี่ยนหลายอย่างพร้อมกัน คุณไม่รู้ว่า improvement มาจากอะไร

## 7. Equation → Executable Specification

ทุกสมการทำ 7 ขั้น:

1. define symbols
2. define shapes
3. define domain/units
4. scalar example
5. vectorized form
6. implementation
7. invariant tests

ตัวอย่าง:

~~~text
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
~~~

ต้องระบุ Q/K/V shapes, score/output shapes และ mask semantics

## 8. Reproduction Types

### Exact reproduction
match dataset, split, model, training recipe, metric และ compute ใกล้ original

### Close reproduction
protocol หลักเหมือน แต่ runtime/GPU/library version ต่างแบบควบคุมได้

### Concept reproduction
ทดสอบ mechanism หลักบน scale เล็ก

Local laptop มักทำ close/concept reproduction; ห้ามอ้างว่า tiny result เท่ากับ original benchmark

## 9. Reproduction Contract Table

| Item | Paper | Our run | Difference |
|---|---|---|---|
| dataset | original | local subset | scale |
| tokenizer | original | same concept | smaller vocab |
| model | large | tiny | params |
| optimizer | reported | matched | none |
| metric | reported | same | none |
| seeds | reported/unknown | 3+ when feasible | documented |

## 10. Baseline Fairness

ตรวจ architecture capacity, data/tokens, augmentation, optimizer/LR schedule, early stopping, hyperparameter search effort, inference budget และ decoding settings

LLM ต้องเพิ่ม context length, tokenizer, prompt template และ test-time compute

## 11. Seed Statistics

~~~text
mean = (1/n) Σ x_i
s = sqrt( Σ(x_i - mean)^2 / (n-1) )
SE = s / sqrt(n)
approx 95% interval = mean ± 1.96 SE
~~~

สำหรับ n เล็ก Student-t interval เหมาะกว่า normal approximation; utility บทนี้ใช้ normal approximationเพื่อสอน mechanics

## 12. Result Gap

~~~text
absolute_error = |reproduced-reference|

relative_error =
|reproduced-reference| / max(|reference|, epsilon)
~~~

อย่าใช้ relative error อย่างเดียวเมื่อ reference ใกล้ศูนย์

## 13. Improvement Recovery

~~~text
recovery =
(reproduced - baseline) /
(claimed - baseline)
~~~

baseline=70, paper=80, ours=77 → recovery=0.7

นี่หมายถึง recover 70% ของ reported improvement ภายใต้ protocol นี้เท่านั้น

## 14. Ablation

สำหรับ metric ที่สูงดีกว่า:

~~~text
ablation_effect = full_metric - ablated_metric
~~~

เปลี่ยน factor เดียวเมื่อทำได้ เช่น remove positional encoding, change optimizer, remove reranker, reduce LoRA rank

## 15. Negative Results

เก็บ config, seed, failure, diagnostics และ next hypothesis

ห้ามลบ failed runs เพียงเพราะผลไม่สวย หากมันเปลี่ยน interpretation

## 16. Reproducibility Metadata

~~~yaml
paper:
code_commit:
dataset:
dataset_version:
seed:
hardware:
os:
python:
dependencies:
model_config:
optimizer:
learning_rate:
batch_size:
steps_or_epochs:
metric:
evaluation_command:
notes:
~~~

NeurIPS paper checklist ปัจจุบันเน้น reproducibility, training details, compute/resources และ instructions สำหรับทำซ้ำ ดู references.md

## 17. Experiment Fingerprint

~~~text
canonical JSON
    ↓
SHA-256
    ↓
experiment fingerprint
~~~

ช่วย detect config drift และผูก result กับ config แต่ไม่แทน Git commit/dataset checksum

## 18. Leakage Audit

ตรวจ duplicate ข้าม split, temporal/user leakage, tokenizer fitting, scaler/statistics fitting และ benchmark contamination

## 19. Compute Accounting

~~~text
accelerator_hours =
accelerator_count * wall_clock_hours
~~~

ถ้าใช้ TDP ประเมิน energy ต้องระบุว่าเป็น proxy ไม่ใช่ power measurement จริง

## 20. Hardware-Aware Reproduction

~~~yaml
Expected hardware:
  CPU: Ryzen 7 5825U or similar
  RAM: 16 GB
  GPU: RTX 3050 Ti Laptop GPU optional
  VRAM: check locally; do not assume exact capacity
  Approximate dataset size: KB–hundreds of MB locally
  Recommended batch size: start small and measure
~~~

Local: equation/unit tests, tiny Transformer, small ablations, 3–5 seeds when cheap

Scaled: original dataset/model, full benchmark, distributed compute, original-like protocol

## 21. Reproduction Workflow

~~~text
1 choose claim
2 freeze paper/version
3 build claim table
4 extract protocol
5 build environment
6 write unit tests
7 implement minimum method
8 reproduce baseline
9 reproduce proposal
10 run seeds
11 ablate
12 compare
13 investigate gaps
14 write report
~~~

**Baseline first**: ถ้า baseline ยังผิด การเทียบ method ใหม่ไม่มีฐานที่เชื่อถือได้

## 22. Debugging Gaps

ตรวจตามลำดับ: split/data → preprocessing → objective → shapes/masks → optimizer/LR → clipping → initialization → train/eval → metric → decoding → appendix details

สร้าง diagnostic test แทน random hyperparameter tweaking

## 23. Research Integrity

ห้าม cherry-pick seed, เปลี่ยน metric หลังเห็นผล, tune baseline น้อยกว่า proposal, discard failures แบบไร้เกณฑ์ หรืออ้าง exact reproduction ทั้งที่ลด scale มาก

## 24. From-Scratch Implementation

`src/paper_engineering.py`:

- relative_error
- result_within_tolerance
- normalized_improvement_recovery
- ablation_effect
- seed_statistics
- accelerator_hours
- experiment_fingerprint
- audit_reproducibility_metadata

## 25. Experiment

1. Seed sensitivity — 5 seeds
2. Single ablation — เปลี่ยน component เดียว
3. Budget match — baseline/proposal ที่ steps/tokens ใกล้กัน

## 26. Failure Cases

1. code ไม่มี
2. dataset version หาย
3. preprocessing ไม่ชัด
4. defaults เปลี่ยน
5. hardware ต่าง
6. seed ไม่ระบุ
7. evaluation script ต่าง
8. contamination
9. metric ต่าง
10. hidden tuning
11. compute ไม่พอ
12. benchmark drift

เป้าหมายคือระบุ uncertainty/source of discrepancy ไม่ใช่บังคับให้เลขตรง

## 27. Production Perspective

ใช้ทักษะนี้ verify vendor/model claims, regression test model upgrades, reproduce incidents, compare inference kernels และสร้าง audit trail

## 28. Research Perspective

งานวิจัยที่ตรวจได้ต้องบอก exactly what was tested, under what conditions, with what uncertainty และ limitations

## 29. Common Mistakes

1. อ่าน abstract แล้วเริ่ม code
2. ไม่ reproduce baseline
3. split ต่าง
4. budget ต่าง
5. seed เดียว
6. metric ต่าง
7. train/eval ผิด
8. เปลี่ยนหลาย factors
9. cherry-pick best run
10. relative error near zero
11. copy figure แต่ไม่อ่าน loss
12. ignore appendix
13. ไม่บันทึก commit/config
14. overclaim tiny reproduction
15. สรุปว่า paper ผิดก่อนตรวจ implementation

## 30. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Paper Reproduction Project](mini-project/README.md)

## 31. Interview Questions

1. reproduction vs replication?
2. ทำไม baseline ต้องผ่านก่อน?
3. seed เดียวมีปัญหาอะไร?
4. ablation ตอบอะไร?
5. leakage ทำให้ benchmark optimistic อย่างไร?
6. preprocessing ไม่ระบุทำอย่างไร?
7. compute fairness คืออะไร?
8. exact vs concept reproduction?
9. fingerprint ช่วยอะไร?
10. debug reproduced gap อย่างไร?

## 32. Checklist

- [ ] claim/evidence/assumption
- [ ] reproduction contract
- [ ] equations/shapes
- [ ] baseline
- [ ] metadata
- [ ] multiple seeds
- [ ] uncertainty
- [ ] ablation
- [ ] leakage audit
- [ ] compute accounting
- [ ] reproduction type
- [ ] discrepancy analysis
- [ ] limitations/report

## 33. Summary

~~~text
claim → contract → implementation → controlled experiment
→ uncertainty → audit → report
~~~

## 34. What's Next

Chapter 80 รวม Programming → Math → Data → Transformer → LLM → Alignment → Systems → Deployment เป็น capstone เดียว
