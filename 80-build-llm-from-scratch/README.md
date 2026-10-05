# Chapter 80 — Build Your Own LLM From Scratch

## 1. Why This Matters

นี่คือ Final Capstone ของหลักสูตร

~~~text
Raw Text
↓
Cleaning / Deduplication
↓
Dataset Split
↓
Tokenizer Training
↓
Token IDs
↓
Packing / DataLoader
↓
Decoder Transformer
↓
Causal Attention
↓
Pretraining
↓
Checkpoint
↓
Evaluation
↓
Instruction Tuning
↓
Preference Optimization
↓
Quantization
↓
Generation
↓
Inference API
↓
Container / Deployment
↓
Monitoring
~~~

**Primary model ต้อง train จาก random initialization** ห้ามใช้ pretrained weights เป็น model หลัก

เป้าหมายคือเข้าใจ lifecycle ทั้งระบบ ไม่ใช่สร้าง model ใหญ่ที่สุด

## 2. Prerequisites

บทนี้ integrate:

- 01–03 Programming/Math/Scientific Python
- 04–05 Data/ML
- 18–23 NN/Backprop/Optimization/PyTorch
- 29–30 Attention/Transformer
- 32–36 NLP/GPT
- 45–48 Retrieval/Agents/LLM architecture
- 49 Tokenizer
- 50 Pretraining
- 51 Data pipeline
- 52–54 Distributed/GPU/Mixed precision
- 55–60 Fine-tuning/Alignment/Evaluation
- 61–67 Quantization/Inference/Deployment/MLOps/Systems
- 68 Safety
- 70 Compression
- 72 Long context
- 79 Research Engineering

## 3. Learning Objectives

By the end of this chapter you can:

- build corpus with provenance
- clean/dedup/split without leakage
- train tokenizer
- pack causal sequences
- build decoder-only Transformer
- calculate weights/optimizer/KV memory
- pretrain from random weights
- validate/checkpoint/generate
- run tiny SFT
- run DPO-style preference step
- analyze INT8 quantization
- define serving/container/monitoring contracts
- distinguish local smoke from production-scale LLM training

## 4. Hardware Contract

~~~yaml
Expected hardware:
  CPU: AMD Ryzen 7 5825U or similar
  RAM: 16 GB
  GPU: NVIDIA RTX 3050 Ti Laptop GPU optional
  VRAM: check with nvidia-smi; do not assume exact capacity
  Approximate dataset size: KB to low MB for smoke run
  Recommended batch size: start 1-8 and measure
~~~

ห้ามรับประกัน training time เพราะ thermals, power, drivers, runtime, sequence length และ model size ต่างกัน

## 5. Local vs Scaled

### Local

- KB–few MB text
- byte-BPE small merge count
- 1–4 blocks
- d_model ~32–256
- short context
- batch 1–8
- gradient accumulation if needed
- tiny SFT/DPO
- quantization analysis
- single-process API

### Scaled

- licensed large corpus
- distributed preprocessing
- large vocab/model
- long context
- DDP/FSDP/ZeRO/tensor/pipeline parallel
- BF16/FP8 where validated
- large SFT/preference datasets
- continuous batching/paged KV cache
- replicated serving/rollout/rollback

principles เหมือนกัน แต่ systems engineering scale ขึ้นมาก

## 6. Corpus

document record:

~~~json
{
  "document_id": "doc-123",
  "source": "source-name",
  "text": "..."
}
~~~

เก็บ provenance, license/permission, version, acquisition date และ cleaning policy

## 7. Cleaning

ขั้นต่ำ:

- Unicode/newline normalization
- invalid-document filtering
- exact dedup
- optional near-dedup
- quality filters
- PII/sensitive-data policy
- provenance retention

อย่า clean รุนแรงโดยไม่วัด distribution shift

## 8. Split Before Learned Preprocessing

~~~text
documents
↓
deduplicate
↓
train/validation/test split
↓
fit tokenizer on TRAIN only
↓
freeze tokenizer
↓
encode validation/test
~~~

นี่เป็น leakage boundary สำคัญ

## 9. Train Tokenizer

Capstone reuse byte-level BPE จาก Chapter 49:

~~~text
UTF-8 bytes
↓
count adjacent pairs
↓
merge most frequent pair
↓
repeat
↓
frozen merge table
~~~

ต้อง test UTF-8 round-trip โดยเฉพาะ multilingual text

## 10. Causal Windows

token sequence:

~~~text
[t0,t1,t2,t3,t4]
~~~

sequence_length=4:

~~~text
input  = [t0,t1,t2,t3]
target = [t1,t2,t3,t4]
~~~

เรียน p(t_{i+1} | t_0...t_i)

## 11. Decoder Transformer

~~~text
token ids
↓ embedding
↓ N × decoder block
↓ final norm
↓ tied LM head
↓ logits
~~~

block:

~~~text
x
├─ RMSNorm → causal attention ─┐
└──────────────────────────────+→ x'
x'
├─ RMSNorm → SwiGLU FFN ───────┐
└───────────────────────────────+→ output
~~~

## 12. Attention Mathematics

~~~text
Q=XW_Q
K=XW_K
V=XW_V

S = QK^T / sqrt(d_k)
S_ij = -∞ when j > i
A = softmax(S)
O = AV
~~~

causal mask ป้องกัน future-token leakage

## 13. Cross-Entropy / Perplexity

~~~text
p_j = exp(z_j) / Σ_k exp(z_k)

L = -log p_y

PPL = exp(mean token loss)
~~~

perplexity เปรียบเทียบตรง ๆ ได้เมื่อ tokenizer/evaluation protocol comparable

## 14. Parameter Memory

~~~text
weight_bytes =
parameter_count * bytes_per_parameter
~~~

10M FP32 params ≈ 40 MB weights แต่ training ยังมี gradients, optimizer states, activations และ temporary buffers

## 15. Adam-State Estimate

educational rough accounting:

~~~text
weights
+ gradients
+ first moment
+ second moment
~~~

ถ้าทุกส่วนเป็น FP32 แบบง่าย ๆ ≈ 16 bytes/parameter ก่อน activations/buffers

implementation details จริงอาจต่างตาม optimizer/dtype/runtime

## 16. KV Cache

~~~text
KV bytes ≈
2 * layers * batch * sequence_length *
kv_heads * head_dim * bytes_per_element
~~~

2 มาจาก K และ V

long context จึงเพิ่ม memory แม้ weights เท่าเดิม

## 17. Scale Progression

~~~text
1K
↓
10K
↓
100K
↓
1M
↓
10M+
~~~

scale ต่อเมื่อ:

- tests ผ่าน
- loss finite
- tiny-batch overfit ได้
- validation pipeline เชื่อถือได้
- checkpoint restore ได้
- memory estimate ผ่าน

## 18. Tiny-Batch Overfit Test

same tiny batch repeated updates → training loss ควรลด

ถ้าไม่ลดตรวจ:

- target shift
- causal mask
- LR/optimizer
- gradient None/zero/NaN
- architecture wiring

นี่คือ diagnostic test ไม่ใช่ final evaluation

## 19. Training Loop

~~~text
zero_grad
forward
loss
backward
clip gradients
optimizer step
scheduler/logging
~~~

accumulation:

~~~text
(loss / accumulation_steps).backward()
repeat microbatches
optimizer.step()
~~~

## 20. Checkpoint Contract

อย่างน้อย:

- model state
- optimizer state
- step
- model config
- tokenizer artifact/version
- data manifest
- RNG state หากต้องการ full resume fidelity

PyTorch docs แนะนำ state_dict-oriented save/load pattern สำหรับ weights

## 21. Validation

- held-out documents
- no optimizer update
- model.eval()
- same frozen tokenizer
- record mean loss/protocol

อย่า tune บน test ซ้ำแล้วเรียก test ว่า untouched

## 22. Generation

~~~text
prompt tokens
↓ forward
↓ last logits
↓ greedy/temperature/top-k/top-p
↓ append
↓ repeat
~~~

tiny local model คุณภาพจำกัดโดย data/model scale เป็น expected result

## 23. SFT

~~~text
instruction/chat examples
↓ template
↓ causal LM
↓ supervised token loss
~~~

production SFT มัก mask non-assistant regions ตาม policy/template

smoke capstone แค่พิสูจน์ pipeline ไม่ใช่สร้าง assistant คุณภาพสูง

## 24. Preference Optimization

preference pair:

~~~text
prompt, chosen, rejected
~~~

~~~text
Delta_policy =
log pi(chosen|x) - log pi(rejected|x)

Delta_ref =
log pi_ref(chosen|x) - log pi_ref(rejected|x)

L_DPO =
-log sigmoid(beta * (Delta_policy - Delta_ref))
~~~

reference policy frozen

## 25. Quantization

symmetric INT8:

~~~text
scale = max_abs / 127
q = clip(round(w/scale), -127, 127)
w_hat = q * scale
~~~

วัด reconstruction error + task quality + actual runtime latency

NumPy quantize/dequantize ไม่ได้ทำให้ inference เร็วขึ้นเอง ต้องมี kernel/runtime รองรับ

## 26. Safety Gate

ก่อน serve:

- input/output limits
- auth/rate limit where needed
- no arbitrary code execution
- secrets outside source
- privacy-aware logging
- dependency/model/data license audit
- tool permissions if agents added

## 27. Serving Contract

~~~json
POST /generate
{
  "prompt": "hello",
  "max_new_tokens": 32
}
~~~

response:

~~~json
{
  "text": "...",
  "model_version": "...",
  "latency_ms": 12.3
}
~~~

`src/serve.py` เป็น skeleton และ intentionally ไม่ load arbitrary untrusted model pickle

## 28. Monitoring

System:
- request rate
- latency p50/p95/p99
- errors
- CPU/RAM/GPU memory
- throughput

Model:
- input/output length
- stop reason
- sampled quality
- drift slices

Safety:
- rejected requests
- permission failures
- policy-trigger rates

## 29. Reproducibility Manifest

~~~yaml
experiment:
git_commit:
dataset_manifest:
tokenizer_fingerprint:
seed:
model_config:
parameter_count:
optimizer:
learning_rate:
batch_size:
sequence_length:
steps:
device:
dtype:
train_loss:
validation_loss:
notes:
~~~

## 30. Code in This Chapter

### Framework independent

`src/capstone_utils.py`:

- causal_windows
- model_weight_bytes
- adam_training_state_bytes
- kv_cache_bytes
- symmetric_int8_quantize/dequantize
- dpo_loss_from_log_ratios
- document_fingerprint
- manifest_fingerprint

### PyTorch smoke

`src/tiny_llm_capstone.py`:

- train Chapter 49 byte-BPE on train corpus
- held-out validation windows
- random-init decoder LM
- causal SDPA
- pretraining
- validation
- state_dict round-trip
- SFT smoke
- DPO-style preference step
- INT8 reconstruction
- bounded generation

**ไม่มี pretrained weights**

## 31. Experiment Matrix

| Experiment | Change | Measure |
|---|---|---|
| A | context 16 vs 32 | loss/memory |
| B | dim 32 vs 64 | loss/params |
| C | 1 vs 2 blocks | loss/params |
| D | LR | stability |
| E | BPE merges | bytes/token/loss |
| F | dtype | memory/speed |
| G | INT8 | reconstruction/task quality |
| H | before/after SFT | behavior |
| I | DPO beta | preference margin |

## 32. Debugging

Shapes:
- IDs [B,T]
- embeddings [B,T,D]
- q/k/v [B,H,T,Dh]
- logits [B,T,V]

Check:
- token dtype integer
- target range < vocab
- loss finite
- grad finite
- split leakage
- tokenizer mismatch
- GPU memory

## 33. Failure Cases

1. tokenizer fit on all splits
2. wrong target shift
3. missing causal mask
4. LR too high
5. seed not recorded
6. checkpoint missing tokenizer/config
7. SFT template mismatch
8. chosen/rejected reversed
9. DPO reference not frozen
10. quantization no quality check
11. API unlimited context/output
12. generated quality overclaimed
13. serving artifact differs from evaluated artifact
14. perplexity compared across tokenizers blindly
15. pretrained model used while claiming from-scratch

## 34. Performance

full attention grows roughly with O(T²D) in the score/value interaction regime

สำหรับ laptop:

1. reduce context
2. reduce batch
3. reduce dim/layers
4. accumulate gradients
5. use mixed precision only when supported/stable
6. measure VRAM
7. checkpoint/offload only when needed

## 35. Production Perspective

เพิ่ม distributed ingestion, immutable datasets, registries, distributed training, failure recovery, serving scheduler/batching/KV management, autoscaling, rollout/rollback, observability และ security review

## 36. Research Perspective

capstone เปิด research questions ด้าน data quality, tokenizer efficiency, architecture, optimizer, long context, alignment, quantization, inference-time compute, retrieval และ multimodal

เลือกหนึ่ง dimension แล้วใช้ Chapter 79 ทำ controlled research ต่อได้

## 37. Common Mistakes

1. เริ่ม model ใหญ่
2. ไม่ test tokenizer round-trip
3. leakage-prone split
4. target shift ผิด
5. mask ผิด
6. ไม่ overfit tiny batch
7. save weights แต่ไม่ save config/tokenizer
8. validation พังแต่ scale ต่อ
9. compare perplexity คนละ tokenizer
10. generation params ไม่บันทึก
11. SFT data ต่ำคุณภาพ
12. preference labels reversed
13. reference ไม่ frozen
14. quantization no re-eval
15. API no limits
16. no monitoring
17. pretrained primary model
18. overclaim local tiny result

## 38. Exercises / Final Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Final Mini Project](mini-project/README.md)

## 39. Interview Questions

1. tokenizer เป็นส่วนของ model artifact เพราะอะไร?
2. causal shift คืออะไร?
3. causal mask ป้องกัน leakage อย่างไร?
4. perplexity มีข้อจำกัดอะไร?
5. training memory มีอะไรบ้าง?
6. KV cache โตตามอะไร?
7. tiny-batch overfit test debug อะไร?
8. SFT vs pretraining?
9. DPO reference model ทำไม?
10. quantized file เล็กแต่ inference อาจไม่เร็วเพราะอะไร?

## 40. Checklist

- [ ] corpus/provenance
- [ ] dedup/split
- [ ] tokenizer train/round-trip
- [ ] causal windows
- [ ] decoder model/causal attention
- [ ] CE/perplexity
- [ ] memory estimates
- [ ] pretraining/validation
- [ ] checkpoint restore
- [ ] generation
- [ ] SFT
- [ ] preference optimization
- [ ] quantization analysis
- [ ] API/deployment
- [ ] monitoring/security
- [ ] reproducibility manifest/report

## 41. Summary

~~~text
data
→ representation
→ optimization
→ model
→ evaluation
→ alignment
→ compression
→ inference
→ deployment
→ monitoring
→ research iteration
~~~

## 42. What's Next

ไม่มี Chapter 81

อ่าน FINAL_CURRICULUM_AUDIT.md แล้วเลือก specialization และทำ Paper Reproduction + Final LLM Capstone ให้ผ่าน mastery gate
