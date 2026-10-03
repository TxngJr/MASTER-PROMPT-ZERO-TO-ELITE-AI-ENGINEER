# Batch 07 Integration Project — Tiny Deep Learning Framework

รวม Chapters 19–21:

~~~text
NumPy data
   ↓
Tensor graph
   ↓
Linear → Tanh → Linear
   ↓
logits
   ↓
stable BCE-with-logits
   ↓
dLoss/dLogits
   ↓
Tensor.backward()
   ↓
parameter gradients
   ↓
AdamW.step()
   ↓
learned nonlinear classifier
~~~

## Goal

พิสูจน์ว่าคุณเข้าใจ training loop โดยไม่พึ่ง PyTorch autograd

framework components ถูกดึงจาก:
- Chapter 19 Tensor/autodiff
- Chapter 20 optimizer
- Chapter 21 stable loss

## Run

~~~bash
source .venv/bin/activate
python -m pip install -r requirements-batch07.txt

python integration-project-batch07/src/tiny_dl_lab.py   --output-dir reports/batch07   --seed 42
~~~

## Dataset

ใช้ nonlinear two-moons classification

เหตุผล:
single linear boundaryไม่พอ จึงบังคับให้ hidden nonlinearityมีความหมาย

## Architecture

~~~text
2
↓
Linear(2,16)
↓
Tanh
↓
Linear(16,1)
↓
logit
~~~

## Training

~~~text
zero_grad
forward
stable BCE-with-logits
backward from dL/dlogit
AdamW step
repeat
~~~

## Why Loss Is External To Tensor Graph?

Chapter 21 สร้าง stable fused BCE-with-logits พร้อม analytic gradient

lab นี้จึง:
1. compute stable scalar loss
2. compute exact dLoss/dLogits
3. seed Tensor.backward with that upstream gradient

นี่แสดงให้เห็นว่า custom differentiable operation เชื่อมกับ reverse-mode engineอย่างไร

## Required Extensions

1. implement BCEWithLogits as a native Tensor operation
2. compare SGD/Momentum/Adam/AdamW
3. replace Tanh with ReLU/GELU/SiLU
4. add mini-batches
5. add shuffling
6. add gradient clipping
7. add cosine LR schedule
8. multiclass softmax/CE
9. gradient-norm logging
10. parameter checkpoint save/load

## Mastery Questions

- ทำไม backward ต้องเริ่มจาก dLoss/dLogits?
- ทำไม parameter gradients ต้อง zero ก่อน step ใหม่?
- AdamW state มีอะไร?
- ทำไม fused BCE-with-logits stable กว่า sigmoid+log?
- ถ้าเอา Tanh ออก network นี้กลายเป็นอะไร?

## Exit Gate

ก่อน PyTorch:
- trace forward shapes
- explain every gradient path
- gradient-check primitives
- optimizer updates verified
- tiny nonlinear classifier learns
