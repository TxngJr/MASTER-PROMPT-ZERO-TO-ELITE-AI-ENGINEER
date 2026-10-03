# Chapter 20 — Optimizers: SGD, Momentum, AdaGrad, RMSProp, Adam, AdamW

## 1. Goal

Chapter 19 ทำให้เราได้ gradient:

~~~text
parameter.grad = dLoss/dParameter
~~~

แต่ gradient ยังไม่เปลี่ยน parameter จนกว่า optimizer จะทำ update

~~~text
forward
→ loss
→ backward
→ gradients
→ optimizer.step()
→ new parameters
~~~

## 2. Gradient Descent

full-batch gradient descent:

~~~text
θ ← θ - η ∇J(θ)
~~~

η คือ learning rate

ถ้า learning rate:
- เล็กเกิน → ช้ามาก
- ใหญ่เกิน → oscillate/diverge
- เหมาะ → loss ลดอย่างเสถียร

## 3. Batch / Mini-Batch / Stochastic

### Batch GD
ใช้ dataset ทั้งหมดต่อ update

### SGD แบบ strict
ใช้ sample เดียว

### Mini-batch SGD
ใช้ batch ขนาดเล็ก เช่น 32/64/256

ใน deep learning คำว่า SGD มักหมายถึง mini-batch SGD ด้วย

## 4. Why Mini-Batches

- vectorization
- memory manageable
- gradient noise ช่วย exploration บางส่วน
- update บ่อยกว่าฝึก full batch

batch size เป็น optimization/system hyperparameter

## 5. SGD

~~~text
θ_t
=
θ_(t-1)
-
η g_t
~~~

ข้อดี:
- simple
- low memory
- strong baseline

ข้อเสีย:
- noisy
- sensitive to learning rate
- ravines/curvature ทำให้ zig-zag

## 6. Momentum

velocity:

~~~text
v_t = β v_(t-1) + g_t
θ_t = θ_(t-1) - η v_t
~~~

intuition:
- gradients direction เดิมสะสม
- oscillating directions cancel บางส่วน
- accelerate along consistent descent direction

β มักอยู่ใกล้ 0.9 แต่ไม่ใช่กฎตายตัว

## 7. Nesterov Momentum

Nesterov ใช้ look-ahead-style correction

มีหลาย algebraically equivalent formulationsขึ้นกับ velocity convention

แนวคิดสำคัญ:
- ordinary momentum ใช้ accumulated velocity
- Nesterov ประเมิน/correct gradient โดยคำนึงถึงตำแหน่งที่ momentum กำลังพาไป

อย่าจำสูตรโดยไม่ดู implementation convention

## 8. AdaGrad

accumulate squared gradients:

~~~text
G_t = G_(t-1) + g_t²

θ_t =
θ_(t-1)
-
η g_t / (sqrt(G_t)+ε)
~~~

parameter ที่ gradient ใหญ่บ่อยจะได้ effective learning rate เล็กลง

ข้อเสีย:
G โต monotonically → learning rate อาจ decay จนเล็กเกิน

## 9. RMSProp

ใช้ exponential moving average แทน cumulative sum:

~~~text
v_t =
β v_(t-1)
+
(1-β) g_t²

θ_t =
θ_(t-1)
-
η g_t/(sqrt(v_t)+ε)
~~~

ช่วยแก้ AdaGrad learning-rate decay ที่แรงเกิน

## 10. Adam

รวม first moment + second moment:

~~~text
m_t = β1 m_(t-1) + (1-β1) g_t
v_t = β2 v_(t-1) + (1-β2) g_t²
~~~

bias correction:

~~~text
m_hat = m_t/(1-β1^t)
v_hat = v_t/(1-β2^t)
~~~

update:

~~~text
θ ← θ - η m_hat/(sqrt(v_hat)+ε)
~~~

## 11. Why Bias Correction

m และ v เริ่ม 0

ช่วงแรก exponential moving averages จึง biased toward zero

หารด้วย:
- 1-β1^t
- 1-β2^t

เพื่อแก้ initial bias

## 12. Epsilon

epsilon ไม่ใช่ random magic constant

มันป้องกัน denominator เล็ก/ศูนย์:

~~~text
sqrt(v_hat)+eps
~~~

ค่า epsilon ต่างกันอาจมีผลใน mixed precision หรือ gradients scale เล็กมาก

## 13. L2 Regularization vs Weight Decay

สำหรับ plain SGD ในบาง formulations:
L2 penalty และ multiplicative weight decayสัมพันธ์กันใกล้เคียง

แต่กับ adaptive optimizers เช่น Adam ความเท่ากันนี้ไม่ถือในแบบเดียวกัน

## 14. AdamW

AdamW แยก weight decay ออกจาก gradient-based adaptive update

concept:

~~~text
θ ← θ * (1 - η λ)
θ ← AdamUpdate(θ, g)
~~~

เรียกว่า decoupled weight decay

PyTorch มี AdamW แยก optimizer state/update และ weight-decay behavior โดยตรงใน current API

## 15. Weight Decay Exclusions

production networks มักพิจารณาไม่ decay:
- biases
- normalization scale/bias

แต่ policy นี้ architecture/framework-specific

tiny optimizer ในบท decay ทุก parameter ที่ส่งเข้า optimizer เพื่อให้ concept ชัด

## 16. zero_grad

gradients accumulate จาก backward

ดังนั้น training loop:

~~~text
optimizer.zero_grad()
loss.backward()
optimizer.step()
~~~

ถ้าลืม zero_grad:
gradient update จะรวม gradients จาก previous steps โดยไม่ตั้งใจ

gradient accumulation ทำได้ intentionally แต่ต้อง scale/plan ชัดเจน

## 17. Optimizer State Memory

SGD:
- params
- grads

Momentum:
- + one velocity tensor

Adam/AdamW:
- + first moment m
- + second moment v

ดังนั้น optimizer states มี memory cost สูงขึ้น

สำหรับ large LLMs state memory เป็นประเด็นสำคัญมาก

## 18. Learning-Rate Schedules Preview

optimizer ≠ scheduler

scheduler เปลี่ยน learning rate ตาม training step

ตัวอย่าง:
- step decay
- cosine decay
- warmup
- one-cycle

จะกลับมาใน framework/training chapters

## 19. Gradient Clipping Preview

ถ้า gradient norm ใหญ่มาก:

~~~text
g ← g * max_norm / ||g||
~~~

เมื่อ norm เกิน threshold

มีประโยชน์ใน RNN/large modelsบางกรณี

แต่ clipping ไม่ควรใช้เพื่อซ่อน bug

## 20. Parameter Groups Preview

framework optimizersรองรับ groups:
- different lr
- different weight decay

เช่น pretrained backbone vs new head

tiny optimizer ยัง intentionally simple

## 21. From Scratch

src/optimizers.py มี:
- SGD
- Momentum
- AdaGrad
- RMSProp
- Adam
- AdamW

ทุก optimizer รับ objects ที่มี:
- data ndarray
- grad ndarray
- zero_grad()

จึงใช้กับ Tensor จาก Chapter 19 ได้

## 22. Optimizer Debugging

ถ้า loss ไม่ลด:
- gradient zero?
- gradients finite?
- lr?
- zero_grad ถูกที่?
- parameter data updateจริงไหม?
- loss/activation stable?
- labels shape?
- initialization?

ถ้า loss NaN:
- check forward first
- inspect grad norm
- lower lr
- stable losses
- clipping only when justified

## 23. Common Mistakes

1. step before backward
2. forget zero_grad
3. learning rate too large
4. Adam = always best
5. AdamW = Adam + L2 gradient — ไม่ตรง concept
6. decay bias/norm blindly
7. compare optimizersด้วยคนละ training budget
8. ignore optimizer state memory
9. tune final test
10. clip gradientsโดยไม่หา root cause

## 24. Exercises

- derive one SGD update
- derive first Adam step bias correction
- plot quadratic optimization paths
- compare SGD vs Momentum
- compare AdaGrad vs RMSProp
- compare Adam vs AdamW with weight decay
- inspect state memory bytes
- add global gradient clipping

## 25. Mini Project

[Optimizer Landscape Lab](mini-project/README.md)

## 26. Exit Checklist

- [ ] SGD
- [ ] mini-batches
- [ ] Momentum
- [ ] Nesterov
- [ ] AdaGrad
- [ ] RMSProp
- [ ] Adam moments
- [ ] bias correction
- [ ] AdamW decoupled decay
- [ ] zero_grad
- [ ] optimizer state memory

## 27. Next

Chapter 21 จะลง activation/loss design และ numerical stability ก่อนนำ Tensor + optimizer + losses มาประกอบเป็น trainable tiny MLP
