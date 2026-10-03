# Chapter 21 — Activations & Loss Functions

## 1. Why This Chapter Matters

Activation และ loss ไม่ใช่แค่สูตรปลายทาง

มันกำหนด:
- geometry ของ model
- gradient flow
- numerical stability
- output interpretation
- optimization difficulty

เลือกผิดคู่:
- training ไม่ดี
- gradients vanish/explode
- probabilitiesผิด
- loss NaN

## 2. Hidden Activations vs Output Functions

Hidden activation:
- ReLU
- LeakyReLU
- ELU
- GELU
- SiLU/Swish
- Tanh

Output depends on task:

binary classification:
~~~text
one logit
→ sigmoid conceptually
→ BCE-with-logits loss
~~~

multiclass:
~~~text
K logits
→ softmax conceptually
→ cross-entropy-from-logits
~~~

regression:
~~~text
often linear output
~~~

## 3. ReLU

~~~text
ReLU(x)=max(0,x)
~~~

derivative:
~~~text
1 if x>0
0 if x<0
~~~

ข้อดี:
- cheap
- good gradient on positive side

ปัญหา:
- dead ReLU
- zero negative gradient

## 4. Leaky ReLU

~~~text
f(x)=x       if x>=0
f(x)=αx      if x<0
~~~

negative side ยังมี gradient เล็ก

## 5. ELU

~~~text
x                  if x>0
α(exp(x)-1)        otherwise
~~~

smooth-ish negative saturation แต่ต้องใช้ exp

## 6. GELU

concept:

~~~text
GELU(x)=x Φ(x)
~~~

deep transformersนิยมใช้ GELU variants

implementation มักใช้ exact/approximate forms

## 7. SiLU / Swish

~~~text
SiLU(x)=x sigmoid(x)
~~~

smooth non-monotonic behaviorบางช่วง

ใช้มากใน modern deep networks

## 8. Sigmoid

~~~text
σ(x)=1/(1+e^-x)
~~~

เหมาะ output binary probability parameterization

hidden sigmoid มี saturation:
- very positive → gradient ~0
- very negative → gradient ~0

## 9. Tanh

~~~text
tanh(x)
~~~

range (-1,1), zero-centered แต่ยัง saturate

## 10. Softmax

~~~text
p_k=exp(z_k)/Σ exp(z_j)
~~~

ต้อง subtract max:

~~~text
z_shift = z-max(z)
~~~

เพื่อลด overflow

## 11. Logits

logit = raw model score ก่อน probability transform

สำคัญมาก:

~~~text
model output logits
loss consumes logits directly
~~~

อย่า sigmoid/softmax ก่อน loss ถ้า API เป็น logits-aware

## 12. Regression Losses

### MSE

~~~text
mean((prediction-target)^2)
~~~

large errors ถูก penalize quadratic

### MAE

~~~text
mean(|prediction-target|)
~~~

robust กว่า MSE ต่อ extreme residuals แต่ derivativeไม่ smooth ที่ 0

### Huber

quadratic near zero, linear for large errors

combine smooth optimization + robustness

## 13. Binary Cross Entropy

probability-space formula:

~~~text
-[y log p + (1-y)log(1-p)]
~~~

แต่ถ้า p กลายเป็น 0 หรือ 1 exact:
- log(0)
- inf / NaN

## 14. BCE With Logits

stable formula ต่อ logit x:

~~~text
max(x,0)
-
x*y
+
log(1+exp(-|x|))
~~~

gradient:

~~~text
sigmoid(x)-y
~~~

หลัง mean reduction หารด้วย N

PyTorch current BCEWithLogitsLoss ก็รวม sigmoid และ BCE ใน operation เดียวเพื่อใช้ log-sum-exp-style numerical stability

## 15. Multiclass Cross Entropy

logits z shape:

~~~text
(N,C)
~~~

log-sum-exp:

~~~text
LSE(z)
=
m + log Σ exp(z-m)

m=max(z)
~~~

per-sample loss:

~~~text
-log softmax(z)[target]
~~~

gradient:

~~~text
softmax(z)-one_hot(target)
~~~

แล้วหาร batch sizeถ้า mean reduction

## 16. Label Smoothing

แทน one-hot target:

~~~text
true class = 1
others = 0
~~~

ใช้:

~~~text
target_distribution
=
(1-epsilon)*one_hot
+
epsilon/C
~~~

ช่วยลด overconfidence บาง setting

แต่:
- ไม่ใช่ calibration guarantee
- epsilon ต้อง validate
- task บางอย่างไม่เหมาะ

## 17. Class Weighting

imbalanced classification อาจ weight classes

แต่:
- changes training objective
- probability calibration อาจเปลี่ยน
- threshold ยังต้อง validate

## 18. Focal Loss Preview

ลด weight ของ easy examples เพื่อ focus hard examples

ใช้ใน dense detection/imbalanceบางแบบ

แต่ไม่ควรใช้แทนการเข้าใจ sampling/threshold/metric

## 19. Numerical Stability Rules

1. subtract max before softmax
2. prefer log-sum-exp
3. prefer logits-aware BCE/CE
4. avoid log(sigmoid(x)) naive สำหรับ extreme logits
5. inspect finite loss/gradients
6. use dtype appropriate to scale

## 20. Activation Gradient Flow

Sigmoid/Tanh:
- saturating regions

ReLU:
- positive gradients survive
- negative gradient zero

GELU/SiLU:
- smooth gates

activation choice interacts with:
- initialization
- normalization
- architecture
- optimizer

## 21. Loss/Task Pairing

binary single-label:
- one logit + BCEWithLogits

multiclass single-label:
- C logits + CrossEntropy

multi-label:
- C independent logits + BCEWithLogits

regression:
- linear output + MSE/MAE/Huber depending error model

## 22. From Scratch

src/activations_losses.py มี:
- stable sigmoid
- ReLU / LeakyReLU / ELU
- GELU approximation
- SiLU
- stable softmax
- MSE + gradient
- MAE + subgradient convention
- Huber + gradient
- BCE-with-logits + gradient
- multiclass cross-entropy + gradient
- label smoothing

## 23. Gradient Checking

loss gradient functionsต้อง compare finite differences

โดยเฉพาะ:
- BCE extreme logits
- CE logits
- Huber transition

## 24. Common Mistakes

1. sigmoid before BCEWithLogits
2. softmax before CrossEntropy
3. unstable exp on huge logits
4. log(0)
5. wrong target shape
6. binary vs multiclass confusion
7. MAE derivative at zero assumption unstated
8. label smoothing twice
9. class weights without metric review
10. activation chosen from trend rather than architecture/task

## 25. Exercises

- derive sigmoid derivative
- derive softmax Jacobian
- derive CE gradient
- implement logsumexp
- compare BCE probability vs logits stability
- compare MSE/MAE/Huber influence on outlier
- plot ReLU/GELU/SiLU derivatives
- test label smoothing

## 26. Mini Project

[Loss Stability Lab](mini-project/README.md)

## 27. Exit Checklist

- [ ] logits
- [ ] stable sigmoid
- [ ] stable softmax
- [ ] ReLU family
- [ ] GELU
- [ ] SiLU
- [ ] MSE/MAE/Huber
- [ ] BCE-with-logits
- [ ] multiclass CE
- [ ] label smoothing
- [ ] task/loss pairing

## 28. Next

หลัง Chapter 21 เราจะรวม Tensor + optimizer + stable loss เป็น tiny trainable MLP ก่อนเข้าสู่ Chapter 22 PyTorch
