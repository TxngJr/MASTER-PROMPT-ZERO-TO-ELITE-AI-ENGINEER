# Chapter 07 — Logistic Regression & Classification Fundamentals

## 1. Why This Matters

Regression ทำนายค่าต่อเนื่อง แต่ classification ต้องตัดสิน class และมักต้องการ probability

Logistic Regression เป็นสะพานสำคัญจาก linear model ไปสู่ probabilistic classification:

```text
x
 ↓
linear score z = Xw + b
 ↓
sigmoid
 ↓
P(y=1|x)
 ↓
threshold
 ↓
class prediction
```

## 2. Prerequisites

- Linear Regression
- dot product / matrix multiplication
- derivative + chain rule
- probability
- train/validation/test
- standardization

## 3. Learning Objectives

เมื่อจบบทนี้คุณควรทำได้:

- อธิบาย binary classification
- แยก score, logit, probability และ class
- derive sigmoid derivative
- derive binary cross-entropy / log loss
- derive gradient ของ logistic regression
- implement binary logistic regression จากศูนย์
- อธิบาย decision threshold
- คำนวณ confusion matrix
- คำนวณ precision / recall / specificity / F1
- อธิบาย ROC-AUC และ PR-AUC ในระดับ foundation
- จัดการ class imbalance เบื้องต้น
- อธิบาย multiclass logistic regression / softmax ในระดับแนวคิด
- ใช้ scikit-learn แบบ leakage-safe

## 4. Binary Classification

ให้ target:

```text
y ∈ {0,1}
```

เช่น:

- spam / not spam
- disease / no disease
- fraud / legitimate

Linear score:

```text
z = wᵀx + b
```

ค่า z อยู่ได้ตั้งแต่ `-∞` ถึง `+∞`

เราต้อง map ไปช่วง `(0,1)`

## 5. Sigmoid

```text
σ(z) = 1 / (1 + e^(-z))
```

properties:

```text
z → +∞  => σ(z) → 1
z = 0   => σ(z) = 0.5
z → -∞  => σ(z) → 0
```

ดังนั้น:

```text
p = P(y=1|x) = σ(wᵀx+b)
```

ภายใต้ model assumptions

## 6. Logit

ถ้า:

```text
p = σ(z)
```

แก้กลับ:

```text
log(p/(1-p)) = z
```

`p/(1-p)` คือ odds และ log ของ odds คือ **logit**

Logistic Regression จึงเป็น linear model ใน log-odds space

## 7. Why Not MSE?

ใช้ MSE กับ sigmoid ได้ในเชิงคณิตศาสตร์ แต่ logistic regression ตาม probabilistic maximum-likelihood formulation นำไปสู่ log loss ที่มี optimization properties เหมาะสมกว่า

สำหรับ Bernoulli target:

```text
P(y|p)=p^y(1-p)^(1-y)
```

Likelihood หลาย samples:

```text
L = Π p_i^y_i (1-p_i)^(1-y_i)
```

log-likelihood:

```text
log L = Σ[y_i log p_i + (1-y_i)log(1-p_i)]
```

เราต้อง maximize likelihood เท่ากับ minimize negative log-likelihood

## 8. Binary Cross-Entropy / Log Loss

```text
J = -(1/n) Σ [
  y_i log(p_i)
  + (1-y_i) log(1-p_i)
]
```

ถ้า model มั่นใจผิดมาก loss จะสูงมาก

## 9. Sigmoid Derivative

```text
σ(z)=1/(1+e^-z)
```

ผล derivative ที่สำคัญ:

```text
σ'(z)=σ(z)(1-σ(z))
```

ให้ `p=σ(z)`:

```text
dp/dz = p(1-p)
```

## 10. Gradient of Logistic Regression

เมื่อรวม sigmoid + BCE แล้ว gradient simplify สวยมาก:

```text
∇w J = (1/n)Xᵀ(p-y)
∂J/∂b = (1/n)Σ(p-y)
```

นี่คล้าย linear regression gradient แต่ residual คือ `p-y`

## 11. Gradient Descent

```text
w ← w - η∇wJ
b ← b - η∂J/∂b
```

ต้องตรวจ:

- loss ลดหรือไม่
- probabilities finite ไหม
- features scale เหมาะหรือไม่

## 12. Numerical Stability

คำนวณ sigmoid แบบ naive:

```python
1 / (1 + np.exp(-z))
```

เมื่อ |z| ใหญ่มากอาจ overflow

implementation ในบทใช้ stable branch และ clip probabilities ก่อน log

## 13. Decision Threshold

default มักใช้:

```text
p >= 0.5 → class 1
```

แต่ 0.5 ไม่ใช่กฎธรรมชาติ

ถ้า false negative แพงมาก อาจลด threshold เพื่อเพิ่ม recall

ถ้า false positive แพงมาก อาจเพิ่ม threshold เพื่อเพิ่ม precision/specificity

threshold ต้องเลือกบน validation data ไม่ใช่ test

## 14. Confusion Matrix

```text
                 Predicted
              0             1
Actual 0      TN            FP
Actual 1      FN            TP
```

จาก 4 ค่านี้สร้าง metrics ได้หลายชนิด

## 15. Accuracy

```text
Accuracy = (TP+TN)/(TP+TN+FP+FN)
```

ถ้า dataset imbalance มาก accuracy อาจหลอก

## 16. Precision

```text
Precision = TP/(TP+FP)
```

คำถาม:

> ในสิ่งที่ model บอกว่า positive มีกี่ส่วนที่ positive จริง?

## 17. Recall / Sensitivity

```text
Recall = TP/(TP+FN)
```

คำถาม:

> positive จริงทั้งหมด model จับได้กี่ส่วน?

## 18. Specificity

```text
Specificity = TN/(TN+FP)
```

## 19. F1

```text
F1 = 2PR/(P+R)
```

เป็น harmonic mean ของ precision และ recall

อย่าใช้ F1 โดยอัตโนมัติ ต้องดูว่าโจทย์ต้องการ trade-off แบบใด

## 20. ROC Curve

เปลี่ยน threshold แล้ว plot:

```text
x = False Positive Rate
y = True Positive Rate
```

ROC-AUC สรุป ranking ability ข้าม thresholds

ใน class imbalance มาก PR curve มักช่วยเห็น precision-recall trade-off ชัดกว่า

## 21. Probability Calibration

Probability 0.8 ที่ calibrate ดีควรหมายถึงกลุ่ม prediction ใกล้ 0.8 มี positive จริงประมาณ 80% ใน setting ที่เหมาะสม

classification accuracy ดีไม่ได้แปลว่า probability calibrated

จะกลับมาใน evaluation ขั้นสูง

## 22. Class Imbalance

ตัวเลือก:

- stratified split
- class-weighted objective
- threshold tuning
- over/under-sampling
- metric ที่เหมาะสม

อย่าทำ resampling ก่อน split เพราะอาจ leakage

## 23. Multiclass

ถ้ามี K classes สามารถใช้:

- One-vs-Rest
- Multinomial Logistic Regression

Multinomial version ใช้ softmax:

```text
P(y=k|x)=exp(z_k)/Σ_j exp(z_j)
```

จะ derive softmax/cross-entropy ลึกอีกครั้งใน neural networks

## 24. Regularization

Logistic Regression ใน production มัก regularized

L2:

```text
J_total = log_loss + λ||w||²
```

scikit-learn `LogisticRegression` ใช้ regularization โดย default และ parameter `C` เป็น inverse regularization strength

## 25. Implementation

ดู:

- [src/logistic_regression.py](src/logistic_regression.py)
- [src/classification_metrics.py](src/classification_metrics.py)

จากศูนย์:

- stable sigmoid
- binary log loss
- LogisticRegressionGD
- confusion matrix
- precision/recall/F1

## 26. scikit-learn

สำหรับ scaled numeric data:

```python
Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression())
])
```

scikit-learn 1.9 รองรับ multinomial classification ใน solvers หลัก และ regularization ถูกเปิดโดย default

## 27. Failure Cases

- decision boundary nonlinear มาก
- classes overlap สูง
- extreme outliers
- perfect/quasi separation
- severe imbalance
- wrong threshold
- leakage
- poor probability calibration
- distribution shift

## 28. Debugging

ถ้า loss = NaN:

- inspect logits
- stable sigmoid
- clip log arguments
- check NaN in X/y
- reduce learning rate
- scale features

ถ้า accuracy สูงแต่ recall ต่ำ:

- inspect confusion matrix
- class balance
- threshold
- metric choice

## 29. Common Mistakes

1. ใช้ accuracy metric เดียวกับ imbalance
2. tune threshold บน test
3. เรียก sigmoid output ว่า calibrated probability เสมอ
4. ไม่ scale features ก่อน iterative solver
5. ใช้ class weights ก่อนเข้าใจ base rate
6. resample ก่อน split
7. log(0) เพราะไม่ clip
8. threshold 0.5 แบบไม่คิด cost
9. ตีความ coefficient เป็น causality
10. target leakage
11. compare F1 คนละ positive label
12. ใช้ precision/recall โดยไม่รู้ denominator

## 30. Exercises & Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 31. Checklist

- [ ] sigmoid / logit
- [ ] derive BCE จาก likelihood
- [ ] derive gradient
- [ ] stable implementation
- [ ] confusion matrix
- [ ] precision/recall/F1
- [ ] threshold tuning
- [ ] imbalance reasoning
- [ ] sklearn pipeline

## 32. What's Next

Chapter 08 จะเปลี่ยนจาก parametric linear decision boundary ไปเป็น instance-based learning: **K-Nearest Neighbors**
