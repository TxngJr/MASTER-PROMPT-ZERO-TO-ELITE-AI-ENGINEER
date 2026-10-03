# Chapter 05 — Machine Learning Fundamentals

## 1. Why This Matters

Machine Learning ไม่ได้หมายถึง “เรียก `.fit()`” แต่คือการกำหนด:

1. data
2. hypothesis/model family
3. objective/loss
4. optimization procedure
5. evaluation protocol
6. assumptions

แล้วถามว่า model ที่เรียนจากข้อมูลหนึ่งชุดจะ **generalize** ไปยังข้อมูลที่ไม่เคยเห็นได้หรือไม่

## 2. Prerequisites

- Chapter 02: functions, derivatives, gradient, probability/statistics
- Chapter 04: train/validation/test, leakage, preprocessing
- NumPy vectorization

## 3. Learning Objectives

By the end of this chapter you can:

- แยก supervised / unsupervised / semi-supervised / self-supervised
- แยก parameter กับ hyperparameter
- อธิบาย loss, objective, metric
- คำนวณ MSE, MAE, R²
- อธิบาย empirical risk minimization
- อธิบาย gradient descent
- แยก underfitting / overfitting
- อธิบาย bias–variance intuition
- ใช้ regularization เป็นแนวคิด
- สร้าง baseline
- ใช้ validation และ cross-validation อย่างถูกต้อง
- ออกแบบ experiment ที่ reproducible

## 4. Mental Model

```text
training data
     ↓
 model family f(x; θ)
     ↓
 predictions
     ↓
 loss(y, ŷ)
     ↓
 optimizer changes θ
     ↺
```

หลัง training:

```text
unseen data
   ↓
same preprocessing
   ↓
f(x; learned θ)
   ↓
prediction
   ↓
metric
```

## 5. What Is Learning?

ให้ dataset:

```text
D = {(x₁,y₁),...,(xₙ,yₙ)}
```

model:

```text
ŷ = f(x;θ)
```

training คือหาค่า parameter `θ` ที่ทำให้ objective ต่ำลงบน training data โดยหวังว่า pattern ที่ได้ generalize

## 6. Parameters vs Hyperparameters

**Parameters** เรียนจากข้อมูล:

- regression weights
- neural network weights

**Hyperparameters** เรากำหนด/เลือก:

- learning rate
- polynomial degree
- regularization strength
- tree depth

validation set ใช้ช่วยเลือก hyperparameters

## 7. Learning Paradigms

### Supervised

มี target `y`

- regression
- classification

### Unsupervised

ไม่มี target โดยตรง

- clustering
- dimensionality reduction

### Semi-supervised

มี labeled data บางส่วน + unlabeled จำนวนมาก

### Self-supervised

สร้าง training signal จาก structure ของข้อมูลเอง เช่น masked-token prediction / next-token prediction

## 8. Loss vs Metric vs Objective

### Loss

ค่าที่วัด error ต่อ sample หรือ aggregate และมักใช้ optimize

### Metric

ค่าที่ใช้ตีความ performance เช่น MAE, accuracy, F1

### Objective

สิ่งที่ optimizer minimize/maximize อาจเป็น:

```text
data loss + regularization penalty
```

metric ไม่จำเป็นต้อง differentiable

## 9. Mean Squared Error

```text
MSE = (1/n) Σ(yᵢ-ŷᵢ)²
```

squared error ลงโทษ residual ใหญ่แรงขึ้น

## 10. Mean Absolute Error

```text
MAE = (1/n) Σ|yᵢ-ŷᵢ|
```

MAE ทน extreme residual มากกว่า MSE ในเชิง loss shape

## 11. R²

```text
R² = 1 - SS_res / SS_tot

SS_res = Σ(yᵢ-ŷᵢ)²
SS_tot = Σ(yᵢ-ȳ)²
```

interpretation:

- 1 = perfect predictions
- 0 = เทียบเท่า mean baseline ภายใต้นิยามทั่วไป
- negative = แย่กว่า mean baseline ได้

อย่าคิดว่า R² ต้องอยู่ 0..1 เสมอ

## 12. Empirical Risk

ถ้า loss ต่อ sample คือ `L`:

```text
R_emp(θ) = (1/n) Σ L(yᵢ, f(xᵢ;θ))
```

training ส่วนใหญ่ optimize quantity ที่ประมาณจาก finite training sample นี้

แต่สิ่งที่เราอยากได้จริงคือ low expected error บน distribution ที่ deployment จะเจอ

## 13. Gradient Descent

ถ้า objective `J(θ)`:

```text
θ_{t+1} = θ_t - η ∇J(θ_t)
```

- `η` = learning rate
- gradient = local direction of steepest increase
- minus gradient = local descent direction

learning rate:

- เล็กเกิน → ช้า
- ใหญ่เกิน → overshoot/diverge

## 14. Underfitting

training error สูง และ validation error สูง

สาเหตุได้แก่:

- model ง่ายเกิน
- feature ไม่พอ
- optimization ยังไม่ดี
- regularization แรงเกิน

## 15. Overfitting

training error ต่ำ แต่ validation/test error แย่

model เรียนรายละเอียด/ noise ของ training sample มากเกินไป

วิธีลด:

- data มากขึ้น/ดีขึ้น
- model complexity ต่ำลง
- regularization
- early stopping
- augmentation (บาง domain)
- better validation protocol

## 16. Bias–Variance Intuition

**High bias**: model family จำกัดเกิน → systematic error  
**High variance**: model sensitive ต่อ training sample มาก → performance แกว่ง

นี่เป็น conceptual decomposition; ในงานจริง noise, distribution shift และ optimization ก็มีบทบาทด้วย

## 17. Regularization

เพิ่ม preference ต่อ solution บางแบบ

ตัวอย่าง:

```text
J(θ) = data_loss + λ penalty(θ)
```

`λ` ใหญ่ → regularization แรง

Chapter 06:
- Ridge = L2
- Lasso = L1

## 18. Baselines

ก่อน model ซับซ้อน ให้ baseline

Regression:
- predict training mean
- predict training median

Classification:
- majority class
- stratified/random baseline เมื่อเหมาะสม

ถ้า model แพ้ baseline ต้อง debug ก่อน scale

## 19. Validation

Training set ใช้ fit parameter  
Validation ใช้เลือก model/hyperparameter  
Test ใช้ final estimate

ถ้าเปิดดู test ซ้ำแล้วปรับ model ตาม test คุณกำลัง overfit test set ทางอ้อม

## 20. Cross-Validation

K-fold:

```text
Fold 1: VALID | TRAIN TRAIN TRAIN
Fold 2: TRAIN | VALID TRAIN TRAIN
Fold 3: TRAIN TRAIN | VALID TRAIN
Fold 4: TRAIN TRAIN TRAIN | VALID
```

train/evaluate หลายรอบแล้วสรุป distribution ของ metric

preprocessing ต้อง fit ใหม่ภายในแต่ละ training fold

## 21. Reproducibility

บันทึก:

- dataset version
- split seed
- code commit
- package versions
- hyperparameters
- metrics
- hardware เมื่อ relevant

Random seed ช่วย reproduce random sequence แต่ไม่ได้รับประกัน bitwise determinism ของทุก hardware/library operation

## 22. Experiment Design

หนึ่ง experiment ควรตอบคำถามชัด:

> เพิ่ม polynomial degree จาก 1 เป็น 3 ช่วย validation MSE หรือไม่?

ไม่ควรเปลี่ยนพร้อมกัน 8 อย่างแล้วสรุปว่าอะไรเป็นสาเหตุ

## 23. Distribution Shift

Train/test จาก distribution คล้ายกันยังไม่พอถ้า production เปลี่ยน

ตัวอย่าง:

- user population เปลี่ยน
- sensor เปลี่ยน
- economic regime เปลี่ยน
- language/domain เปลี่ยน

นี่คือเหตุผลที่ production monitoring สำคัญ

## 24. Metrics Can Mislead

metric เดียวอาจซ่อน:

- subgroup failures
- asymmetric costs
- outliers
- temporal degradation
- calibration problems

ต้องเลือก metric ตามงาน

## 25. From Scratch

[src/evaluation.py](src/evaluation.py) มี:

- MSE
- MAE
- R²
- mean baseline
- deterministic K-fold indices

เขียนเองเพื่อเข้าใจนิยามก่อนใช้ sklearn

## 26. Common Mistakes

1. optimize test set
2. ไม่มี baseline
3. report train score อย่างเดียว
4. preprocessing นอก CV
5. metric ไม่สัมพันธ์กับ business/scientific goal
6. seed ไม่บันทึก
7. เปลี่ยนหลาย hyperparameters พร้อมกันแล้วหาเหตุผลมั่ว
8. สรุป causation จาก predictive model
9. model ใหญ่ขึ้นทันทีเมื่อ score ต่ำ
10. ignore subgroup errors
11. ใช้ R² เป็นเปอร์เซ็นต์ความแม่นยำ
12. คิด loss ต่ำ = deployment ดีเสมอ

## 27. Exercises / Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 28. Checklist

- [ ] loss vs metric vs objective
- [ ] parameter vs hyperparameter
- [ ] MSE/MAE/R² ด้วยมือ
- [ ] gradient descent intuition
- [ ] under/overfitting
- [ ] bias/variance
- [ ] baseline
- [ ] K-fold
- [ ] reproducible experiment
- [ ] distribution shift

## 29. What's Next

Chapter 06 จะใช้ทุก concept นี้กับ Regression และ derive linear regression จากสมการไปถึง implementation จริง
