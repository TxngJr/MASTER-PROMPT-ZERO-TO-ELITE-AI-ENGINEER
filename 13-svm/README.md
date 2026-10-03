# Chapter 13 — Support Vector Machines (SVM)

## 1. Why This Matters

SVM ทำให้ geometry ของ classification ชัดมาก:

```text
data points
   ↓
separating hyperplane
   ↓
maximize margin
   ↓
support vectors define boundary
```

SVM เชื่อม Linear Algebra, Geometry, Optimization และ Kernel Methods เข้าด้วยกัน

## 2. Prerequisites

- vectors / dot products
- linear decision boundaries
- optimization / regularization
- classification metrics
- scaling
- train / validation / test

## 3. Learning Objectives

เมื่อจบบทนี้คุณควร:

- อธิบาย hyperplane และ signed distance
- อธิบาย margin
- derive hard-margin intuition
- อธิบาย soft margin และ slack
- เข้าใจ hinge loss
- implement linear SVM ด้วย subgradient descent
- อธิบาย role ของ C
- อธิบาย support vectors
- เข้าใจ kernel trick
- อธิบาย linear / polynomial / RBF kernels
- tune C และ gamma บน validation/CV
- อธิบาย scaling importance
- compare LinearSVC vs SVC conceptually

## 4. Hyperplane

binary linear classifier:

```text
f(x)=wᵀx+b
```

prediction:

```text
sign(f(x))
```

decision boundary:

```text
wᵀx+b=0
```

vector `w` ตั้งฉากกับ hyperplane

## 5. Distance to Hyperplane

signed distance:

```text
(wᵀx+b) / ||w||
```

absolute distance:

```text
|wᵀx+b| / ||w||
```

## 6. Margin

ใช้ labels:

```text
y∈{-1,+1}
```

constraint สำหรับ separable case:

```text
y_i(wᵀx_i+b) >= 1
```

support hyperplanes:

```text
wᵀx+b=+1
wᵀx+b=-1
```

ระยะระหว่างสองเส้น:

```text
2 / ||w||
```

maximize margin = minimize `||w||`

## 7. Hard-Margin Objective

```text
min 1/2 ||w||²

subject to
y_i(wᵀx_i+b) >= 1
```

ใช้ได้เมื่อข้อมูล separable แบบเหมาะสม

ในโลกจริงมักมี noise/outliers จึงต้อง soft margin

## 8. Soft Margin

เพิ่ม slack variables:

```text
ξ_i >= 0
```

constraint:

```text
y_i(wᵀx_i+b) >= 1-ξ_i
```

objective:

```text
min 1/2||w||² + C Σ ξ_i
```

C ใหญ่:
- penalize violations มาก
- fit training เข้มขึ้น
- regularization อ่อนลงใน common SVM formulation

C เล็ก:
- ยอม margin violations มากขึ้น
- regularization แรงขึ้น

## 9. Hinge Loss

```text
L_i = max(0, 1-y_i f(x_i))
```

ถ้า:

```text
y_i f(x_i) >= 1
```

sample อยู่นอก/บน margin ถูกด้าน → hinge loss 0

ถ้าอยู่ใน marginหรือผิดด้าน → positive loss

## 10. Primal Objective

educational linear objective:

```text
J(w,b)
=
1/2 ||w||²
+
C/n Σ max(0,1-y_i(wᵀx_i+b))
```

## 11. Subgradient

สำหรับ sample ที่:

```text
y_i f(x_i) < 1
```

contribution ต่อ gradient ของ hinge term:

```text
-w.r.t. w: -y_i x_i
-w.r.t. b: -y_i
```

ถ้า margin satisfied:

```text
0
```

รวม regularization gradient:

```text
∇w = w + C/n Σ_active (-y_i x_i)
```

## 12. Support Vectors

support vectors คือ points ที่กำหนด margin/boundary

ใน kernel SVM final decision function พึ่ง subset ของ training points ผ่าน dual coefficients

points ไกล boundary จำนวนมากไม่มีผลโดยตรงต่อ optimum แบบเดียวกับ support vectors

## 13. Why Scaling Matters

SVM ใช้ dot products / distances ใน feature space

ถ้า feature scales ต่างกันมาก:
- geometry ถูก feature scale ใหญ่ dominate
- C/gamma tuning แปลก
- optimization แย่ลง

ดังนั้น:

```text
StandardScaler → SVM
```

ควรอยู่ใน Pipeline เพื่อป้องกัน leakage

## 14. Kernel Trick

linear model ใน transformed feature space:

```text
φ(x)
```

แต่แทนคำนวณ `φ(x)ᵀφ(z)` ตรง ๆ ใช้:

```text
K(x,z)
```

ถ้า kernel สอดคล้องกับ inner product ใน feature space ที่เหมาะสม เราสามารถสร้าง nonlinear boundary โดยไม่ explicit feature expansion

## 15. Linear Kernel

```text
K(x,z)=xᵀz
```

เหมาะเมื่อ boundary approximately linear หรือ feature dimension สูง

## 16. Polynomial Kernel

```text
K(x,z)=(γxᵀz+r)^d
```

สร้าง interaction nonlinear ตาม degree

degree สูงอาจ overfit และ scale sensitive

## 17. RBF Kernel

```text
K(x,z)=exp(-γ||x-z||²)
```

gamma:
- สูง → influence local มาก → boundary ซับซ้อน
- ต่ำ → influence กว้าง → smoother boundary

interaction:

```text
C ↔ gamma
```

ต้อง tune ร่วมกัน

## 18. LinearSVC vs SVC

### LinearSVC
- linear only
- optimized สำหรับ linear SVM
- scale ได้ดีกว่า kernel SVC บน large/high-dimensional data
- API/solver formulation ต่างจาก SVC(kernel="linear") บางส่วน

### SVC
- kernelized
- linear/poly/RBF/custom kernels
- training cost โตเร็วกับ sample count

## 19. Multiclass

SVM binary โดยธรรมชาติ แต่ libraries ใช้ decomposition strategies เช่น:
- one-vs-rest
- one-vs-one

scikit-learn SVC ใช้ one-vs-one internally

## 20. Probability Estimates

SVC `probability=True` ต้องทำ calibration-like additional fitting และทำให้ fit ช้าลง

อย่าเปิดเพียงเพราะอยากได้เลข probability โดยไม่รู้ว่าใช้ทำอะไร

## 21. From Scratch

[src/linear_svm.py](src/linear_svm.py)

มี:

- hinge loss
- linear SVM subgradient training
- decision function
- binary prediction

เป้าหมายคือเข้าใจ primal geometry ไม่ใช่เขียน SMO production solver

## 22. scikit-learn

linear:

```python
Pipeline([
    ("scale", StandardScaler()),
    ("model", LinearSVC(C=1.0))
])
```

RBF:

```python
Pipeline([
    ("scale", StandardScaler()),
    ("model", SVC(C=1.0, gamma="scale", kernel="rbf"))
])
```

## 23. Complexity

Kernel SVC:
- memory/time สามารถโตเร็วกับ n_samples
- prediction cost พึ่งจำนวน support vectors

LinearSVC:
- เหมาะ large sparse/high-dimensional problems มากกว่าในหลายกรณี

## 24. Failure Cases

- unscaled features
- huge dataset + kernel SVC
- noisy labels + C สูงมาก
- wrong gamma
- class imbalance
- leakage
- overlapping classes
- probability misuse

## 25. Debugging

ถ้า model ทาย class เดียว:
- check labels
- scaling
- C
- imbalance

ถ้า RBF overfit:
- lower gamma
- lower C
- inspect CV

ถ้า fit ช้ามาก:
- sample count?
- kernel?
- switch LinearSVC / approximate kernels

## 26. Common Mistakes

1. ไม่ scale
2. C สูง = regularization สูง — ผิดสำหรับ common SVM C interpretation
3. tune C/gamma บน test
4. kernel SVC กับ dataset ใหญ่มากโดยไม่ benchmark
5. probability=True โดยไม่จำเป็น
6. support vector = outlier เสมอ
7. coefficient = causality
8. compare models คนละ split
9. ignore imbalance
10. kernel trick = free unlimited complexity โดยไม่มี computational cost

## 27. Exercises / Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 28. Checklist

- [ ] hyperplane
- [ ] margin
- [ ] hard vs soft margin
- [ ] hinge loss
- [ ] C
- [ ] subgradient
- [ ] support vectors
- [ ] kernels
- [ ] RBF gamma
- [ ] scaling
- [ ] LinearSVC vs SVC

## 29. What's Next

Chapter 14 เปลี่ยนจาก supervised classification ไปเป็น unsupervised learning: **Clustering**
