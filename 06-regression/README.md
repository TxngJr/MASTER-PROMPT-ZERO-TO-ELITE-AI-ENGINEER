# Chapter 06 — Regression: Linear, Polynomial, Ridge, Lasso

## 1. Why This Matters

Regression เป็นจุดแรกที่ทุก foundation มารวมกัน:

```text
data
+ vectors/matrices
+ calculus
+ loss
+ optimization
+ regularization
+ evaluation
= trainable model
```

ถ้าเข้าใจ regression จากศูนย์ การเข้าใจ neural network จะง่ายขึ้น เพราะ linear layer ก็ยังเป็น `Xw+b`

## 2. Prerequisites

- matrix multiplication
- derivative / gradient
- MSE / R²
- train/validation/test
- standardization
- NumPy

## 3. Learning Objectives

By the end of this chapter you can:

- derive simple linear regression gradients
- เขียน multivariate prediction `Xw+b`
- fit OLS ด้วย least-squares solution
- fit linear regression ด้วย gradient descent
- สร้าง polynomial features
- อธิบาย underfit/overfit จาก polynomial degree
- derive Ridge objective
- อธิบาย L1 vs L2
- เข้าใจ Lasso sparsity intuition
- อธิบาย Elastic Net
- evaluate regression ด้วย MSE/MAE/RMSE/R²
- compare from-scratch implementation กับ scikit-learn
- ใช้ Pipeline เพื่อกัน preprocessing leakage

## 4. Mental Model

```text
features X
   ↓
linear combination
Xw + b
   ↓
ŷ
   ↓ compare with y
residual = ŷ-y
   ↓
loss
   ↓
gradient / solver
   ↓
update w,b
```

## 5. Simple Linear Regression

หนึ่ง feature:

```text
ŷ = wx + b
```

- `w` = slope
- `b` = intercept

Residual:

```text
eᵢ = ŷᵢ-yᵢ
```

MSE:

```text
J(w,b) = (1/n) Σ (wxᵢ+b-yᵢ)²
```

## 6. Derive Gradients

ให้:

```text
eᵢ = wxᵢ+b-yᵢ
J = (1/n)Σeᵢ²
```

ตาม chain rule:

```text
∂J/∂w
= (1/n)Σ 2eᵢ ∂eᵢ/∂w
= (2/n)Σ eᵢxᵢ
```

และ:

```text
∂J/∂b
= (2/n)Σ eᵢ
```

gradient descent:

```text
w ← w - η ∂J/∂w
b ← b - η ∂J/∂b
```

## 7. Multivariate Linear Regression

สำหรับ:

```text
X ∈ R^(n×d)
w ∈ R^d
b ∈ R
```

prediction:

```text
ŷ = Xw + b
```

shape:

```text
(n×d)(d,) → (n,)
```

MSE gradient:

```text
∇w J = (2/n) Xᵀ(Xw+b-y)
∂J/∂b = (2/n) Σ(Xw+b-y)
```

นี่คือ vectorized version ของ scalar derivation

## 8. Least Squares Solution

ถ้าใส่ intercept เข้า design matrix:

```text
X̃ = [1  X]
θ = [b, w₁,...,w_d]^T
```

objective:

```text
min ||X̃θ-y||²
```

ตั้ง gradient เป็นศูนย์:

```text
X̃ᵀX̃θ = X̃ᵀy
```

textbook form:

```text
θ = (X̃ᵀX̃)⁻¹X̃ᵀy
```

แต่ numerical code ไม่ควรสร้าง inverse ตรง ๆ โดยไม่จำเป็น

ใน implementation ใช้ least-squares solver เช่น `np.linalg.lstsq` ซึ่งจัดการ rank-deficient cases ได้ดีกว่า naive inverse

## 9. Residuals

```text
residual = y - ŷ
```

ตรวจ residual plot เพื่อหา:

- nonlinear structure
- heteroscedasticity
- outliers
- missing features

Regression metric ที่ดีไม่แทน residual diagnosis

## 10. Polynomial Regression

Polynomial regression ยังเป็น **linear in parameters**

degree 2:

```text
ŷ = b + w₁x + w₂x²
```

เราเปลี่ยน feature:

```text
x → [x, x²]
```

แล้วใช้ linear regression

degree สูงขึ้น:
- fit curve ซับซ้อนขึ้น
- variance สูงขึ้น
- extrapolation อันตรายขึ้น

## 11. Feature Interactions

สอง features degree 2 อาจมี:

```text
x₁, x₂, x₁², x₁x₂, x₂²
```

จำนวน features โตเร็วตาม degree และ input dimensionality

## 12. Ridge Regression — L2

Objective:

```text
J(w,b)
=
(1/n)||Xw+b-y||²
+
λ||w||²₂
```

โดยทั่วไปไม่ regularize intercept

gradient weight:

```text
∇w J
=
(2/n)Xᵀ(Xw+b-y)
+
2λw
```

ผล:
- shrink coefficients
- ช่วยเมื่อ features correlated
- ลด variance ได้
- coefficient มักไม่เป็นศูนย์เป๊ะ

## 13. Ridge Closed Form

เมื่อ intercept ถูกจัดการเหมาะสม:

```text
θ = (X̃ᵀX̃ + αR)⁻¹X̃ᵀy
```

`R` เป็น identity-like matrix ที่ element ของ intercept เป็น 0 เพื่อไม่ penalize intercept

ใน code ใช้ `solve` มากกว่า explicit inverse

## 14. Lasso — L1

Objective:

```text
J(w,b)
=
(1/(2n))||Xw+b-y||²
+
λ||w||₁
```

```text
||w||₁ = Σ|w_j|
```

L1 มีจุดไม่ differentiable ที่ 0

ใช้:
- subgradient
- coordinate descent
- proximal methods

Lasso มีแนวโน้มสร้าง exact-zero coefficients จึงใช้เป็น sparse model/feature selection ได้ แต่ selection อาจ unstable เมื่อ features correlated มาก

## 15. L1 Subgradient

สำหรับ `w_j ≠ 0`:

```text
d|w_j|/dw_j = sign(w_j)
```

ที่ 0 subgradient เป็นช่วง `[-1,1]`

educational implementation ในบทนี้ใช้ subgradient descent เพื่อให้เห็น concept; production library ใช้ solver ที่เหมาะกว่า

## 16. Elastic Net

ผสม L1 + L2:

```text
data loss
+ λ₁||w||₁
+ λ₂||w||²₂
```

ช่วยสร้าง sparsity พร้อม stabilization เมื่อ predictors correlated

## 17. Why Scaling Matters

Ridge/Lasso penalty วัด magnitude ของ coefficients

ถ้า feature หนึ่งมีหน่วย 0–1 และอีก feature 0–1,000,000 coefficient magnitudes เปรียบเทียบไม่ยุติธรรม

ดังนั้นมัก:

```text
split
→ fit scaler on train
→ transform
→ fit regularized regression
```

## 18. Metrics

MSE:
```text
(1/n)Σ(y-ŷ)²
```

RMSE:
```text
sqrt(MSE)
```

MAE:
```text
(1/n)Σ|y-ŷ|
```

R²:
```text
1 - SS_res/SS_tot
```

ใช้หลาย metric เมื่อ error cost ไม่ชัดเจน

## 19. Closed Form vs Gradient Descent

Closed/least-squares solver:
- เหมาะกับ linear least squares ขนาดไม่ใหญ่มาก
- ไม่มี learning-rate tuning
- numerical linear algebra สำคัญ

Gradient descent:
- generalize ไป objective/models จำนวนมาก
- scale ไป large datasets/NN
- ต้องเลือก learning rate/steps
- convergence diagnosis สำคัญ

## 20. Implementation Files

- [src/regression.py](src/regression.py)
- [src/sklearn_comparison.py](src/sklearn_comparison.py)

สร้างเอง:

- LinearRegressionClosedForm
- LinearRegressionGD
- RidgeRegressionClosedForm
- LassoRegressionSubgradient
- polynomial_features_1d

## 21. scikit-learn

ใช้:

- `LinearRegression`
- `PolynomialFeatures`
- `Ridge`
- `Lasso`
- `ElasticNet`
- `Pipeline`

หลักสำคัญคือ pipeline ต้อง encapsulate preprocessing ที่ fit จาก training data

## 22. Failure Cases

Linear model พังได้เมื่อ:

- relationship nonlinear แต่ features ไม่ represent
- strong outliers
- extrapolation ไกล training range
- multicollinearity ทำ coefficients unstable
- leakage
- distribution shift
- omitted variables
- target noise สูง

## 23. Debugging

ถ้า loss diverge:

- scale features
- ลด learning rate
- ตรวจ sign ของ gradient
- ตรวจ MSE implementation
- ตรวจ NaN/Inf
- plot loss history

ถ้า train/test ต่างมาก:

- leakage?
- overfit?
- split shift?
- polynomial degree สูงเกิน?
- insufficient data?

## 24. Interpretation Caveat

Coefficient:

```text
+2.5
```

ไม่ได้พิสูจน์ว่า feature “ทำให้” target เพิ่ม 2.5

predictive association ≠ causal effect

## 25. Complexity

Prediction linear regression:

```text
O(nd)
```

Polynomial expansion เพิ่ม effective feature dimension `p` แล้ว cost ขึ้นตาม `p`

least-squares solver cost ขึ้นแรงเมื่อ feature count โต จึงต้องระวัง polynomial explosion

## 26. Common Mistakes

1. explicit matrix inverse ทุกครั้ง
2. ไม่ scale ก่อน L1/L2
3. regularize intercept โดยไม่ตั้งใจ
4. polynomial degree สูงมาก
5. fit polynomial/scaler ก่อน split
6. evaluate train only
7. R² เป็น negative แล้วคิด library bug
8. extrapolate polynomial ไกล
9. coefficient = causation
10. compare models คนละ split
11. tune alpha บน test
12. ignore baseline
13. gradient descent ไม่มี loss history
14. ใช้ Lasso sparsity เป็น “truth” ของ feature importance

## 27. Exercises / Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 28. Interview Questions

1. OLS optimize อะไร?
2. derive gradient ของ MSE สำหรับ w
3. ทำไมไม่ควร inverse `XᵀX` ตรง ๆ?
4. polynomial regression ยัง linear อย่างไร?
5. Ridge ต่างจาก Lasso อย่างไร?
6. ทำไม scaling สำคัญกับ regularization?
7. R² ติดลบได้อย่างไร?
8. multicollinearity กระทบ coefficients อย่างไร?
9. regularization ลด overfitting อย่างไร?
10. closed form กับ GD เลือกต่างกันอย่างไร?

## 29. Checklist

- [ ] derive MSE gradients
- [ ] explain matrix shapes
- [ ] fit OLS from scratch
- [ ] fit GD model
- [ ] polynomial features
- [ ] Ridge objective
- [ ] Lasso/L1 intuition
- [ ] compare sklearn
- [ ] residual diagnosis
- [ ] regularized pipeline without leakage

## 30. What's Next

Chapter 07 จะเปลี่ยน continuous prediction เป็น classification ด้วย Logistic Regression โดยนำ linear score ไปผ่าน sigmoid และใช้ probabilistic loss
