# Chapter 12 — Boosting: AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost

## 1. Why This Matters

Random Forest สร้าง trees จำนวนมากค่อนข้างอิสระแล้ว average

Boosting ทำตรงข้าม:

```text
weak model 1
   ↓ inspect mistakes / gradients
weak model 2
   ↓ correct remaining error
weak model 3
   ↓
...
   ↓
weighted / additive ensemble
```

Boosting เป็นหนึ่งในตระกูลที่แข็งแรงที่สุดสำหรับ tabular ML

## 2. Prerequisites

- Decision Trees
- derivatives / gradients
- classification/regression losses
- residuals
- regularization
- bias–variance
- validation / early stopping

## 3. Learning Objectives

เมื่อจบบทนี้คุณควร:

- อธิบาย boosting vs bagging
- derive AdaBoost sample-weight update
- implement AdaBoost with decision stumps
- อธิบาย additive models
- derive gradient boosting สำหรับ squared loss
- implement Gradient Boosting Regressor จาก scratch
- อธิบาย learning_rate × n_estimators trade-off
- อธิบาย stochastic gradient boosting
- อธิบาย histogram boosting
- อธิบาย XGBoost second-order objective
- อธิบาย LightGBM histogram + leaf-wise growth
- อธิบาย CatBoost ordered categorical statistics/boosting
- เลือก validation/early stopping protocol โดยไม่ leakage

## 4. Bagging vs Boosting

### Bagging

```text
Tree 1 ─┐
Tree 2 ─┼→ average/vote
Tree 3 ─┘
```

เป้าหมายหลัก: variance reduction

### Boosting

```text
F₀
 ↓ fit weak learner to current mistakes
F₁
 ↓ fit weak learner to current mistakes
F₂
 ↓ ...
```

เป้าหมาย: สร้าง strong additive model จาก weak learners

## 5. AdaBoost Intuition

เริ่มให้น้ำหนักแต่ละ sample เท่ากัน

weak classifier ผิด sample ไหน:
- เพิ่ม weight ของ sample นั้น

ถูก sample ไหน:
- ลด relative influence

รอบถัดไปจึง “สนใจ” examples ที่ยากขึ้น

## 6. AdaBoost Binary Setup

ใช้ labels:

```text
y_i ∈ {-1,+1}
```

weak learner:

```text
h_t(x) ∈ {-1,+1}
```

weighted error:

```text
ε_t = Σ_i w_i [y_i ≠ h_t(x_i)]
```

ถ้า `ε_t < 0.5` learner ดีกว่าสุ่ม

## 7. AdaBoost Learner Weight

```text
α_t = 1/2 ln((1-ε_t)/ε_t)
```

error ต่ำ → alpha สูง

error ใกล้ 0.5 → alpha ใกล้ 0

## 8. Sample Weight Update

```text
w_i ← w_i exp(-α_t y_i h_t(x_i))
```

ถ้าทายถูก:

```text
y_i h_t(x_i)=+1
→ multiply by exp(-α)
```

ถ้าทายผิด:

```text
y_i h_t(x_i)=-1
→ multiply by exp(+α)
```

จากนั้น normalize weights ให้ sum=1

## 9. Final AdaBoost Prediction

```text
F(x)=Σ_t α_t h_t(x)
ŷ=sign(F(x))
```

## 10. Decision Stumps

AdaBoost มักอธิบายด้วย depth-1 trees หรือ **stumps**

stump ถามคำถามเดียว:

```text
x_j <= threshold
```

แม้แต่ learner ง่ายมาก เมื่อ combine sequentially ก็สร้าง boundary ซับซ้อนได้

## 11. Gradient Boosting — General View

สร้าง additive model:

```text
F_M(x)=F_0(x)+Σ_{m=1}^M η h_m(x)
```

แต่ละ weak learner fit ทิศที่ลด loss

แนวคิด:

```text
pseudo_residual
=
-negative gradient of loss
```

## 12. Squared Error Case

loss ต่อ sample:

```text
L(y,F)=1/2(y-F)²
```

derivative ตาม prediction F:

```text
∂L/∂F = F-y
```

negative gradient:

```text
-(F-y)=y-F
```

ซึ่งก็คือ residual

ดังนั้น squared-loss gradient boosting:

1. เริ่ม `F₀=mean(y)`
2. residual `r=y-F`
3. fit regression tree ให้ residual
4. `F←F+ηh(x)`
5. repeat

## 13. Why Learning Rate?

```text
F←F+ηh
```

`η` เล็ก:
- update conservative
- มักต้อง trees มากขึ้น
- อาจ generalize ดีขึ้น

`η` ใหญ่:
- fit เร็ว
- overfit/overshoot ง่ายขึ้น

interaction สำคัญ:

```text
learning_rate ↔ n_estimators
```

## 14. Tree Depth in Boosting

boosting มักใช้ shallow trees

depth:
- 1 → additive stumps
- 2–5 → model interactions
- deep trees → powerful แต่ overfit/slow ง่าย

## 15. Stochastic Gradient Boosting

fit แต่ละ boosting stage ด้วย subsample ของ rows:

```text
subsample < 1
```

เพิ่ม randomness คล้าย bagging บางส่วน และช่วย regularization/speed

## 16. Early Stopping

monitor validation metric:

```text
iteration 1  validation loss ↓
...
iteration 85 best
...
iteration 110 no improvement
→ stop
```

ต้องใช้ validation data ที่ไม่ถูก fit เป็น training target ในรอบนั้น

## 17. Histogram Gradient Boosting

แทนการลอง thresholds จากทุก unique value:

```text
continuous values
→ bins
→ gradient/hessian histograms
→ evaluate splits by bins
```

ข้อดี:
- เร็ว
- memory efficient
- scale ไป dataset ใหญ่กว่า classic exact split search

scikit-learn มี `HistGradientBoostingClassifier/Regressor`

## 18. XGBoost — Core Idea

XGBoost = regularized gradient tree boosting system

มันเพิ่มหลายอย่างเหนือ vanilla Gradient Boosting:

- second-order Taylor approximation
- explicit tree complexity regularization
- shrinkage
- row/column subsampling
- missing-value handling
- histogram/approximate split algorithms
- efficient systems implementation

## 19. XGBoost Objective

ที่ boosting round t:

```text
Obj^(t)
=
Σ_i L(y_i, ŷ_i^(t-1)+f_t(x_i))
+
Ω(f_t)
```

second-order approximation:

```text
≈
Σ_i [
g_i f_t(x_i)
+
1/2 h_i f_t(x_i)²
]
+
Ω(f_t)
+ constant
```

โดย:

```text
g_i = ∂L/∂ŷ
h_i = ∂²L/∂ŷ²
```

จึงใช้ทั้ง gradient และ curvature

## 20. XGBoost Tree Regularization

conceptual form:

```text
Ω(f)=γT + 1/2 λΣ_j w_j²
```

- T = number of leaves
- γ penalizes adding leaves
- λ = L2 leaf-weight regularization

XGBoost ยังมี L1 ผ่าน `reg_alpha`

## 21. XGBoost Tree Methods

official documentation ปัจจุบันมี:

- `exact`
- `approx`
- `hist`

และ `auto` ใช้ behavior เดียวกับ `hist` ใน current docs

`hist` เป็น histogram-optimized approximate greedy algorithm

สำหรับ laptop/tabular work มักเริ่มจาก hist

## 22. LightGBM

LightGBM เน้น efficiency:

- histogram-based training
- histogram subtraction
- sparse optimizations
- leaf-wise / best-first growth
- native categorical split strategies
- distributed/GPU options

## 23. Leaf-Wise Growth

depth-wise:

```text
split all nodes level by level
```

leaf-wise:

```text
เลือก leaf ที่ลด loss ได้มากที่สุดแล้ว split ต่อ
```

ข้อดี:
- loss ลดเร็วภายใต้ leaf budget

ข้อเสีย:
- tree อาจลึก/ไม่สมดุล
- small data overfit ง่าย

LightGBM จึงควบคุม complexity ด้วย:

- `num_leaves`
- `min_data_in_leaf`
- `max_depth`
- regularization

## 24. CatBoost

CatBoost ถูกออกแบบมาให้ categorical features เป็น first-class citizens

จุดสำคัญ:

- native categorical handling
- target-statistic-like numerical features แบบ ordered
- strategies เพื่อลด target leakage/prediction shift
- Ordered vs Plain boosting modes
- symmetric/oblivious-tree style เป็นองค์ประกอบสำคัญของ implementation

อย่า one-hot categorical data ล่วงหน้าโดยอัตโนมัติเมื่อใช้ CatBoost; official docs แนะนำให้ส่ง categorical features ให้ CatBoost จัดการ

## 25. Ordered Categorical Statistics Intuition

naive target encoding:

```text
category mean target
```

ถ้าคำนวณจาก row เดียวกัน target ของ row อาจ leak เข้า feature

ordered strategy ใช้ข้อมูล “ก่อนหน้า” ตาม permutation/order เพื่อสร้าง statistic สำหรับ current row โดยไม่ใช้ current target โดยตรง

นี่คือ intuition สำคัญของ CatBoost

## 26. Comparing Families

### sklearn GradientBoosting
- classic boosting
- transparent
- good educational baseline
- slower on very large data

### sklearn HistGradientBoosting
- histogram
- efficient
- supports modern large-tabular workflows

### XGBoost
- strong regularized boosting ecosystem
- gradient + hessian
- hist/approx systems
- extensive tuning controls

### LightGBM
- histogram
- leaf-wise
- fast on large tabular data
- native categorical options

### CatBoost
- excellent categorical workflow
- ordered category statistics
- less preprocessing for raw categorical data

ไม่มี winner ตายตัว ต้อง validate บนโจทย์จริง

## 27. Optional Libraries

Core CI ไม่บังคับติดตั้ง external boosting libraries เพื่อให้ course lightweight

เมื่อต้องการ:

```bash
python -m pip install -r requirements-batch04-extras.txt
```

ดู [theory/02-xgboost-lightgbm-catboost.md](theory/02-xgboost-lightgbm-catboost.md)

## 28. From Scratch

[src/boosting.py](src/boosting.py) มี:

- regression stump
- GradientBoostingRegressorFromScratch
- binary decision stump
- AdaBoostBinaryClassifierFromScratch

## 29. Hardware-Aware Advice

บน laptop:

- เริ่ม 100–300 trees
- shallow depth
- CPU threads จำกัดตามความร้อน
- histogram algorithms เมื่อ data ใหญ่
- monitor RAM
- อย่า grid search ขนาดมหาศาล

สำหรับ external libraries ให้เริ่ม CPU ก่อน แล้วค่อยเรียน GPU configs ใน Chapter 53

## 30. Failure Cases

- noisy labels + too many boosting rounds
- high learning rate
- deep base trees
- target leakage
- validation over-tuning
- severe distribution shift
- categorical encoding leakage
- tiny datasets + aggressive leaf-wise growth

## 31. Common Mistakes

1. boosting = bagging
2. residual = error แบบเดียวกับทุก loss
3. tune on test
4. learning rate ลดแต่ไม่เพิ่ม rounds
5. deep trees ทุก round
6. early stopping บน training loss
7. target encode full dataset
8. assume library probabilities calibrated
9. compare packages ด้วย hyperparameters ที่ชื่อเหมือนแต่ semantics ต่าง
10. use feature importance as causality
11. install every libraryก่อนเข้าใจ algorithm
12. run huge tuning grid บน laptop

## 32. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 33. Checklist

- [ ] AdaBoost error/alpha/weights
- [ ] additive model
- [ ] negative gradient
- [ ] squared-loss residual
- [ ] learning rate
- [ ] stochastic boosting
- [ ] histogram boosting
- [ ] XGBoost g/h + regularization
- [ ] LightGBM leaf-wise
- [ ] CatBoost ordered categorical handling
- [ ] early stopping / validation

## 34. What's Next

Batch 05 จะเรียน SVM, Clustering และ Dimensionality Reduction ซึ่งกลับไปใช้ geometry, margins, distances, eigenvectors และ local/global structure ของข้อมูล
