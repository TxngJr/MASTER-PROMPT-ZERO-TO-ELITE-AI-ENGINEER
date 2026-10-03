# Batch 02 Review — Chapters 04–06

## Chapters

- 04 — Data Fundamentals
- 05 — Machine Learning Fundamentals
- 06 — Regression

## Concepts Learned

### Data

- samples/features/targets
- schema/data contracts
- missing values
- duplicates/outliers
- numeric/categorical preprocessing
- train/validation/test
- time/group leakage
- feature engineering
- imbalance foundations

### Machine Learning

- model/hypothesis
- parameters/hyperparameters
- loss/metric/objective
- empirical risk
- gradient descent
- baselines
- underfitting/overfitting
- bias/variance
- regularization
- K-fold validation
- reproducibility
- distribution shift

### Regression

- OLS
- MSE gradients
- least-squares solution
- gradient descent
- polynomial regression
- Ridge
- Lasso
- Elastic Net
- residual diagnosis
- regression metrics

## Algorithms Implemented From Scratch

- train/validation/test index split
- Standardizer
- MedianImputer
- OneHotEncoder concept
- MSE
- MAE
- R²
- K-fold splitter
- gradient descent on a scalar objective
- OLS linear regression
- GD linear regression
- Ridge closed-form
- Lasso subgradient
- polynomial feature expansion

## Framework Skills

scikit-learn:

- `Pipeline`
- `ColumnTransformer`
- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`
- `LinearRegression`
- `Ridge`
- `Lasso`
- `ElasticNet`

## Integration Project

[Leakage-Safe Regression System](integration-project-batch02/README.md)

Pipeline:

```text
mixed tabular data
→ split
→ train-only preprocessing
→ baseline
→ model candidates
→ validation selection
→ final test
→ report
```

## Adversarial Audit

### Data leakage

Pass condition:
- preprocessing fit หลัง split
- stateful transforms อยู่ใน model Pipeline
- test ไม่ใช้เลือก candidate
- mean baseline ใช้ training mean

### Math

Pass condition:
- MSE gradient derive จาก chain rule
- matrix dimensions ระบุ
- Ridge/Lasso penalties แยกชัด
- R² ไม่ถูกตีความเป็น accuracy percentage

### Code

Pass condition:
- invalid shapes/empty data ถูก reject ใน from-scratch code
- deterministic seeds ใน splits/synthetic data
- unit tests compare OLS กับ sklearn
- unknown category inference ไม่ crash

### Concept boundaries

Batch 02 ยังไม่ครอบคลุม classification metrics, logistic loss, nearest neighbors หรือ Bayes classifier เพราะอยู่ Chapters 07–09

## Mastery Checklist

- [ ] อธิบาย leakage อย่างน้อย 5 แบบ
- [ ] เขียน fit/transform class เอง
- [ ] อธิบาย loss vs metric vs objective
- [ ] คำนวณ MSE/MAE/R²
- [ ] อธิบาย K-fold
- [ ] derive OLS gradient
- [ ] implement OLS
- [ ] อธิบาย Ridge vs Lasso
- [ ] ทำ polynomial degree experiment
- [ ] integration project test ผ่าน
- [ ] อธิบายว่าทำไม coefficient ไม่พิสูจน์ causality

## Exit Gate

ก่อน Chapter 07:

1. `pytest` ของ Batch 02 ผ่าน
2. integration project รันได้
3. ทำ coding exercises อย่างน้อย 80%
4. สามารถสร้าง regression pipeline ใหม่จาก CSV ที่ไม่เคยเห็นได้โดยไม่ leakage
