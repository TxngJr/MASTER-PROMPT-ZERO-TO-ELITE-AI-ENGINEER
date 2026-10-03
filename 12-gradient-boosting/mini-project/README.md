# Mini Project — Boosting Systems Lab

## Core Models

เปรียบเทียบบน dataset เดียวกัน:

- Decision Tree
- Random Forest
- AdaBoost
- GradientBoosting
- HistGradientBoosting

## Optional External Models

เมื่อติดตั้ง:

```bash
python -m pip install -r requirements-batch04-extras.txt
```

เพิ่ม:

- XGBoost
- LightGBM
- CatBoost

## Protocol

1. fixed train/validation/test
2. no test tuning
3. same primary metric
4. early stopping เมื่อ library support
5. record runtime
6. record model size approximation
7. repeat ≥3 seeds เมื่อ practical

## Required Experiments

### Shrinkage

```text
learning_rate:
0.3, 0.1, 0.05, 0.01
```

ปรับ n_estimators ให้สัมพันธ์

### Tree complexity

shallow vs deeper weak learners

### Subsampling

compare full rows vs stochastic boosting

### Histogram

compare classic GradientBoosting vs HistGradientBoosting เมื่อ dataset ใหญ่ขึ้น

## External-library comparison

อย่าใช้ parameter names เท่ากันแล้วถือว่า fair

บันทึก:
- growth policy
- leaves/depth
- categorical preprocessing
- early stopping policy
- thread count
- CPU/GPU
- runtime/RAM

## Report

ตอบ:

- boosting ชนะ single tree ตรงไหน?
- forest vs boosting ต่างกันด้าน bias/variance อย่างไร?
- early stopping iteration เท่าไร?
- XGBoost/LightGBM/CatBoost มี inductive/system differences อะไร?
