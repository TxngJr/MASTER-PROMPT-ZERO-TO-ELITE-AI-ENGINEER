# Batch 02 Integration Project — Leakage-Safe Regression System

โปรเจกต์นี้รวม Chapters 04–06:

```text
raw tabular data
      ↓
schema / missing / feature audit       Chapter 04
      ↓
train / validation / test
      ↓
fit preprocessing on TRAIN only        Chapter 04
      ↓
baseline + metrics + experiment design Chapter 05
      ↓
Linear / Ridge / Lasso                 Chapter 06
      ↓
validation model selection
      ↓
one final test evaluation
      ↓
JSON report
```

## Goal

สร้าง regression experiment ที่ไม่รั่วข้อมูลและ reproduce ได้

Starter implementation ใช้ synthetic tabular data ที่มี:

- numeric features
- categorical feature
- missing values
- noise
- known ground-truth structure

เหตุผลที่ใช้ synthetic data: เรารู้ว่าข้อมูลถูกสร้างมาอย่างไร จึงตรวจ pipeline และ reasoning ได้ง่ายก่อนใช้ real-world dataset

## Run

```bash
source .venv/bin/activate
python -m pip install -r requirements-batch02.txt

python integration-project-batch02/src/regression_pipeline.py \
  --output-dir reports/batch02 \
  --seed 42
```

Output:

```text
reports/batch02/
└── report.json
```

## What the code demonstrates

### Chapter 04

- DataFrame schema
- numeric/categorical feature separation
- missing numeric values
- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`
- `ColumnTransformer`
- split before fit

### Chapter 05

- mean baseline
- MSE / MAE / R²
- validation-based model selection
- reproducible seed
- final test used after selection

### Chapter 06

- Linear Regression
- Ridge
- Lasso
- coefficient regularization concept

## Critical leakage rule

Pipeline ถูก `fit` ด้วย training set เท่านั้น

scikit-learn จะ fit:

- numeric median
- numeric mean/std
- categorical vocabulary
- regression parameters

จาก train ใน call เดียว

จากนั้น `.predict(validation)` และ `.predict(test)` ใช้ state เดิม

## Required extensions

ก่อนถือว่าผ่าน Batch 02 ให้เพิ่ม:

1. Polynomial Regression candidate
2. Elastic Net candidate
3. K-fold validation แทน validation split เดียว
4. residual plot
5. prediction-vs-truth plot
6. coefficient table หลัง preprocessing
7. command สำหรับอ่าน CSV จริง
8. data-quality warnings
9. test unknown categorical value ใน test
10. test extreme test value ไม่เปลี่ยน scaler state

## Mastery Questions

1. ถ้า validation ดีกว่า test มาก คุณจะตรวจอะไร?
2. ทำไม preprocessing ต้องอยู่ใน Pipeline?
3. ทำไม mean baseline ต้องใช้ mean ของ train?
4. alpha ของ Ridge/Lasso ควรเลือกจากข้อมูลชุดไหน?
5. ถ้า coefficient เป็นบวก เราสรุป causality ได้ไหม?

ตอบได้พร้อมอธิบาย code path ถือว่าพร้อมไป Chapter 07
