# Batch 03 Integration Project — Classification Model Lab

รวม Chapters 07–09:

```text
classification dataset
      ↓
stratified split
      ↓
train-only scaling
      ↓
Logistic Regression
KNN
Gaussian Naive Bayes
      ↓
validation probabilities
      ↓
model + threshold selection
      ↓
one final test
      ↓
JSON report
```

## Goal

สร้าง reproducible binary-classification experiment ที่:

- มี class imbalance
- ใช้ stratified split
- compare models บน split เดียวกัน
- ใช้ F1 เป็น selection metric ตัวอย่าง
- tune threshold บน validation เท่านั้น
- ใช้ test เพียง final evaluation

## Run

```bash
source .venv/bin/activate
python -m pip install -r requirements-batch03.txt

python integration-project-batch03/src/classification_lab.py \
  --output-dir reports/batch03 \
  --seed 42
```

## Models

- Logistic Regression
- KNN
- Gaussian Naive Bayes

## Metrics

- accuracy
- precision
- recall
- F1
- specificity
- ROC-AUC
- PR-AUC

## Why threshold tuning?

ทุก model สร้าง score/probability ได้ แต่ decision policy ไม่จำเป็นต้องใช้ 0.5

โปรเจกต์จะ:

1. fit model ด้วย train
2. predict probability บน validation
3. sweep thresholds
4. เลือก threshold ที่ F1 สูงสุดบน validation
5. เลือก model จาก validation F1
6. ใช้ model+threshold นั้น evaluate test

## Required Extensions

1. เพิ่ม `class_weight="balanced"` Logistic Regression candidate
2. tune K ของ KNN ด้วย CV
3. เพิ่ม Bernoulli/MultinomialNB experiment บน count data
4. plot confusion matrix
5. plot ROC curve
6. plot PR curve
7. calibration curve
8. cost-sensitive threshold objective
9. subgroup evaluation
10. repeated splits เพื่อวัด variance

## Mastery Questions

- accuracy สูงแต่ recall ต่ำหมายถึงอะไร?
- threshold ถูกเลือกจากชุดไหน?
- KNN ต้อง scale เพราะอะไร?
- GaussianNB model likelihood ต่างจาก Logistic อย่างไร?
- Naive Bayes probability ทำไมอาจ overconfident?
- model ที่ validation F1 สูงสุด guaranteed production ดีที่สุดไหม?

ถ้าตอบได้พร้อม trace code ได้ ถือว่าพร้อม Batch 04
