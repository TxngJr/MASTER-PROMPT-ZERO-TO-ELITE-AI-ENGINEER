# Batch 04 Integration Project — Tree Ensemble Benchmark

รวม Chapters 10–12:

```text
same tabular classification data
        ↓
train / validation / test
        ↓
Decision Tree
Random Forest
AdaBoost
Gradient Boosting
Histogram Gradient Boosting
        ↓
same validation metric
        ↓
select model
        ↓
one final test
        ↓
runtime + metrics report
```

## Goal

ตอบคำถามเชิงวิศวกรรม ไม่ใช่แค่ “ตัวไหน score สูงสุด”:

- single tree overfit แค่ไหน?
- forest ลด variance อย่างไร?
- boosting ลด bias อย่างไร?
- histogram boosting เร็วขึ้นเมื่อ data โตหรือไม่?
- model complexity / latency ต่างกันอย่างไร?

## Run

```bash
source .venv/bin/activate
python -m pip install -r requirements-batch04.txt

python integration-project-batch04/src/tree_ensemble_lab.py \
  --output-dir reports/batch04 \
  --seed 42
```

## Core Models

- DecisionTreeClassifier
- RandomForestClassifier
- AdaBoostClassifier
- GradientBoostingClassifier
- HistGradientBoostingClassifier

## Selection Protocol

1. generate fixed synthetic data
2. stratified train/validation/test
3. fit each model on train
4. compare validation F1 + ROC-AUC
5. select by F1 then ROC-AUC
6. evaluate selected model on test once
7. report fit/predict time

## Why no StandardScaler?

Tree split models generally depend on order/thresholds rather than Euclidean distance or coefficient scale

ดังนั้นโปรเจกต์นี้ intentionally ไม่ standardize เพื่อ reinforce ว่า preprocessing ต้องสัมพันธ์กับ model

## Optional Extension

ติดตั้ง:

```bash
python -m pip install -r requirements-batch04-extras.txt
```

แล้วเพิ่ม:

- XGBoost
- LightGBM
- CatBoost

แต่ comparison ต้องใช้ package-appropriate preprocessing และ comparable tuning budget

## Required Extensions

1. repeated seeds
2. confidence interval / score distribution
3. OOB score ของ Random Forest
4. staged validation curve ของ GradientBoosting
5. early stopping comparison
6. noise-feature experiment
7. label-noise experiment
8. class-imbalance experiment
9. model-size serialization comparison
10. optional XGBoost/LightGBM/CatBoost

## Mastery Questions

- bagging กับ boosting ต่างกันเชิง error correlation อย่างไร?
- ทำไม forest train trees independent-ish แต่ boosting ไม่ได้?
- max_depth มีบทบาทต่างกันใน forest vs boosting อย่างไร?
- HistGradientBoosting ใช้ histogram เพื่ออะไร?
- XGBoost ใช้ Hessian เพราะอะไร?
- LightGBM leaf-wise เสี่ยงอะไร?
- CatBoost ลด category target leakage อย่างไร?
