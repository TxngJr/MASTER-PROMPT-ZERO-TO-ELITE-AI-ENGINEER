# XGBoost vs LightGBM vs CatBoost — Engineering Comparison

เอกสารนี้เน้น conceptual/system differences มากกว่าการจำ defaults ซึ่งเปลี่ยนตาม versions

## XGBoost

Official docs:
https://xgboost.readthedocs.io/en/stable/

### Tree construction

current docs มี `exact`, `approx`, `hist`; `auto` maps to `hist`

### Regularization

สำคัญ:

- `reg_lambda` — L2
- `reg_alpha` — L1
- `gamma` / minimum split loss concept
- `min_child_weight`
- row/column subsampling

### Growth

`grow_policy`:
- depthwise
- lossguide

### Minimal optional example

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="logloss",
    random_state=42,
)
```

ต้อง tune/early-stop ด้วย validation protocol

---

## LightGBM

Official docs:
https://lightgbm.readthedocs.io/en/latest/

### Histogram

continuous values ถูก bucket เป็น bins

split evaluation ทำบน histogram statistics

### Leaf-wise growth

เลือก leaf ที่มี largest loss reduction ก่อน

ดังนั้น `num_leaves` เป็น complexity control สำคัญ

### Categorical

มี native categorical splitting strategies จึงไม่จำเป็นต้อง one-hot ทุกกรณี

### Minimal optional example

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
)
```

ระวัง small dataset + num_leaves มาก → overfit

---

## CatBoost

Official docs:
https://catboost.ai/docs/

### Categorical first

CatBoost รับ categorical columns โดยตรง

official docs เตือนว่าไม่ควร one-hot categorical features ล่วงหน้าแบบทั่วไป เพราะ CatBoost มี processing ของตัวเอง

### Ordered statistics

สร้าง categorical-derived statistics โดยหลีกเลี่ยงการใช้ current target ตรง ๆ ใน feature ของ current row ตาม ordered/permutation idea

### Ordered boosting

มี Ordered และ Plain boosting schemes ตาม mode/configuration

### Minimal optional example

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    verbose=False,
    random_seed=42,
)

model.fit(
    X_train,
    y_train,
    cat_features=["city", "device_type"],
)
```

---

## Do Not Compare Parameters by Name Alone

ตัวอย่าง:

```text
XGBoost max_depth=8
LightGBM num_leaves=255
CatBoost depth=8
```

ไม่ได้แปลว่า models มี complexity เท่ากัน

growth policy, tree structure, categorical handling, regularization และ binning ต่างกัน

## Fair Comparison Protocol

1. same raw split
2. package-appropriate preprocessing
3. same primary metric
4. comparable tuning budget
5. early stopping on same validation policy
6. final test once
7. record CPU/RAM/runtime/model size
8. repeat seeds เมื่อ randomness มีผล

## Laptop Strategy

เริ่ม:

```text
100–300 iterations
small/moderate depth
single experiment at a time
CPU first
early stopping
```

อย่าเริ่มด้วย thousands of trials หรือ exhaustive grid
