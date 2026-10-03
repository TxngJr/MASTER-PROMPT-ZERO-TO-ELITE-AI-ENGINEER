# Mini Project — Forest Variance Lab

## Required

เปรียบเทียบ:

- DecisionTree
- Bagging trees with all features
- Random Forest
- ExtraTrees (sklearn)

บน dataset เดียวกันและ repeated seeds

## Measure

- train score
- validation score
- score standard deviation across seeds
- OOB score
- fit time
- prediction time
- model size approximation

## Hyperparameter grid

```text
n_estimators: 10, 50, 100, 200
max_features: sqrt, log2, all
max_depth: 3, 6, None
min_samples_leaf: 1, 5, 20
```

อย่ารัน full Cartesian grid ถ้าเครื่องร้อน/ช้า ให้ทำ controlled experiments ทีละตัวแปร

## Report

ตอบว่า:

- forest ลด variance จาก single tree จริงไหม?
- max_features ต่ำเกินทำให้เกิดอะไร?
- OOB ใกล้ validation score แค่ไหน?
- trees เพิ่มถึงจุดไหนเริ่ม diminishing returns?
