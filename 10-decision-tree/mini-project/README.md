# Mini Project — Tree Complexity Lab

## Goal

ศึกษาความสัมพันธ์ระหว่าง tree complexity กับ generalization

## Required

ใช้ dataset classification เดียวกัน แล้วทดลอง:

```text
max_depth = 1,2,3,5,8,None
min_samples_leaf = 1,2,5,10,20
```

เก็บ:

- train accuracy/F1
- validation accuracy/F1
- number of leaves
- depth
- fit time

## Experiments

1. scaled vs unscaled features
2. add irrelevant noise feature
3. inject label noise
4. inspect feature importances
5. visualize decision boundary เมื่อมี 2 features

## Report

ตอบ:

- depth จุดไหนเริ่ม overfit?
- min_samples_leaf เปลี่ยน boundary อย่างไร?
- scaling มีผลหรือไม่?
- noise feature ได้ importance หรือไม่?
- importance ตีความได้แค่ไหน?
