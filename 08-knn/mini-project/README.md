# Mini Project — KNN Geometry Lab

## Goal

ศึกษาว่า geometry และ preprocessing เปลี่ยน KNN อย่างไร

## Required Experiments

1. 2D synthetic classification
2. k = 1,3,5,9,15,31
3. scaled vs unscaled features
4. Euclidean vs Manhattan
5. uniform vs distance weights
6. validation F1/accuracy
7. prediction latency
8. decision-boundary plots

## Curse of Dimensionality Experiment

เริ่มจาก informative 2 features แล้วเพิ่ม random noise features:

```text
2 → 5 → 10 → 25 → 50 → 100 dimensions
```

วัด:

- validation accuracy/F1
- mean nearest-neighbor distance
- ratio nearest/farthest distance โดยประมาณ
- prediction runtime

อธิบายผล ไม่ใช่แค่ plot
