# Mini Project — Regression Laboratory

## Goal

สร้าง controlled experiment เปรียบเทียบ:

- mean baseline
- Linear Regression
- Polynomial Regression
- Ridge
- Lasso
- Elastic Net

## Dataset

สร้าง synthetic dataset ที่รู้ ground-truth เช่น:

```text
y = 3x₁ - 2x₂ + 0.5x₁² + noise
```

## Required protocol

1. generate ด้วย fixed seed
2. split train/validation/test
3. fit preprocessing เฉพาะ train
4. baseline
5. train models
6. tune degree/alpha บน validation
7. evaluate chosen configuration บน test ครั้งสุดท้าย
8. residual plots
9. coefficient table
10. report assumptions/limitations

## Required plots

- prediction vs truth
- residual vs prediction
- polynomial degree vs train/validation MSE
- regularization strength vs coefficient magnitude

## Report questions

- model ไหน underfit?
- model ไหน overfit?
- regularization เปลี่ยน coefficients อย่างไร?
- test result แตกต่างจาก validation มากไหม?
- มีอะไรบ้างที่ predictive regression ตอบไม่ได้เชิง causality?
