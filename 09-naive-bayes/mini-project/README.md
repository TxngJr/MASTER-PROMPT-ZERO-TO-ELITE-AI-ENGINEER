# Mini Project — Naive Bayes Evidence Lab

## Part A — Gaussian Data

1. generate 2–3 classes
2. inspect feature distributions ต่อ class
3. train GaussianNB
4. compare from-scratch vs sklearn
5. inspect failures เมื่อ distribution ไม่ Gaussian

## Part B — Tiny Text Classifier

สร้างข้อความตัวอย่างเองอย่างน้อย 40–100 records

pipeline:

```text
text
→ tokenize
→ vocabulary fit on train
→ count vectors
→ MultinomialNB
→ evaluation
```

สำคัญ: vocabulary ต้อง fit บน train เท่านั้น

## Experiments

- alpha = 0.1, 0.5, 1, 2, 10
- duplicate correlated feature
- class imbalance
- unknown words

## Report

อธิบาย:
- smoothing ทำอะไร
- log space สำคัญอย่างไร
- independence assumption fail ตรงไหน
- Gaussian vs Multinomial ต่างกันอย่างไร
