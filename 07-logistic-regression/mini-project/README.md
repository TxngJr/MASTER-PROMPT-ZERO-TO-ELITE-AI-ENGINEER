# Mini Project — Threshold-Aware Binary Classifier

## Goal

สร้าง binary classification experiment ด้วย Logistic Regression

## Required

1. synthetic imbalanced dataset
2. stratified train/validation/test split
3. StandardScaler fit เฉพาะ train
4. Logistic Regression
5. probability predictions
6. threshold sweep ตั้งแต่ 0.05–0.95
7. confusion matrix
8. precision/recall/F1/specificity
9. เลือก threshold บน validation
10. final test evaluation เพียงครั้งเดียว

## Report

อธิบาย:

- ทำไม threshold ที่เลือกไม่จำเป็นต้อง 0.5
- metric ไหนเป็น objective
- false positive / false negative ต่างกันอย่างไร
- test result ต่างจาก validation หรือไม่
- probability calibration ยังไม่ได้พิสูจน์จาก accuracy อย่างไร
