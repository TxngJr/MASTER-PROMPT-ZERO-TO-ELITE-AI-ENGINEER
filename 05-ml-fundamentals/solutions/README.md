# Chapter 05 Solutions

1. parameter เรียนจาก training; hyperparameter ถูกกำหนด/เลือกนอก inner fitting
2. loss มักเป็น quantity ที่ optimize; metric ใช้ประเมินและอาจไม่ differentiable
3. `(1/n)Σ(y-ŷ)^2`
4. fit training มากแต่ generalize แย่
5. reference performance ขั้นต่ำที่ model ควรชนะ

6. residual sum of squares อาจมากกว่า total variation รอบ mean
7. model อาจ memorize training sample
8. bias สูง=systematic simplicity error; variance สูง=sensitive ต่อ sample
9. จะทำให้ unbiased final estimate ถูกใช้ตัดสินใจและไม่ unbiased อีก
10. ใช้หลาย validation folds ลด dependency ต่อ split เดียวและใช้ข้อมูลคุ้มขึ้น

## RMSE

```python
rmse = np.sqrt(np.mean((y_true-y_pred)**2))
```

## Learning-rate experiment

สำหรับ `f(x)=(x-3)^2`:

```text
x_new = x - η 2(x-3)
```

ให้ทดลองจริง ไม่ควรจำผลจากคำตอบ เพราะ behavior เป็นบทเรียนเรื่อง optimization
