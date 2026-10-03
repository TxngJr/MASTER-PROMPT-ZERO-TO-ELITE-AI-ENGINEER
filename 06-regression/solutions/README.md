# Chapter 06 Solutions

1. `ŷ=Xw+b`
2. difference ระหว่าง observed กับ predicted
3. residual sum of squares / squared-error objective
4. L2 / squared coefficient norm
5. L1 / absolute coefficient norm

6. `(2/n)Σ(wx+b-y)x`
7. features nonlinear in x ได้ แต่ prediction ยังเป็น linear combination ของ parameters
8. shrink solution ทำให้ sensitivity ต่อ correlated directions ลดลง
9. penalty ทำงานบน coefficient magnitude จึงต้องทำ feature scales comparable
10. inverse แพง/unstable กว่า numerical solvers ที่แก้ least-squares/system โดยตรง

## RMSE

```python
rmse = np.sqrt(np.mean((y_true-y_pred)**2))
```

## R² method

ใช้ training mean เฉพาะในความหมาย baseline ของชุดที่ metric กำลังวัดตามนิยาม R²; อย่าสับสนกับ preprocessing mean

ข้อ challenge ให้ implement และ test เองเพื่อเห็น solver behavior
