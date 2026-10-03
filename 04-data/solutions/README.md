# Chapter 04 Solutions

## Key answers

1. sample=หนึ่ง observation, feature=input, target=สิ่งที่ทำนาย
2. schema=contract ของโครงสร้าง/ชนิด/rules
3. `z=(x-μ_train)/σ_train`
4. nominal categorical variables
5. information ที่ไม่ควรมีตอน prediction หลุดเข้าสู่ training/evaluation

6. ถ้า fit ก่อน split validation/test statistics จะ influence transform
7. numeric representation ไม่ได้แปลว่าระยะห่างมีความหมาย
8. median ทน extreme values กว่า mean
9. future records อาจหลุดเข้า train
10. row เหมือนกันอาจยังเป็นเหตุการณ์จริงคนละครั้ง

## Min-max pattern

```python
minimum = X_train.min(axis=0)
maximum = X_train.max(axis=0)
scale = np.where(maximum == minimum, 1.0, maximum - minimum)
X_train_scaled = (X_train - minimum) / scale
X_test_scaled = (X_test - minimum) / scale
```

จุดสำคัญคือ test ใช้ `minimum/maximum` จาก train

Challenge ควร implement พร้อม tests เอง
