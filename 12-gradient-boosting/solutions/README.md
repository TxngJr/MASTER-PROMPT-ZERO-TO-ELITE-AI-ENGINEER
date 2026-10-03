# Chapter 12 Solutions

1. bagging train models parallel-ish แล้ว aggregate; boosting ต่อ models sequentially
2. `Σ w_i [y_i != h(x_i)]`
3. `0.5 ln((1-ε)/ε)`
4. negative gradient / pseudo-residual ของ loss ปัจจุบัน
5. shrink contribution ของ weak learner แต่ละรอบ

6. correct → multiply `exp(-α)`; incorrect → `exp(+α)`
7. `L=0.5(y-F)^2`, ดังนั้น `-dL/dF=y-F`
8. learning rate ต่ำมักต้อง rounds มากขึ้น
9. aggregate split statistics ต่อ bins แทน scan raw unique thresholds ซ้ำ
10. best-first growth อาจสร้าง branch ลึกมากบน small data

## XGBoost leaf-weight hint

ถ้า leaf j มี gradients `G_j=Σg_i` และ Hessians `H_j=Σh_i`, L2 λ:

```text
w*_j = -G_j / (H_j + λ)
```

derive โดย differentiate local quadratic objective แล้วตั้งเป็นศูนย์

Challenge ควร derive/implementเองก่อนดู library
