# Chapter 09 Solutions

1. `P(y|x)=P(x|y)P(y)/P(x)`
2. belief/base rate ก่อนเห็น current features
3. probability/density ของ observed features เมื่อกำหนด class
4. features conditionally independent given class
5. เพิ่ม pseudo-count เพื่อป้องกัน zero probability

6. product probability เล็กอาจ underflow; log เปลี่ยน product เป็น sum
7. feature ต่อ class มี Gaussian distribution ตาม model
8. non-negative counts/frequencies เช่น word counts
9. Bernoulli model binary presence/absence; Multinomial model counts
10. independence assumption อาจ double-count correlated evidence

## Log-sum-exp idea

ถ้า logits/log-scores `a`:

```text
log Σ exp(a_i)
=
m + log Σ exp(a_i-m)

m=max(a)
```

ช่วย numerical stability

Challenge ให้ implement พร้อม test เอง
