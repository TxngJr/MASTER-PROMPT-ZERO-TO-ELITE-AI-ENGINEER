# Chapter 07 Solutions

## Core answers

1. `σ(z)=1/(1+e^-z)`
2. `log(p/(1-p))`
3. `-[y log p + (1-y)log(1-p)]`
4. `TP+FP`
5. `TP+FN`

6. `σ'(z)=σ(z)(1-σ(z))`
7. take log of Bernoulli likelihood แล้ว negate/average
8. threshold เป็น decision policy ที่ขึ้นกับ cost
9. majority-class prediction อาจ accuracy สูง
10. ลด threshold มักเพิ่ม recall แต่เพิ่ม false positives

## Balanced accuracy

binary:

```text
(recall + specificity)/2
```

## Threshold sweep

สำหรับ threshold หลายค่า:
1. convert probabilities → labels
2. compute confusion metrics
3. เลือกบน validation setตาม objective

Challenge ให้เขียนเองพร้อม tests
