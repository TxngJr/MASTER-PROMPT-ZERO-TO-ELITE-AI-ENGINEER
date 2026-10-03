# Chapter 10 Solutions

1. `1-Σp_k²`
2. `-Σp_k log₂p_k`
3. parent impurity minus weighted child impurity
4. terminal node ที่คืน prediction
5. จำนวน split levels สูงสุด

6. p0=0.75, p1=0.25:
```text
Gini=1-0.75²-0.25²=0.375
```

7. monotonic scaling ไม่เปลี่ยน ordering ของ numeric values
8. partitions เล็กลงจน memorize noise
9. เลือกดีที่สุดเฉพาะ node ปัจจุบัน ไม่ search tree structures ทั้งหมด
10. bias ต่อ features ที่มี split opportunities มาก และไม่ใช่ causality

Challenge ควร implement พร้อม tests เอง
