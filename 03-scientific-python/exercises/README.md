# Chapter 03 Exercises

## Level 1 — Recall

1. `shape`, `ndim`, `dtype` หมายถึงอะไร?
2. `*` กับ `@` ใน NumPy ต่างกันอย่างไร?
3. broadcasting คืออะไร?
4. DataFrame คืออะไร?
5. histogram ใช้ดูอะไร?

## Level 2 — Understanding

6. อธิบายผล shape ของ `(100,4) @ (4,3)`
7. ทำไม `axis=0` reduction บน `(100,4)` จึงได้ shape `(4,)`?
8. ทำไม vectorization มักเร็วกว่า Python loop?
9. ทำไม missing values ไม่ควรถูก drop แบบอัตโนมัติ?
10. ทำไม correlation matrix ไม่พิสูจน์ causation?

## Level 3 — Coding

11. standardize matrix ต่อ column โดย NumPy และ handle std=0
12. filter rows ของ DataFrame ด้วย boolean mask
13. group sample dataset ตาม group แล้วหา mean score
14. plot study_hours vs score
15. เพิ่ม median และ quantiles ลง analyzer summary

## Level 4 — Debugging

16. `X @ w` shape error: X=(100,4), w=(3,) — diagnose
17. broadcasting ให้ผล shape ไม่ตั้งใจ — เขียน shape alignment ก่อนแก้
18. Pandas numeric column กลายเป็น object เพราะ bad token — หาและ clean โดยไม่ซ่อนจำนวน invalid values

## Level 5 — Challenge

19. benchmark loop vs NumPy บนหลาย input sizes แล้ว plot runtime
20. เขียน report ที่รวม missing/duplicates/statistics/correlation/plots จาก CSV ใหม่
