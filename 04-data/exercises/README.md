# Chapter 04 Exercises

## Level 1 — Recall

1. sample, feature, target ต่างกันอย่างไร?
2. schema คืออะไร?
3. standardization สูตรอะไร?
4. one-hot encoding ใช้กับข้อมูลแบบใด?
5. data leakage คืออะไร?

## Level 2 — Understanding

6. ทำไมต้อง split ก่อน fit scaler?
7. ทำไม ID ไม่ควรถือเป็น continuous feature โดยอัตโนมัติ?
8. median imputation มีข้อดีอะไรเมื่อมี outlier?
9. random split ผิดอย่างไรใน forecasting?
10. duplicate row กับ duplicate event ต่างกันอย่างไร?

## Level 3 — Coding

11. เพิ่ม MinMaxScaler ใน `preprocessing.py`
12. เพิ่ม most-frequent categorical imputer
13. เขียน function ตรวจ numeric range ตาม schema
14. split DataFrame ตาม indices ที่สร้าง
15. สร้าง missing-indicator feature

## Level 4 — Debugging

16. scaler ถูก fit บน full dataset — อธิบาย leakage และแก้
17. one-hot production พังเมื่อ category ใหม่ — เสนอ policy 2 แบบ
18. time series score ดีผิดปกติหลัง random shuffle — วิเคราะห์

## Level 5 — Challenge

19. สร้าง group-aware splitter ที่ไม่ให้ group เดียวกันข้าม split
20. สร้าง data-quality audit CLI ที่คืน non-zero exit code เมื่อ schema fail
