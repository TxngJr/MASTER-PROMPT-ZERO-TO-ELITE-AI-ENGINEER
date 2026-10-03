# Chapter 11 Solutions

1. sample n records with replacement from n training records
2. train estimators on bootstrap samples แล้ว aggregate
3. จำกัด candidate features ที่แต่ละ split
4. row ที่ไม่ได้ถูกเลือกใน bootstrap ของ tree นั้น
5. average probabilities / vote

6. correlated errors ไม่หายด้วย averaging เท่า independent-ish errors
7. ใช้ limit `(1+a/n)^n→e^a` ด้วย a=-1
8. variance มักลดและ stabilize; bias ของ base learnerไม่เปลี่ยนโดยตรงมาก
9. OOB bootstrap ทำลาย temporal ordering
10. Extra Trees randomize split selection/thresholds เพิ่ม

Challenge ให้ implement พร้อม tests
