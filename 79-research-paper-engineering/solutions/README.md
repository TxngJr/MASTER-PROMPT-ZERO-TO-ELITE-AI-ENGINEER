# Chapter 79 Solutions

1. claim คือข้อกล่าว, evidence คือข้อมูลสนับสนุน, assumption คือเงื่อนไข
2. ablation วัด contribution ด้วย controlled change
3. seed เปลี่ยน initialization/order/dropout/sampling
4. exact match scale/protocol; concept ทดสอบ mechanism scale เล็ก
5. fingerprint detect config drift
6. baseline ไม่ match = pipeline ยังไม่น่าเชื่อถือ
7. multiple seeds แสดง variance; best-of-N มี selection bias
8. close/concept reproduction พร้อมระบุ scale difference
9. held-out information เข้า training ทำให้ metric optimistic
10. compute เพิ่มอาจเป็นสาเหตุ improvement

Coding guidance: รายงาน mean/std/CI, ใช้ absolute tolerance เมื่อ reference ใกล้ศูนย์, canonical JSON, paired differences และ bootstrap resampling

Debugging: ตรวจ duplicate/split/tokenizer fitting/contamination; evaluation preprocessing/model.eval/metric/checkpoint/decoding; ซ่อม baseline ก่อน interpret proposal

Challenge rubric: claim, differences, baseline, controlled factor, seeds/uncertainty, limitations และ artifact/config IDs
