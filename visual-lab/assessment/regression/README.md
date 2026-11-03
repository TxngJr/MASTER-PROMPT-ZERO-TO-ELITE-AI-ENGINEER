# ประเมินโค้ดที่ผู้เรียนเขียน — Regression

การผ่าน suite นี้ยืนยันเฉพาะพฤติกรรมที่ทดสอบ ไม่แทนการอธิบายเหตุผล โจทย์ใหม่ หรือการตรวจโดยผู้สอน ไม่รับรองความเข้าใจ 100% หรือทุก input ที่เป็นไปได้

1. คัดลอก `starter.py` ไปยังโฟลเดอร์ผลงานของคุณ เช่น `outputs/my-regression/solution.py` แล้วเติม 12 ฟังก์ชันตาม docstrings
2. ก่อนรัน บันทึกสมการ/การทำนาย/shape/กรณีล้มเหลวในหน้า HTML ให้ครบ รัน tests ที่คุณเขียนเองก่อน
3. รันจาก repository root ด้วย environment Python 3.12 ที่เตรียมไว้:

```bash
.venv/frameworks/bin/python tools/assess_regression.py \
  --submission outputs/my-regression/solution.py \
  --seed 20261015 \
  --output outputs/my-regression/code-run.json
```

ตัวตรวจ import และเรียกโค้ดในไฟล์ที่ระบุบนเครื่องของคุณ ไม่ใช่ sandbox และไม่ใช้ paid API เปลี่ยน seed เพื่อได้ค่าทดสอบใหม่; seed ใหม่ไม่ยืนยันเองว่าไม่เคยเห็นโจทย์นั้น ผู้ตรวจต้องกำกับเงื่อนไขการทำงานจริง

Expected: `15/15 checks passed`, `failed=0`, `skipped=0`, exit code 0; starter ที่ยังไม่เขียนต้องไม่ผ่าน รายงานมี hash ของไฟล์ผู้เรียน suite/version/seed/environment และค่าที่ตรวจจริง เมื่อไม่ผ่านให้ดู check แรกที่ผิดและกลับไปแก้ฟังก์ชันนั้น ห้ามแก้เฉพาะค่า expected หรือเปลี่ยนสถานะใน JSON

4. นำรายงาน JSON เข้า panel “หลักฐานตามเกณฑ์” ใน HTML ระบบตรวจรูปแบบ/เวอร์ชัน/รายการ checks แต่ browser ไม่สามารถยืนยันย้อนหลังว่าไฟล์ JSON ไม่ถูกแต่งเองหรือผู้ตรวจเป็นใคร
5. ส่งไฟล์ source ที่ hash ตรงกับรายงาน พร้อม export หลักฐานราย objective ให้ผู้ตรวจตรวจโค้ด สมการ และคำอธิบาย ทำคนละกรณีในรอบ transfer และหลังอย่างน้อย 7 วันในรอบ retention ตามเกณฑ์หลักสูตร

ผู้ตรวจควรสร้าง seed ใหม่ กำกับการไม่ใช้ hints/เฉลย ตรวจ first incorrect operation และรัน suite กับ source ที่ส่งอีกครั้ง รายงานของเครื่องหลักสูตรเองไม่ใช่ผลงานของผู้เรียน
