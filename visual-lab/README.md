# AI Engineer Learning Studio v3

หลักสูตร HTML 80 บทสำหรับผู้ที่เขียน Python ได้แล้ว มีคำอธิบายภาษาไทย ตัวอย่างตัวเลข ต้นฉบับประกอบ source code/tests แบบฝึกหลัก 1,600 ข้อ และโจทย์พิสูจน์ราย objective ใช้งาน local โดยไม่มี paid API หรือบริการรัน Python เบื้องหลัง

## เปิดเรียน

เปิด [index.html](index.html) หรือ `animation.html` ของแต่ละบท Assets เป็น classic scripts และ relative paths; เนื้อหาถูกสร้างลง HTML แล้ว ไม่มีการดึง Markdown ตอนเรียน

การเปิดผ่าน HTTP บนเครื่องเป็นเส้นทางที่ทดสอบ browser จริง:

```bash
# รันจาก repository root
python3 -m http.server 8765 --bind 127.0.0.1
```

เปิด `http://127.0.0.1:8765/06-regression/animation.html` เพื่อเริ่ม Regression หรือ `/visual-lab/index.html` เพื่อเริ่มบท 1 หยุด server ด้วย Ctrl+C

`#chapter-29`, `#chapter-6/section/gradient-descent` และ anchors ของเนื้อหาต้นฉบับรุ่นก่อนยังใช้ได้ ปุ่มอ่านต่อเก็บหัวข้อและตำแหน่งเลื่อน การเปิด `file://` ยังไม่ได้ทดสอบผ่านเครื่องมือ browser; browser บางตัวแยก localStorage ตามไฟล์ จึงแนะนำ HTTP local สำหรับ progress เดียวกันทุกบท

## เรียนให้เกิดหลักฐาน

1. อ่านโจทย์ นิยามสัญลักษณ์ หน่วย และ shape/state ก่อนดูสูตร
2. ทำนายผลแล้วเปลี่ยน slider/ช่องตัวเลขหนึ่งตัว ตรึง baseline เพื่อตรวจผลต่าง
3. เดิน operation และอ่านการเน้นคอลัมน์/tensor ที่เชื่อมกับสมการ
4. กด update เองเมื่อพร้อม; แก้ weight ด้วยมือไม่เท่ากับ optimizer update
5. ถ้าติด ให้เปิด **ยังไม่เข้าใจตรงนี้** แล้วเลือกศัพท์ สมการ ภาพ โค้ด หรือผลการทดลอง คำอธิบายอยู่ใน HTML และใช้ offline ได้
6. ทำคำถาม 6 ประเภทและกรณีใหม่ หากกรอกค่าระหว่าง operation เว็บจะชี้ operation แรกที่ตรวจพบว่าไม่ตรงจากค่าที่กรอก
7. ทำแบบฝึกหลัก 20 ข้อก่อนเปิดคำใบ้ 3 ระดับ/เฉลย แล้วใช้โจทย์ราย objective ตรวจคำอธิบาย กรณีใหม่ และกรณีล้มเหลว
8. บันทึกงานใน **พิสูจน์ความสามารถทีละ objective**: คำอธิบาย กรณีใหม่ โค้ด/ผลรันจริง และกรณีล้มเหลว ตรวจ checklist แล้วกดส่งออก JSON เพื่อสำรองหรือส่งผู้ตรวจ เนื้อหาเปลี่ยนจะเก็บข้อความเดิมพร้อมป้ายรุ่นก่อนและให้ตรวจ checklist ใหม่
9. รัน local lab/mini-project/integration พร้อมเพิ่ม tests ของงานที่ดัดแปลงเอง ส่ง source/config/ผลทดสอบและข้อจำกัดตาม rubric สามารถใช้ผลงานเดียวรองรับหลาย objectives เมื่อระบุหลักฐานแต่ละข้อครบ

Regression เริ่ม `w=1,b=0,MSE=3,gradient=(-4,-2)`; update `η=.1` ได้ `w=1.4,b=.2,MSE=1.36` ตัวอย่าง static ระบุค่าเริ่มต้น ส่วนห้องทดลองคำนวณจากสถานะปัจจุบัน

DBSCAN คำนวณ neighborhoods/core/border/noise จริง; Attention lab เพิ่ม projections/split heads/concat; bootstrap สุ่ม indices แบบ replacement และคำนวณ statistic ใหม่ทุก replicate

## Progress บอกอะไร

- **อ่านแล้ว** เป็นเครื่องหมายของผู้เรียน
- **ผ่านแบบฝึกเว็บ** มาจากคำถามรุ่นปัจจุบัน
- **ผ่านกรณีใหม่** ต้องตอบกรณีอื่นถูกในครั้งแรกก่อนเปิดผลของกรณีนั้น ไม่ใช่การรับรองทักษะทั้งหมด
- **บันทึกหลักฐานและตรวจ checklist ครบ** เป็นการเตรียมผลงานให้ผู้ตรวจ ไม่ใช่คะแนนผ่านหรือการยืนยันว่าโค้ดรันแล้ว
- **ตรวจแบบฝึกด้วยตนเอง** และ **ยืนยัน local lab** เป็น self report; เว็บไม่ได้รันหรือตรวจ Python ของผู้เรียน
- ผลจาก v1/v2 ถูกเก็บในประวัติ ไม่กลายเป็นคะแนนผ่าน v3 การเปลี่ยนเวอร์ชันคำถามเก็บผลก่อนหน้าใน archive แล้วเริ่มการประเมินใหม่ ผล local รุ่นก่อนติดป้ายแยก

คำตอบเก็บด้วย semantic option ID จึงไม่เปลี่ยนความหมายหลังสลับตัวเลือก/reload ข้อมูลทั้งหมดอยู่ใน localStorage ของ browser/origin นั้น; อย่าล้าง site data ถ้าต้องการรักษาประวัติ

## ประเมินตามเกณฑ์ที่มีหลักฐาน

panel **หลักฐานตามเกณฑ์การเรียนรู้** ครอบคลุม 1,144 objectives ทุก objective มี rubric 6 ด้าน คะแนน 0–2 และ 3 รอบ: initial, transfer, retention หลังอย่างน้อย 7 วัน ทุกด้านต้องได้ 2 พร้อมเหตุผลและตำแหน่งหลักฐาน ไม่เฉลี่ยกลบด้านที่ยังทำไม่ได้ ไม่ใช้ read/self-check/local checkbox ให้ผ่าน

7 วันเป็นนโยบายทบทวนของหลักสูตร ไม่ใช่การรับประกันการจำตลอดไป ผู้ตรวจต้องเป็นคนอื่นตามที่ระบุในรายงาน และกำกับโจทย์ใหม่/การไม่ใช้ hints จริง เว็บ offline ตรวจรูปแบบ เวอร์ชัน และ checksum ได้ แต่ไม่ได้ยืนยันตัวตนผู้ตรวจ ความเป็นเจ้าของโค้ด หรือว่า JSON ไม่ถูกสร้างเอง

เริ่ม Regression ด้วย [starter และวิธีรัน](assessment/regression/README.md) เพื่อเขียน 12 ฟังก์ชัน จากนั้นรัน:

```bash
.venv/frameworks/bin/python tools/assess_regression.py --submission outputs/my-regression/solution.py --seed 20261015 --output outputs/my-regression/code-run.json
```

suite ตรวจ 15 กลุ่มพฤติกรรมด้วยข้อมูลหลายขนาด/seed, finite differences, scalar MSE 3→1.36, rank-deficient OLS, mean-loss Ridge, cross terms, KKT/sparsity, metrics และ train-only scaler โดยรันไฟล์ผู้เรียนใน child Python บนเครื่อง ไม่ใช่ sandbox ไม่มี paid API

สำหรับบทอื่น วาง src ที่ผู้เรียนทำไว้ครบทั้งบท แล้วใช้ tests ของหลักสูตรที่คัดลอกไปโดยไม่แก้ไข:

```bash
.venv/frameworks/bin/python tools/assess_local_submission.py --chapter 22 --submission-src outputs/my-pytorch/src --output outputs/my-pytorch/code-run.json
```

output ต้องอยู่นอก src ผลตรวจระบุ source/suite hashes และ failed/skipped; บท 80 ใช้ tokenizer จากบท 49 เป็น **reference dependency** ที่ระบุ hash แยก ไม่อ้างว่าเป็น tokenizer ที่ผู้เรียนเขียน

นำเข้า code-run JSON ใน HTML แล้วดาวน์โหลดแบบฟอร์มให้ผู้ตรวจ ส่ง source ที่ hash ตรงรายงานและงานราย objective ให้ตรวจ ผู้ตรวจกรอกคะแนนจริง/กรณีใหม่/วันที่/หลักฐานตาม [คู่มือผู้ตรวจ](assessment/REVIEW_GUIDE.md) แล้วนำรายงานกลับเข้าเว็บ ข้อที่ยังขาดจะแสดงเหตุผลและลิงก์กลับไปหัวข้อที่เกี่ยวข้อง การแก้งานหลังตรวจทำให้ snapshot เดิมไม่ตรงและต้องตรวจใหม่

ผลลัพธ์ใช้ถ้อยคำ **ตามรายงานผู้ตรวจ: ครบเกณฑ์ X/Y objectives** ไม่ออกใบรับรองความเข้าใจ 100% หรือใบรับรองจากสถาบัน ผลงานที่ยังไม่ได้ส่งไม่มีสิทธิ์ผ่านอัตโนมัติ และการผ่าน tests จำนวนจำกัดไม่ครอบคลุมทุก input/ทุก objective

## Python 3.12 และการตรวจซ้ำ

ใน workspace นี้เตรียม `.venv/frameworks` ด้วย CPython 3.12.15 แล้ว สำหรับติดตั้งใหม่ใช้ Python 3.12 และชุด CPU ที่ทดสอบ:

```bash
python3.12 -m venv .venv-course
source .venv-course/bin/activate
python -m pip install -r requirements-visual-lab-cpu.txt
python visual-lab/python/regression_walkthrough.py
(cd 22-pytorch && python -m pytest tests -q -rs)
(cd 23-tensorflow-keras && python -m pytest tests -q -rs)
```

ไฟล์ requirements นี้ pin เวอร์ชันที่ตรวจบน CPU และใช้ official PyTorch CPU index ส่วน requirements ราย batch เดิมยังอยู่สำหรับงานแต่ละชุด GPU และ external deployment ต้องตรวจแยก

```bash
# Build โดยใช้ environment ที่มี Markdown
python tools/build_visual_book.py

# fresh tests ทุกบท/โครงงาน พร้อม source hashes
.venv/frameworks/bin/python tools/verify_course_runtime.py

# ตรวจ copied-src assessment runner ทั้ง 80 บท (QA เท่านั้น ไม่ให้คะแนนผู้เรียน)
.venv/frameworks/bin/python tools/verify_learner_runner.py

# ตรวจ HTML, numerical/assessment tests, rubric gates และตัวตรวจงานผู้เรียน
.venv/frameworks/bin/python tools/audit_visual_course.py
```

`node tools/build_visual_lab.mjs` อัปเดต chapter metadata แล้วเรียก builder เดียวกัน ไม่ต้อง author เนื้อหาซ้ำ

## แหล่งข้อมูลที่แก้ไขได้

- `course/authored/answers-*.md`: เฉลยเฉพาะข้อ 1,600 ข้อ
- `objective-map.txt`, `exercise-map.txt`, `section-exercises.txt`: การเชื่อม ID/เลขหัวข้อที่ระบุโดยตรง ไม่มีการค้นคำอังกฤษเพื่อเดา topic
- `regression-teaching.json`, `glossary.json`: คำอธิบายต้นแบบและอภิธานศัพท์
- `course/chapters.json`: ชุดทดลองและ presets; `course/contracts.json`: contracts ที่ build แล้ว
- `assets/course-math.js`, `course-models*.js`: pure calculations; `course-render.js`: วาดจาก snapshots; `course-assessment.js`: variants/grading/migration
- `course-mastery*.js`, `mastery-policy.json`: เกณฑ์/รายงานผู้ตรวจและผลรันโค้ดผู้เรียน; `course-app.js`: controls/history/reading/progress; `template.html`: โครงหน้า

Builder ปฏิเสธเฉลยหาย, rubric ทั่วไปแทนคำตอบ, topic/section/evidence ID ไม่ตรง และการนำ KNN กลับมาแทน DBSCAN ที่ต้องมีในบท 14

คำใบ้ที่เขียนเฉพาะเรื่อง 32 ข้อครอบคลุม Regression ทั้งบทและจุดสำคัญของ DBSCAN/Attention/Bootstrap ส่วนอีก 1,568 ข้อเปิด excerpt ของเฉลยเพิ่มทีละส่วน ยังต้องตรวจคุณภาพการชี้นำรายข้อโดยผู้สอน

## ขอบเขตหลักฐาน

ดู [VALIDATION.md](VALIDATION.md), [audit.json](course/audit.json), [runtime-v3.json](course/runtime-v3.json) และ [browser-qa-v3.json](course/browser-qa-v3.json) ผล v2 เป็นหลักฐานประวัติ ไม่ใช้ยืนยัน build นี้

249 ชุดทดลองใช้กลไกขนาดเล็กที่คำนวณจริง การทดลองที่เป็น toy model/สมมติฐาน queue/state machine ระบุขอบเขตไว้ ไม่ใช้เป็น benchmark ระบบ production ความเข้าใจจริงต้องประเมินจากคำตอบใหม่ เหตุผล และผลงานของผู้เรียน ยังไม่มีการทดลองกับผู้เรียนจริง จึงไม่รับรองว่าอ่านครบหรือผ่าน tests ของต้นแบบแล้วเข้าใจทุกเรื่อง 100%
