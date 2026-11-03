# Visual Course v3 — หลักฐานตรวจรับ

ตรวจวันที่ **2026-10-15** รายงานแยกโครงสร้าง คณิตศาสตร์ การสอน local runtime และการทดลองกับผู้เรียน ไม่ใช้จำนวน tests สรุปว่าผู้เรียนเข้าใจแล้ว

| ขอบเขต | ผลและข้อจำกัด |
|---|---|
| โครงสร้าง | HTML 80 บท, แบบฝึกหลัก 1,600 ข้อ, objective/evidence contracts 1,144 รายการ; ไม่มี placeholders, IDs ซ้ำ หรือ local assets หาย |
| เนื้อหา | เฉลยหลักเขียนเฉพาะข้อใหม่; 3 hints/ข้อ (เขียนเฉพาะเรื่อง 32 ข้อ; อีก 1,568 ข้อเป็น excerpt ของเฉลยที่ค่อย ๆ เปิดเพิ่มและยังต้องตรวจวิธีชี้นำโดยผู้สอน), explicit source mappings, glossary และ offline help; โจทย์ราย objective มี reference และเกณฑ์ส่งงาน |
| คำถามในเว็บ | 480 question families, 1,110 versioned cases; 6 ประเภท/บท, shape/state ของ operation จริง, seeded Fisher–Yates shuffle, semantic answer IDs |
| Numerical + assessment | **70 tests ผ่าน**, 642 default/control-boundary/select cases และ 249 presets; independent NumPy/sklearn/Torch CPU oracles, finite differences และ history/input invariants |
| Content / learner assessment tests | **37 tests ผ่าน** รวม 15 content checks และ 22 learner-runner checks: เฉลยผิดหัวข้อ, framework/PCA shapes, fresh dimensions, negative implementations และ copied-src workspace |
| เกณฑ์ทักษะ | 1,144 objectives × 6 ด้าน × 3 รอบ; ทุกด้านต้องได้ 2, คนละ case/fingerprint, retention ≥7 วันตามนโยบายหลักสูตร; ไม่ให้คะแนนจาก self report |
| Learner runner QA | copied-src workspace 80/80 บทผ่านการตรวจ; รวม PyTorch/TensorFlow CPU และ dependency capstone บท 80; ทั้งหมดเป็น QA ไม่ใช่ผลงานผู้เรียน |
| Local chapter runtime | **406 tests ผ่าน**, 80 บท, 0 skipped; Python 3.12.15 |
| Integration runtime | **110 tests ผ่าน**, 27 โครงงาน, 0 skipped |
| Browser | ตรวจผ่าน localhost; desktop 1440×960 และ mobile 390×844, ครบ 80 บท; 249 experiments แสดงผลได้; ไม่พบ console errors ในรอบตรวจ |
| การสอน | authored reference review และ rubric พร้อมใช้งาน; โครงสร้างอ้างอิงครบไม่ใช่การยืนยันประสิทธิผลการเรียน |
| ผู้เรียนจริง | **ยังไม่ทดลอง**; ต้องวัดโจทย์ใหม่ คำอธิบาย และผลงานจริงตาม rubric |
| GPU / production | **ยังไม่ทดสอบ** CUDA/multi-GPU, cluster/network serving, external model integrations และ physical devices; CPU smoke tests ไม่ใช้แทนหลักฐานเหล่านี้ |

รวม tests ในขอบเขตข้างต้น **623 ผ่าน ไม่มี skipped** รายละเอียดคำสั่ง/output/source hashes อยู่ใน [audit.json](course/audit.json) และ [runtime-v3.json](course/runtime-v3.json) เวลา runtime กับเวลา audit แยกกัน; audit ยอมรับผล runtime เฉพาะเมื่อ hashes ของ source/tests ยังตรง

## ปัญหาที่แก้และหลักฐาน

- Regression: MSE `3 → 1.36`, gradients `(-4,-2)`, weights `(1.4,.2)`; NumPy comparisons และ finite differences ยังผ่าน ตัวอย่าง slope/bias/MSE/derivation เชื่อมเฉพาะเรื่อง แก้การอ้าง standardization/RMSE ผิดหัวข้อที่พบตอนอ่าน browser
- CNN: output2×2รับindex0–3; index8แสดง shape error ไม่วนกลับ และไม่เพิ่ม history ของ snapshot ผิด; index3กู้การคำนวณได้
- Optimizer: เปิดหน้าθ2,L4,updates0; manualηไม่ฝึก; SGDหนึ่งก้าวθ1.6,L2.56; method changeล้างmomentsที่currentθเดิม
- DBSCAN: ε=.35,min_samples3ได้2clusters,core6,border1,noise1; ε=.05,min_samples1ได้8clustersและภาพยังแสดงครบ8จุด; เทียบ sklearn
- Attention: Q/K/V projections, head splitting/masks/concat เทียบ Torch SDPA; แก้ WV แล้ว A คงเดิม
- Bootstrap: LCG-seeded resampling with replacement; indices/statistics/linear quantilesเทียบ NumPy ไม่ใช้ Wilson แทน
- Input domain: λ/η/Cไม่ติดลบ, temperature/h/epsilonที่เป็นตัวหารต้องบวก, gates/probabilitiesอยู่ในdomain และ indexตรวจตามshape
- Assessments: จับคำถามซ้ำและตัวเลือกซ้ำ, แก้ PCA dimension ที่ใช้ไม่ได้, แยก PyTorch weight orientation, ตรวจ GPU block boundary; version changeล้าง exposure history พร้อม archive
- Migration: v2 read/local self-reportคงอยู่ แต่ legacychecks2ข้อไม่ให้ผ่านv3; practice/new-case/reloadตรวจผ่านผ่าน UI
- Intermediate worksheet: final answerถูกแต่predictionarrayผิด ชี้ operation แรกที่ตรวจพบจากข้อมูลที่กรอก; JSON syntaxแสดงวิธีแก้
- Evidence notebook: บันทึกคำอธิบาย/กรณีใหม่/โค้ดและผลรัน/กรณีล้มเหลว พร้อม self-review ราย objective; reload/export JSON ตรวจผ่าน เก็บงานรุ่นก่อนและให้ทบทวน rubric ใหม่เมื่อเนื้อหาเปลี่ยน ไม่ให้คะแนนผ่านอัตโนมัติ
- Hints: แก้ 126 ข้อที่ซ้ำหรือเปิดเฉลยเต็มก่อนเวลา; Regression 20 ข้อมีคำใบ้แนวคิด → สมการ → คำนวณบางส่วนที่เขียนแยกจากเฉลย
- Criterion gates: ทดสอบ self review/คะแนนไม่ครบ/หลักฐานหาย/โจทย์ซ้ำ/hints/วันที่อนาคต/วันที่ไม่มีจริง/เลขรอบที่ไม่ใช่วันเวลา/retentionเร็วเกินไป/รายงานเก่า/source hashไม่ตรง/QA reports ไม่ให้สถานะครบเกณฑ์; ไม่นำรายงานคนละชื่อผู้เรียนมารวมกัน; reviewer identity ยังเป็น unverified
- Learner source runner: starter 0/15 checks, QA adapter 15/15 checks แต่ tag QA ไม่ให้ credit ผู้เรียน; fresh seeded dimensions 1/2/3/4 จับโค้ดที่ hardcode dimension ได้
- Navigation/accessibility: legacy anchor `source-06-regression-canonical-6-derive-gradients`, resume, keyboard Enter/Escape, menu focus และ reduced-motion stopตรวจผ่าน

## เวอร์ชันและการอ่านรายงาน

[contracts.json](course/contracts.json) ระบุเวอร์ชันต่อบท; [audit.json](course/audit.json) ระบุ page hash และผลแยก 27 batches `engineeringChecks` เป็นสถานะระบบ ไม่ใช่การผ่านทักษะของผู้เรียน รายงาน browser ของเกณฑ์ประเมินรอบปัจจุบันอยู่ใน [browser-qa-mastery.json](course/browser-qa-mastery.json); [browser-qa-v3.json](course/browser-qa-v3.json) เป็นรอบก่อนเพิ่มเกณฑ์นี้; [browser-qa.json](course/browser-qa.json) และภาพ v2 เป็นหลักฐานจากรอบก่อน

[preview-v3.jpg](preview-v3.jpg) เป็นภาพ browser จริงของต้นแบบ ไม่ใช่ mockup การเปิด file:// ไม่ได้ทดสอบผ่าน browser automation; static assets ไม่มี runtime fetch/CDN/API และทดสอบการใช้งานผ่าน HTTP local แล้ว

รายละเอียดเกณฑ์และข้อจำกัดการรับรองอยู่ใน [คู่มือผู้ตรวจ](assessment/REVIEW_GUIDE.md) และ [ผลตรวจ runner](course/learner-runner-qa.json) ผล test suite ยืนยันระบบ/กรณีที่ตรวจ ไม่ใช่ความเข้าใจของผู้เรียน ยังไม่มีผลประเมินผู้เรียนจริงหรือการยืนยันตัวตนผู้ตรวจ
