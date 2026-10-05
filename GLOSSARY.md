# Glossary

| Term | ความหมาย |
|---|---|
| Algorithm | ขั้นตอนแก้ปัญหาที่กำหนดไว้อย่างชัดเจน |
| API | ข้อตกลงสำหรับเรียกใช้งาน software/component |
| Array | โครงสร้างข้อมูลแบบลำดับ; NumPy รองรับหลายมิติ |
| Batch | กลุ่มตัวอย่างที่ประมวลผลพร้อมกัน |
| CLI | Command-Line Interface |
| Dataset | ชุดข้อมูลสำหรับวิเคราะห์หรือฝึกโมเดล |
| dtype | ชนิดข้อมูลสมาชิกใน array/tensor |
| Feature | input variable |
| Function | mapping จาก input ไป output |
| Gradient | เวกเตอร์ของ partial derivatives |
| Label | target/คำตอบของ supervised sample |
| Matrix | ตารางตัวเลข 2 มิติ |
| Parameter | ค่าภายในโมเดลที่เรียนรู้จากข้อมูล |
| Scalar | ตัวเลขเดี่ยว |
| Shape | ขนาดตามแต่ละแกน |
| Tensor | โครงสร้างตัวเลขหลายมิติ |
| Vector | ลำดับตัวเลข 1 มิติ |
| Vectorization | ใช้ array operation แทน loop ระดับ Python |
| Virtual environment | environment แยก Python dependencies |


## Batch 27 additions

**Reproduction Contract** — เอกสารที่กำหนด claim, data, model, training และ evaluation conditions ที่จะทำซ้ำ

**Ablation Study** — controlled experiment ที่เอาหรือเปลี่ยน component เพื่อวัด contribution

**Experiment Fingerprint** — deterministic hash ของ canonical experiment configuration ใช้ตรวจ config drift

**Accelerator-Hours** — accelerator count × wall-clock hours ใช้เป็น compute-accounting measure แบบง่าย

**Concept Reproduction** — การทำซ้ำกลไกหลักบน scale เล็กโดยไม่อ้าง benchmark equivalence

**Causal Window** — input/target token window ที่เลื่อนหนึ่งตำแหน่งสำหรับ next-token prediction

**Preference Pair** — prompt พร้อม chosen/rejected response สำหรับ preference-learning objective

**Reproducibility Manifest** — record ที่ผูก code, data, tokenizer, model config, seed, environment และ metrics
