# Chapter 80 Exercises

## Level 1 — Recall
1. ทำไม split ก่อน fit tokenizer?
2. causal target shift คืออะไร?
3. causal mask ป้องกันอะไร?
4. perplexity มาจากอะไร?
5. KV cache เก็บอะไร?

## Level 2 — Understanding
6. ทำไม weights ไม่ใช่ training memory ทั้งหมด?
7. context เพิ่มแล้ว attention แพงเร็วเพราะอะไร?
8. SFT ต่างจาก pretraining อย่างไร?
9. DPO reference model มีบทบาทอะไร?
10. quantized file เล็กแต่ latency อาจไม่ลดเพราะอะไร?

## Level 3 — Coding
11. เพิ่ม top-k sampling
12. export loss curve
13. checkpoint metadata JSON
14. gradient accumulation
15. configurable layers/dim/heads

## Level 4 — Debugging
16. loss NaN หลัง 3 steps
17. validation loss ต่ำผิดปกติ
18. tokenizer ภาษาไทยไม่ round-trip

## Level 5 — Challenge
19. scale ~100K → ~1M params ภายใต้ memory budget
20. compare FP32/mixed precision/INT8 ด้วย quality/memory/latency ที่ fair
