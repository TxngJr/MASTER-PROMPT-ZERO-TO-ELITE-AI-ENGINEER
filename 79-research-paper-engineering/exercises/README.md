# Chapter 79 Exercises

## Level 1 — Recall
1. claim, evidence และ assumption ต่างกันอย่างไร?
2. ablation study ตอบคำถามอะไร?
3. random seed กระทบ experiment อย่างไร?
4. exact reproduction ต่างจาก concept reproduction อย่างไร?
5. experiment fingerprint ใช้ทำอะไร?

## Level 2 — Understanding
6. ทำไมต้อง reproduce baseline ก่อน?
7. ทำไม mean จาก 5 seeds ดีกว่า best-of-5?
8. original ใช้ 8 GPUs แต่ local ใช้ laptop GPU ควรเรียก reproduction แบบใด?
9. leakage ทำให้ conclusion ผิดได้อย่างไร?
10. compute budget ต่างกันทำให้ comparison ไม่ fair อย่างไร?

## Level 3 — Coding
11. ใช้ seed_statistics กับ 5 runs
12. test relative_error near zero
13. พิสูจน์ fingerprint invariant ต่อ key order
14. implement paired differences
15. implement bootstrap CI

## Level 4 — Debugging
16. accuracy สูงกว่า paper ผิดปกติ: leakage checklist
17. train loss match แต่ validation ไม่ match: 5 สาเหตุ
18. baseline ต่ำกว่า paper 15 จุด: ทำไมยังสรุป proposal ไม่ได้?

## Level 5 — Challenge
19. เขียน reproduction contract 1 หน้า
20. ออกแบบ 3 ablations ภายใต้ compute จำกัด
