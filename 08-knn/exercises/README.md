# Chapter 08 Exercises

## Level 1 — Recall

1. Euclidean distance สูตรอะไร?
2. Manhattan distance สูตรอะไร?
3. k เล็กมีแนวโน้ม bias/variance อย่างไร?
4. ทำไมต้อง scaling?
5. KNN training ทำอะไร?

## Level 2 — Understanding

6. ทำไม irrelevant feature ทำ KNN แย่ลง?
7. curse of dimensionality คืออะไร?
8. uniform กับ distance voting ต่างกันอย่างไร?
9. prediction complexity brute force ประมาณเท่าไร?
10. KDTree ไม่ได้ช่วยทุก high-dimensional problem เพราะอะไร?

## Level 3 — Coding

11. implement pairwise distances แบบ vectorized
12. เพิ่ม `predict_proba`
13. implement KNN regressor
14. benchmark k=1..31
15. compare p=1 vs p=2

## Level 4 — Debugging

16. KNN แย่มากหลังเพิ่ม income feature
17. query ตรง training point แต่ distance vote ให้ class อื่น
18. latency โตมากเมื่อ dataset โต 100 เท่า

## Level 5 — Challenge

19. implement chunked brute-force prediction เพื่อลด peak memory
20. visualize decision boundary บน 2D dataset
