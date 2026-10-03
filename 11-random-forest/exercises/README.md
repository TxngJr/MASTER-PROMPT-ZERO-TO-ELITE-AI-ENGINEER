# Chapter 11 Exercises

## Level 1 — Recall

1. bootstrap sampling คืออะไร?
2. bagging คืออะไร?
3. max_features ทำอะไร?
4. OOB sample คืออะไร?
5. Random Forest aggregate อย่างไร?

## Level 2 — Understanding

6. ทำไม trees ต้อง decorrelate?
7. derive approximation `(1-1/n)^n → e^-1`
8. n_estimators เพิ่มแล้ว bias/variance เปลี่ยนอย่างไร?
9. OOB ใช้กับ time series ทำไมอันตราย?
10. Extra Trees ต่างจาก Random Forest อย่างไร?

## Level 3 — Coding

11. เพิ่ม OOB confusion matrix
12. เพิ่ม `max_features=None`
13. เพิ่ม feature importance จาก impurity decrease
14. benchmark n_estimators 1..200
15. plot OOB score vs number of trees

## Level 4 — Debugging

16. ทุก tree เหมือนกันเพราะใช้ seed ซ้ำผิด
17. forest probability sum ไม่เท่ากับ 1
18. training เร็วแต่ inference ช้ามาก

## Level 5 — Challenge

19. implement RandomForestRegressor
20. implement permutation importance บน held-out set
