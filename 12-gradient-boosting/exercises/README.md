# Chapter 12 Exercises

## Level 1 — Recall

1. boosting ต่างจาก bagging อย่างไร?
2. AdaBoost weighted error คืออะไร?
3. AdaBoost alpha สูตรอะไร?
4. gradient boosting fit อะไรในแต่ละ round?
5. learning_rate ทำอะไร?

## Level 2 — Understanding

6. derive AdaBoost weight direction สำหรับ correct/incorrect sample
7. squared loss negative gradient ทำไมเท่ากับ residual?
8. learning_rate ต่ำต้องสัมพันธ์กับ n_estimators อย่างไร?
9. histogram boosting ลด cost ตรงไหน?
10. LightGBM leaf-wise มี overfit risk อย่างไร?

## Level 3 — Coding

11. เพิ่ม `staged_predict` ให้ gradient booster
12. เพิ่ม validation early stopping
13. implement stochastic row subsampling
14. plot train/validation loss per iteration
15. compare stump vs depth-2 sklearn boosting

## Level 4 — Debugging

16. AdaBoost weak learner error > 0.5
17. validation loss เริ่มสูงหลัง iteration 80
18. CatBoost pipeline ถูก one-hot encode ก่อนส่งเข้า libraryโดยไม่จำเป็น

## Level 5 — Challenge

19. implement binary logistic-loss gradient boosting
20. derive XGBoost optimal leaf weight จาก gradient/hessian + L2 penalty
