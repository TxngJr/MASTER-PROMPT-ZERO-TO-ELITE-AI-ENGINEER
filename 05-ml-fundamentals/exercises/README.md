# Chapter 05 Exercises

## Level 1 — Recall

1. parameter กับ hyperparameter ต่างกันอย่างไร?
2. loss กับ metric ต่างกันอย่างไร?
3. MSE สูตรอะไร?
4. overfitting คืออะไร?
5. baseline มีไว้ทำอะไร?

## Level 2 — Understanding

6. ทำไม R² ติดลบได้?
7. ทำไม train score อย่างเดียวไม่พอ?
8. อธิบาย bias–variance intuition
9. ทำไม test set ไม่ควรถูกใช้ tune model?
10. K-fold ช่วยอะไร?

## Level 3 — Coding

11. implement RMSE
12. implement median baseline
13. เพิ่ม repeated K-fold
14. plot `f(x)=(x-3)^2` กับ gradient-descent trajectory
15. ทดลอง learning rate 0.01, 0.1, 0.9, 1.1

## Level 4 — Debugging

16. validation ดีผิดปกติเพราะ scaler fit ก่อน CV
17. model train MSE ต่ำมากแต่ validation MSE สูง
18. experiment เปรียบเทียบ model แต่ split seed เปลี่ยนทุก run

## Level 5 — Challenge

19. ทำ learning curve: train ด้วย 10%, 20%, ..., 100% แล้ว plot train/validation error
20. ทำ K-fold evaluator ที่รับ callback `fit_predict(train_idx,val_idx)`
