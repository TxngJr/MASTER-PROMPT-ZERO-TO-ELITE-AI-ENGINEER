# Chapter 06 Exercises

## Level 1 — Recall

1. linear regression prediction equation?
2. residual คืออะไร?
3. OLS minimize อะไร?
4. Ridge ใช้ penalty แบบใด?
5. Lasso ใช้ penalty แบบใด?

## Level 2 — Understanding

6. derive `∂MSE/∂w` ใน scalar case
7. ทำไม polynomial regression ยัง linear in parameters?
8. ทำไม Ridge ช่วย multicollinearity?
9. ทำไม standardization สำคัญกับ Lasso/Ridge?
10. ทำไม explicit inverse ไม่ใช่ default numerical strategy?

## Level 3 — Coding

11. implement RMSE
12. เพิ่ม `score_r2` method
13. implement polynomial features สำหรับหลาย input features
14. plot loss history ของ GD
15. compare degree 1..10 ด้วย validation MSE

## Level 4 — Debugging

16. GD loss กลายเป็น inf — diagnose
17. Ridge fit ดีแต่ coefficients แปลกเพราะ features คนละ scale
18. polynomial test score แย่มากแต่ train scoreเกือบ perfect

## Level 5 — Challenge

19. implement coordinate-descent Lasso
20. implement Elastic Net GD/proximal variant แล้วเทียบ sklearn
