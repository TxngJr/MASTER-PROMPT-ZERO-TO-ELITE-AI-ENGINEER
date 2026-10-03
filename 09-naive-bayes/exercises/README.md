# Chapter 09 Exercises

## Level 1 — Recall

1. Bayes theorem สูตรอะไร?
2. prior คืออะไร?
3. likelihood คืออะไร?
4. naive assumption คืออะไร?
5. Laplace smoothing คืออะไร?

## Level 2 — Understanding

6. ทำไมใช้ log probabilities?
7. GaussianNB assume อะไรต่อ feature?
8. MultinomialNB เหมาะกับข้อมูลแบบใด?
9. BernoulliNB ต่างจาก MultinomialNB อย่างไร?
10. ทำไม posterior probability อาจ overconfident?

## Level 3 — Coding

11. เพิ่ม `predict_log_proba` ให้ GaussianNB
12. เพิ่ม `predict_proba` ด้วย log-sum-exp normalization
13. implement BernoulliNB
14. ทดลอง alpha หลายค่า
15. compare class priors แบบ empirical vs fixed

## Level 4 — Debugging

16. probability product กลายเป็น 0
17. Gaussian variance ของ feature หนึ่งเป็น 0
18. MultinomialNB รับ StandardScaler output แล้ว error/ให้ผลผิด

## Level 5 — Challenge

19. ทำ tiny bag-of-words tokenizer + MultinomialNB
20. วิเคราะห์ผลเมื่อ duplicate correlated features เพื่อเห็น independence assumption breakdown
