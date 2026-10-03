# Mini Project — Model Evaluation Lab

## Goal

สร้าง experiment runner ขนาดเล็กที่ยังไม่ผูกกับ model ชนิดใดชนิดหนึ่ง

## Required

1. โหลด/generate targets
2. mean baseline
3. candidate predictions อย่างน้อย 3 แบบ
4. คำนวณ MSE/MAE/R²
5. K-fold indices
6. report mean/std ของ metric ข้าม folds
7. เก็บ config/seed
8. save JSON report

## Optimization experiment

ใช้ `gradient_descent_quadratic` แล้ว plot:

- objective curve
- trajectory
- final error

ทดลอง learning rate หลายค่าและอธิบาย:

- convergence
- oscillation
- divergence

## Mastery question

ถ้า model A validation MSE ดีกว่า B เพียงเล็กน้อย:

- split uncertainty มีเท่าไร?
- baseline เป็นเท่าไร?
- difference stable ข้าม folds ไหม?
- computational cost ต่างกันไหม?

อย่าเลือก model จากเลขเดียวโดยไม่มี context
