# Regularization — Ridge, Lasso, Elastic Net

## Why penalize weights?

เมื่อ model มีหลาย solutions ที่ fit training ใกล้กัน เราอาจ prefer solution ที่ coefficient ไม่สุดโต่ง

## Ridge

```text
J(w)=MSE(w)+λΣw_j²
```

gradient penalty:

```text
∂/∂w_j λw_j² = 2λw_j
```

weights จึงถูกผลักเข้าหา 0 อย่าง smooth

## Lasso

```text
J(w)=data_loss+λΣ|w_j|
```

L1 geometry มีมุมบน axes จึงเกิด exact-zero solution ได้บ่อย

นี่คือ intuition เชิง geometry ไม่ใช่คำรับประกันว่า feature ที่เป็นศูนย์ “ไม่มีความสำคัญจริง”

## Elastic Net

```text
J(w)=data_loss+λ₁||w||₁+λ₂||w||²₂
```

เหมาะเมื่ออยากได้ sparsity แต่ features correlated ทำให้ pure Lasso unstable

## Standardization

penalty ทำงานบน coefficient magnitude

จึงต้องทำให้ feature scales comparable เมื่อ interpretation/penalty ต้องการ

## Alpha selection

อย่าเลือก alpha จาก test

ใช้:
- validation set
- cross-validation

จากนั้น retrain ตาม protocol ที่กำหนดและ evaluate test ครั้งสุดท้าย
