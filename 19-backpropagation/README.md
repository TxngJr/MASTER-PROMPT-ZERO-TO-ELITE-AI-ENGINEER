# Chapter 19 — Backpropagation & Reverse-Mode Autodiff

## 1. Goal

Chapter 18 ทำได้เฉพาะ forward pass:

~~~text
x → Linear → ReLU → Linear → prediction → loss
~~~

แต่ model ยังเรียนไม่ได้ เพราะเรายังไม่รู้ว่า parameter แต่ละตัวทำให้ loss เปลี่ยนอย่างไร

Chapter 19 เพิ่ม:

~~~text
loss
 ↓ chain rule
gradients
 ↓
parameters
~~~

เป้าหมายคือสร้าง tiny reverse-mode autodiff engine จากศูนย์

## 2. Derivative

ถ้า y=f(x), derivative dy/dx บอก local sensitivity ของ y ต่อ x

ตัวอย่าง:

~~~text
y=x²
dy/dx=2x
~~~

## 3. Partial Derivative

ถ้า z=f(x,y) จะมี partial derivatives ต่อ input แต่ละตัว

Neural network มี parameters จำนวนมาก จึงต้องจัดการ partial derivatives เป็นระบบ

## 4. Chain Rule

ถ้า:

~~~text
a=f(x)
L=g(a)
~~~

แล้ว:

~~~text
dL/dx = dL/da × da/dx
~~~

นี่คือแก่นของ backpropagation

## 5. Computational Graph

~~~text
x ─┐
   × → z → ReLU → a → loss
w ─┘
~~~

forward คำนวณ values  
backward คำนวณ local gradients และ compose ด้วย chain rule

## 6. Local Gradient

สำหรับ z=x*y:

~~~text
∂z/∂x = y
∂z/∂y = x
~~~

ถ้า upstream gradient g=∂L/∂z:

~~~text
∂L/∂x = g*y
∂L/∂y = g*x
~~~

## 7. Addition

สำหรับ z=x+y local gradient ต่อ input ทั้งสองเท่ากับ 1

gradient จึงถูกส่งต่อไปทั้งสอง branches

## 8. Gradient Accumulation

ถ้า x ถูกใช้หลาย path:

~~~text
a=x*x
b=x+3
L=a+b
~~~

gradient จากทุก path ต้องบวกกัน ไม่ใช่ overwrite

PyTorch autograd ปัจจุบันก็สะสม gradient ใน grad ของ leaves เมื่อ backward ถูกเรียก จึงต้อง zero/reset ก่อน optimization step ถัดไป

## 9. Reverse-Mode Automatic Differentiation

เมื่อมี scalar loss หนึ่งตัวและ parameters จำนวนมาก reverse mode มีประสิทธิภาพ เพราะ propagate gradient จาก output ย้อนกลับไป inputs ทั้งหมดใน pass เดียว

## 10. Topological Order

1. DFS จาก loss
2. สร้าง topological list
3. seed loss gradient
4. traverse reversed list
5. call local backward function

## 11. Scalar Loss Seed

สำหรับ scalar loss:

~~~text
dL/dL = 1
~~~

non-scalar output ต้องมี upstream gradient/Jacobian-vector-product specification

## 12. Broadcasting

forward:

~~~text
X shape (32,64)
b shape (64,)
Z = X + b
~~~

NumPy broadcast b ไปทุก row

backward ต้อง reverse broadcasting:

~~~text
db = sum(dZ, axis=0)
~~~

## 13. Matrix Multiplication

~~~text
Z = XW
G = dL/dZ

dL/dX = G Wᵀ
dL/dW = Xᵀ G
~~~

นี่คือหัวใจของ Dense layer backprop

## 14. Activation Backward

ReLU:

~~~text
1 if x>0
0 if x<0
~~~

Sigmoid:

~~~text
dσ/dx = σ(x)(1-σ(x))
~~~

Tanh:

~~~text
d tanh(x)/dx = 1-tanh²(x)
~~~

## 15. Sum / Mean Backward

sum กระจาย upstream gradient ไปทุก element

mean เหมือน sum แต่หารด้วยจำนวน elements ที่ reduce

## 16. Gradient Checking

central difference:

~~~text
df/dx ≈ [f(x+ε)-f(x-ε)]/(2ε)
~~~

ใช้ตรวจ analytical gradient

gradient checking ช้าและใช้เพื่อ debugging ไม่ใช่ training production

## 17. Relative Error

~~~text
|g_analytic-g_numeric|
/
max(1, |g_analytic|, |g_numeric|)
~~~

## 18. Dynamic Graph

tiny engine ของเราสร้าง graph ระหว่าง forward execution

PyTorch autograd ก็ใช้ reverse automatic differentiation บน computation graph และสร้าง graph จาก operations ระหว่าง forward

## 19. Tensor Engine

ดู src/tensor.py

รองรับ:
- add/subtract
- multiply/divide
- power
- matrix multiplication
- sum/mean
- reshape
- exp/log
- tanh
- sigmoid
- ReLU
- broadcasting-aware gradients
- gradient accumulation
- reverse-mode backward

## 20. Intentional Limits

ยังไม่มี:
- GPU
- convolutions
- sparse tensors
- mixed precision
- graph optimization
- distributed autograd
- higher-order gradients
- in-place safety engine

เป้าหมายคือเข้าใจ mechanism

## 21. Common Mistakes

1. overwrite gradients แทน accumulate
2. backward wrong graph order
3. forget transpose in matmul
4. ignore broadcasting
5. seed non-scalar backward incorrectly
6. stale parameter gradients
7. gradient check ด้วย epsilon แย่มาก
8. mutate data used by backward closure
9. expect autodiff to fix unstable forward math

## 22. Mini Project

[Autodiff Gradient Lab](mini-project/README.md)

## 23. Exit Checklist

- [ ] chain rule
- [ ] local gradient
- [ ] reverse mode
- [ ] topological sort
- [ ] accumulation
- [ ] broadcasting
- [ ] matmul gradients
- [ ] numerical checking
- [ ] Tensor.backward()

## 24. Next

Chapter 20 จะใช้ gradients เหล่านี้เพื่อ update parameters ด้วย SGD, Momentum, RMSProp, Adam และ AdamW
