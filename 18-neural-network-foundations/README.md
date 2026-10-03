# Chapter 18 — Neural Network Foundations

## 1. Why This Matters

Neural Network ดูซับซ้อนเพราะมี layers จำนวนมาก แต่แก่นพื้นฐานคือ:

~~~text
linear transformation
→ nonlinearity
→ linear transformation
→ nonlinearity
→ ...
~~~

บทนี้ยังไม่ทำ backpropagation เต็มรูปแบบ

เป้าหมายคือให้คุณเข้าใจ **forward pass** ทุกบรรทัดก่อน Chapter 19

## 2. Learning Objectives

เมื่อจบบทนี้คุณควร:

- อธิบาย neuron
- implement perceptron
- อธิบาย weights / bias
- ใช้ matrix form ของ dense layer
- implement Linear layer
- implement ReLU / Sigmoid / Tanh
- อธิบาย why nonlinear activation is necessary
- implement MLP forward pass
- calculate MSE / binary cross-entropy
- trace tensor/array shapes
- understand parameter count
- explain computational graph concept
- understand initialization basics
- explain why XOR breaks single linear separator

## 3. Artificial Neuron

input:

~~~text
x = [x1, x2, ..., xd]
~~~

weights:

~~~text
w = [w1, w2, ..., wd]
~~~

pre-activation:

~~~text
z = wᵀx + b
~~~

activation:

~~~text
a = φ(z)
~~~

## 4. Weight Meaning

weight บอก sensitivity/direction ของ input ใน linear combination

แต่:
- learned weight ไม่เท่ากับ causality
- scale ของ features มีผลต่อ magnitude
- deep network weights ต้องตีความร่วมกับ layers อื่น

## 5. Bias

bias ทำให้ boundary/activation shift ออกจาก origin

ถ้าไม่มี bias:

~~~text
z = wᵀx
~~~

hyperplane ต้องผ่าน origin

## 6. Perceptron

classic binary perceptron:

~~~text
ŷ = sign(wᵀx+b)
~~~

ถ้าทายผิด:

~~~text
w ← w + η y x
b ← b + η y
~~~

โดย y ∈ {-1,+1}

perceptron converge เมื่อ data linearly separable ภายใต้ assumptions ที่เหมาะสม

## 7. Perceptron Limitation

XOR:

~~~text
0 0 → 0
0 1 → 1
1 0 → 1
1 1 → 0
~~~

ไม่สามารถแยกด้วยเส้นตรงเส้นเดียว

นี่เป็นเหตุผลหนึ่งที่ต้องมี hidden layers + nonlinear activations

## 8. Dense / Linear Layer

batch input:

~~~text
X ∈ R^(N × Din)
~~~

weights:

~~~text
W ∈ R^(Din × Dout)
~~~

bias:

~~~text
b ∈ R^(Dout)
~~~

output:

~~~text
Z = XW + b
~~~

shape:

~~~text
(N × Din)(Din × Dout)
→
N × Dout
~~~

bias ใช้ broadcasting across batch

## 9. Parameter Count

dense layer:

~~~text
Din × Dout weights
+
Dout biases
~~~

total:

~~~text
Din*Dout + Dout
~~~

ตัวอย่าง 100→64:

~~~text
100*64 + 64 = 6464 parameters
~~~

## 10. Why Nonlinearity?

ถ้า stack linear layers:

~~~text
XW1W2W3
~~~

ทั้งหมดรวมเป็น matrix เดียวได้:

~~~text
XW_equivalent
~~~

ดังนั้น deep stack ที่ไม่มี nonlinear activation ยังคงเป็น linear transformation

## 11. ReLU

~~~text
ReLU(x)=max(0,x)
~~~

ข้อดี:
- simple
- cheap
- non-saturating positive side

ข้อเสีย:
- negative side gradient 0 ใน backprop
- dead neurons เป็นไปได้

gradient จะเรียน Chapter 19

## 12. Sigmoid

~~~text
σ(x)=1/(1+e^-x)
~~~

range:

~~~text
(0,1)
~~~

เหมาะกับ binary output probability parameterization

hidden layers ปัจจุบันมักใช้ activation อื่นมากกว่า sigmoid เพราะ saturation/optimization

## 13. Tanh

~~~text
tanh(x)
~~~

range:

~~~text
(-1,1)
~~~

zero-centered กว่า sigmoid แต่ยัง saturate

## 14. Softmax Preview

multiclass logits:

~~~text
z1,...,zK
~~~

softmax:

~~~text
p_k = exp(z_k)/Σ_j exp(z_j)
~~~

ต้องใช้ stable version:

~~~text
z ← z - max(z)
~~~

รายละเอียด cross-entropy/backprop จะเรียนต่อ

## 15. Hidden Layer

MLP 1 hidden layer:

~~~text
X
↓ Linear(Din,H)
Z1
↓ ReLU
A1
↓ Linear(H,Dout)
Z2
~~~

classification อาจตามด้วย sigmoid/softmax ตาม task

## 16. Forward Pass

forward pass = คำนวณจาก input ไป output

ไม่มี parameter update ในขั้นนี้

~~~text
input
→ layer 1
→ activation
→ layer 2
→ logits/output
~~~

## 17. Loss

loss แปลง prediction quality เป็น scalar objective

### MSE

~~~text
MSE = mean((y-ŷ)^2)
~~~

### Binary Cross-Entropy

~~~text
BCE =
-mean(
 y log(p)
 +(1-y)log(1-p)
)
~~~

บท 19 จะถามว่า loss เปลี่ยนอย่างไรเมื่อ weight แต่ละตัวเปลี่ยน

## 18. Computational Graph

สมการ:

~~~text
z = x*w + b
a = ReLU(z)
L = (a-y)^2
~~~

มองเป็น graph:

~~~text
x ─┐
   × → + → ReLU → loss
w ─┘   ↑
       b
~~~

แต่ละ node:
- receives values
- computes output
- later can propagate derivatives backward

บทนี้ trace values  
บท 19 trace gradients

## 19. Initialization

ถ้า weights ทุกตัวเริ่มเหมือนกันใน hidden layer:
- neurons อาจเรียนเหมือนกัน
- symmetry ไม่ถูก break

จึง random initialize

แต่ scale สำคัญ:
- ใหญ่เกิน → activations explode/saturate
- เล็กเกิน → signal shrink

Xavier/He initialization จะเรียนลึกขึ้นใน Deep Learning chapters

## 20. Batch Dimension

อย่าเขียน code ที่เข้าใจแค่ sample เดียว

single sample:

~~~text
(Din,)
~~~

batch:

~~~text
(N,Din)
~~~

deep learning frameworksใช้ batch เป็นหลัก

## 21. Feature / Hidden / Output Dimensions

ตัวอย่าง:

~~~text
Input:  (32, 10)
W1:     (10, 64)
A1:     (32, 64)
W2:     (64, 1)
Output: (32, 1)
~~~

shape tracing เป็น skill debugging สำคัญมาก

## 22. Classification Output

binary:
- one logit
- sigmoid
- threshold

multiclass:
- K logits
- softmax
- argmax

แต่ production loss APIs บาง frameworkรวม stable sigmoid/softmax กับ cross-entropy ภายใน จึงห้าม double-apply โดยไม่อ่าน docs

## 23. Regression Output

มักใช้ linear output:

~~~text
ŷ = Z_last
~~~

ไม่จำเป็นต้อง activation ถ้า target unrestricted real value

## 24. Universal Approximation — Caution

MLP ที่มี hidden layerและ nonlinear activationสามารถ approximate functions กว้างมากภายใต้ conditions

แต่ theorem ไม่ได้แปลว่า:
- train ง่าย
- data น้อยก็พอ
- generalize ดี
- architecture ไหนก็เหมาะ

## 25. From Scratch

ดู src/neural_foundations.py

มี:
- PerceptronClassifier
- Linear
- relu
- sigmoid
- tanh
- softmax
- mse_loss
- binary_cross_entropy
- TinyMLP forward pass

ไม่มี backprop intentionally

## 26. Training Perceptron vs Training MLP

Perceptron:
- update rule directlyจาก misclassification

MLP:
- parameters หลาย layers
- ต้องใช้ chain rule/backprop
- optimizer update

นั่นคือ Chapters 19–20

## 27. Common Mistakes

1. matrix shapes ไม่ตรง
2. bias shape ผิด
3. stack linear layersโดยไม่มี activationแล้วคิดว่า nonlinear
4. sigmoid overflow
5. softmax overflow
6. BCE log(0)
7. output activation ไม่ตรง task
8. double sigmoid/softmax กับ combined loss API
9. parameter countผิดเพราะลืม bias
10. คิด forward pass = training

## 28. Exercises / Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 29. Checklist

- [ ] neuron
- [ ] perceptron
- [ ] dense layer
- [ ] matrix shapes
- [ ] ReLU/Sigmoid/Tanh
- [ ] softmax stability
- [ ] MLP forward
- [ ] loss
- [ ] parameter count
- [ ] computational graph
- [ ] initialization intuition

## 30. What's Next

Chapter 19 — Backpropagation จะ derive chain rule ผ่าน computational graph แล้วสร้าง tiny autodiff/backprop engine
