# Chapter 24 — Convolutional Neural Networks

## 1. Why CNNs?

A fully connected layer treats every input position as unrelated.

Images have spatial structure:
- nearby pixels interact
- edges repeat across locations
- local patterns combine into larger patterns

CNNs exploit this with:
- local connectivity
- weight sharing
- hierarchical receptive fields

## 2. Learning Objectives

By the end of this chapter you should be able to:

- calculate convolution output shapes
- explain channels and filters
- implement 2D cross-correlation from scratch
- explain stride and padding
- explain receptive fields
- implement max pooling
- count Conv2D parameters
- distinguish NCHW and NHWC
- build CNNs in PyTorch and Keras
- understand pooling vs strided convolution
- explain translation-related inductive bias
- debug CNN shape errors
- train a small image classifier

## 3. Convolution vs Cross-Correlation

Deep-learning libraries usually implement cross-correlation:

~~~text
output[i,j]
=
sum over kernel window:
input[i+u,j+v] * kernel[u,v]
~~~

Classical mathematical convolution flips the kernel first.

In deep learning, learned kernels make this naming distinction less important operationally, but you should know it.

## 4. Single-Channel Example

Input:

~~~text
H × W
~~~

Kernel:

~~~text
KH × KW
~~~

valid output:

~~~text
(H-KH+1) × (W-KW+1)
~~~

for stride 1 and no padding.

## 5. General Output Shape

For one spatial dimension:

~~~text
out =
floor(
  (input + 2*padding - dilation*(kernel-1) - 1)
  / stride
) + 1
~~~

With dilation=1:

~~~text
out =
floor((input + 2P - K)/S) + 1
~~~

Always calculate before coding.

## 6. Channels

RGB image:

~~~text
C_in = 3
~~~

A filter spans all input channels:

~~~text
weight shape:
C_out × C_in × KH × KW
~~~

Each output filter produces one output channel.

## 7. Conv Parameter Count

~~~text
C_out * C_in * KH * KW
+
C_out biases
~~~

Example:

~~~text
3 input channels
16 output channels
3×3 kernel

16*3*3*3 + 16
= 448 parameters
~~~

Weight sharing keeps parameter count much smaller than a dense layer over all pixels.

## 8. NCHW vs NHWC

PyTorch commonly:

~~~text
N × C × H × W
~~~

TensorFlow/Keras default:

~~~text
N × H × W × C
~~~

Never silently reshape one into the other.

Use explicit transpose/permutation.

## 9. Padding

No padding shrinks spatial resolution.

For odd kernel K and stride 1, common same-like padding is:

~~~text
P = (K-1)/2
~~~

for dilation 1.

Example:
- K=3 → P=1
- K=5 → P=2

## 10. Stride

Stride controls window movement.

Stride 2 roughly downsamples spatial dimensions by about 2.

Larger stride:
- lower compute
- lower spatial resolution
- may discard details

## 11. Dilation

Dilation inserts gaps between kernel elements.

Effective kernel size:

~~~text
K_eff = dilation*(K-1)+1
~~~

It expands receptive field without increasing kernel parameter count.

## 12. Receptive Field

A neuron in later layers depends on a region of the original image.

Stacking 3×3 convolutions grows receptive field while preserving nonlinear depth.

Receptive field depends on:
- kernel
- stride
- dilation
- previous layer jumps

## 13. Why Weight Sharing Matters

The same filter scans across image positions.

This encodes an inductive bias:

> a useful local pattern may matter regardless of its exact location

CNNs are not perfectly translation invariant by default, but convolution provides translation-equivariant structure before pooling/boundary effects and other operations.

## 14. Activation

Typical block:

~~~text
Conv
→ ReLU
→ Conv
→ ReLU
→ Pool/Stride
~~~

Without nonlinearities, stacked linear convolution layers still collapse into a linear mapping.

## 15. Max Pooling

For each local window:

~~~text
output = max(window)
~~~

Effects:
- downsampling
- local feature selection
- reduced compute downstream

But pooling discards information.

Modern architectures sometimes prefer strided convolution or global average pooling.

## 16. Average Pooling

~~~text
output = mean(window)
~~~

Global Average Pooling:

~~~text
N × C × H × W
→
N × C
~~~

reduces each feature map to one value.

It can replace large fully connected heads.

## 17. Feature Hierarchy

Early layers often respond to simple local patterns.

Deeper layers combine earlier representations into more complex structures.

Do not over-literalize filters as human concepts; representation is distributed.

## 18. Normalization Preview

CNN stacks often use:
- BatchNorm
- LayerNorm
- GroupNorm

These are covered more deeply later.

## 19. Dropout

Dropout may regularize dense/CNN heads, but placement and value matter.

Do not add dropout automatically before diagnosing overfitting.

## 20. PyTorch CNN

~~~python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
~~~

## 21. Keras CNN

~~~python
model = keras.Sequential([
    keras.layers.Input(shape=(8, 8, 1)),
    keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10),
])
~~~

Same concept, different default image layout.

## 22. From Scratch

src/cnn_numpy.py includes:
- conv2d_nchw
- max_pool2d_nchw
- output-size helper

The implementation uses loops deliberately for transparency.

Production frameworks use optimized kernels and accelerator libraries.

## 23. Common Bugs

1. NCHW/NHWC mismatch
2. wrong output shape formula
3. wrong channel count
4. bias broadcast on wrong axis
5. accidental integer image dtype
6. forgetting normalization
7. softmax before cross-entropy logits loss
8. flattening too early
9. oversized dense head
10. comparing models with different splits

## 24. Performance

Convolution cost roughly grows with:

~~~text
N * Cout * Hout * Wout * Cin * KH * KW
~~~

Actual performance depends on:
- memory layout
- kernel implementation
- accelerator
- dtype
- batch size

## 25. Your RTX 3050 Ti

For 8×8/28×28 educational CNNs the GPU may not always beat CPU because launch/transfer overhead matters.

Use GPU when:
- batch/model/image size grows
- profiling shows benefit

Do not judge accelerator quality from a tiny toy kernel alone.

## 26. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 27. Checklist

- [ ] output shape
- [ ] channels
- [ ] parameter count
- [ ] padding/stride/dilation
- [ ] receptive field
- [ ] pooling
- [ ] NCHW/NHWC
- [ ] PyTorch CNN
- [ ] Keras CNN
- [ ] debug shapes

## 28. What's Next

Batch 09 continues into sequence models: RNN, LSTM/GRU, and Autoencoders/VAEs.
