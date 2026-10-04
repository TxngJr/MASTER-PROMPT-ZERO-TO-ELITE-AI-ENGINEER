# Chapter 22 — PyTorch

## 1. Why PyTorch Now?

Chapters 18–21 built the ideas manually:

~~~text
Tensor
→ computational graph
→ backward
→ gradients
→ optimizer
→ parameter update
~~~

PyTorch gives you the production-grade version of that stack:

~~~text
torch.Tensor
→ autograd
→ nn.Module
→ loss
→ torch.optim
→ DataLoader
→ CPU / CUDA
~~~

The goal is not to forget the from-scratch implementation. The goal is to map every PyTorch abstraction back to concepts you already understand.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- create tensors with explicit dtype/device
- reason about shape, stride, view/reshape, broadcasting
- move tensors/models between CPU and CUDA
- use requires_grad and backward
- inspect gradients
- build models with nn.Module
- register Parameters correctly
- use train/eval modes
- write a training loop
- use Dataset and DataLoader
- save/load state_dict
- disable gradients for inference
- debug device/dtype/shape mismatches
- understand CUDA memory basics

## 3. Installation on Fedora

Do not hard-code an old CUDA wheel command.

Use the current official PyTorch install selector:
https://pytorch.org/get-started/locally/

Choose:
- Linux
- Pip
- Python
- the CUDA build appropriate for the currently supported PyTorch release and your NVIDIA driver

Verify:

~~~python
import torch

print(torch.__version__)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
~~~

The official docs currently require Python 3.10+ for latest stable PyTorch.

## 4. Device Mental Model

A Tensor has:
- data
- dtype
- shape
- device
- optional gradient history

Example:

~~~python
x = torch.randn(32, 128, device=device)
~~~

CPU and CUDA tensors cannot generally participate in the same operation without moving one of them.

~~~python
x = x.to(device)
model = model.to(device)
~~~

## 5. Tensor Shapes

Always trace:

~~~text
batch × features
batch × channels × height × width
batch × sequence × embedding
~~~

PyTorch CNN convention is usually NCHW:

~~~text
N = batch
C = channels
H = height
W = width
~~~

## 6. dtype

Common:
- float32
- float16
- bfloat16
- int64 for many class-index targets
- bool masks

Do not cast labels to float if CrossEntropyLoss expects class indices.

## 7. Autograd

~~~python
x = torch.tensor(3.0, requires_grad=True)
y = x * x + 2 * x
y.backward()
print(x.grad)
~~~

Autograd performs reverse-mode differentiation through the dynamic graph.

## 8. Gradient Accumulation

PyTorch accumulates gradients.

Typical loop:

~~~python
optimizer.zero_grad()
loss.backward()
optimizer.step()
~~~

If zero_grad is omitted, gradients from previous steps accumulate.

## 9. nn.Module

A model subclasses nn.Module:

~~~python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
~~~

Calling model(x) invokes Module machinery around forward.

## 10. Parameter Registration

Assign nn.Parameter or child nn.Module as attributes.

Bad pattern:
- storing raw tensors that require gradients but are not registered intentionally

Check:

~~~python
for name, p in model.named_parameters():
    print(name, p.shape)
~~~

## 11. train() vs eval()

~~~python
model.train()
~~~

sets training behavior for modules such as Dropout/BatchNorm.

~~~python
model.eval()
~~~

sets evaluation behavior.

eval does NOT disable gradients.

For inference use:

~~~python
model.eval()
with torch.inference_mode():
    ...
~~~

## 12. Loss Functions

Binary classification:
- BCEWithLogitsLoss

Multiclass:
- CrossEntropyLoss

Regression:
- MSELoss
- L1Loss
- SmoothL1Loss

Prefer combined logits losses when available because they implement stable math.

## 13. CrossEntropyLoss

For multiclass logits shape:

~~~text
logits: (N, C)
target: (N,) integer class ids
~~~

Do not manually softmax before CrossEntropyLoss.

## 14. Dataset / DataLoader

Dataset defines sample access.

DataLoader handles:
- batching
- shuffling
- workers
- collation
- optional pinned memory

For training:
- shuffle=True is normal for IID datasets

For time series:
- preserve causal split and be careful about sequence construction

## 15. A Canonical Training Loop

~~~python
model.train()

for x, y in loader:
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
~~~

## 16. Validation Loop

~~~python
model.eval()

with torch.inference_mode():
    for x, y in validation_loader:
        ...
~~~

No optimizer steps.

## 17. state_dict

Save learned state:

~~~python
torch.save(model.state_dict(), "model.pt")
~~~

Load:

~~~python
model.load_state_dict(torch.load("model.pt", map_location=device))
~~~

For long-term reproducibility also record:
- architecture/config
- preprocessing
- class mapping
- framework version
- training metadata

## 18. Reproducibility

Seed:
- Python
- NumPy
- torch
- CUDA where relevant

But exact GPU determinism can depend on kernels/hardware/settings.

Reproducible does not mean statistically robust; still repeat experiments.

## 19. CUDA Memory

Important concepts:
- allocated tensor memory
- cached allocator memory
- activations
- gradients
- optimizer states
- model parameters

Useful:

~~~python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
~~~

Do not assume empty_cache fixes a memory leak.

## 20. Your RTX 3050 Ti Laptop GPU

Start with small batches.

For early chapters:
- batch 32–128
- float32
- small CNN/MLP
- num_workers 0–2 initially

Later chapters cover:
- mixed precision
- gradient accumulation
- checkpointing
- CUDA profiling

## 21. Debug Checklist

If you get a device error:
- print x.device
- print next(model.parameters()).device

If loss is NaN:
- inspect input range
- logits
- learning rate
- dtype
- target encoding

If shapes fail:
- print every intermediate shape

If CUDA unavailable:
- verify nvidia-smi
- verify torch.cuda.is_available()
- reinstall using the current official selector rather than guessing CUDA wheel versions

## 22. Source

See src/pytorch_basics.py

It provides:
- choose_device
- seed_everything
- TinyClassifier
- one train step
- parameter counting

## 23. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 24. Checklist

- [ ] Tensor/device/dtype
- [ ] autograd
- [ ] zero_grad
- [ ] nn.Module
- [ ] parameters
- [ ] train/eval
- [ ] inference_mode
- [ ] Dataset/DataLoader
- [ ] state_dict
- [ ] CUDA debugging

## 25. What's Next

Chapter 23 maps the same concepts to TensorFlow/Keras and compares framework philosophies.
