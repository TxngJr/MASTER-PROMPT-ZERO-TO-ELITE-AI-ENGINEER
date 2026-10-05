# Hardware Lab Matrix

Canonical local target: Fedora on Acer Aspire 7 A715-43G-class hardware, Ryzen 7 5825U, 16 GB system RAM, optional RTX 3050 Ti Laptop GPU. Do not assume VRAM capacity; confirm it with nvidia-smi before choosing a batch size.

General rule: begin with the smallest dataset/model/batch that proves the mechanism. Increase only after a successful smoke run. Never promise a fixed training time.

| Chapters | Local learning run | RAM / GPU guidance | Scaled version |
|---|---|---|---|
| 18–21 | NumPy tiny networks | CPU; small working set | larger MLPs, accelerators |
| 22–23 | tiny framework MLP | CPU or GPU; start batch 8–64 | larger datasets/models |
| 24 | small CNN | GPU optional; start batch 16–32 | ImageNet-class training |
| 25–26 | short sequences, small hidden sizes | batch 8–32 initially | long sequences/larger recurrent models |
| 27–28 | tiny latent/generative models | GPU optional; batch 8–32 initially | larger generative training |
| 29–36 | tiny attention/Transformer models | context 32–128, batch 2–16 initially | long context/pretrained-scale models |
| 37 | small graphs | CPU first | large graph sampling/distributed GNN |
| 38 | toy environments | CPU sufficient for tabular/tiny agents | parallel simulators/larger policies |
| 40–42 | tiny diffusion/multimodal/audio examples | reduce resolution/audio duration first; batch 1–8 | multi-GPU training |
| 43–47 | small retrieval/recommendation/agent data | CPU first | production retrieval/services |
| 48–51 | tiny random-init language models | context 32–128, batch 1–8 | large pretraining clusters |
| 52–54 | simulation plus local accelerator | single-GPU concepts locally | multi-GPU/multi-node |
| 55–60 | tiny post-training/eval smoke tests | batch 1–8; use parameter-efficient methods for larger models | distributed post-training |
| 61–63 | quantized/tiny inference and serving | measure resident memory and KV cache | production serving fleets |
| 64–69 | small deployment/MLOps/safety/XAI examples | CPU-first CI; optional GPU inference | managed clusters |
| 70–75 | tiny compression/MoE/context/reasoning/multimodal/audio demos | batch 1–4 when memory bound | specialized accelerators |
| 76–78 | tiny world/robotics/edge simulations | CPU and simulated constraints first | robotics/embedded hardware |
| 79 | paper reproduction smoke experiments | smallest reproducible baseline | author-scale reproduction |
| 80 | tiny random-init LLM capstone | context 32–128, batch 1–4 initially; confirm VRAM | distributed pretraining |

## Memory estimation
- parameters approximately equal parameter_count times bytes_per_parameter
- gradients add roughly one parameter-sized copy when stored
- optimizer state depends on optimizer and precision; Adam-family methods add multiple state tensors
- activations depend on batch, sequence/image size, layer widths and checkpointing
- autoregressive inference adds KV-cache memory that grows with batch and context

If a run approaches system or GPU memory limits, reduce batch size, sequence/resolution, model width/depth, or dataset working set before using more aggressive techniques.
