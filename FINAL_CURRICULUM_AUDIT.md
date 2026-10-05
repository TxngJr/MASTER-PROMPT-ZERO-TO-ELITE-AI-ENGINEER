# Final Curriculum Audit — Zero to Elite AI Engineer

## Audit Scope

ตรวจ Chapters 01–80 กับเป้าหมายเดิม

เอกสารนี้ **ไม่อ้างว่าครอบคลุม AI ทุกสิ่งที่มนุษย์เคยสร้าง** เพราะ field เปลี่ยนต่อเนื่อง

Status:
- Core complete
- Covered
- Conceptual
- Supplement recommended

## 1. Programming / Systems Foundations — Core complete

Python, C/C++ overview, memory, Linux/Fedora, Bash, Git/GitHub, environments, processes, testing/debugging

## 2. Mathematics — Core complete

algebra, functions/logs/summation, vectors/matrices, matrix multiplication, eigen/SVD foundations, calculus/chain rule/gradients, probability/distributions, expectation/variance/covariance, statistics/hypothesis foundations

Math ถูกวนกลับมาใช้ใน regression, PCA, backprop, attention, RL, LoRA, alignment และ research evaluation

## 3. Scientific Python / Data — Core complete

NumPy/Pandas/Matplotlib/SciPy/Jupyter, vectorization, cleaning, leakage, splitting, cross-validation, imbalance, data engineering และ LLM data pipelines

## 4. Classical ML — Core complete

Regression, Logistic, KNN, Naive Bayes, Trees, Random Forest, Boosting, SVM, clustering, PCA/SVD/t-SNE/UMAP, anomaly detection, time series

## 5. Deep Learning — Core complete

MLP, computational graphs, backprop/autograd, optimizers, activations/losses, PyTorch, TensorFlow/Keras comparison, CNN/RNN/LSTM/GRU, AE/VAE/GAN

## 6. Computer Vision — Covered

CNN, ViT, object detection, IoU/NMS, Faster R-CNN/YOLO concepts, semantic/instance segmentation, U-Net/Mask R-CNN concepts, VLM links

## 7. NLP / Language Modeling — Core complete

Unicode/tokenization, n-grams, embeddings, Word2Vec/GloVe/FastText, BERT, GPT, T5, BPE, pretraining

## 8. Transformers / LLM Architecture — Core complete

Q/K/V, scaled attention, masks, MHA, Transformer blocks, RMSNorm, RoPE, GQA, SwiGLU, KV cache, prefill/decode, LM head, sampling

## 9. Generative AI — Core complete

autoregressive, latent-variable models, GAN, VAE, diffusion/DDPM/DDIM, generative evaluation

## 10. Reinforcement Learning — Covered

MDP, Bellman, Q-learning, SARSA concepts, DQN, policy gradient, actor-critic, PPO, RLHF links, world models/planning

## 11. Speech / Audio — Covered

waveform/STFT/mel/MFCC concepts, CTC/STT, TTS/vocoders, neural codecs/audio tokens, streaming ASR/TTS, VAD/endpointing

## 12. Multimodal — Covered

dual encoders, contrastive embeddings, fusion, cross-attention, image tokens/projectors, VLM, audio-language concepts

## 13. Recommenders / Retrieval / RAG — Core complete

collaborative/content-based, matrix factorization, BPR/two-tower, embeddings/vector search, ANN/IVF/HNSW/PQ concepts, hybrid retrieval, reranking, chunking/context/citations/evaluation

## 14. Agents — Covered

tool/function calling, state, planning, bounded loops, memory concepts, MCP concepts, multi-agent concepts, recovery, permissions, injection defenses

## 15. LLM Data / Fine-Tuning / Alignment — Core complete

data acquisition/cleaning/dedup/sharding/packing, full fine-tuning, LoRA/QLoRA/PEFT, SFT, reward models, RLHF/PPO/KL, DPO, IPO/ORPO/KTO concepts, evaluation/safety

## 16. LLM Systems — Core complete

quantization, prefill/decode, KV cache, batching, FlashAttention/PagedAttention/speculative concepts, serving, FastAPI, Docker, Kubernetes, MLOps, monitoring, data/distributed systems

## 17. Distributed / GPU Systems — Covered

DDP, FSDP/ZeRO concepts, tensor/pipeline parallel, collectives, CUDA hierarchy/memory/coalescing/occupancy concepts, mixed precision, scaling efficiency

## 18. Safety / Explainability / Compression — Covered

prompt injection, poisoning/jailbreak concepts, secure tools/sandbox/permissions, SHAP/LIME/saliency, pruning/distillation/quantization/low-rank methods

## 19. Advanced Architectures — Covered

MoE, long context, RoPE/ALiBi concepts, reasoning/test-time compute, verifiers/search, world models, robotics/VLA concepts, edge/TinyML

## 20. Research Engineering — Core complete

Chapter 79 adds paper reading, claim extraction, equations, reproduction contracts, baselines, seeds/uncertainty, ablation, metadata และ technical reports

## 21. Final End-to-End Integration — Core complete

Chapter 80 connects:

~~~text
Programming
→ Math
→ Data
→ ML/DL
→ Transformer
→ Tokenizer
→ LLM Pretraining
→ SFT
→ Preference Optimization
→ Quantization
→ Serving
→ Deployment
→ Monitoring
→ Research iteration
~~~

Primary capstone model trains from random initialization

## 22. Hardware Feasibility Audit

หลักสูตรใช้ tiny datasets/models, CPU fallbacks, optional laptop GPU และ separate scaled concepts

สำหรับ 16 GB RAM + laptop GPU ใช้:
- reduce batch/context/model
- accumulation
- mixed precision when stable
- quantization
- checkpoint/offloading concepts
- measure before scaling

## 23. Reproducibility Audit

มี seeds, tests, CI, manifests, checkpoint concepts, package requirements, experiment tracking และ hardware reporting

แต่ bitwise equivalence ข้าม GPU/runtime/library versions ไม่รับประกัน; ต้องรายงาน environment และ tolerance

## 24. Security Audit

ครอบคลุม secrets/API keys, supply-chain awareness, data/model safety, prompt injection, tool permissions, sandboxing, unsafe-deserialization awareness และ deployment limits

## 25. Missing / Adjacent Topics

หัวข้อเหล่านี้เป็น **supplement recommended** ไม่ใช่ prerequisite ที่ขาดของ main AI Engineer path:

### A. Classical symbolic AI
search/CSP/SAT planning, knowledge representation, logic programming

### B. Causal inference
DAGs, interventions, confounding, potential outcomes, treatment effects

### C. Probabilistic graphical models
Bayesian networks, deeper HMMs, factor graphs, message passing

### D. Federated/privacy-preserving ML
federated averaging, secure aggregation concepts, differential privacy/privacy accounting

### E. Scientific ML
PINNs, neural operators, surrogate modeling

### F. Neuromorphic/spiking AI
optional specialization

การระบุ gap เหล่านี้ตรง ๆ ดีกว่าการอ้างว่า course ครอบคลุมทุก subfield

## 26. Dependency Audit

critical chain coherent:

~~~text
math/programming
→ ML
→ neural nets/backprop
→ attention/transformer
→ NLP/GPT
→ tokenizer/pretraining/data
→ systems/fine-tuning/alignment
→ optimization/serving/MLOps
→ safety/advanced
→ research engineering
→ final capstone
~~~

ไม่มี Chapter 81 ที่จำเป็น

## 27. Adversarial Beginner Audit

Strengths:
- objectives/checklists/exercises/projects
- batch integrations
- executable tests
- glossary/references
- local/scaled split

Risk:
course ใหญ่มาก; การ run tests ผ่านไม่เท่ากับ mastery

## 28. Senior Engineer Audit

Strengths:
testing/debugging, performance/memory, reproducibility, deployment/security, distributed/system concerns

Risk:
production maturity ต้องเพิ่มประสบการณ์ incidents, code review, live SLOs และ real data governance

## 29. Mathematician Audit

Strengths:
applied derivations เชื่อมกับ implementation

Risk:
ผู้มุ่ง theoretical ML research ควรเพิ่ม real analysis, optimization theory, measure-theoretic probability และ information theory

## 30. ML Researcher Audit

Strengths:
paper reproduction, ablation, uncertainty, baselines, research connections

Risk:
novel research ต้อง literature depth และ open-ended problem formulation เพิ่มจาก fixed exercises

## 31. Career Mapping

- Data Scientist — data/statistics/classical ML/evaluation
- ML Engineer — training/software/deployment/MLOps
- AI Engineer — RAG/agents/multimodal/end-to-end
- Deep Learning Engineer — architectures/optimization/frameworks
- CV Engineer — CNN/ViT/detection/segmentation
- NLP Engineer — tokenizer/embeddings/BERT/GPT/T5
- LLM Engineer — pretraining/fine-tuning/alignment/RAG/serving
- MLOps Engineer — tracking/registry/pipelines/CI/CD/monitoring
- AI Infrastructure Engineer — distributed/GPU/serving/data systems
- Research Engineer — from-scratch + Chapter 79
- AI Researcher — strong engineering base; advanced math/literature specialization recommended

## 32. Final Mastery Standard

Completion ≠ read all chapters

Mastery means:

### Explain
teach concept without hiding behind library calls

### Derive
derive central equations

### Implement
build representative algorithms

### Debug
diagnose data/math/code/system failures

### Modify
change architecture/training/data safely

### Compare
fair controlled evaluation

### Apply
solve a new problem

### Research
reproduce a paper claim

### Ship
deploy and monitor a bounded AI system

## 33. Final Verdict

Roadmap Programming → Math → ML → DL → Transformer → LLM → Alignment → Systems → Multimodal → Robotics → Edge ถูกเชื่อมครบใน Chapters 01–80 และ integration projects

Repository นี้ควรถูกเรียกว่า **broad end-to-end AI engineering curriculum** ไม่ใช่คำกล่าวว่า exhaustive ต่อ AI ทุก subfield
