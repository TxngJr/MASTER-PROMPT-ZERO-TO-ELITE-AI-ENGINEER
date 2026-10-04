# Zero to Elite AI Engineer

หลักสูตรภาษาไทยแบบ **from first principles** สำหรับการเรียน AI ตั้งแต่ Programming/Math ไปจนถึงสร้าง LLM, fine-tune, align, optimize และ deploy โดยใช้ Fedora Linux เป็นสภาพแวดล้อมหลัก

> เป้าหมายของ repository นี้ไม่ใช่การจำ API แต่คือ เข้าใจ → derive → implement → debug → experiment → build

## Target machine

หลักสูตรออกแบบให้ทำ Local Lab ได้บน laptop ระดับ:

- Fedora Linux
- AMD Ryzen 7 5825U
- NVIDIA RTX 3050 Ti Laptop GPU
- RAM 16 GB
- SSD 512 GB

งานที่ใหญ่เกินเครื่องจะมี **Local Version** และ **Scaled Version** แยกกันในบทที่เกี่ยวข้อง

## Completed curriculum

### Batch 01 — Foundations

1. [Chapter 01 — Programming Foundations](01-programming/README.md)
2. [Chapter 02 — Mathematics Foundations](02-mathematics/README.md)
3. [Chapter 03 — Scientific Python](03-scientific-python/README.md)

Integration:
[Mini Data Science Engine](integration-project/README.md)

### Batch 02 — Data, ML Fundamentals & Regression

4. [Chapter 04 — Data Fundamentals](04-data/README.md)
5. [Chapter 05 — Machine Learning Fundamentals](05-ml-fundamentals/README.md)
6. [Chapter 06 — Regression](06-regression/README.md)

Integration:
[Leakage-Safe Regression System](integration-project-batch02/README.md)

### Batch 03 — Classification Foundations

7. [Chapter 07 — Logistic Regression](07-logistic-regression/README.md)
8. [Chapter 08 — K-Nearest Neighbors](08-knn/README.md)
9. [Chapter 09 — Naive Bayes](09-naive-bayes/README.md)

Integration:
[Classification Model Lab](integration-project-batch03/README.md)

### Batch 04 — Trees & Ensembles

10. [Chapter 10 — Decision Trees](10-decision-tree/README.md)
11. [Chapter 11 — Random Forest](11-random-forest/README.md)
12. [Chapter 12 — Gradient Boosting](12-gradient-boosting/README.md)

Integration:
[Tree Ensemble Benchmark](integration-project-batch04/README.md)

Optional boosting libraries:

~~~bash
python -m pip install -r requirements-batch04-extras.txt
~~~

### Batch 05 — Geometry, Clustering & Representation

13. [Chapter 13 — Support Vector Machines](13-svm/README.md)
14. [Chapter 14 — Clustering](14-clustering/README.md)
15. [Chapter 15 — Dimensionality Reduction](15-dimensionality-reduction/README.md)

Integration:
[Geometry & Representation Lab](integration-project-batch05/README.md)

Optional UMAP:

~~~bash
python -m pip install -r requirements-batch05-extras.txt
~~~

### Batch 06 — Anomalies, Time & Neural Foundations

16. [Chapter 16 — Anomaly Detection](16-anomaly-detection/README.md)
17. [Chapter 17 — Time Series Forecasting](17-time-series/README.md)
18. [Chapter 18 — Neural Network Foundations](18-neural-network-foundations/README.md)

Integration:
[Anomaly-Aware Forecasting & Neural Baseline Lab](integration-project-batch06/README.md)

Optional Prophet:

~~~bash
python -m pip install -r requirements-batch06-extras.txt
~~~

### Batch 07 — Backpropagation, Optimizers & Stable Objectives

19. [Chapter 19 — Backpropagation & Reverse-Mode Autodiff](19-backpropagation/README.md)
20. [Chapter 20 — Optimizers](20-optimizers/README.md)
21. [Chapter 21 — Activations & Loss Functions](21-activations-losses/README.md)

Integration:
[Tiny Deep Learning Framework](integration-project-batch07/README.md)

### Batch 08 — Frameworks & Convolution

22. [Chapter 22 — PyTorch](22-pytorch/README.md)
23. [Chapter 23 — TensorFlow / Keras](23-tensorflow-keras/README.md)
24. [Chapter 24 — Convolutional Neural Networks](24-cnn/README.md)

Integration:
[Cross-Framework CNN Lab](integration-project-batch08/README.md)

### Batch 09 — Sequence Models & Latent Variables

25. [Chapter 25 — Recurrent Neural Networks](25-rnn/README.md)
26. [Chapter 26 — LSTM / GRU](26-lstm-gru/README.md)
27. [Chapter 27 — Autoencoders / Variational Autoencoders](27-autoencoder-vae/README.md)

Integration:
[Sequence & Latent Representation Lab](integration-project-batch09/README.md)

### Batch 10 — GAN, Attention & Transformer

28. [Chapter 28 — Generative Adversarial Networks](28-gan/README.md)
29. [Chapter 29 — Attention](29-attention/README.md)
30. [Chapter 30 — Transformer](30-transformer/README.md)

Chapter 28 includes:
- Generator / Discriminator
- minimax objective
- non-saturating generator loss
- stable BCE with logits
- alternating optimization
- detach semantics
- mode collapse
- DCGAN principles
- Wasserstein / gradient-penalty intuition

Chapter 29 includes:
- Query / Key / Value
- scaled dot-product attention
- stable softmax
- causal masks
- padding masks
- self-attention / cross-attention
- multi-head attention
- PyTorch scaled_dot_product_attention

Chapter 30 includes:
- token embeddings
- positional encodings
- residual connections
- LayerNorm
- feed-forward networks
- encoder / decoder blocks
- pre-norm / post-norm
- decoder-only causal Transformer
- vocabulary projection
- next-token objective
- autoregressive generation

Integration:
[Adversarial & Transformer Lab](integration-project-batch10/README.md)

The Batch 10 integration contains:
- tiny PyTorch GAN on a 2D multimodal distribution
- tiny decoder-only character Transformer language model

### Batch 11 — Vision Transformers & NLP Foundations

31. [Chapter 31 — Vision Transformer](31-vision-transformer/README.md)
32. [Chapter 32 — NLP Fundamentals](32-nlp-fundamentals/README.md)
33. [Chapter 33 — Word2Vec / GloVe / FastText](33-word-embeddings/README.md)

Chapter 31 includes:
- patchification
- patch embedding
- CLS token
- learned positional embeddings
- Transformer encoder
- patch-size / attention-cost trade-offs
- Tiny ViT on sklearn digits

Chapter 32 includes:
- Unicode and normalization
- tokenization levels
- vocabulary / UNK / PAD
- n-gram language models
- smoothing
- next-token windows
- cross-entropy / perplexity
- padding and masks
- corpus leakage / quality

Chapter 33 includes:
- CBOW / Skip-Gram
- negative sampling
- unigram^0.75 distribution
- cosine similarity
- co-occurrence matrices
- GloVe
- FastText character n-grams
- OOV subword representations
- intrinsic vs extrinsic evaluation

Integration:
[Vision & Static Embedding Lab](integration-project-batch11/README.md)

The Batch 11 integration contains:
- a tiny PyTorch Vision Transformer classifier
- a NumPy Skip-Gram/GloVe/FastText embedding benchmark

### Batch 12 — BERT, GPT & T5

34. [Chapter 34 — BERT / Encoder Models](34-bert-encoder-models/README.md)
35. [Chapter 35 — GPT / Decoder Models](35-gpt-decoder-models/README.md)
36. [Chapter 36 — Encoder-Decoder / T5](36-encoder-decoder-t5/README.md)

Chapter 34 includes:
- encoder-only bidirectional Transformers
- Masked Language Modeling
- original BERT 15% / 80-10-10 corruption policy
- CLS / SEP / MASK / PAD
- segment embeddings
- encoder fine-tuning

Chapter 35 includes:
- decoder-only causal Transformers
- shifted next-token targets
- causal LM objective
- temperature / top-k / top-p sampling
- context windows
- KV-cache foundations

Chapter 36 includes:
- encoder-decoder architecture
- decoder causal self-attention
- encoder-decoder cross-attention
- teacher forcing / shift-right
- text-to-text framing
- T5 span corruption
- sentinel tokens

Integration:
[BERT vs GPT vs T5 Lab](integration-project-batch12/README.md)

The Batch 12 integration trains:
- a tiny BERT-style masked encoder
- a tiny GPT-style causal language model
- a tiny T5-style encoder-decoder model

### Batch 13 — Graphs, Reinforcement Learning & Generative Foundations

37. [Chapter 37 — Graph Neural Networks](37-graph-neural-networks/README.md)
38. [Chapter 38 — Reinforcement Learning](38-reinforcement-learning/README.md)
39. [Chapter 39 — Generative AI Foundations](39-generative-ai-foundations/README.md)

Chapter 37 includes:
- graph / adjacency / edge-list representations
- message passing
- GCN normalization
- GraphSAGE
- Graph Attention Networks
- node / graph / link prediction
- oversmoothing / oversquashing
- transductive vs inductive learning

Chapter 38 includes:
- MDPs
- returns and discounting
- Bellman equations
- Q-Learning
- DQN
- replay buffers
- target networks
- REINFORCE
- actor-critic
- PPO clipping

Chapter 39 includes:
- generative vs discriminative models
- explicit vs implicit density models
- likelihood / NLL
- autoregressive models
- latent-variable models / ELBO
- GANs
- energy-based models
- score / diffusion foundations
- mode coverage and generative evaluation

Integration:
[Graph, Control & Generative Lab](integration-project-batch13/README.md)

The Batch 13 integration contains:
- a tiny GCN node classifier
- a tiny DQN chain-control agent
- explicit-density and forward-noising generative diagnostics

### Batch 14 — Diffusion, Multimodal & Speech

40. [Chapter 40 — Diffusion Models](40-diffusion-models/README.md)
41. [Chapter 41 — Multimodal Models](41-multimodal-models/README.md)
42. [Chapter 42 — Speech: STT / TTS](42-speech-stt-tts/README.md)

Chapter 40 includes:
- beta / alpha / alpha_bar schedules
- forward noising q(x_t|x_0)
- epsilon prediction
- DDPM reverse process
- DDIM intuition
- classifier-free guidance
- U-Net and latent diffusion
- image / audio / video diffusion

Chapter 41 includes:
- dual encoders
- CLIP-style contrastive learning
- shared embedding spaces
- image↔text retrieval
- early / late / intermediate fusion
- cross-attention
- image tokens
- VLM projectors
- multimodal ablations

Chapter 42 includes:
- waveform / sample rate
- framing and STFT
- mel / log-mel features
- CTC
- seq2seq STT
- Whisper-style concepts
- TTS acoustic models
- vocoders
- WER / CER

Integration:
[Diffusion, Multimodal & Speech Lab](integration-project-batch14/README.md)

The Batch 14 integration contains:
- a tiny 2D DDPM epsilon predictor and reverse sampler
- a CLIP-style dual-encoder retrieval experiment
- a synthetic-tone CTC speech recognizer

### Batch 15 — Recommenders, Vision Localization & Vector Retrieval

43. [Chapter 43 — Recommender Systems](43-recommender-systems/README.md)
44. [Chapter 44 — Object Detection / Segmentation](44-object-detection-segmentation/README.md)
45. [Chapter 45 — Embeddings / Vector Databases](45-embeddings-vector-databases/README.md)

Chapter 43 includes:
- explicit / implicit feedback
- collaborative filtering
- matrix factorization
- BPR
- two-tower retrieval
- ranking metrics
- temporal evaluation
- cold start and feedback loops

Chapter 44 includes:
- bounding boxes and IoU
- NMS
- anchor-based / anchor-free detection
- one-stage / two-stage detectors
- Faster R-CNN / YOLO concepts
- semantic / instance / panoptic segmentation
- U-Net / Mask R-CNN
- AP / mAP concepts

Chapter 45 includes:
- embedding geometry
- cosine / dot / L2
- exact nearest-neighbor search
- ANN Recall@K
- IVF
- HNSW
- Product Quantization
- metadata filtering
- hybrid retrieval
- reranking
- vector-index lifecycle

Integration:
[Recommendation, Vision & Vector Retrieval Lab](integration-project-batch15/README.md)

The Batch 15 integration contains:
- a two-tower recommender trained with a BPR-style objective
- a tiny U-Net segmentation experiment
- an exact-vs-IVF vector retrieval benchmark

### Batch 16 — RAG, Agents & Modern LLM Architecture

46. [Chapter 46 — Retrieval-Augmented Generation](46-rag/README.md)
47. [Chapter 47 — AI Agents / Tool Calling](47-ai-agents-tool-calling/README.md)
48. [Chapter 48 — Modern LLM Architecture](48-llm-architecture/README.md)

Chapter 46 includes:
- parsing / chunking / overlap
- metadata and provenance
- dense / sparse retrieval
- hybrid retrieval and RRF
- MMR diversification
- reranking
- context packing
- citation/source tracking
- Recall@K / MRR
- RAG failure taxonomy
- authorization-aware retrieval

Chapter 47 includes:
- agent loops
- structured tool schemas
- argument validation
- tool allow-lists
- explicit state
- step/tool budgets
- retries and idempotency
- read/write permission boundaries
- prompt/tool-output injection defenses
- approval gates
- observability and agent evaluation

Chapter 48 includes:
- token embeddings
- residual stream
- RMSNorm
- RoPE
- causal scaled dot-product attention
- MHA / MQA / GQA
- SwiGLU
- final normalization
- weight-tied LM head
- KV cache
- prefill vs decode

Integration:
[RAG, Agent & Modern LLM Lab](integration-project-batch16/README.md)

The Batch 16 integration contains:
- dense+sparse RAG with RRF/context packing
- a schema-validated allow-listed tool agent
- a tiny modern decoder with RMSNorm, RoPE, GQA, causal SDPA and SwiGLU

### Batch 17 — Tokenizer, Pretraining & LLM Data Pipeline

49. [Chapter 49 — Tokenizer From Scratch](49-tokenizer-from-scratch/README.md)
50. [Chapter 50 — LLM Pretraining From Scratch](50-llm-pretraining-from-scratch/README.md)
51. [Chapter 51 — LLM Dataset / Data Pipeline](51-llm-dataset-pipeline/README.md)

Chapter 49 includes:
- Unicode / UTF-8 bytes
- byte-level vocabulary
- BPE pair counting and deterministic merges
- merge ranks
- encode / decode
- special-token design
- multilingual compression evaluation
- tokenizer artifact/versioning

Chapter 50 includes:
- causal LM input/target shifting
- cross-entropy / perplexity
- AdamW
- warmup + cosine decay
- gradient accumulation
- gradient clipping
- validation
- tokens seen / throughput
- checkpoint/resume concepts

Chapter 51 includes:
- document schema / provenance
- Unicode normalization
- exact deduplication
- near-dedup concepts
- document-level splitting
- frozen-tokenizer contract
- token packing
- sharding / streaming
- data mixtures
- dataset manifests
- contamination auditing

Integration:
[Raw Text to Tiny LLM](integration-project-batch17/README.md)

The Batch 17 integration contains:
- normalized/deduplicated document pipeline
- deterministic document-level train/validation/test split
- byte-level BPE trained on train documents only
- packed causal token sequences
- tiny modern LLM pretraining with AdamW, LR scheduling, clipping and checkpoint serialization

### Batch 18 — Distributed Training, CUDA & Mixed Precision

52. [Chapter 52 — Distributed Training](52-distributed-training/README.md)
53. [Chapter 53 — GPU / CUDA Fundamentals](53-gpu-cuda-fundamentals/README.md)
54. [Chapter 54 — Mixed Precision: FP32 / FP16 / BF16](54-mixed-precision/README.md)

Chapter 52 includes:
- rank / world size / process groups
- all-reduce / all-gather / reduce-scatter / broadcast
- DDP
- rank-aware data sharding
- global batch size
- tensor parallelism
- pipeline parallelism
- FSDP / ZeRO concepts
- communication and scaling efficiency

Chapter 53 includes:
- CUDA grid / block / thread hierarchy
- SMs and 32-thread warps
- SIMT / divergence
- registers / global / shared memory
- coalescing
- bank conflicts
- synchronization
- streams / events
- occupancy
- arithmetic intensity / roofline
- GPU profiling

Chapter 54 includes:
- FP32 / FP16 / BF16 formats
- range vs precision
- underflow / overflow
- autocast
- loss scaling
- current torch.amp APIs
- GradScaler
- unscale / clipping order
- BF16 training
- mixed-precision memory accounting

Integration:
[Distributed, CUDA & Precision Lab](integration-project-batch18/README.md)

The Batch 18 integration contains:
- a distributed-memory/communication planning experiment
- CUDA runtime/device inspection with CPU-safe fallback
- BF16 CPU autocast and CUDA FP16 + GradScaler paths

### Batch 19 — Fine-Tuning, LoRA/QLoRA/PEFT & Instruction Tuning

55. [Chapter 55 — Fine-Tuning](55-fine-tuning/README.md)
56. [Chapter 56 — LoRA / QLoRA / PEFT](56-lora-qlora-peft/README.md)
57. [Chapter 57 — Instruction Tuning](57-instruction-tuning/README.md)

Chapter 55 includes:
- full fine-tuning
- freezing / unfreezing
- trainable-parameter accounting
- discriminative learning rates
- early stopping
- checkpoint selection
- domain shift
- catastrophic forgetting
- retained-capability evaluation

Chapter 56 includes:
- LoRA low-rank updates
- rank / alpha / dropout
- target modules
- adapter merge / unmerge
- PEFT checkpointing
- QLoRA frozen quantized base
- NF4 / double-quantization concepts
- generic codebook quantization
- current Hugging Face PEFT configuration concepts

Chapter 57 includes:
- instruction / chat schemas
- model-specific chat templates
- role/control tokens
- full-sequence vs assistant-only loss
- ignore-index masking
- multi-turn supervision
- EOS / boundaries
- packing / truncation
- SFT data quality / deduplication
- held-out instruction evaluation

Integration:
[Fine-Tuning, LoRA & Instruction SFT Lab](integration-project-batch19/README.md)

The Batch 19 integration contains:
- full fine-tuning on a controlled adaptation problem
- LoRA on the same base/target mapping
- assistant-only causal SFT on chat-like token sequences

Optional Hugging Face PEFT labs:

~~~bash
python -m pip install -r requirements-batch19-peft.txt
~~~

### Batch 20 — RLHF, DPO / Preference Optimization & LLM Evaluation

58. [Chapter 58 — RLHF](58-rlhf/README.md)
59. [Chapter 59 — DPO / Preference Optimization](59-dpo-preference-optimization/README.md)
60. [Chapter 60 — LLM Evaluation](60-llm-evaluation/README.md)

Chapter 58 includes:
- preference-pair datasets
- Bradley-Terry reward modeling
- pairwise reward accuracy
- policy / reference models
- exact categorical KL
- sampled policy-reference log-ratios
- KL-shaped rewards
- advantages
- PPO ratios / clipping
- reward hacking / overoptimization

Chapter 59 includes:
- completion log-probabilities
- policy / reference preference margins
- standard sigmoid DPO loss
- beta
- completion masking
- length audits
- reference log-probability caching
- IPO / ORPO / KTO concepts
- current TRL DPO configuration concepts

Chapter 60 includes:
- exact match / token F1
- multiple-choice evaluation
- pairwise evaluation
- Brier score / ECE calibration
- human / LLM-as-judge concepts
- bootstrap confidence intervals
- paired model comparisons
- contamination audits
- slice evaluation
- reproducibility manifests
- latency / throughput / memory evaluation

Integration:
[Preference Alignment & Evaluation Lab](integration-project-batch20/README.md)

The Batch 20 integration contains:
- a Bradley-Terry reward-model experiment
- DPO against a frozen reference policy
- paired-bootstrap model comparison
- normalized exact-hash contamination auditing

Optional Hugging Face TRL labs:

~~~bash
python -m pip install -r requirements-batch20-trl.txt
~~~

### Batch 21 — Quantization, Inference Optimization & LLM Serving

61. [Chapter 61 — Quantization](61-quantization/README.md)
62. [Chapter 62 — Inference Optimization](62-inference-optimization/README.md)
63. [Chapter 63 — LLM Serving](63-llm-serving/README.md)

Chapter 61 includes:
- symmetric / asymmetric quantization
- scale / zero point
- per-tensor / per-channel / grouped quantization
- weight-only / activation quantization
- calibration
- GPTQ / AWQ concepts
- GGUF distinction
- storage/error accounting

Chapter 62 includes:
- prefill vs decode
- KV-cache memory
- MHA / GQA / MQA cache trade-offs
- Dynamic / Static / Offloaded / Quantized caches
- continuous batching
- paged KV memory
- prefix caching
- chunked prefill
- FlashAttention / SDPA
- speculative decoding
- TTFT / TPOT / throughput

Chapter 63 includes:
- Transformers local generation
- vLLM
- llama.cpp / GGUF
- OpenAI-compatible APIs
- streaming / cancellation
- readiness / liveness
- metrics / percentiles
- backpressure / rate limits
- autoscaling
- canary / shadow rollouts
- serving security boundaries

Integration:
[Quantized Inference & Serving Capacity Lab](integration-project-batch21/README.md)

The Batch 21 integration contains:
- INT8/INT4/per-channel quantization quality comparisons
- KV-cache and paged-memory capacity planning
- serving SLO / replica / canary planning
- PyTorch weight-only reconstruction smoke

Optional lightweight serving environment:

~~~bash
python -m pip install -r requirements-batch21-serving.txt
~~~

vLLM and llama.cpp remain hardware/runtime-specific and are not forced into CPU CI.

### Batch 22 — Deploy AI, MLOps & AI Data Engineering

64. [Chapter 64 — Deploy AI](64-deploy-ai/README.md)
65. [Chapter 65 — MLOps](65-mlops/README.md)
66. [Chapter 66 — AI Data Engineering](66-ai-data-engineering/README.md)

Chapter 64 includes:
- API deployment contracts
- Docker images / immutable artifacts
- Kubernetes Deployment / Service
- Gateway vs Ingress
- ConfigMap / Secret
- startup / readiness / liveness probes
- GPU scheduling / device plugins
- rolling updates
- HPA / custom metrics
- canary / blue-green rollout
- cloud deployment concepts
- rollback / graceful termination

Chapter 65 includes:
- experiment tracking
- run / artifact fingerprinting
- model registry
- versions / aliases / tags
- promotion gates
- CI / CD / CT
- offline / online metrics
- data / prediction / concept drift
- PSI
- training-serving skew
- champion / challenger
- lineage / rollback

Chapter 66 includes:
- ETL / ELT
- data lake / warehouse / lakehouse concepts
- Parquet row groups / column chunks
- projection / predicate pruning
- partitioning / compaction
- batch / streaming
- event time / processing time
- watermarks / late data
- idempotency
- data contracts / schema evolution
- backfills / lineage
- point-in-time correctness

Integration:
[Production ML Lifecycle Lab](integration-project-batch22/README.md)

The Batch 22 integration contains:
- contract validation / dedup / partition / lineage
- experiment fingerprint / drift / promotion gate
- HPA-style scaling / rollout / GPU capacity / canary planning
- one end-to-end rollout gate across data, model and infrastructure

Optional MLflow environment:

~~~bash
python -m pip install -r requirements-batch22-mlops.txt
~~~

Optional PyArrow environment:

~~~bash
python -m pip install -r requirements-batch22-data.txt
~~~

## Framework Installation

Core dependencies through Batch 22:

~~~bash
python -m pip install -r requirements-batch22.txt
~~~

PyTorch remains outside the core dependency chain.

For Fedora + NVIDIA, use the current official PyTorch selector:
https://pytorch.org/get-started/locally/

CPU-only CI uses PyTorch's CPU package index.

TensorFlow from Batch 08 remains in its dedicated framework environment.

## Learning loop

ทุกบทใช้วงจร:

~~~text
Theory
  ↓
Intuition
  ↓
Mathematics / Systems Model
  ↓
From-Scratch Implementation
  ↓
Library Implementation
  ↓
Experiment
  ↓
Debug
  ↓
Project
  ↓
Review
~~~

อย่าข้าม exercises และอย่าอ่าน solutions ก่อนพยายามทำเอง

## Setup

อ่าน [SETUP_FEDORA.md](SETUP_FEDORA.md) ก่อน

สร้าง environment:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-batch22.txt
~~~

Core tests:

~~~bash
pytest -q
~~~

Framework-specific tests require the corresponding framework environment.

## Recommended study order

~~~text
01 Programming
↓
02 Mathematics
↓
03 Scientific Python
↓
Batch 01 Integration
↓
04 Data
↓
05 ML Fundamentals
↓
06 Regression
↓
Batch 02 Integration
↓
07 Logistic Regression
↓
08 KNN
↓
09 Naive Bayes
↓
Batch 03 Integration
↓
10 Decision Trees
↓
11 Random Forest
↓
12 Gradient Boosting
↓
Batch 04 Integration
↓
13 Support Vector Machines
↓
14 Clustering
↓
15 Dimensionality Reduction
↓
Batch 05 Integration
↓
16 Anomaly Detection
↓
17 Time Series
↓
18 Neural Network Foundations
↓
Batch 06 Integration
↓
19 Backpropagation
↓
20 Optimizers
↓
21 Activations & Loss Functions
↓
Batch 07 Tiny Deep Learning Framework
↓
22 PyTorch
↓
23 TensorFlow / Keras
↓
24 CNN
↓
Batch 08 Cross-Framework CNN Lab
↓
25 RNN
↓
26 LSTM / GRU
↓
27 Autoencoder / VAE
↓
Batch 09 Sequence & Latent Representation Lab
↓
28 GAN
↓
29 Attention
↓
30 Transformer
↓
Batch 10 Adversarial & Transformer Lab
↓
31 Vision Transformer
↓
32 NLP Fundamentals
↓
33 Word2Vec / GloVe / FastText
↓
Batch 11 Vision & Static Embedding Lab
↓
34 BERT / Encoder Models
↓
35 GPT / Decoder Models
↓
36 Encoder-Decoder / T5
↓
Batch 12 BERT vs GPT vs T5 Lab
↓
37 Graph Neural Networks
↓
38 Reinforcement Learning
↓
39 Generative AI Foundations
↓
Batch 13 Graph, Control & Generative Lab
↓
40 Diffusion Models
↓
41 Multimodal Models
↓
42 Speech: STT / TTS
↓
Batch 14 Diffusion, Multimodal & Speech Lab
↓
43 Recommender Systems
↓
44 Object Detection / Segmentation
↓
45 Embeddings / Vector Databases
↓
Batch 15 Recommendation, Vision & Vector Retrieval Lab
↓
46 Retrieval-Augmented Generation
↓
47 AI Agents / Tool Calling
↓
48 Modern LLM Architecture
↓
Batch 16 RAG, Agent & Modern LLM Lab
↓
49 Tokenizer From Scratch
↓
50 LLM Pretraining From Scratch
↓
51 LLM Dataset / Data Pipeline
↓
Batch 17 Raw Text to Tiny LLM Lab
↓
52 Distributed Training
↓
53 GPU / CUDA Fundamentals
↓
54 Mixed Precision: FP32 / FP16 / BF16
↓
Batch 18 Distributed, CUDA & Precision Lab
↓
55 Fine-Tuning
↓
56 LoRA / QLoRA / PEFT
↓
57 Instruction Tuning
↓
Batch 19 Fine-Tuning, LoRA & Instruction SFT Lab
↓
58 RLHF
↓
59 DPO / Preference Optimization
↓
60 LLM Evaluation
↓
Batch 20 Preference Alignment & Evaluation Lab
↓
61 Quantization
↓
62 Inference Optimization
↓
63 LLM Serving
↓
Batch 21 Quantized Inference & Serving Capacity Lab
↓
64 Deploy AI
↓
65 MLOps
↓
66 AI Data Engineering
↓
Batch 22 Production ML Lifecycle Lab
~~~

## Reviews

- [BATCH_REVIEW.md](BATCH_REVIEW.md) — Batch 01
- [BATCH_02_REVIEW.md](BATCH_02_REVIEW.md) — Batch 02
- [BATCH_03_REVIEW.md](BATCH_03_REVIEW.md) — Batch 03
- [BATCH_04_REVIEW.md](BATCH_04_REVIEW.md) — Batch 04
- [BATCH_05_REVIEW.md](BATCH_05_REVIEW.md) — Batch 05
- [BATCH_06_REVIEW.md](BATCH_06_REVIEW.md) — Batch 06
- [BATCH_07_REVIEW.md](BATCH_07_REVIEW.md) — Batch 07
- [BATCH_08_REVIEW.md](BATCH_08_REVIEW.md) — Batch 08
- [BATCH_09_REVIEW.md](BATCH_09_REVIEW.md) — Batch 09
- [BATCH_10_REVIEW.md](BATCH_10_REVIEW.md) — Batch 10
- [BATCH_11_REVIEW.md](BATCH_11_REVIEW.md) — Batch 11
- [BATCH_12_REVIEW.md](BATCH_12_REVIEW.md) — Batch 12
- [BATCH_13_REVIEW.md](BATCH_13_REVIEW.md) — Batch 13
- [BATCH_14_REVIEW.md](BATCH_14_REVIEW.md) — Batch 14
- [BATCH_15_REVIEW.md](BATCH_15_REVIEW.md) — Batch 15
- [BATCH_16_REVIEW.md](BATCH_16_REVIEW.md) — Batch 16
- [BATCH_17_REVIEW.md](BATCH_17_REVIEW.md) — Batch 17
- [BATCH_18_REVIEW.md](BATCH_18_REVIEW.md) — Batch 18
- [BATCH_19_REVIEW.md](BATCH_19_REVIEW.md) — Batch 19
- [BATCH_20_REVIEW.md](BATCH_20_REVIEW.md) — Batch 20
- [BATCH_21_REVIEW.md](BATCH_21_REVIEW.md) — Batch 21
- [BATCH_22_REVIEW.md](BATCH_22_REVIEW.md) — Batch 22

## Definition of mastery

บทหนึ่งถือว่า “เข้าใจ” เมื่อทำได้อย่างน้อย:

1. **Explain** — อธิบายด้วยคำของตัวเอง
2. **Derive** — ไล่ที่มาของสมการสำคัญ
3. **Implement** — เขียนแก่น algorithm
4. **Debug** — หา root cause
5. **Modify** — เปลี่ยนโจทย์แล้วปรับ implementation
6. **Apply** — ใช้กับข้อมูลใหม่

## Current status

- [x] Batch 01 — Chapters 01–03
- [x] Batch 02 — Chapters 04–06
- [x] Batch 03 — Chapters 07–09
- [x] Batch 04 — Chapters 10–12
- [x] Batch 05 — Chapters 13–15
- [x] Batch 06 — Chapters 16–18
- [x] Batch 07 — Chapters 19–21
- [x] Batch 08 — Chapters 22–24
- [x] Batch 09 — Chapters 25–27
- [x] Batch 10 — Chapters 28–30
- [x] Batch 11 — Chapters 31–33
- [x] Batch 12 — Chapters 34–36
- [x] Batch 13 — Chapters 37–39
- [x] Batch 14 — Chapters 40–42
- [x] Batch 15 — Chapters 43–45
- [x] Batch 16 — Chapters 46–48
- [x] Batch 17 — Chapters 49–51
- [x] Batch 18 — Chapters 52–54
- [x] Batch 19 — Chapters 55–57
- [x] Batch 20 — Chapters 58–60
- [x] Batch 21 — Chapters 61–63
- [x] Batch 22 — Chapters 64–66
- [ ] Batch 23 — Chapters 67–69
- [ ] Batch 24+ — รอ batch ก่อนหน้าผ่าน quality audit

บทถัดไป:

- Chapter 67 — AI Distributed Systems
- Chapter 68 — AI Safety / Alignment / Guardrails
- Chapter 69 — Interpretability / XAI
