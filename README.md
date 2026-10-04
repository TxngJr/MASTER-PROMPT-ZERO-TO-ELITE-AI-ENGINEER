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

## Framework Installation

Core dependencies through Batch 14:

~~~bash
python -m pip install -r requirements-batch14.txt
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
python -m pip install -r requirements-batch14.txt
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
- [ ] Batch 15 — Chapters 43–45
- [ ] Batch 16+ — รอ batch ก่อนหน้าผ่าน quality audit

บทถัดไป:

- Chapter 43 — Recommender Systems
- Chapter 44 — Object Detection / Segmentation
- Chapter 45 — Embeddings / Vector Databases
