# Batch 14 Integration Project — Diffusion, Multimodal & Speech Lab

Batch 14 contains three framework experiments.

## Part A — 2D DDPM

~~~text
4-mode 2D data
↓
sample random timestep
↓
add Gaussian noise
↓
MLP epsilon predictor
↓
MSE noise objective
↓
reverse DDPM sampler
~~~

Tracks:
- training loss
- parameter count
- generated mean/covariance
- number of diffusion steps

## Part B — CLIP-Style Alignment

Two synthetic modalities are generated from the same hidden semantic vector.

~~~text
image features → image encoder ─┐
                               ├→ contrastive loss
text features  → text encoder ─┘
~~~

Tracks:
- symmetric contrastive loss
- image→text top-1 retrieval
- text→image top-1 retrieval

## Part C — Synthetic Speech CTC

~~~text
toy token sequence
↓
tone waveform
↓
torch.stft + log power
↓
BiGRU acoustic model
↓
CTC logits
↓
greedy CTC decode
~~~

This verifies:
- waveform frontend
- explicit Hann window
- complex STFT output
- CTC log-probability shapes
- blank-token decoding

It is not a real human speech recognizer.

## Run

~~~bash
python integration-project-batch14/src/diffusion_multimodal_speech_lab.py --mode diffusion --steps 100
python integration-project-batch14/src/diffusion_multimodal_speech_lab.py --mode multimodal --steps 100
python integration-project-batch14/src/diffusion_multimodal_speech_lab.py --mode speech --steps 40
~~~

## Required Extensions

1. cosine alpha-bar schedule
2. DDIM sampler
3. conditional diffusion + classifier-free guidance
4. multimodal Recall@K
5. token-level cross-attention fusion
6. visual-token-to-LM projector
7. mel filter-bank speech frontend
8. CTC beam-search concept implementation
9. seq2seq STT model
10. toy mel-to-waveform vocoder experiment
