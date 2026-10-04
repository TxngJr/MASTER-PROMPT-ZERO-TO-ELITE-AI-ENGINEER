# Batch 14 Review — Chapters 40–42

## Chapters

- 40 — Diffusion Models
- 41 — Multimodal Models
- 42 — Speech: STT / TTS

## Diffusion Skills

- beta / alpha / alpha_bar schedules
- forward Markov noising
- closed-form q(x_t|x_0)
- SNR intuition
- epsilon prediction
- x0 reconstruction
- posterior variance
- DDPM reverse mean
- ancestral sampling
- DDIM intuition
- classifier / classifier-free guidance
- U-Net conditioning
- latent diffusion
- image/audio/video diffusion

## Multimodal Skills

- modality-specific encoders
- shared embedding spaces
- cosine similarity
- contrastive temperature
- symmetric CLIP-style loss
- image↔text retrieval
- early / late / intermediate fusion
- cross-attention
- visual tokens
- VLM projectors
- frozen vs joint training
- modality ablations

## Speech Skills

- sample rate / Nyquist
- framing / hop length
- Hann window
- STFT
- magnitude / power spectrogram
- mel scale / filterbank
- log-mel features
- CTC
- CTC greedy decoding
- seq2seq STT
- streaming considerations
- TTS acoustic models
- vocoders
- WER / CER

## Implemented From Scratch

### Chapter 40
- linear_beta_schedule
- alpha_terms
- q_sample
- predict_x0_from_epsilon
- posterior_variance
- ddpm_reverse_mean

### Chapter 41
- l2_normalize
- similarity_logits
- stable_cross_entropy
- symmetric_contrastive_loss
- retrieval_top1_accuracy
- simple_cross_attention

### Chapter 42
- frame_signal
- stft
- hz_to_mel / mel_to_hz
- mel_filterbank
- log_mel_spectrogram
- ctc_greedy_decode
- word_error_rate

## Integration Project

Three modes:

1. 2D DDPM epsilon-prediction + reverse sampler
2. CLIP-style paired-modality contrastive retrieval
3. synthetic-tone CTC speech recognizer

## PyTorch API Audit

Speech integration uses:
- explicit Hann window
- torch.stft(..., return_complex=True)
- complex magnitude/power features
- log_softmax before CTCLoss
- log_probs shaped as T,N,C
- blank id excluded from targets

These choices match the current stable PyTorch APIs used by the course.

## Methodology Audit

### Diffusion
- random timestep training
- exact forward noising formula
- separate reverse sampling loop
- no fresh noise at final step
- generated moments reported without claiming full distribution quality

### Multimodal
- paired train/test split
- normalized embeddings
- symmetric image→text / text→image objective
- retrieval evaluated on held-out pairs

### Speech
- synthetic data avoids external downloads
- target token IDs exclude CTC blank
- fixed target lengths are explicit
- STFT frontend settings are explicit
- toy tones are not presented as real speech

## Interpretation Audit

- diffusion loss alone is not sample-quality evaluation
- contrastive similarity does not prove grounded understanding
- high retrieval performance on synthetic pairs does not imply real VLM capability
- CTC greedy decoding is not globally optimal decoding
- toy tone recognition is only a mechanics test
- TTS identity generation requires consent-aware deployment practices

## Exit Gate

Before Chapter 43:

1. Batch 14 Core CI passes
2. Batch 14 PyTorch smoke passes
3. derive q(x_t|x_0)
4. derive epsilon-to-x0 reconstruction
5. explain DDPM vs DDIM intuition
6. derive symmetric multimodal contrastive loss
7. explain cross-attention fusion
8. derive STFT/mel frontend shape flow
9. explain CTC collapse and blank behavior
10. explain STT vs TTS model pipelines
