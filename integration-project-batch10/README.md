# Batch 10 Integration Project — Adversarial & Transformer Lab

Batch 10 combines two different generative paradigms:

## Part A — GAN on a 2D Distribution

~~~text
noise z
  ↓
Generator
  ↓
fake 2D points

real 2D points ─┐
fake 2D points ─┼→ Discriminator
~~~

Track:
- generator loss
- discriminator loss
- real/fake mean
- real/fake covariance
- mode coverage
- fixed latent probes

## Part B — Tiny Decoder-Only Transformer

~~~text
text corpus
↓
character vocabulary
↓
context windows
↓
token embedding + position
↓
causal Transformer blocks
↓
vocabulary logits
↓
next-token cross-entropy
~~~

Track:
- train loss
- validation loss
- parameter count
- generated sample
- context length
- device

## Why Put These Together?

GANs and autoregressive Transformers are both generative models, but their training signals differ radically:

GAN:
- adversarial game
- no explicit likelihood

Transformer LM:
- next-token conditional likelihood
- teacher-forced cross-entropy

The comparison teaches that “generative AI” is not one objective.

## Run GAN

~~~bash
python integration-project-batch10/src/generative_transformer_lab.py \
  --mode gan \
  --steps 200 \
  --output-dir reports/batch10-gan
~~~

## Run Transformer LM

~~~bash
python integration-project-batch10/src/generative_transformer_lab.py \
  --mode transformer \
  --steps 200 \
  --output-dir reports/batch10-transformer
~~~

## Methodology Rules

### GAN
- fixed seed
- fixed latent probe
- separate G/D optimizer steps
- fake detached for D update
- fake not detached for G update

### Transformer
- train/validation split by corpus position
- no validation tokens inserted into training windows
- causal attention
- raw logits into cross-entropy
- generation separated from training

## Required Extensions

1. WGAN-GP toy experiment
2. GAN mode-coverage metric
3. top-k sampling
4. temperature sampling sweep
5. learned vs sinusoidal positions
6. pre-norm vs post-norm
7. context-length sweep
8. attention-map inspection
9. tokens/second benchmark
10. KV-cache preview implementation

## Mastery Questions

- Why can GAN losses oscillate while training still improves?
- Why does fake.detach() belong in the D step?
- Why must decoder self-attention be causal?
- Why is teacher forcing different from generation?
- Why does dense attention grow quadratically with context length?
