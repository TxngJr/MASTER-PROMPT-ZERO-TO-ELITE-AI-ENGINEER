# Chapter 28 — Generative Adversarial Networks

## 1. Why GANs?

A GAN trains two networks in competition:

~~~text
random noise z
    ↓
Generator G
    ↓
fake sample
    ↓
Discriminator D
    ↓
real or fake?
~~~

The discriminator also receives real data.

The generator improves by learning to create samples that the discriminator scores as realistic.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- explain generator vs discriminator roles
- derive the original minimax objective
- explain non-saturating generator loss
- understand logits-based stable losses
- implement GAN losses from scratch
- explain alternating optimization
- diagnose mode collapse
- explain why GAN training can oscillate
- understand DCGAN principles
- explain Wasserstein intuition
- understand gradient penalty conceptually
- build a tiny PyTorch GAN
- evaluate GANs beyond visual cherry-picking

## 3. Generator

The generator maps noise to data space:

~~~text
z ~ p(z)
x_fake = G(z)
~~~

Typical latent prior:

~~~text
z ~ N(0,I)
~~~

The generator does not receive a direct target image for each z.

## 4. Discriminator

The discriminator estimates whether a sample came from real data or the generator.

In the original probabilistic view:

~~~text
D(x) ∈ (0,1)
~~~

Modern implementations often output logits and use a stable logits-based binary loss.

## 5. Original Minimax Objective

~~~text
min_G max_D

E_x~pdata [
    log D(x)
]
+
E_z~pz [
    log(1 - D(G(z)))
]
~~~

The discriminator wants:
- real → high score
- fake → low score

The generator tries to make fake samples harder to distinguish.

## 6. Discriminator BCE Form

With logits:

~~~text
L_D
=
BCEWithLogits(real_logits, 1)
+
BCEWithLogits(fake_logits, 0)
~~~

Usually average the two terms or otherwise choose a consistent reduction.

## 7. Saturating Generator Objective

Direct minimax generator form:

~~~text
min_G
E [
    log(1-D(G(z)))
]
~~~

Early in training, if D easily rejects fake samples, gradients can become weak.

## 8. Non-Saturating Generator Loss

Common practical objective:

~~~text
L_G
=
BCEWithLogits(
    D(G(z)),
    1
)
~~~

Equivalent intuition:

> make the discriminator classify fake samples as real.

This provides stronger early gradients than the saturating minimax form.

## 9. Why Logits?

Do not manually apply sigmoid before BCEWithLogitsLoss.

PyTorch's BCEWithLogitsLoss combines sigmoid and BCE in a numerically stable formulation.

## 10. Alternating Optimization

A canonical training step:

~~~text
1. sample real batch
2. sample z
3. update discriminator
4. sample another z or reuse intentionally
5. update generator
~~~

When updating D:
- fake samples should not backpropagate into G if G is not being updated

PyTorch pattern:

~~~python
fake = generator(z).detach()
~~~

When updating G:
- discriminator parameters need not be stepped
- gradient must flow through D operations into G output

## 11. Training Is a Game

Ordinary supervised learning minimizes one stable objective.

GAN training is a two-player game.

The target changes while both networks learn.

Possible behavior:
- oscillation
- one side overpowering the other
- cycling
- unstable gradients

Loss curves are therefore harder to interpret than standard classifier loss.

## 12. Mode Collapse

Generator may map many z values to nearly the same output:

~~~text
z1 ─┐
z2 ─┼→ almost same sample
z3 ─┘
~~~

It may fool D locally while failing to represent full data diversity.

Symptoms:
- repeated generated examples
- low diversity
- latent interpolation changes little

## 13. Discriminator Too Strong

If D becomes nearly perfect immediately:
- G may receive poor learning signal
- training can stall

Potential causes:
- D capacity much larger
- learning-rate mismatch
- poor normalization
- easy artifacts in G outputs

Do not mechanically weaken D without measuring the actual dynamics.

## 14. Generator Too Strong / Weak D

If D cannot learn:
- G receives uninformative feedback
- generated quality may drift

Balance is empirical.

## 15. DCGAN Principles

Classic DCGAN-style image GANs popularized:
- convolutional generator/discriminator
- strided convolution
- BatchNorm in many layers
- ReLU in generator
- LeakyReLU in discriminator
- tanh image output when data scaled to [-1,1]

These are historical design principles, not universal modern rules.

## 16. Scaling Real Images

If generator output uses tanh:

~~~text
range ≈ [-1,1]
~~~

then real training data should be scaled consistently.

Mismatch:

~~~text
real [0,1]
fake [-1,1]
~~~

lets discriminator exploit trivial range differences.

## 17. Label Smoothing

One-sided real-label smoothing may use target such as 0.9 instead of 1.0.

It can reduce overconfidence in some setups.

Do not smooth labels blindly or use it to hide deeper instability.

## 18. Wasserstein GAN Intuition

WGAN replaces binary discrimination with a critic scoring real/fake samples.

It is motivated by learning a distance related to Wasserstein-1 / Earth-Mover distance.

The critic is constrained to be approximately 1-Lipschitz.

## 19. Weight Clipping vs Gradient Penalty

Original WGAN used weight clipping.

WGAN-GP instead penalizes gradient norm on interpolated samples:

~~~text
λ (
    ||∇_x D(x_hat)||_2 - 1
)^2
~~~

This generally provides a better Lipschitz constraint than crude weight clipping.

## 20. GAN Evaluation

Never evaluate only by selecting the best-looking generated samples.

Useful ideas:
- sample grids generated with fixed latent seeds
- diversity measurements
- nearest-neighbor inspection
- FID / KID for image distributions
- precision/recall-style generative metrics

Metrics themselves have assumptions and failure modes.

## 21. Fixed Noise

Keep a fixed latent batch:

~~~text
z_fixed
~~~

Across epochs decode the same z values.

This helps visualize learning progression without changing the probe every time.

## 22. From Scratch

src/gan_math.py includes:
- stable BCE with logits
- discriminator loss
- non-saturating generator loss
- saturating generator loss

The PyTorch integration project trains a tiny GAN.

## 23. Common Mistakes

1. sigmoid before BCEWithLogitsLoss
2. forgetting detach during D update
3. detaching fake during G update
4. data/output range mismatch
5. reading D/G loss like ordinary supervised loss
6. cherry-picking samples
7. no fixed-noise monitoring
8. confusing mode collapse with overfitting only
9. changing multiple GAN hyperparameters at once
10. claiming better generation from one sample grid

## 24. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 25. Checklist

- [ ] minimax objective
- [ ] D loss
- [ ] non-saturating G loss
- [ ] logits stability
- [ ] alternating updates
- [ ] detach semantics
- [ ] mode collapse
- [ ] DCGAN concepts
- [ ] WGAN intuition
- [ ] GAN evaluation

## 26. What's Next

Chapter 29 leaves adversarial training and introduces the mechanism that transformed modern sequence modeling: Attention.
