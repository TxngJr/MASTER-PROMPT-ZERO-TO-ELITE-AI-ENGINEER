# Chapter 27 — Autoencoders & Variational Autoencoders

## 1. Why Representation Learning?

An autoencoder learns:

~~~text
x
↓ encoder
z
↓ decoder
x_hat
~~~

The network is trained to reconstruct its input.

If the latent representation is constrained appropriately, the model is forced to capture useful structure instead of simply copying the input.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- explain encoder / latent / decoder
- distinguish undercomplete and overcomplete autoencoders
- explain reconstruction loss
- understand denoising autoencoders
- understand sparse/regularized autoencoders conceptually
- explain why a plain autoencoder is not automatically generative
- derive the VAE latent distribution parameterization
- implement the reparameterization trick
- calculate Gaussian KL divergence
- explain ELBO
- implement a small VAE in PyTorch
- sample from the prior
- interpolate in latent space
- recognize posterior collapse

## 3. Plain Autoencoder

Encoder:

~~~text
z = f_encoder(x)
~~~

Decoder:

~~~text
x_hat = f_decoder(z)
~~~

Optimize:

~~~text
L_reconstruction(x, x_hat)
~~~

Examples:
- MSE for continuous normalized values
- BCE-like likelihood for Bernoulli-style pixels

Loss choice should match the assumed observation model.

## 4. Bottleneck

An undercomplete autoencoder uses:

~~~text
dim(z) < dim(x)
~~~

This forces compression.

But compression alone does not guarantee semantically meaningful features.

## 5. Overcomplete Autoencoder

If latent dimension is large:

~~~text
dim(z) >= dim(x)
~~~

the network may learn an identity-like solution unless regularized.

Possible constraints:
- noise
- sparsity
- weight regularization
- architecture
- contractive penalties

## 6. Denoising Autoencoder

Train on corrupted input but clean target:

~~~text
x
↓ corrupt
x_tilde
↓ encoder/decoder
x_hat
↓ compare with
x
~~~

The model learns to remove certain perturbations.

This can encourage robust representations.

## 7. Latent Space

A useful latent space may:
- cluster related samples
- interpolate smoothly
- discard nuisance variation
- support downstream prediction

But a plain autoencoder does not force the aggregate latent distribution to match a simple prior.

Random z samples may decode to nonsense.

## 8. Why VAE?

A Variational Autoencoder introduces a probabilistic latent model.

Generative story:

~~~text
z ~ p(z)
x ~ p_theta(x | z)
~~~

Common prior:

~~~text
p(z) = N(0, I)
~~~

The encoder approximates the intractable posterior:

~~~text
q_phi(z | x)
~~~

## 9. Gaussian Encoder

A common encoder outputs:

~~~text
mu(x)
logvar(x)
~~~

representing:

~~~text
q(z|x)
=
N(
    mu,
    diag(exp(logvar))
)
~~~

Use log variance because:
- variance must be positive
- unconstrained neural output can represent logvar
- numerical manipulation is convenient

## 10. Sampling Problem

Naively:

~~~text
z ~ N(mu, sigma²)
~~~

sampling appears to block straightforward gradient flow through mu/sigma.

## 11. Reparameterization Trick

Sample:

~~~text
epsilon ~ N(0,I)
~~~

then:

~~~text
sigma = exp(0.5 * logvar)

z =
mu
+
sigma ⊙ epsilon
~~~

Now randomness is isolated in epsilon, while z remains differentiable with respect to mu/logvar.

## 12. ELBO

We want high log likelihood:

~~~text
log p(x)
~~~

but direct marginalization over z is generally difficult.

The evidence lower bound:

~~~text
ELBO
=
E_q [
    log p_theta(x|z)
]
-
KL(
    q_phi(z|x)
    ||
    p(z)
)
~~~

Maximize ELBO.

Equivalent loss often written:

~~~text
VAE loss
=
reconstruction loss
+
KL divergence
~~~

with sign/convention depending on implementation.

## 13. KL to Standard Normal

For diagonal Gaussian:

~~~text
q(z|x)
=
N(mu, diag(sigma²))

p(z)
=
N(0,I)
~~~

KL per sample:

~~~text
KL
=
-1/2 * sum(
    1
    + logvar
    - mu²
    - exp(logvar)
)
~~~

This encourages approximate posterior distributions toward the prior.

## 14. Reconstruction Term

For binary-valued/normalized Bernoulli-style observations, one can model:

~~~text
p(x|z)
~~~

with Bernoulli logits and use BCE-with-logits reconstruction loss.

For real-valued data, Gaussian likelihood often corresponds to an MSE-like objective under assumptions.

The reconstruction term is a likelihood model, not an arbitrary cosmetic choice.

## 15. Reconstruction vs Regularization Trade-Off

Strong reconstruction pressure:
- preserves detail
- latent may drift from prior

Strong KL pressure:
- latent follows prior
- reconstructions may degrade
- posterior collapse may occur

## 16. beta-VAE

A common modification:

~~~text
L =
reconstruction
+
beta * KL
~~~

beta > 1 increases latent regularization pressure.

This may encourage certain disentanglement behavior but can hurt reconstruction.

Disentanglement is not guaranteed automatically.

## 17. Posterior Collapse

The encoder may learn:

~~~text
q(z|x) ≈ p(z)
~~~

for all x, so decoder effectively ignores z.

Symptoms:
- KL near zero
- poor latent information
- powerful decoder reconstructs using other structure

Mitigations can include:
- KL warmup/annealing
- free bits
- decoder capacity choices
- objective modifications

## 18. Sampling

After training:

~~~text
z ~ N(0,I)
x_generated = decoder(z)
~~~

This is why matching latent posteriors toward the prior matters.

## 19. Latent Interpolation

Given z1 and z2:

~~~text
z(alpha)
=
(1-alpha)z1
+
alpha z2
~~~

decode intermediate points.

Smooth outputs are evidence about decoder geometry, not proof of semantic disentanglement.

## 20. Autoencoder vs PCA

A linear autoencoder with squared loss and appropriate constraints is closely related to the PCA principal subspace.

Nonlinear autoencoders can learn nonlinear mappings.

But they:
- need optimization
- can overfit
- do not automatically provide orthogonal components
- do not automatically give explained-variance semantics

## 21. VAE vs Deterministic Autoencoder

Plain AE:
- deterministic z
- reconstruction objective
- no required simple prior match

VAE:
- distribution q(z|x)
- stochastic latent sample
- KL regularization
- explicit generative latent-variable interpretation

## 22. PyTorch VAE Structure

Encoder:

~~~python
h = encoder(x)
mu = fc_mu(h)
logvar = fc_logvar(h)
~~~

Reparameterize:

~~~python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + eps * std
~~~

Decoder:

~~~python
logits = decoder(z)
~~~

## 23. Stable Reconstruction

Prefer logits-based losses:

~~~python
F.binary_cross_entropy_with_logits(
    logits,
    x,
    reduction="sum",
)
~~~

Do not manually sigmoid then use a logits loss.

## 24. From Scratch

src/vae_math.py includes:
- stable sigmoid
- reparameterize
- diagonal Gaussian KL
- BCE-with-logits
- ELBO-style loss helper

The actual trainable neural VAE is exercised in the PyTorch integration lab.

## 25. Evaluation

Autoencoder:
- reconstruction loss
- downstream latent utility
- anomaly score behavior
- robustness

VAE:
- reconstruction
- KL
- ELBO estimate
- sample quality
- latent interpolation
- downstream utility

A prettier latent plot is not sufficient evaluation.

## 26. Common Mistakes

1. calling any autoencoder generative
2. sampling random z from a plain AE latent space
3. using variance directly without positivity constraint
4. forgetting 0.5 in exp(0.5*logvar)
5. wrong KL sign
6. summing/averaging inconsistently
7. double sigmoid with BCEWithLogits
8. interpreting KL=0 as always good
9. judging VAE only by reconstruction sharpness
10. calling latent dimensions disentangled without evidence

## 27. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 28. Checklist

- [ ] encoder/decoder
- [ ] bottleneck
- [ ] denoising
- [ ] plain AE limits
- [ ] q(z|x)
- [ ] mu/logvar
- [ ] reparameterization
- [ ] KL
- [ ] ELBO
- [ ] beta-VAE
- [ ] posterior collapse
- [ ] prior sampling

## 29. What's Next

Batch 10 moves into adversarial and attention-based generative modeling: GANs, Attention, and Transformers.
