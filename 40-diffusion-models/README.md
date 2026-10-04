# Chapter 40 — Diffusion Models

## 1. Core Idea

Diffusion models learn to reverse a gradual noising process.

~~~text
clean x_0
↓ add small Gaussian noise repeatedly
x_1
↓
...
↓
x_T ≈ Gaussian noise

learned reverse process:

noise
↓ denoise
...
↓
sample
~~~

## 2. Learning Objectives

By the end of this chapter you should be able to:

- define beta, alpha and cumulative alpha schedules
- derive q(x_t | x_0)
- sample arbitrary diffusion timesteps directly
- explain epsilon prediction
- derive the DDPM reverse mean
- explain posterior variance
- implement a training objective
- implement ancestral sampling
- explain DDIM intuition
- explain classifier and classifier-free guidance
- understand timestep embeddings
- explain U-Net usage
- distinguish pixel-space vs latent diffusion
- understand diffusion for image/audio/video

## 3. Forward Markov Process

A common DDPM forward transition:

~~~text
q(x_t | x_(t-1))
=
N(
sqrt(alpha_t) x_(t-1),
beta_t I
)
~~~

where:

~~~text
alpha_t = 1 - beta_t
~~~

## 4. Cumulative Product

Define:

~~~text
alpha_bar_t =
Π_(s=1)^t alpha_s
~~~

Then there is a useful closed form:

~~~text
q(x_t | x_0)
=
N(
sqrt(alpha_bar_t) x_0,
(1-alpha_bar_t) I
)
~~~

## 5. Direct Noising

Therefore:

~~~text
epsilon ~ N(0,I)

x_t =
sqrt(alpha_bar_t) x_0
+
sqrt(1-alpha_bar_t) epsilon
~~~

This lets training sample any timestep without simulating every earlier noising step.

## 6. Signal-to-Noise Ratio

A useful measure:

~~~text
SNR_t =
alpha_bar_t
/
(1-alpha_bar_t)
~~~

Early timesteps:
- higher signal

Late timesteps:
- more noise

## 7. Beta Schedule

A schedule chooses beta_t over timesteps.

Common educational choices:
- linear beta schedule
- cosine-inspired alpha_bar schedules

The schedule changes task difficulty across time.

## 8. Noise-Prediction Objective

Sample:
- clean x_0
- timestep t
- noise epsilon

Construct x_t.

Train network:

~~~text
epsilon_theta(x_t,t)
≈
epsilon
~~~

Typical simplified objective:

~~~text
L =
E[
||epsilon - epsilon_theta(x_t,t)||^2
]
~~~

## 9. Why Predict Noise?

Given x_t, t and estimated epsilon, one can estimate the clean sample:

~~~text
x0_hat
=
(
x_t
-
sqrt(1-alpha_bar_t) epsilon_hat
)
/
sqrt(alpha_bar_t)
~~~

Equivalent parameterizations may predict:
- noise epsilon
- clean x_0
- velocity v

Different systems choose differently.

## 10. Timestep Conditioning

The denoiser must know the noise level.

Typical input:

~~~text
time embedding
→ MLP
→ inject into network blocks
~~~

Sinusoidal timestep embeddings are common.

## 11. Reverse Process

Learn:

~~~text
p_theta(x_(t-1) | x_t)
~~~

as a Gaussian parameterized using the model prediction.

Sampling begins from:

~~~text
x_T ~ N(0,I)
~~~

and repeatedly moves toward lower-noise states.

## 12. DDPM Reverse Mean

Using epsilon prediction, a common reverse mean is:

~~~text
mu_theta(x_t,t)
=
1/sqrt(alpha_t)
[
x_t
-
beta_t
/
sqrt(1-alpha_bar_t)
epsilon_theta(x_t,t)
]
~~~

Then sample:

~~~text
x_(t-1)
=
mu_theta
+
sigma_t z
~~~

for t > 1.

## 13. Posterior Variance

The true forward posterior has variance related to:

~~~text
beta_tilde_t
=
beta_t
(1-alpha_bar_(t-1))
/
(1-alpha_bar_t)
~~~

Educational samplers often use this posterior variance.

## 14. Final Step

At t=1 there is no need to add fresh noise when producing x_0.

An off-by-one error here can degrade sample quality.

## 15. DDIM Intuition

DDIM constructs a non-Markovian sampling path that can be deterministic when stochasticity parameter eta=0.

Benefits:
- fewer sampling steps
- deterministic sampling path under some settings

DDIM does not require retraining the same epsilon-prediction network in the basic setting.

## 16. Classifier Guidance

A separate classifier can modify the score toward a desired class.

Conceptually:

~~~text
conditional score
=
unconditional score
+
guidance strength
*
gradient log p(class|x_t)
~~~

Requires a noise-aware classifier.

## 17. Classifier-Free Guidance

Train the denoiser sometimes with condition and sometimes without it.

At sampling:

~~~text
eps_guided
=
eps_uncond
+
w(
eps_cond - eps_uncond
)
~~~

Higher w often strengthens conditioning but can reduce diversity or create artifacts.

## 18. U-Net

Image diffusion commonly uses a U-Net-like denoiser:

- downsampling path
- bottleneck
- upsampling path
- skip connections
- residual blocks
- timestep conditioning
- attention at selected resolutions

## 19. Latent Diffusion

Instead of denoising raw pixels:

~~~text
image
↓ encoder
latent z
↓ diffusion in latent space
↓ decoder
image
~~~

This reduces spatial/computational cost.

## 20. Diffusion for Audio

Possible representations:
- waveform
- spectrogram
- learned latent audio representation

The diffusion machinery can operate over many continuous data spaces.

## 21. Diffusion for Video

Video adds:
- spatial dimensions
- temporal dimension
- much larger compute/memory requirements
- temporal consistency challenges

Architectures may use temporal attention/convolution or latent video spaces.

## 22. Sampling Cost

Unlike one-pass GAN generation, basic diffusion performs many denoising evaluations.

This motivates:
- DDIM
- distillation
- consistency models
- solver-based samplers
- fewer-step models

## 23. Training vs Sampling

Training:
- randomly choose t
- one noising operation
- one denoiser prediction

Sampling:
- start from noise
- run many reverse timesteps

Training is parallel across examples; generation is iterative across diffusion time.

## 24. From Scratch

src/diffusion_numpy.py includes:

- linear_beta_schedule
- alpha_terms
- q_sample
- predict_x0_from_epsilon
- posterior_variance
- ddpm_reverse_mean

## 25. Common Mistakes

1. confusing alpha with alpha_bar
2. timestep indexing off by one
3. using wrong alpha_bar in q_sample
4. adding noise at final reverse step
5. forgetting timestep conditioning
6. mixing epsilon/x0/v parameterizations
7. guidance scale interpreted as probability
8. schedule arrays with incompatible indexing
9. evaluating only cherry-picked samples
10. assuming fewer steps always preserves quality

## 26. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 27. Checklist

- [ ] beta / alpha / alpha_bar
- [ ] q(x_t|x_0)
- [ ] epsilon prediction
- [ ] x0 reconstruction
- [ ] posterior variance
- [ ] DDPM mean
- [ ] sampling loop
- [ ] DDIM
- [ ] CFG
- [ ] U-Net
- [ ] latent diffusion

## 28. What's Next

Chapter 41 combines multiple modalities such as text and vision in shared or connected representation spaces.
