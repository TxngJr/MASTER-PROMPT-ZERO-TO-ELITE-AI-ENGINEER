# Chapter 39 — Generative AI Foundations

## 1. What Makes a Model Generative?

A generative model learns a distribution over data or a process that can produce new samples.

Conceptually:

~~~text
training data
↓
learn distribution / sampler
↓
new samples
~~~

This differs from a purely discriminative model that maps inputs directly to labels.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- distinguish generative vs discriminative modeling
- explain explicit vs implicit density models
- explain likelihood and negative log-likelihood
- compare autoregressive models
- compare latent-variable models
- compare GANs
- explain energy-based models conceptually
- explain score/diffusion modeling conceptually
- understand sampling vs density evaluation
- understand mode coverage vs sample quality
- identify major generative evaluation traps

## 3. Discriminative Modeling

Typical supervised classifier:

~~~text
P(y|x)
~~~

Goal:
- predict labels/targets given input

It does not necessarily model the distribution of x itself.

## 4. Generative Modeling

A generative model may learn:

~~~text
P(x)
~~~

or:

~~~text
P(x,y)
~~~

or an implicit sampling procedure whose density is not tractable.

## 5. Explicit Density Models

Models where a likelihood/density can be computed or bounded.

Examples:
- autoregressive models
- normalizing flows
- many latent-variable models via ELBO bounds

## 6. Implicit Generative Models

Models may define sampling without a tractable normalized likelihood.

Classic GANs are a major example.

Evaluation therefore requires distributional/sample diagnostics rather than direct NLL alone.

## 7. Maximum Likelihood

Given samples x_i and model p_theta(x):

~~~text
maximize
Σ_i log p_theta(x_i)
~~~

Equivalent:

~~~text
minimize
-Σ_i log p_theta(x_i)
~~~

Negative log-likelihood is a core training/evaluation quantity for explicit density models.

## 8. Autoregressive Models

Factorize:

~~~text
P(x_1,...,x_T)
=
Π_t P(x_t | x_<t)
~~~

Examples:
- GPT-like text models
- pixel autoregressive models
- autoregressive audio models

Pros:
- tractable next-step likelihood
- straightforward teacher-forced training

Cons:
- sequential generation

## 9. Latent Variable Models

Introduce latent z:

~~~text
p(x)
=
∫ p(x|z)p(z) dz
~~~

Examples:
- VAE

Latent variables can capture lower-dimensional hidden structure.

Exact marginal likelihood may be difficult.

## 10. ELBO

VAE-style training optimizes a lower bound:

~~~text
log p(x)
>=
E_q(z|x)[log p(x|z)]
-
KL(q(z|x)||p(z))
~~~

The ELBO trades reconstruction likelihood with latent regularization.

## 11. Adversarial Models

GAN:

~~~text
z → Generator → fake x
                   ↓
real x ───────→ Discriminator
~~~

The generator learns through a learned discriminator/critic rather than explicit log-likelihood.

## 12. Energy-Based Models

An energy model assigns:

~~~text
E_theta(x)
~~~

Lower energy can represent higher compatibility/probability.

Normalized density:

~~~text
p_theta(x)
=
exp(-E_theta(x))
/
Z(theta)
~~~

Partition function:

~~~text
Z(theta)
=
∫ exp(-E_theta(x)) dx
~~~

can be difficult to compute.

## 13. Score Function

For continuous density:

~~~text
score(x)
=
∇_x log p(x)
~~~

It points toward increasing log density locally.

Score-based generative models learn score fields at different noise levels.

## 14. Diffusion Preview

Forward process gradually adds noise:

~~~text
x_0
→ x_1
→ ...
→ x_T ≈ noise
~~~

A learned reverse/denoising process reconstructs samples from noise.

Chapter 40 derives and implements this in detail.

## 15. Forward Gaussian Noising

A common closed form:

~~~text
x_t
=
sqrt(alpha_bar_t) x_0
+
sqrt(1-alpha_bar_t) epsilon
~~~

where:

~~~text
epsilon ~ N(0,I)
~~~

As alpha_bar decreases, the sample contains less original signal.

## 16. Flow-Based Models Preview

Normalizing flows use invertible transforms:

~~~text
z ~ simple density
x = f(z)
~~~

Change-of-variables gives exact likelihood when Jacobian determinants are tractable.

Flows trade architectural constraints for exact density evaluation.

## 17. Likelihood Is Not Everything

High likelihood does not guarantee:
- perceptual quality
- semantic usefulness
- safe outputs

Likewise visually appealing samples do not prove correct density coverage.

## 18. Mode Coverage

A true distribution may have several modes.

A model should ideally represent all important modes.

Mode dropping:

~~~text
real modes:
A B C D

generated:
A A A C
~~~

can still produce individually plausible samples while missing diversity.

## 19. Precision vs Recall Intuition

Generative precision:
- are generated samples realistic / on-support?

Generative recall:
- does the model cover the variety of real data?

A model can have:
- high precision, low recall
- broad recall, low precision

## 20. Distribution Shift

Generative models learn from training distribution.

Out-of-distribution prompts/conditions can produce unpredictable behavior.

Document:
- training domain
- known limitations
- evaluation domain

## 21. Conditional Generation

Condition c:

~~~text
p(x|c)
~~~

Examples:
- class-conditioned image
- text-conditioned image
- prompt-conditioned language
- speaker-conditioned audio

Conditioning changes the target distribution, not the basic definition of generation.

## 22. Guidance Preview

Conditional diffusion can alter sampling using:
- classifier guidance
- classifier-free guidance

These are covered in the dedicated diffusion chapter.

## 23. Sampling Quality

Sampling hyperparameters matter.

Examples:
- temperature in autoregressive models
- truncation/top-p
- diffusion step count/guidance
- GAN latent sampling

Evaluation must specify sampling procedure.

## 24. Memorization

A model can reproduce or nearly reproduce training examples.

Generative evaluation should consider:
- nearest training examples
- duplicate detection
- privacy/memorization risk

Novel-looking output alone does not prove non-memorization.

## 25. Evaluation Families

Text:
- NLL/perplexity
- task/human evaluation
- diversity / factuality / safety diagnostics

Images:
- FID/KID-style distribution metrics
- precision/recall
- human/task evaluation

Audio/video:
- domain-specific quality and alignment metrics

No single metric captures all quality dimensions.

## 26. From Scratch

src/generative_math.py includes:

- gaussian_nll
- categorical_nll
- forward_diffusion_sample
- energy_to_probability
- effective_sample_size

These primitives connect major generative objectives.

## 27. Generative Family Map

~~~text
Generative Models
├── Autoregressive
│   └── GPT-style
├── Latent Variable
│   └── VAE
├── Adversarial
│   └── GAN
├── Energy-Based
├── Flow-Based
└── Score / Diffusion
~~~

Real systems can combine multiple ideas.

## 28. Common Mistakes

1. equating generation with GANs only
2. assuming sample quality implies mode coverage
3. comparing likelihood across incompatible data/tokenization
4. calling ELBO exact likelihood
5. treating GAN discriminator score as universal quality metric
6. ignoring sampling hyperparameters
7. ignoring training-data memorization
8. assuming diffusion is just "add random noise"
9. evaluating with one metric only
10. confusing conditioning with guaranteed controllability

## 29. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 30. Checklist

- [ ] generative vs discriminative
- [ ] explicit vs implicit
- [ ] NLL / likelihood
- [ ] autoregressive
- [ ] latent variable / ELBO
- [ ] GAN
- [ ] energy model
- [ ] flow concept
- [ ] score function
- [ ] diffusion preview
- [ ] mode coverage
- [ ] evaluation limitations

## 31. What's Next

Chapter 40 derives diffusion models in detail: forward noising, reverse denoising, schedules, epsilon prediction and sampling.
