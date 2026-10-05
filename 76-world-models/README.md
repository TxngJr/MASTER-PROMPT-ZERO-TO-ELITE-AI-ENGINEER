# Chapter 76 — World Models

## 1. What Is a World Model?

A world model learns enough structure about environment dynamics to predict useful future representations/outcomes and support planning or control.

~~~text
observation
↓ encoder
latent state
+ action
↓ dynamics model
predicted future latent
↓ reward/value/decoder/planner
decision
~~~

Not every world model must reconstruct pixels.

## 2. Learning Objectives

- distinguish observation, state, latent state and belief state
- implement a simple learned dynamics model
- roll out imagined trajectories
- plan over model-predicted futures
- calculate model error across rollout horizons
- distinguish pixel prediction vs latent prediction
- understand model-based RL
- understand Dreamer-style latent imagination
- understand JEPA-style predictive representations
- reason about compounding model error
- evaluate world models beyond reconstruction quality

## 3. Markov State

In an ideal Markov state:

~~~text
P(s_{t+1} | history, a_t)
=
P(s_{t+1} | s_t, a_t)
~~~

Raw observations are often partially observable, so the model may need memory/belief state.

## 4. Latent Dynamics

Encode observation o_t:

~~~text
z_t = encoder(o_t)
~~~

Predict:

~~~text
z_hat_{t+1} = f(z_t, a_t)
~~~

A simple linear educational model:

~~~text
z_{t+1} = A z_t + B a_t
~~~

## 5. Multi-Step Rollout

~~~text
z_t
+ a_t → z_hat_{t+1}
+ a_{t+1} → z_hat_{t+2}
...
~~~

Errors compound because later predictions consume earlier predictions.

## 6. Model-Based Planning

Given candidate action sequences:

1. imagine trajectory under each plan
2. compute predicted reward/cost
3. choose highest predicted return
4. execute only the first action
5. observe real environment
6. replan

This receding-horizon idea reduces reliance on long open-loop prediction.

## 7. Discounted Return

~~~text
G = sum_t gamma^t r_t
~~~

World-model planning is only as good as dynamics plus reward/value estimates.

## 8. Pixel Prediction

Predict future pixels directly.

Advantages:
- easy to visualize

Challenges:
- expensive high-dimensional outputs
- many pixel details irrelevant to control
- blurry/uncertain futures

## 9. Latent Prediction

Predict task-relevant representations rather than exact pixels.

This can discard unpredictable nuisance detail while preserving useful structure.

## 10. JEPA-Style Prediction

Joint-Embedding Predictive Architectures predict representations of unseen/target content rather than reconstructing raw input exactly.

Meta's current V-JEPA 2 release explicitly frames this direction as world-model research and highlights robotics/physical-world applications. (see references.md)

An educational objective:

~~~text
L = || predictor(context, action) - target_embedding ||^2
~~~

Real JEPA systems include richer masking/encoder/target-encoder designs.

## 11. Video World Models

Video supplies temporal supervision:
- object permanence
- motion
- interactions
- scene dynamics

But passive video alone may not reveal action-conditioned controllability.

## 12. Action-Conditioned Models

For robotics/control:

~~~text
future = f(current_state, action)
~~~

Actions let the model distinguish what the agent can cause from what merely happens.

## 13. Dreamer-Style Concepts

Dreamer-family approaches learn a compact latent dynamics model and improve behavior using imagined trajectories inside that learned latent space.

Key concepts:
- representation model
- transition/dynamics model
- reward/value prediction
- imagination rollouts
- actor/value learning

## 14. Uncertainty

A deterministic model can be overconfident when multiple futures are plausible.

Approaches include:
- stochastic latent variables
- ensembles
- distributions over next states

Planning should account for model uncertainty in safety-critical domains.

## 15. Compounding Error

One-step MSE can be low while 50-step rollouts drift badly.

Evaluate prediction error versus horizon:

~~~text
horizon 1, 2, 4, 8, 16, ...
~~~

## 16. Model Exploitation

A planner can discover action sequences that exploit inaccuracies in the learned model and look excellent only in imagination.

Mitigations:
- short-horizon replanning
- uncertainty penalties
- real-environment validation
- conservative constraints

## 17. World Model vs Simulator

A simulator is usually hand-engineered/physics-based.

A learned world model is estimated from data.

Hybrid systems can combine both.

## 18. Evaluation

Measure:
- one-step prediction
- multi-step rollout error
- representation quality
- planning/control success
- uncertainty calibration
- robustness under distribution shift

Pixel realism alone is insufficient.

## 19. From Scratch

`src/world_model.py` implements:
- one_step_mse
- fit_linear_dynamics
- linear_step
- rollout_linear_dynamics
- discounted_return
- goal_reward
- enumerate_action_plans
- plan_by_model
- jepa_prediction_loss

## 20. Common Mistakes

1. one-step accuracy treated as long-horizon accuracy
2. pixel reconstruction treated as complete world understanding
3. action conditioning omitted for controllable environments
4. planner allowed to exploit model errors
5. uncertainty ignored
6. imagined return treated as real return
7. open-loop long rollout used without replanning
8. latent representation never evaluated
9. train/test trajectories leak
10. simulation success assumed to transfer to reality

## 21. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 22. Checklist

- [ ] latent state
- [ ] transition model
- [ ] action conditioning
- [ ] rollout
- [ ] planning
- [ ] compounding error
- [ ] uncertainty
- [ ] JEPA concepts
- [ ] Dreamer concepts
- [ ] world-model evaluation

## 23. What's Next

Chapter 77 turns learned perception/world representations into safe robot actions, primarily in simulation for this course.