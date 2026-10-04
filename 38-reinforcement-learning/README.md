# Chapter 38 — Reinforcement Learning

## 1. What Is Reinforcement Learning?

An agent repeatedly interacts with an environment:

~~~text
state
↓
agent chooses action
↓
environment
↓
reward + next state
↓
repeat
~~~

Unlike supervised learning, correct actions are not directly labeled.

The learning signal comes from rewards over time.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- define an MDP
- explain state/action/reward/transition
- explain return and discounting
- derive Bellman expectation/optimality equations
- implement tabular value iteration
- implement Q-Learning
- explain exploration vs exploitation
- explain replay buffers
- explain target networks
- understand DQN
- derive REINFORCE
- explain advantage functions
- understand actor-critic
- explain PPO clipped objective
- build a tiny PyTorch RL agent

## 3. Markov Decision Process

An MDP contains:

~~~text
(S, A, P, R, gamma)
~~~

where:
- S: states
- A: actions
- P: transition dynamics
- R: reward
- gamma: discount factor

## 4. Markov Property

The next-state distribution depends on the current state/action rather than the entire history:

~~~text
P(s_(t+1) | history, a_t)
=
P(s_(t+1) | s_t, a_t)
~~~

If the observation does not contain enough state information, the problem may be partially observable.

## 5. Return

Discounted return from time t:

~~~text
G_t
=
r_t
+
gamma r_(t+1)
+
gamma^2 r_(t+2)
+ ...
~~~

## 6. Discount Factor

~~~text
0 <= gamma <= 1
~~~

Smaller gamma:
- emphasizes near-term rewards

Larger gamma:
- values longer-term rewards

The correct choice depends on task horizon/objective.

## 7. Policy

A policy defines action selection:

~~~text
pi(a|s)
~~~

Deterministic:

~~~text
a = pi(s)
~~~

Stochastic:

~~~text
a ~ pi(.|s)
~~~

## 8. State Value

~~~text
V^pi(s)
=
E_pi[G_t | s_t=s]
~~~

Expected return when starting at state s and following policy pi.

## 9. Action Value

~~~text
Q^pi(s,a)
=
E_pi[G_t | s_t=s,a_t=a]
~~~

## 10. Bellman Expectation Equation

~~~text
V^pi(s)
=
Σ_a pi(a|s)
Σ_s' P(s'|s,a)
[
R(s,a,s')
+
gamma V^pi(s')
]
~~~

## 11. Bellman Optimality

~~~text
Q*(s,a)
=
E[
r
+
gamma max_a' Q*(s',a')
]
~~~

This recursion motivates Q-Learning.

## 12. Value Iteration

Repeatedly apply:

~~~text
V_(k+1)(s)
=
max_a
Σ_s'
P(s'|s,a)
[
R + gamma V_k(s')
]
~~~

until convergence for a finite known MDP.

## 13. Q-Learning

Model-free TD control:

~~~text
target
=
r
+
gamma max_a' Q(s',a')
~~~

Update:

~~~text
Q(s,a)
←
Q(s,a)
+
alpha
[
target - Q(s,a)
]
~~~

For terminal transitions, do not bootstrap beyond terminal.

## 14. Temporal-Difference Error

~~~text
delta
=
r
+
gamma V(s')
-
V(s)
~~~

It measures mismatch between current estimate and one-step bootstrapped target.

## 15. Exploration vs Exploitation

Greedy exploitation chooses the currently best action.

Exploration tries alternatives.

Common baseline:

~~~text
epsilon-greedy
~~~

With probability epsilon:
- random action

Otherwise:
- argmax Q

## 16. Epsilon Schedule

Often start with more exploration and decay epsilon.

Do not decay too quickly or the agent may stop discovering useful behavior.

## 17. DQN

Replace tabular Q with neural network:

~~~text
state
↓
Q-network
↓
Q(s,a_1)...Q(s,a_k)
~~~

Training target:

~~~text
y =
r
+
gamma
max_a' Q_target(s',a')
~~~

## 18. Replay Buffer

Store transitions:

~~~text
(s,a,r,s',done)
~~~

Sample mini-batches later.

Benefits:
- breaks strong temporal correlation
- reuses experience
- improves data efficiency

## 19. Target Network

Using the same rapidly-changing network on both sides of the TD target can destabilize learning.

Maintain:
- online network
- target network

Periodically copy or softly update target parameters.

## 20. DQN Loss

Common:

~~~text
Huber(
Q_online(s,a),
target
)
~~~

Huber/SmoothL1 reduces sensitivity to large TD errors compared with pure squared loss.

## 21. Double DQN Preview

Standard DQN uses max over target values and can overestimate.

Double DQN separates:
- action selection by online network
- action evaluation by target network

## 22. Policy Gradient

Instead of learning Q then choosing argmax, optimize policy directly.

REINFORCE objective gradient:

~~~text
grad J(theta)
=
E[
G_t
grad log pi_theta(a_t|s_t)
]
~~~

## 23. Why log-probability?

For sampled action:

~~~text
log pi(a|s)
~~~

gives a differentiable surrogate.

Positive return:
- increase probability

Negative/low advantage:
- decrease probability

## 24. Baseline

Subtract a baseline b(s):

~~~text
G_t - b(s_t)
~~~

This can reduce gradient variance without changing the expected policy-gradient direction when used correctly.

## 25. Advantage

~~~text
A(s,a)
=
Q(s,a)
-
V(s)
~~~

Actor-critic methods estimate how much better an action was than the state's baseline expectation.

## 26. Actor-Critic

Actor:
- policy pi(a|s)

Critic:
- V(s) or Q(s,a)

Critic provides lower-variance learning signal.

## 27. Generalized Advantage Estimation

GAE combines TD residuals across time:

~~~text
A_t^GAE
=
delta_t
+
gamma lambda delta_(t+1)
+ ...
~~~

lambda controls bias/variance trade-off.

## 28. PPO

PPO reuses sampled trajectories while constraining large policy updates.

Probability ratio:

~~~text
r_t(theta)
=
pi_theta(a_t|s_t)
/
pi_old(a_t|s_t)
~~~

Clipped objective:

~~~text
min(
r_t A_t,
clip(r_t,1-epsilon,1+epsilon) A_t
)
~~~

Maximize this surrogate.

## 29. Why Clipping?

Large policy changes can destroy behavior learned from the data that generated the trajectory.

Clipping limits incentive to move the action probability ratio too far.

It is not a hard guarantee that the policy changes by a fixed amount.

## 30. Entropy Bonus

Policy entropy:

~~~text
H(pi(.|s))
~~~

can encourage exploration.

Too much entropy pressure prevents policy concentration.

## 31. On-Policy vs Off-Policy

On-policy:
- train mainly from data collected by current/recent policy
- examples: REINFORCE, PPO

Off-policy:
- can learn from data generated by different behavior policy
- examples: Q-Learning, DQN

## 32. Reward Design

Bad rewards can produce unintended policies.

Always inspect behavior, not only aggregate reward.

Reward hacking/specification gaming becomes increasingly important in alignment chapters.

## 33. Evaluation

Separate:
- training exploration returns
- deterministic/stochastic evaluation returns

Use multiple seeds because RL variance can be high.

## 34. From Scratch

src/rl_numpy.py includes:

- discounted_returns
- bellman_optimality_backup
- q_learning_update
- epsilon_greedy
- ppo_clipped_surrogate

## 35. Common Mistakes

1. bootstrapping terminal states
2. mixing reward and return
3. epsilon never decays or decays too fast
4. replay target with wrong done mask
5. target network never updated
6. action tensor used with wrong Q dimension
7. policy loss sign reversed
8. advantage not detached when intended
9. PPO ratio built from wrong old log-probabilities
10. reporting one lucky random seed

## 36. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 37. Checklist

- [ ] MDP
- [ ] return / gamma
- [ ] V / Q
- [ ] Bellman equations
- [ ] Q-Learning
- [ ] epsilon-greedy
- [ ] DQN
- [ ] replay / target network
- [ ] REINFORCE
- [ ] advantage
- [ ] actor-critic
- [ ] PPO clipping

## 38. What's Next

Chapter 39 unifies the major families of generative modeling before the dedicated diffusion chapter.
