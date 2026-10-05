# SARSA — On-Policy Temporal-Difference Control

SARSA is named after the transition tuple:

~~~text
State, Action, Reward, next State, next Action
~~~

For an on-policy action-value estimate:

~~~text
target = r + gamma * Q(s_next, a_next)
delta  = target - Q(s, a)

Q(s,a) <- Q(s,a) + alpha * delta
~~~

For terminal transitions, the bootstrap term is zero.

## SARSA vs Q-Learning

Q-Learning target:

~~~text
r + gamma * max_a Q(s_next,a)
~~~

SARSA target:

~~~text
r + gamma * Q(s_next,a_next)
~~~

where `a_next` is sampled from the current behavior policy (for example epsilon-greedy). Therefore SARSA is on-policy while basic Q-Learning is off-policy.

## Why behavior matters

If an epsilon-greedy policy sometimes takes risky exploratory actions, SARSA learns values that include those exploratory outcomes. Q-Learning instead backs up the greedy next action. In environments with dangerous states this can produce different learned paths.

## Correctness checks

- terminal transition does not bootstrap;
- `a_next` must come from the same behavior policy used to collect data;
- fixed seed reproduces a tiny tabular trajectory;
- setting epsilon to zero can make SARSA and greedy behavior more similar but does not change their formal update definitions.

## Exercise

Implement one grid-world update for both SARSA and Q-Learning and compare the targets on a state where the behavior policy selected a non-greedy next action.
