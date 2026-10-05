# SARSA

SARSA is an on-policy temporal-difference control algorithm. The name comes from the transition tuple:

S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}

Update:

Q(S_t,A_t) <- Q(S_t,A_t) + alpha [R_{t+1} + gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]

The next action A_{t+1} is sampled from the same behavior policy currently interacting with the environment, commonly epsilon-greedy.

## SARSA vs Q-Learning
- SARSA: on-policy target uses the action actually selected next.
- Q-Learning: off-policy target uses max_a Q(S_{t+1},a).
- With exploratory behavior, SARSA learns values that include the consequences of that exploration.
- Q-Learning instead targets the greedy policy even while behavior explores.

## Terminal state
When S_{t+1} is terminal, the bootstrapped term is zero.

## Minimal pseudocode
1. Initialize Q.
2. Choose A from policy derived from Q.
3. Observe R and S'.
4. Choose A' from the same policy.
5. Apply SARSA update.
6. Set S <- S', A <- A' and repeat.

## Experiment
Compare SARSA and Q-Learning on the same small grid environment with identical seeds, epsilon schedule, alpha and gamma. Plot return and unsafe/undesired state visits separately.
