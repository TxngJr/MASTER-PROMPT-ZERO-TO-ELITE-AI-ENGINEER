# Batch 13 Integration Project — Graph, Control & Generative Lab

Batch 13 contains three experiments.

## Part A — Tiny GCN

Synthetic homophilous communities:

~~~text
node features + graph
↓
GCN layer
↓
GCN layer
↓
node logits
~~~

Tracks:
- node/edge counts
- parameter count
- validation accuracy
- final test accuracy

## Part B — Tiny DQN

Chain-control environment:

~~~text
state
↓
Q-network
↓
left / right action
↓
reward + next state
~~~

Uses:
- replay buffer
- target network
- epsilon-greedy exploration
- Huber TD loss
- terminal masking

Tracks:
- success rate
- recent return
- replay size

## Part C — Generative Distribution Diagnostics

Synthetic four-mode 2D data.

Fit a single Gaussian explicit density and compare:
- fitted mean/covariance
- NLL
- generated sample moments

Also inspect forward Gaussian noising at several alpha_bar values as preparation for Chapter 40 Diffusion.

## Run

~~~bash
python integration-project-batch13/src/graph_rl_generative_lab.py --mode gnn --steps 30
python integration-project-batch13/src/graph_rl_generative_lab.py --mode dqn --steps 30
python integration-project-batch13/src/graph_rl_generative_lab.py --mode generative
~~~

## Required Extensions

1. GraphSAGE edge-list model
2. GAT implementation
3. heterophily stress test
4. DQN Double-DQN target
5. prioritized replay concept experiment
6. REINFORCE chain agent
7. PPO discrete-control agent
8. Gaussian-mixture explicit density
9. GAN comparison from Batch 10
10. diffusion denoiser in Chapter 40
