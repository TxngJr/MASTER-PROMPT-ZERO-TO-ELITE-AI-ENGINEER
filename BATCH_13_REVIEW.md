# Batch 13 Review — Chapters 37–39

## Chapters

- 37 — Graph Neural Networks
- 38 — Reinforcement Learning
- 39 — Generative AI Foundations

## GNN Skills

- graph / adjacency / edge-list representations
- node / edge / graph features
- message passing
- permutation-invariant aggregation
- GCN symmetric normalization
- GraphSAGE
- GAT
- node / graph / link prediction
- transductive vs inductive learning
- oversmoothing / oversquashing
- homophily caveats

## Reinforcement-Learning Skills

- MDP
- return / discounting
- policy
- state/action values
- Bellman expectation / optimality
- Q-Learning
- epsilon-greedy exploration
- replay buffers
- target networks
- DQN
- REINFORCE
- actor-critic
- advantage / GAE
- PPO clipping

## Generative-AI Foundation Skills

- generative vs discriminative modeling
- explicit vs implicit density models
- likelihood / NLL
- autoregressive models
- latent-variable models / ELBO
- GANs
- energy-based models
- score functions
- diffusion preview
- flow-model concepts
- mode coverage
- generative precision / recall intuition
- sampling/evaluation limitations

## Implemented From Scratch

### Chapter 37
- adjacency_from_edges
- add_self_loops
- symmetric_normalize
- gcn_layer
- mean_neighbor_aggregate
- graph_mean_pool

### Chapter 38
- discounted_returns
- bellman_optimality_backup
- q_learning_update
- epsilon_greedy
- ppo_clipped_surrogate

### Chapter 39
- gaussian_nll
- categorical_nll
- forward_diffusion_sample
- energy_to_probability
- effective_sample_size

## Integration Project

Three modes:

1. synthetic-community GCN node classifier
2. DQN on a tiny chain-control environment
3. explicit Gaussian / forward-noising generative diagnostics

## PyTorch API Audit

The integration uses only core PyTorch primitives.

GNN:
- dense normalized adjacency for visibility of the math

DQN:
- replay buffer
- target network
- AdamW
- Huber TD loss

The course intentionally does not require PyTorch Geometric yet.

## Methodology Audit

### GNN
- train/validation/test masks are separate
- labels for held-out nodes are not used in training loss
- validation chooses checkpoint
- test is evaluated after selection

### RL
- terminal transitions do not bootstrap
- replay stores explicit done flags
- target network is updated separately
- epsilon exploration is distinct from greedy evaluation logic

### Generative
- NLL is reported only where density is explicitly available
- multimodal data exposes limitations of a single Gaussian
- diffusion noising is treated as a preview, not a complete generative sampler

## Interpretation Audit

- graph attention weights are not complete explanations
- high RL return from one seed is insufficient evidence
- reward design can create unintended behavior
- sample quality does not guarantee mode coverage
- ELBO is generally a bound rather than exact likelihood
- GANs do not expose a standard tractable likelihood

## Exit Gate

Before Chapter 40:

1. Batch 13 Core CI passes
2. Batch 13 PyTorch smoke passes
3. derive GCN normalization
4. explain message passing and invariant aggregation
5. derive Bellman optimality / Q-Learning update
6. explain DQN replay + target networks
7. derive policy-gradient intuition and PPO ratio clipping
8. classify major generative-model families by objective/sampling behavior
