# Chapter 37 — Graph Neural Networks

## 1. Why Graphs?

Many datasets are not naturally grids or sequences.

Examples:

- social networks
- molecules
- knowledge graphs
- recommendation interactions
- road networks
- dependency graphs

A graph is:

~~~text
G = (V,E)
~~~

with nodes V and edges E.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- represent graphs with adjacency matrices and edge lists
- distinguish directed/undirected graphs
- explain node, edge and graph features
- derive message passing
- implement neighbor aggregation
- derive the GCN normalization
- explain GraphSAGE
- explain Graph Attention Networks
- understand oversmoothing
- understand permutation invariance/equivariance
- build node/graph prediction pipelines
- implement a tiny PyTorch GNN without PyG

## 3. Graph Representation

Adjacency matrix:

~~~text
A_ij = 1
if edge i→j exists
~~~

For N nodes:

~~~text
A ∈ R^(N×N)
~~~

Node features:

~~~text
X ∈ R^(N×F)
~~~

## 4. Edge List

Sparse graphs are often stored as pairs:

~~~text
source_nodes
target_nodes
~~~

or:

~~~text
edge_index:
(2,E)
~~~

This avoids storing N² entries.

## 5. Message Passing

Generic layer:

~~~text
m_ij = MESSAGE(h_i, h_j, e_ij)

m_i = AGGREGATE({m_ij : j in N(i)})

h_i' = UPDATE(h_i, m_i)
~~~

Aggregation should not depend on arbitrary neighbor ordering.

## 6. Permutation Invariance

A node's neighbor set has no canonical order.

Common invariant aggregators:
- sum
- mean
- max

If neighbor rows are permuted, the aggregate should remain unchanged.

## 7. GCN

Classic graph convolution:

~~~text
A_tilde = A + I

D_tilde_ii =
Σ_j A_tilde_ij

H' =
sigma(
D_tilde^(-1/2)
A_tilde
D_tilde^(-1/2)
H W
)
~~~

Self-loops let a node preserve/use its own current features.

## 8. Why Normalize?

Raw sum aggregation gives high-degree nodes larger activation magnitude.

Symmetric normalization controls degree effects:

~~~text
1 / sqrt(d_i d_j)
~~~

for each connected pair.

## 9. GCN Shape Flow

~~~text
H:
(N,F_in)

W:
(F_in,F_out)

normalized adjacency:
(N,N)

output:
(N,F_out)
~~~

## 10. Node Classification

Example:

~~~text
node features + graph
↓
GNN
↓
node embeddings
↓
Linear
↓
class logits per node
~~~

Train only on labeled training nodes.

Validation/test node labels must remain held out.

## 11. Transductive vs Inductive

Transductive:
- graph structure for all nodes may be visible
- labels for validation/test remain hidden

Inductive:
- model must generalize to new nodes/graphs

Be explicit about which setting you evaluate.

## 12. GraphSAGE

GraphSAGE learns an aggregation function:

~~~text
neighbor_summary =
AGGREGATE({h_j})

h_i' =
sigma(
W [
 h_i || neighbor_summary
]
)
~~~

It was designed with inductive representation learning in mind.

## 13. Mean GraphSAGE

Simple variant:

~~~text
neighbor_mean =
mean_j h_j

combined =
concat(h_i, neighbor_mean)

h_i' =
sigma(W combined)
~~~

## 14. Sampling Neighbors

Large graphs can have huge neighborhoods.

GraphSAGE-style systems may sample a fixed number of neighbors per layer.

This bounds computation but introduces sampling variance.

## 15. Graph Attention Network

GAT learns neighbor weights:

~~~text
e_ij =
LeakyReLU(
a^T [Wh_i || Wh_j]
)

alpha_ij =
softmax_j(e_ij)

h_i' =
Σ_j alpha_ij W h_j
~~~

Only graph-connected neighbors participate.

## 16. Multi-Head Graph Attention

Multiple attention heads can:
- concatenate outputs
- average outputs

as defined by architecture/layer.

## 17. Edge Features

Some GNNs include edge information:

~~~text
message_ij =
f(
h_i,
h_j,
e_ij
)
~~~

Important for:
- bond types
- distances
- relation types

## 18. Graph-Level Prediction

Need one vector for an entire graph.

Use a permutation-invariant readout:

- sum pooling
- mean pooling
- max pooling
- learned attention pooling

Then classify/regress the graph representation.

## 19. Link Prediction

Goal:

~~~text
score(i,j)
~~~

predict whether nodes should connect.

Simple decoder:

~~~text
score =
h_i^T h_j
~~~

More expressive decoders can be learned.

## 20. Oversmoothing

Deep message passing repeatedly mixes neighboring features.

Eventually node representations may become too similar.

Symptoms:
- class separation decreases with depth
- embeddings collapse locally/globally

Mitigations:
- residuals
- normalization
- fewer layers
- architecture changes

## 21. Oversquashing

Information from exponentially many distant nodes may be compressed through fixed-size representations/graph bottlenecks.

This differs from oversmoothing.

Graph geometry strongly affects long-range information flow.

## 22. Homophily Caveat

Many classic GNNs work well when connected nodes tend to share labels/features.

In heterophilous graphs, naive neighbor averaging can hurt.

Always inspect graph/task structure.

## 23. PyTorch Without PyG

Message aggregation can be written using edge lists and operations such as indexed accumulation.

Conceptually:

~~~text
messages = transformed[source]
aggregate[target] += messages
~~~

This chapter intentionally avoids requiring PyTorch Geometric so the mechanics stay visible.

## 24. From Scratch

src/gnn_numpy.py includes:

- adjacency_from_edges
- add_self_loops
- symmetric_normalize
- gcn_layer
- mean_neighbor_aggregate
- graph_mean_pool

## 25. Common Mistakes

1. wrong source/target edge convention
2. forgetting self-loops for the intended GCN equation
3. using raw adjacency when normalized adjacency is expected
4. leaking validation/test labels into features
5. assuming every graph is homophilous
6. order-dependent neighbor aggregation
7. graph pooling that ignores batch boundaries
8. comparing transductive vs inductive results directly
9. too many layers causing oversmoothing
10. treating attention weights as complete explanations

## 26. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 27. Checklist

- [ ] adjacency / edge list
- [ ] message passing
- [ ] invariant aggregation
- [ ] GCN normalization
- [ ] GraphSAGE
- [ ] GAT
- [ ] node prediction
- [ ] graph pooling
- [ ] link prediction
- [ ] oversmoothing / oversquashing
- [ ] transductive vs inductive

## 28. What's Next

Chapter 38 moves from supervised prediction to agents that learn from interaction: Reinforcement Learning.
