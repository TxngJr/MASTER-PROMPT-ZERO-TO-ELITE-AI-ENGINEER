# Supplement 03 — Probabilistic Graphical Models

PGMs make conditional-independence structure explicit and connect probability theory to structured inference.

## Learning objectives

- read a Bayesian-network DAG;
- factor a joint distribution by parent sets;
- understand conditional independence and d-separation concepts;
- implement exact enumeration on tiny models;
- understand HMM transition/emission structure;
- implement the forward algorithm;
- understand factor graphs and variable elimination;
- understand sum-product / belief-propagation concepts;
- distinguish exact and approximate inference.

## Bayesian networks

For DAG variables (X_1,dots,X_n):

~~~text
P(X_1,...,X_n) = product_i P(X_i | Parents(X_i))
~~~

The graph encodes a factorization; it does not by itself supply the numerical conditional probability tables.

## Conditional independence

A chain, fork and collider behave differently:

~~~text
chain:    A -> B -> C
fork:     A <- B -> C
collider: A -> B <- C
~~~

Conditioning can block or open paths depending on the structure. Use d-separation rules rather than intuition alone.

## Hidden Markov Models

An HMM assumes:

~~~text
P(z_t | z_1,...,z_(t-1)) = P(z_t | z_(t-1))
P(x_t | z_1,...,z_t,x_<t) = P(x_t | z_t)
~~~

Forward recursion:

~~~text
alpha_1(j) = pi_j * emission_j(x_1)

alpha_t(j) =
emission_j(x_t) * sum_i alpha_(t-1)(i) * A_(i,j)
~~~

The observation likelihood is `sum_j alpha_T(j)`.

For long sequences compute in log-space or rescale to avoid underflow.

## Factor graphs and variable elimination

A factor graph represents a product of local functions. Exact variable elimination multiplies relevant factors and sums out variables in an order. The elimination order can dominate computational cost.

## Message passing

On trees, sum-product belief propagation passes local messages and gives exact marginals. On loopy graphs, iterative belief propagation is approximate and may not converge.

## Research/engineering perspective

- graph structure is a modeling assumption;
- exact inference can be exponential in induced width;
- approximate inference requires convergence/error diagnostics;
- never compare log-likelihoods computed with different normalization conventions.
