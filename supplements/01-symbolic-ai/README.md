# Supplement 01 — Classical Symbolic AI

This supplement closes the classical-AI gap without creating Chapter 81.

## Learning objectives

- formulate state-space search;
- implement BFS, uniform-cost and A-star reasoning;
- explain admissible and consistent heuristics;
- formulate CSPs and backtracking;
- understand propositional SAT/CNF foundations;
- represent facts/rules and perform Horn-clause forward chaining;
- understand STRIPS-style planning concepts;
- compare symbolic search/planning with learned policies.

## State-space search

A search problem contains an initial state, successor function, goal test and path cost.

BFS is complete for finite branching with unit step cost. Uniform-cost search expands lowest path cost. A-star expands by:

~~~text
f(n) = g(n) + h(n)
~~~

An admissible heuristic never overestimates the true remaining cost.

## Constraint Satisfaction Problems

A CSP has variables, domains and constraints. Backtracking assigns variables incrementally. Useful strategies include MRV, degree heuristic, least-constraining value, forward checking and arc-consistency concepts.

## Propositional satisfiability

A propositional formula can be represented in conjunctive normal form. DPLL-style reasoning uses unit propagation, branching and backtracking. Modern solvers add substantially more engineering; this module teaches the foundation.

## Knowledge representation

A Horn-style rule such as `A and B -> C` fires when its premises are known. Forward chaining repeats until a fixed point.

## Planning

STRIPS-like planning models predicates, action preconditions, add/delete effects and goals. Planning can then be treated as structured search.

## Symbolic vs learned AI

Symbolic methods provide explicit constraints and verifiable steps but may require brittle manual models. Learned methods generalize from data but may violate hard constraints. Hybrid systems can use learned proposals with symbolic validation.

## Debugging checklist

- canonicalize states;
- track visited/best-cost state correctly;
- separate path cost and heuristic;
- verify heuristic assumptions on tiny exact cases;
- make CSP constraint checks deterministic;
- ensure inference reaches a fixed point.

## Mini project

Build a route planner with A-star, a small scheduling CSP and a Horn-rule verifier. Compare node expansions and discuss a safe role for a learned heuristic.
