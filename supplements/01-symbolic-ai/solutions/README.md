# Symbolic AI Solutions

1. State-space search explores reachable states through legal actions from an initial state toward a goal.
2. An admissible heuristic never exceeds the true optimal remaining cost.
3. A CSP is variables, domains and constraints.
4. A Horn rule supports implication-style fixed-point inference.
5. BFS is optimal for unit costs; uniform-cost search handles unequal non-negative costs by accumulated path cost.
6. With h=0, A-star priority f=g, giving the same primary ordering as uniform-cost search.
7. MRV chooses the smallest remaining domain; least-constraining value preserves options for neighboring variables.
8. Unit propagation uses a clause with one unresolved literal to force that literal.
9–12. The reference module demonstrates queue/parent reconstruction, best-cost A-star, recursive assignment and fixed-point chaining.
13. Compare against the exact shortest path and document which heuristic assumption was violated.
14. Track visited or best-known cost and skip stale higher-cost frontier entries.
15. Remove inconsistent neighbor-domain values after a tentative assignment and backtrack on an empty domain.
16. Evaluate signed literals correctly, OR within clauses and AND across clauses.
17. Hold graph/start/goal fixed and report expansions plus optimal path cost.
18. Represent predicates as state, actions as precondition/effect transitions and goals as a goal test.
19. Let the learned component prioritize proposals, while the symbolic layer verifies legality and final constraints.
20. Discuss explicit guarantees and brittle modeling versus data-driven generalization and imperfect guarantees.
