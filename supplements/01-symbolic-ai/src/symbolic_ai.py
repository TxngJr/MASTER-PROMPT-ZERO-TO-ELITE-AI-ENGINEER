from __future__ import annotations
from collections import deque
import heapq


def bfs_shortest_path(start, goal, neighbors):
    q = deque([start])
    parent = {start: None}
    while q:
        node = q.popleft()
        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path))
        for nxt in neighbors(node):
            if nxt not in parent:
                parent[nxt] = node
                q.append(nxt)
    return None


def astar(start, goal, neighbors, heuristic):
    frontier = [(heuristic(start, goal), 0.0, start)]
    parent = {start: None}
    best_g = {start: 0.0}
    while frontier:
        _, g, node = heapq.heappop(frontier)
        if g != best_g.get(node):
            continue
        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path)), g
        for nxt, step_cost in neighbors(node):
            if step_cost < 0:
                raise ValueError("A-star requires non-negative edge costs")
            new_g = g + step_cost
            if new_g < best_g.get(nxt, float("inf")):
                best_g[nxt] = new_g
                parent[nxt] = node
                heapq.heappush(frontier, (new_g + heuristic(nxt, goal), new_g, nxt))
    return None


def backtracking_csp(variables, domains, consistent):
    assignment = {}
    def solve(i):
        if i == len(variables):
            return dict(assignment)
        var = variables[i]
        for value in domains[var]:
            if consistent(assignment, var, value):
                assignment[var] = value
                result = solve(i + 1)
                if result is not None:
                    return result
                assignment.pop(var)
        return None
    return solve(0)


def horn_forward_chain(facts, rules):
    known = set(facts)
    changed = True
    while changed:
        changed = False
        for premises, conclusion in rules:
            if set(premises).issubset(known) and conclusion not in known:
                known.add(conclusion)
                changed = True
    return known
