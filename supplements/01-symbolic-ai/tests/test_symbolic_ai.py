from pathlib import Path
import importlib.util

P = Path(__file__).parents[1] / "src" / "symbolic_ai.py"
S = importlib.util.spec_from_file_location("symbolic_ai", P)
m = importlib.util.module_from_spec(S)
S.loader.exec_module(m)


def test_bfs_and_astar():
    graph = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
    assert len(m.bfs_shortest_path("A", "D", lambda x: graph[x])) == 3
    weighted = {"A": [("B", 1), ("C", 4)], "B": [("D", 2)], "C": [("D", 1)], "D": []}
    path, cost = m.astar("A", "D", lambda x: weighted[x], lambda a, b: 0)
    assert path == ["A", "B", "D"]
    assert cost == 3


def test_csp_and_horn():
    variables = ["x", "y"]
    domains = {"x": [1, 2], "y": [1, 2]}
    sol = m.backtracking_csp(variables, domains, lambda a, v, val: all(other != val for other in a.values()))
    assert sol["x"] != sol["y"]
    out = m.horn_forward_chain({"A"}, [(("A",), "B"), (("A", "B"), "C")])
    assert {"A", "B", "C"}.issubset(out)
