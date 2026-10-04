"""World-model prediction, rollout, planning and latent-objective utilities."""

from __future__ import annotations

import itertools
import numpy as np


def one_step_mse(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    pred = np.asarray(predicted, dtype=float)
    tgt = np.asarray(target, dtype=float)

    if pred.shape != tgt.shape or pred.size == 0:
        raise ValueError("matching non-empty arrays required")

    return float(np.mean((pred - tgt) ** 2))


def fit_linear_dynamics(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    *,
    l2: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(states, dtype=float)
    u = np.asarray(actions, dtype=float)
    y = np.asarray(next_states, dtype=float)

    if x.ndim != 2 or u.ndim != 2 or y.ndim != 2:
        raise ValueError("states/actions/next_states must be 2D")
    if not (len(x) == len(u) == len(y)):
        raise ValueError("batch length mismatch")
    if len(x) == 0:
        raise ValueError("at least one transition required")
    if x.shape[1] != y.shape[1]:
        raise ValueError("state dimension mismatch")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")

    design = np.concatenate([x, u], axis=1)
    eye = np.eye(design.shape[1], dtype=float)
    gram = design.T @ design + l2 * eye
    theta = np.linalg.solve(gram, design.T @ y)

    state_dim = x.shape[1]
    a = theta[:state_dim].T
    b = theta[state_dim:].T
    return a, b


def linear_step(
    state: np.ndarray,
    action: np.ndarray,
    *,
    transition_matrix: np.ndarray,
    action_matrix: np.ndarray,
) -> np.ndarray:
    x = np.asarray(state, dtype=float)
    u = np.asarray(action, dtype=float)
    a = np.asarray(transition_matrix, dtype=float)
    b = np.asarray(action_matrix, dtype=float)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matrices must be 2D")
    if a.shape[0] != a.shape[1]:
        raise ValueError("transition_matrix must be square")
    if x.shape != (a.shape[1],):
        raise ValueError("state shape mismatch")
    if b.shape[0] != a.shape[0]:
        raise ValueError("action matrix output mismatch")
    if u.shape != (b.shape[1],):
        raise ValueError("action shape mismatch")

    return a @ x + b @ u


def rollout_linear_dynamics(
    initial_state: np.ndarray,
    actions: np.ndarray,
    *,
    transition_matrix: np.ndarray,
    action_matrix: np.ndarray,
) -> np.ndarray:
    action_sequence = np.asarray(actions, dtype=float)
    if action_sequence.ndim != 2:
        raise ValueError("actions must be [horizon,action_dim]")

    state = np.asarray(initial_state, dtype=float).copy()
    trajectory = [state.copy()]

    for action in action_sequence:
        state = linear_step(
            state,
            action,
            transition_matrix=transition_matrix,
            action_matrix=action_matrix,
        )
        trajectory.append(state.copy())

    return np.stack(trajectory)


def discounted_return(
    rewards: np.ndarray,
    *,
    gamma: float = 1.0,
) -> float:
    values = np.asarray(rewards, dtype=float).reshape(-1)
    if values.size == 0:
        return 0.0
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0,1]")

    discounts = gamma ** np.arange(len(values))
    return float(np.sum(discounts * values))


def goal_reward(
    state: np.ndarray,
    goal: np.ndarray,
    *,
    action: np.ndarray | None = None,
    action_penalty: float = 0.0,
) -> float:
    x = np.asarray(state, dtype=float)
    g = np.asarray(goal, dtype=float)
    if x.shape != g.shape:
        raise ValueError("state/goal shape mismatch")
    if action_penalty < 0:
        raise ValueError("action_penalty must be non-negative")

    cost = float(np.sum((x - g) ** 2))
    if action is not None:
        u = np.asarray(action, dtype=float)
        cost += action_penalty * float(np.sum(u**2))

    return -cost


def enumerate_action_plans(
    action_values: list[float],
    *,
    horizon: int,
    action_dim: int = 1,
) -> list[np.ndarray]:
    if not action_values:
        raise ValueError("action_values must be non-empty")
    if horizon <= 0 or action_dim <= 0:
        raise ValueError("horizon/action_dim must be positive")

    atomic_actions = list(
        itertools.product(action_values, repeat=action_dim)
    )
    plans = []

    for plan in itertools.product(
        atomic_actions,
        repeat=horizon,
    ):
        plans.append(np.asarray(plan, dtype=float))

    return plans


def plan_by_model(
    initial_state: np.ndarray,
    candidate_plans: list[np.ndarray],
    *,
    transition_matrix: np.ndarray,
    action_matrix: np.ndarray,
    goal: np.ndarray,
    gamma: float = 1.0,
    action_penalty: float = 0.0,
) -> tuple[np.ndarray, float]:
    if not candidate_plans:
        raise ValueError("candidate_plans must be non-empty")

    best_plan = None
    best_return = float("-inf")

    for plan in candidate_plans:
        trajectory = rollout_linear_dynamics(
            initial_state,
            plan,
            transition_matrix=transition_matrix,
            action_matrix=action_matrix,
        )

        rewards = np.asarray(
            [
                goal_reward(
                    trajectory[index + 1],
                    goal,
                    action=plan[index],
                    action_penalty=action_penalty,
                )
                for index in range(len(plan))
            ],
            dtype=float,
        )

        score = discounted_return(
            rewards,
            gamma=gamma,
        )

        if score > best_return:
            best_return = score
            best_plan = plan

    assert best_plan is not None
    return best_plan.copy(), float(best_return)


def jepa_prediction_loss(
    predicted_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
) -> float:
    """Simple educational JEPA-style representation prediction loss."""
    return one_step_mse(
        predicted_embeddings,
        target_embeddings,
    )
