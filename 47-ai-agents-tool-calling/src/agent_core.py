"""Small, explicit and safe-by-construction educational agent executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    required: dict[str, type]
    optional: dict[str, type] = field(default_factory=dict)
    read_only: bool = True


@dataclass
class ToolResult:
    ok: bool
    tool: str
    data: Any = None
    error: str | None = None


def validate_arguments(
    spec: ToolSpec,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise TypeError("arguments must be a dictionary")

    allowed = set(spec.required) | set(spec.optional)
    unknown = set(arguments) - allowed
    if unknown:
        raise ValueError(
            f"unknown arguments for {spec.name}: {sorted(unknown)}"
        )

    missing = set(spec.required) - set(arguments)
    if missing:
        raise ValueError(
            f"missing arguments for {spec.name}: {sorted(missing)}"
        )

    validated: dict[str, Any] = {}

    for key, expected_type in {
        **spec.required,
        **spec.optional,
    }.items():
        if key not in arguments:
            continue

        value = arguments[key]

        if expected_type is int and isinstance(value, bool):
            raise TypeError(f"{key} must be int, not bool")

        if not isinstance(value, expected_type):
            raise TypeError(
                f"{key} must be {expected_type.__name__}"
            )

        validated[key] = value

    return validated


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[
            str,
            tuple[ToolSpec, Callable[..., Any]],
        ] = {}

    def register(
        self,
        spec: ToolSpec,
        function: Callable[..., Any],
    ) -> None:
        if not spec.name:
            raise ValueError("tool name must be non-empty")
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._tools[spec.name] = (spec, function)

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        allow_write: bool = False,
    ) -> ToolResult:
        if name not in self._tools:
            return ToolResult(
                ok=False,
                tool=name,
                error="tool_not_allowed",
            )

        spec, function = self._tools[name]

        if not spec.read_only and not allow_write:
            return ToolResult(
                ok=False,
                tool=name,
                error="write_permission_required",
            )

        try:
            validated = validate_arguments(spec, arguments)
            data = function(**validated)
        except (TypeError, ValueError) as exc:
            return ToolResult(
                ok=False,
                tool=name,
                error=f"validation_error: {exc}",
            )
        except Exception as exc:
            return ToolResult(
                ok=False,
                tool=name,
                error=f"tool_error: {type(exc).__name__}",
            )

        return ToolResult(
            ok=True,
            tool=name,
            data=data,
        )


@dataclass
class AgentState:
    goal: str
    observations: list[ToolResult] = field(default_factory=list)
    steps: int = 0
    done: bool = False
    answer: str | None = None


def bounded_agent_loop(
    *,
    goal: str,
    registry: ToolRegistry,
    policy: Callable[[AgentState], dict[str, Any]],
    max_steps: int = 8,
    allow_write: bool = False,
) -> AgentState:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    state = AgentState(goal=goal)

    while not state.done and state.steps < max_steps:
        action = policy(state)

        if not isinstance(action, dict):
            raise TypeError("policy must return a dictionary")

        action_type = action.get("type")

        if action_type == "finish":
            answer = action.get("answer")
            if not isinstance(answer, str):
                raise TypeError("finish answer must be a string")
            state.answer = answer
            state.done = True

        elif action_type == "tool":
            name = action.get("name")
            arguments = action.get("arguments", {})

            if not isinstance(name, str):
                raise TypeError("tool name must be a string")

            result = registry.execute(
                name,
                arguments,
                allow_write=allow_write,
            )
            state.observations.append(result)

        else:
            raise ValueError("unknown action type")

        state.steps += 1

    return state
