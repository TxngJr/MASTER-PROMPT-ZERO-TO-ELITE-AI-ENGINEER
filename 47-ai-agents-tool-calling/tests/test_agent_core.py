from pathlib import Path
import importlib.util
import sys


MODULE_PATH = Path(__file__).parents[1] / "src" / "agent_core.py"
SPEC = importlib.util.spec_from_file_location("agent_core_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_registry_validates_arguments() -> None:
    registry = mod.ToolRegistry()
    registry.register(
        mod.ToolSpec(
            name="add",
            required={"a": int, "b": int},
        ),
        lambda a, b: a + b,
    )

    good = registry.execute("add", {"a": 2, "b": 3})
    bad = registry.execute("add", {"a": 2, "b": "3"})

    assert good.ok and good.data == 5
    assert not bad.ok
    assert "validation_error" in bad.error


def test_unknown_tool_fails_closed() -> None:
    registry = mod.ToolRegistry()
    result = registry.execute("arbitrary_function", {})
    assert not result.ok
    assert result.error == "tool_not_allowed"


def test_write_tool_requires_permission() -> None:
    registry = mod.ToolRegistry()
    registry.register(
        mod.ToolSpec(
            name="save_note",
            required={"text": str},
            read_only=False,
        ),
        lambda text: {"saved": text},
    )

    blocked = registry.execute(
        "save_note",
        {"text": "hello"},
    )
    allowed = registry.execute(
        "save_note",
        {"text": "hello"},
        allow_write=True,
    )

    assert not blocked.ok
    assert blocked.error == "write_permission_required"
    assert allowed.ok


def test_agent_loop_stops_on_finish() -> None:
    registry = mod.ToolRegistry()
    registry.register(
        mod.ToolSpec(
            name="lookup",
            required={"key": str},
        ),
        lambda key: {"value": 42},
    )

    def policy(state):
        if not state.observations:
            return {
                "type": "tool",
                "name": "lookup",
                "arguments": {"key": "answer"},
            }
        return {
            "type": "finish",
            "answer": str(state.observations[-1].data["value"]),
        }

    state = mod.bounded_agent_loop(
        goal="find answer",
        registry=registry,
        policy=policy,
        max_steps=4,
    )

    assert state.done
    assert state.answer == "42"
    assert state.steps == 2


def test_agent_loop_respects_step_budget() -> None:
    registry = mod.ToolRegistry()

    def policy(state):
        return {
            "type": "tool",
            "name": "missing",
            "arguments": {},
        }

    state = mod.bounded_agent_loop(
        goal="loop",
        registry=registry,
        policy=policy,
        max_steps=3,
    )

    assert not state.done
    assert state.steps == 3
    assert len(state.observations) == 3
