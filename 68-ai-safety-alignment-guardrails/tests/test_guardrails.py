from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).parents[1] / "src" / "guardrails.py"
SPEC = importlib.util.spec_from_file_location("guardrails_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_threshold_decision_three_zones() -> None:
    assert mod.threshold_decision(
        0.1,
        allow_below=0.3,
        block_at_or_above=0.8,
    ) == "allow"
    assert mod.threshold_decision(
        0.5,
        allow_below=0.3,
        block_at_or_above=0.8,
    ) == "review"
    assert mod.threshold_decision(
        0.9,
        allow_below=0.3,
        block_at_or_above=0.8,
    ) == "block"


def test_budget_guard() -> None:
    assert mod.budget_allows(
        used=80,
        requested=20,
        limit=100,
    )
    assert not mod.budget_allows(
        used=80,
        requested=21,
        limit=100,
    )


def test_write_tool_requires_approval() -> None:
    decision = mod.permission_decision(
        tool="update_record",
        allowlisted_tools={"read_record", "update_record"},
        write_tools={"update_record"},
        user_approved=False,
    )
    assert decision == "approval_required"

    approved = mod.permission_decision(
        tool="update_record",
        allowlisted_tools={"read_record", "update_record"},
        write_tools={"update_record"},
        user_approved=True,
    )
    assert approved == "allow"


def test_rag_acl_filters_before_use() -> None:
    records = [
        {"id": "public", "allowed_groups": ["students"]},
        {"id": "staff", "allowed_groups": ["staff"]},
    ]
    visible = mod.retrieval_acl_filter(
        records,
        principal_groups={"students"},
    )
    assert [record["id"] for record in visible] == ["public"]


def test_tool_schema_rejects_unknown_fields() -> None:
    errors = mod.validate_tool_arguments(
        {"item_id": "42", "unexpected": "x"},
        required={"item_id": str},
    )
    assert errors == ["unknown field: unexpected"]


def test_safety_metrics_track_false_positives() -> None:
    metrics = mod.safety_metrics(
        [True, True, False, False],
        [True, False, True, False],
    )
    assert metrics["block_recall"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["accuracy"] == 0.5
