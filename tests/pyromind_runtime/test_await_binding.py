"""A parked await is woken by the platform callback, not by an agent turn."""

from __future__ import annotations

from pyromind_runtime.application.conversation_runtime import (
    ConversationRuntime,
    _build_external_task_notification,
)
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.snapshot import ConversationSnapshot, ExternalTaskState
from pyromind_runtime.infrastructure.file_product_store import FileProductStore

from .fake_adapter import FakeAdapter


def _runtime(tmp_path) -> ConversationRuntime:
    (tmp_path / "conv-1").mkdir()
    FileProductStore(tmp_path / "conv-1").create(
        ConversationSnapshot(
            conversation_id="conv-1", capabilities=HarnessCapabilities()
        ),
        user_id="42",
    )
    return ConversationRuntime(tmp_path, FakeAdapter())


def _task(**overrides) -> ExternalTaskState:
    payload = {
        "task_id": "9184",
        "kind": "graph_await",
        "run_id": "run-1",
        "graph_run_id": "run-1",
        "status": "succeeded",
        "submitted_at": "2026-10-22T06:00:00+00:00",
        "updated_at": "2026-10-22T06:30:00+00:00",
    }
    payload.update(overrides)
    return ExternalTaskState.model_validate(payload)


def test_a_bound_graph_task_does_not_trigger_an_agent_turn() -> None:
    notification = _build_external_task_notification(
        _task(), status="succeeded", error_summary=None, auto_run=True
    )
    assert notification.trigger_turn is False
    assert "resumes that run from its checkpoint" in notification.hidden_text
    assert "re-submit" in notification.hidden_text


def test_an_unbound_task_keeps_its_normal_instruction() -> None:
    notification = _build_external_task_notification(
        _task(kind="data_preparation", graph_run_id=None),
        status="succeeded",
        error_summary=None,
        auto_run=True,
    )
    assert notification.trigger_turn is True


def test_parking_on_an_external_task_binds_the_platform_task_to_the_run(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    runtime._bind_parked_await(
        "conv-1",
        {
            "id": "run-1",
            "status": "waiting",
            "waiting": {
                "kind": "external",
                "node": "chain.wait",
                "task_id": "9184",
                "output_dir": "/.pyromind-agent/conv-1/data_preparation/x",
            },
        },
        "waiting",
    )
    tasks = runtime._store("conv-1").load_snapshot().external_tasks
    bound = next(item for item in tasks if item.task_id == "9184")
    assert bound.graph_run_id == "run-1"
    assert bound.run_id == "run-1"
    assert bound.kind == "graph_await"


def test_a_human_gate_is_not_bound_to_an_external_task(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    runtime._bind_parked_await(
        "conv-1",
        {
            "id": "run-1",
            "status": "waiting",
            "waiting": {"kind": "human", "node": "chain.review"},
        },
        "waiting",
    )
    assert runtime._store("conv-1").load_snapshot().external_tasks == ()
