from __future__ import annotations

from uuid import uuid4

import pytest
from harness_adapter.pi_adapter.adapter import PiAdapter, _PiSession
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from pyromind_runtime.application.event_projection import ProductEventProjector
from pyromind_runtime.application.workflow_completion import WorkflowCompletionHook
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_product_store import FileProductStore


@pytest.fixture
def workflow_session(tmp_path):
    root = tmp_path / "conversation"
    (root / "public_data/workflow_canvas").mkdir(parents=True)
    files = PiSessionFiles(root)
    files.initialize({})
    adapter = PiAdapter(tmp_path, terminal_backend="os-sandbox")
    session = _PiSession("conversation", root, files, {}, RequestContext(user_id="42"))
    adapter._sessions[session.session_id] = session
    handle = adapter._handle(session.session_id)
    store = FileProductStore(root)
    store.create(
        ConversationSnapshot(
            conversation_id=session.session_id, capabilities=handle.capabilities
        ),
        user_id="42",
    )
    hook = WorkflowCompletionHook(store, adapter, handle)
    return adapter, session, hook, store


def frame(kind, payload, run_id="run"):
    return {
        "protocolVersion": 2,
        "type": "pi.event",
        "eventId": uuid4().hex,
        "sessionId": "conversation",
        "runId": run_id,
        "kind": kind,
        "payload": payload,
    }


async def drain(session, hook, store):
    while not session.queue.empty():
        event = session.queue.get_nowait()
        if event.type in {"workflow.modified", "run.finished"}:
            await hook.accept(event)
        else:
            product = ProductEventProjector().project(session.session_id, event)
            if product is not None:
                store.append(product)


async def edit(
    adapter,
    session,
    *,
    kind="tool.completed",
    path="public_data/workflow_canvas/workflow.py",
    run_id="run",
):
    await adapter._runner_event(
        session,
        frame(
            kind,
            {
                "tool_name": "edit",
                "tool_call_id": uuid4().hex,
                "arguments": {"path": path},
                "content": [],
            },
            run_id,
        ),
    )


@pytest.mark.parametrize("edits", [0, 1, 3])
@pytest.mark.parametrize(
    "outcome,status",
    [("completed", "idle"), ("cancelled", "paused"), ("failed", "error")],
)
async def test_emits_once_after_dirty_run_for_every_outcome(
    workflow_session, edits, outcome, status
):
    adapter, session, hook, store = workflow_session
    path = session.workspace_root / "public_data/workflow_canvas/workflow.py"
    for index in range(edits):
        path.write_text(f"workflow = Intermediate{index}()")
        await edit(adapter, session)
        await drain(session, hook, store)
        assert not any(event.type == "workflow.updated" for event in store.replay())
    path.write_text("workflow = Final()")
    finish = frame(
        "run.finished",
        {"outcome": {"status": outcome}, "checkpoint_entry_id": "final-leaf"},
    )
    await adapter._runner_event(session, finish)
    await drain(session, hook, store)
    await adapter._runner_event(session, {**finish, "eventId": "duplicate"})
    await drain(session, hook, store)
    output = [event for event in store.replay() if event.type == "workflow.updated"]
    assert len(output) == (1 if edits else 0)
    assert store.load_snapshot().status == status
    if edits:
        assert output[0].payload["dsl"] == "workflow = Final()"
        assert output[0].run_id == "run"
        assert output[0].seq < store.replay()[-1].seq
        assert session.files.load_checkpoint_index()[output[0].event_id] == "final-leaf"
    assert session.files.load_inflight() is None


async def test_failed_or_unrelated_edits_do_not_mark_workflow(workflow_session):
    adapter, session, hook, store = workflow_session
    await edit(adapter, session, kind="tool.failed")
    await edit(adapter, session, path="other.py")
    await drain(session, hook, store)
    assert store.load_workflow_runs() == {}
    assert not any(event.type == "workflow.updated" for event in store.replay())


async def test_same_content_in_later_run_still_emits_and_native_capture_wins(
    workflow_session,
):
    adapter, session, hook, store = workflow_session
    for run_id in ("first", "second"):
        await adapter._runner_event(session, frame("agent.started", {}, run_id))
        await edit(adapter, session, run_id=run_id)
        await adapter._runner_event(
            session,
            frame(
                "run.finished",
                {
                    "outcome": {"status": "completed"},
                    "workflow_dsl": "workflow = Final()",
                    "checkpoint_entry_id": run_id,
                },
                run_id,
            ),
        )
        (session.workspace_root / "public_data/workflow_canvas/workflow.py").write_text(
            "next run modified this"
        )
        await drain(session, hook, store)
    output = [event for event in store.replay() if event.type == "workflow.updated"]
    assert len(output) == 2
    assert all(event.payload["dsl"] == "workflow = Final()" for event in output)
    assert output[0].event_id != output[1].event_id


@pytest.mark.parametrize("missing", [False, True])
async def test_deleted_or_invalid_dsl_does_not_break_final_status(
    workflow_session, missing, monkeypatch
):
    adapter, session, hook, store = workflow_session

    def invalid(_dsl):
        raise ValueError("invalid DSL")

    monkeypatch.setattr(
        "harness_adapter.pi_adapter.adapter.convert_dsl_to_xyflow", invalid
    )
    await edit(adapter, session)
    await adapter._runner_event(
        session,
        frame(
            "run.finished",
            {
                "outcome": {"status": "completed"},
                "workflow_dsl": None if missing else "invalid DSL",
            },
        ),
    )
    await drain(session, hook, store)
    output = [event for event in store.replay() if event.type == "workflow.updated"]
    assert len(output) == (0 if missing else 1)
    if output:
        assert output[0].payload["canvas"] is None
        assert output[0].payload["dsl"] == "invalid DSL"
    assert store.load_snapshot().status == "idle"


async def test_recovery_retries_persisted_completion_without_duplicate_output(
    workflow_session, monkeypatch
):
    adapter, session, hook, store = workflow_session
    await edit(adapter, session)
    await drain(session, hook, store)
    await adapter._runner_event(
        session,
        frame(
            "run.finished",
            {
                "outcome": {"status": "completed"},
                "workflow_dsl": "workflow = Final()",
                "checkpoint_entry_id": "leaf-before-crash",
            },
        ),
    )
    completion = session.queue.get_nowait()
    original = store.save_workflow_run

    def crash_after_publish(state):
        if state.completed:
            raise RuntimeError("simulated process exit after publishing")
        original(state)

    monkeypatch.setattr(store, "save_workflow_run", crash_after_publish)
    with pytest.raises(RuntimeError, match="simulated"):
        await hook.accept(completion)
    recovered_store = FileProductStore(session.workspace_root)
    recovered = WorkflowCompletionHook(
        recovered_store, adapter, adapter._handle(session.session_id)
    )
    await recovered.recover()
    await recovered.accept(completion)
    assert (
        len([event for event in store.replay() if event.type == "workflow.updated"])
        == 1
    )
    assert recovered_store.load_workflow_runs()["run"].completed
    output = next(event for event in store.replay() if event.type == "workflow.updated")
    assert session.files.load_checkpoint_index()[output.event_id] == "leaf-before-crash"


async def test_native_inflight_recovery_keeps_dirty_flag_until_product_finalizes(
    workflow_session,
):
    adapter, session, hook, store = workflow_session
    (session.workspace_root / "public_data/workflow_canvas/workflow.py").write_text(
        "workflow = Recovered()"
    )
    session.files.save_inflight({"run_id": "run", "workflow_modified": True})
    await adapter._recover_inflight(session)
    assert session.files.load_inflight()["completion"]
    await drain(session, hook, store)
    assert store.load_snapshot().current_workflow.dsl == "workflow = Recovered()"
    assert store.load_snapshot().status == "paused"
    assert session.files.load_inflight() is None


async def test_snapshot_storage_failure_preserves_terminal_status_and_can_retry(
    workflow_session, monkeypatch
):
    adapter, session, hook, store = workflow_session
    await edit(adapter, session)
    await drain(session, hook, store)
    await adapter._runner_event(
        session,
        frame(
            "run.finished",
            {
                "outcome": {"status": "failed"},
                "workflow_dsl": "workflow = Final()",
            },
        ),
    )
    completion = session.queue.get_nowait()
    original = adapter.finalize_run

    async def fail(*_args):
        raise OSError("storage temporarily unavailable")

    monkeypatch.setattr(adapter, "finalize_run", fail)
    await hook.accept(completion)
    assert store.load_snapshot().status == "error"
    assert not store.load_workflow_runs()["run"].completed
    monkeypatch.setattr(adapter, "finalize_run", original)
    await hook.recover()
    assert (
        len([event for event in store.replay() if event.type == "workflow.updated"])
        == 1
    )


async def test_completion_does_not_clear_next_runs_native_state(workflow_session):
    adapter, session, hook, store = workflow_session
    await adapter._runner_event(
        session, frame("run.finished", {"outcome": {"status": "completed"}})
    )
    completion = session.queue.get_nowait()
    session.files.save_inflight({"run_id": "next", "operation_id": "next-edit"})
    await hook.accept(completion)
    assert session.files.load_inflight() == {
        "run_id": "next",
        "operation_id": "next-edit",
    }


async def test_finished_run_survives_next_run_overwriting_inflight(workflow_session):
    adapter, session, hook, store = workflow_session
    await edit(adapter, session)
    await adapter._runner_event(
        session,
        frame(
            "run.finished",
            {
                "outcome": {"status": "completed"},
                "workflow_dsl": "previous final",
                "checkpoint_entry_id": "previous-leaf",
            },
        ),
    )
    await adapter._runner_event(session, frame("agent.started", {}, "next"))
    # Lose the in-memory queue before the Product pump has persisted anything.
    while not session.queue.empty():
        session.queue.get_nowait()
    assert store.load_workflow_runs() == {}
    await adapter._recover_inflight(session)
    await drain(session, hook, store)
    output = [event for event in store.replay() if event.type == "workflow.updated"]
    assert len(output) == 1
    assert output[0].run_id == "run"
    assert output[0].payload["dsl"] == "previous final"
    assert session.files.load_checkpoint_index()[output[0].event_id] == "previous-leaf"
    assert store.load_snapshot().status == "paused"
    assert session.files.load_pending_completions() == {}
