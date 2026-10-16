"""Subagent lifecycle is a generic product operation, regardless of harness."""

import json
import tarfile
from pathlib import Path

import pytest
from harness_adapter.pi_adapter.event_translator import translate_runner_event
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.errors import ProductRuntimeError
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.ports.harness import SessionSpec

from .fake_adapter import FakeAdapter


@pytest.mark.parametrize("label", ["pi", "openhands", "future-harness"])
async def test_generic_subagent_operation_holds_session_and_checkpoint(tmp_path, label):
    runtime = ConversationRuntime(tmp_path, FakeAdapter(), idle_eviction_seconds=3600)
    context = RequestContext(user_id="42")
    await runtime.create_conversation(
        SessionSpec(conversation_id=label, user_id="42", workspace_root=str(tmp_path)),
        context,
    )
    store = runtime._store(label)
    store.append(
        ProductEvent(
            conversation_id=label,
            type="operation.started",
            payload={"operation_id": "child", "name": "search", "category": "subagent"},
        )
    )
    active = runtime._active[label]
    assert not runtime._is_releasable(active)
    with pytest.raises(ProductRuntimeError, match="idle"):
        runtime._require_checkpoint_safe(store.load_snapshot())
    store.append(
        ProductEvent(
            conversation_id=label,
            type="operation.completed",
            payload={"operation_id": "child", "output": []},
        )
    )
    assert runtime._is_releasable(active)
    runtime._require_checkpoint_safe(store.load_snapshot())
    await runtime.close()


def test_child_lifecycle_never_changes_parent_status():
    frame = {
        "eventId": "child:start",
        "sessionId": "parent",
        "runId": "child",
        "kind": "subagent.started",
        "payload": {"label": "search"},
    }
    (started,) = translate_runner_event(frame)
    assert started.type == "operation.started"
    assert started.payload["category"] == "subagent"
    (completed,) = translate_runner_event(
        {**frame, "kind": "subagent.finished", "payload": {"status": "complete"}}
    )
    assert completed.type == "operation.completed"
    assert completed.payload["operation_id"] == started.payload["operation_id"]
    (interrupted,) = translate_runner_event(
        {**frame, "kind": "subagent.finished", "payload": {"status": "interrupted"}}
    )
    assert interrupted.type == "operation.failed"


def test_pinned_genome_compatibility_package_preserves_implementation():
    root = Path(__file__).resolve().parents[2] / "harness-adapter/pi-runtime"
    with (
        tarfile.open(root / "vendor/agentgenome-pi-extension-0.6.3.tgz") as old,
        tarfile.open(root / "vendor/agentgenome-pi-extension-0.6.3-pi087.tgz") as new,
    ):
        assert old.getnames() == new.getnames()
        for entry in old.getmembers():
            if not entry.isfile():
                continue
            previous = old.extractfile(entry).read()
            current = new.extractfile(entry.name).read()
            if entry.name == "package/package.json":
                previous, current = json.loads(previous), json.loads(current)
                assert (
                    current["peerDependencies"]["@earendil-works/pi-coding-agent"]
                    == "0.84.2 || 0.87.0"
                )
                current["peerDependencies"] = previous["peerDependencies"]
            assert previous == current
