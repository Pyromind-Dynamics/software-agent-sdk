from pathlib import Path

import pytest
from pyromind_agent_server.external_task_registry import WorkflowExternalTaskRegistry
from pyromind_runtime.application.pipeline_runs import PipelineRuns
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_product_store import FileProductStore


@pytest.fixture
def pipeline_runs(tmp_path: Path):
    conversation_id = "a" * 32
    root = tmp_path / conversation_id
    root.mkdir()
    store = FileProductStore(root)
    store.create(
        ConversationSnapshot(
            conversation_id=conversation_id,
            capabilities=HarnessCapabilities(),
        ),
        user_id="test",
    )
    registry = WorkflowExternalTaskRegistry(tmp_path)
    service = PipelineRuns(conversation_id, store, registry)
    return service, store, registry


def test_pipeline_run_survives_restart_and_requires_terminal_status(pipeline_runs):
    service, store, registry = pipeline_runs
    run = service.reserve({"model_path": "/models/v1"}, None)
    service.finish_submission(run["run_id"], run["revision"], "task-1", "submitted")
    restarted = PipelineRuns(service.conversation_id, store, registry)
    with pytest.raises(ValueError, match="still active"):
        restarted.reserve(run["definition"], run["run_id"])
    registry.update_status(service.conversation_id, "task-1", "failed")
    resumed = restarted.reserve(run["definition"], run["run_id"])
    assert resumed["output_dir"] == run["output_dir"]
    assert resumed["revision"] > run["revision"]
    with pytest.raises(ValueError, match="in progress"):
        restarted.reserve(run["definition"], run["run_id"])


def test_changed_config_and_cross_conversation_resume_rejected(pipeline_runs, tmp_path):
    service, store, registry = pipeline_runs
    run = service.reserve({"model_path": "/models/v1"}, None)
    service.finish_submission(run["run_id"], run["revision"], None, "failed")
    with pytest.raises(ValueError, match="changed"):
        service.reserve({"model_path": "/models/v2"}, run["run_id"])
    other = "b" * 32
    root = tmp_path / other
    root.mkdir()
    other_store = FileProductStore(root)
    other_store.create(
        ConversationSnapshot(
            conversation_id=other,
            capabilities=HarnessCapabilities(),
        ),
        user_id="test",
    )
    other_service = PipelineRuns(other, other_store, registry)
    assert other_service.resolve(run["run_id"]) is None
    with pytest.raises(ValueError, match="unknown"):
        other_service.reserve(run["definition"], run["run_id"])
    assert other_service.task_for(None, run["run_id"], run["output_dir"]) is None
    assert store.load_pipeline_runs()[run["run_id"]].schema_version == 1


def test_stopped_snapshot_allows_resume_before_registry_callback(pipeline_runs):
    service, store, registry = pipeline_runs
    run = service.reserve({"model_path": "/models/v1"}, None)
    service.finish_submission(run["run_id"], run["revision"], "task-1", "submitted")
    payload = registry.resolve(service.conversation_id, "task-1")
    store.append(
        ProductEvent(
            conversation_id=service.conversation_id,
            type="external_task.submitted",
            payload=payload,
        )
    )
    store.append(
        ProductEvent(
            conversation_id=service.conversation_id,
            type="external_task.completed",
            payload={"task_id": "task-1", "status": "stopped"},
        )
    )
    assert service.task_for("task-1", None, None) == "task-1"
    assert service.task_for(None, run["run_id"], None) == "task-1"
    assert service.reserve(run["definition"], run["run_id"])["run_id"] == run["run_id"]


def test_uncertain_submission_is_not_blindly_repeated(pipeline_runs):
    service, _, _ = pipeline_runs
    run = service.reserve({"model_path": "/models/v1"}, None)
    service.finish_submission(run["run_id"], run["revision"], None, "uncertain")
    with pytest.raises(ValueError, match="uncertain"):
        service.reserve(run["definition"], run["run_id"])


def test_product_store_rejects_stale_run_revision(pipeline_runs):
    service, store, _ = pipeline_runs
    run = service.reserve({"model_path": "/models/v1"}, None)
    state = store.load_pipeline_runs()[run["run_id"]]
    service.finish_submission(run["run_id"], run["revision"], None, "failed")
    with pytest.raises(ValueError, match="concurrently"):
        store.save_pipeline_run(state, expected_revision=state.revision)
