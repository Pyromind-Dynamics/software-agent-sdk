from datetime import UTC, datetime
from unittest.mock import Mock
from uuid import uuid4

import pytest

from pyromind_agent_server import project_flywheel
from pyromind_agent_server.echomind_acquisition import (
    AcquisitionBatch,
    AcquisitionSimulation,
    AcquisitionSimulationStore,
    AcquisitionSource,
)
from pyromind_agent_server.echomind_graph_runs import (
    GraphRun,
    GraphRunStore,
    RuntimeFailure,
)
from pyromind_agent_server.echomind_runtime_slots import RuntimeSlotStore
from pyromind_agent_server.echomind_sandbox_storage import SandboxEvaluationStorage
from pyromind_agent_server.echomind_storage import read_model
from pyromind_runtime.domain.batch_schedule import BatchReceipt
from pyromind_runtime.domain.echomind_model_reports import ModelEvaluationRecord
from tests.pyromind_runtime.test_platform_batch_schedule import (
    platform_batch as platform_batch,
)


@pytest.fixture
def completed_batch(tmp_path, platform_batch, monkeypatch):
    runtime, _, _, _, calls = platform_batch
    service = project_flywheel.ProjectFlywheelServices(
        projects_root=tmp_path,
        conversations_dir=tmp_path / "conversations",
        credential_store=None,
        cipher=None,
    )
    project = runtime.project.project
    pid = project.project_id
    service.projects.save(runtime.project)
    assert runtime.slot.tracer
    script = f"/app/trigger/echomind-project-{pid}-{'a' * 64}.sh"
    tracer = runtime.slot.tracer.model_copy(
        update={"custom_runtime_script_path": script}
    )
    RuntimeSlotStore(tmp_path).save(runtime.slot.model_copy(update={"tracer": tracer}))
    runtime.storage = Mock(spec=SandboxEvaluationStorage)
    runtime.admin_key = None
    monkeypatch.setattr(service, "_batch_runtime", lambda *_: runtime)
    monkeypatch.setattr(
        project_flywheel, "load_tracer_admin_key", lambda **_: runtime.admin_key
    )
    archive = Mock(spec=project_flywheel.archive_tracer_batch)
    monkeypatch.setattr(project_flywheel, "archive_tracer_batch", archive)

    batch_id = "batch-00000000"
    run_id, simulation_id = str(uuid4()), str(uuid4())
    batch = AcquisitionBatch(
        index=0,
        claimed=True,
        sample_ids=["image-1"],
        input_directory="/workspace/batch",
    )
    simulation = AcquisitionSimulation(
        simulation_id=simulation_id,
        project_id=pid,
        request_id="test",
        status="graph_started",
        source=AcquisitionSource(
            train_uri="/train", val_uri="/val", image_root="/images"
        ),
        requested_samples=2,
        completed_samples=1,
        attempted_sample_ids=batch.sample_ids,
        completed_sample_ids=batch.sample_ids,
        requested_rounds=2,
        completed_rounds=1,
        splits=["train"],
        tracer_resource_id=runtime.tracer_resource_id,
        batch_ids=[batch_id],
        batches={batch_id: batch},
        graph_run_ids=[run_id],
    )
    AcquisitionSimulationStore(tmp_path).save(simulation)
    run = GraphRun(
        run_id=run_id,
        project_id=pid,
        release_id=runtime.release_id,
        release_digest=runtime.published.content_sha256,
        round_no=1,
        request_id="run",
        trigger="tracer_batch",
        source_tracer_resource_id=runtime.tracer_resource_id,
        source_batch_id=batch_id,
        status="succeeded",
        outcome="candidate_recorded",
        finished_at=datetime.now(UTC),
        inputs={
            "tracer_batch": {
                "resource_id": runtime.tracer_resource_id,
                "simulation_id": simulation_id,
                "batch_id": batch_id,
                "batch_index": batch.index,
            }
        },
    )
    GraphRunStore(tmp_path).save(run)
    candidate = ModelEvaluationRecord(
        candidate_id=run_id,
        run_id=run_id,
        project_id=pid,
        round_no=1,
        release_id=run.release_id,
        release_digest=run.release_digest,
        execution_mode="published",
        model_uri="/models/model",
        deployment_model_uri="/models/merged",
        report_uri="/evaluation/report.json",
        report_artifact_id=str(uuid4()),
        report_status="invalid",
        evaluation_status="invalid",
        metrics={"scorable_rows": 5},
        created_at=run.created_at,
    )
    monkeypatch.setattr(
        service.report_service.store, "list_candidates", lambda _: [candidate]
    )
    path = tmp_path / pid / "acquisition_archives" / f"{run_id}.json"
    return service, simulation, run, archive, calls, runtime, path


def test_archive_and_recheck_before_release_after_authorization(completed_batch):
    service, _, run, archive, calls, runtime, path = completed_batch
    service.poll()
    saved = read_model(path, BatchReceipt)
    assert saved.status == "releasing" and saved.released_at is None
    assert (saved.run_id, saved.tracer_resource_id, saved.batch_index) == (
        run.run_id,
        runtime.tracer_resource_id,
        0,
    )
    archive.assert_called_once_with(
        runtime.storage,
        tracer_resource_id=runtime.tracer_resource_id,
        batch_index=0,
        archive_directory=(
            f"/workspace/.pyromind-agent/{runtime.project.project.conversation_id}"
            f"/project-batches/archives/{runtime.tracer_resource_id}/0"
        ),
        sealed_before=run.created_at,
    )
    with pytest.raises(ValueError, match="validated published report"):
        service._candidate(run.project_id, run.run_id)
    service.poll()
    assert archive.call_count == 1 and not calls

    runtime.admin_key = "private-admin"
    archive.side_effect = OSError("invalid archive")
    service.poll()
    assert calls == [("GET", "/admin/batches")]
    assert read_model(path, BatchReceipt).status == "releasing"

    archive.side_effect = None
    service.poll()
    saved = read_model(path, BatchReceipt)
    assert saved.status == "released" and saved.released_at is not None
    assert calls == [("GET", "/admin/batches")] * 2 + [
        ("POST", "/admin/batches/0/release")
    ]
    service.poll()
    assert len(calls) == 3 and archive.call_count == 3


@pytest.mark.parametrize(
    "reason", ["failed_run", "other_tracer", "partial", "wrong_index"]
)
def test_unmatched_or_incomplete_batch_is_preserved(completed_batch, tmp_path, reason):
    service, simulation, run, archive, calls, runtime, path = completed_batch
    runtime.admin_key = "private-admin"
    if reason == "failed_run":
        failure = RuntimeFailure(category="external", message="Training failed")
        run = run.model_copy(
            update={"status": "failed", "outcome": "failed", "failure": failure}
        )
    elif reason == "other_tracer":
        simulation = simulation.model_copy(update={"tracer_resource_id": "other"})
    elif reason == "partial":
        simulation = simulation.model_copy(
            update={"completed_samples": 0, "completed_sample_ids": []}
        )
    else:
        source = {**run.inputs["tracer_batch"], "batch_index": 1}
        run = run.model_copy(update={"inputs": {"tracer_batch": source}})
    GraphRunStore(tmp_path).save(run)
    AcquisitionSimulationStore(tmp_path).save(simulation)
    service.poll()
    assert not path.exists() and not calls
    archive.assert_not_called()
