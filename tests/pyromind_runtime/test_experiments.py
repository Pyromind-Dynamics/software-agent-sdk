import hashlib
import json
from dataclasses import replace
from unittest.mock import Mock
from uuid import uuid4, uuid5

import httpx
import pytest
from pyromind_agent_server.lineage import main as lineage_main
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.experiments import (
    experiment_catalog,
    experiment_metrics,
)
from pyromind_runtime.application.training_evaluations import (
    TrainingEvaluations,
    digest,
)
from pyromind_runtime.application.versions import Versions
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_plans import ExperimentContext
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.preparation import (
    PreparationPolicy,
    ProjectView,
    WorkflowNodeDefinition,
)
from pyromind_runtime.domain.snapshot import ConversationSnapshot, ExternalTaskState
from pyromind_runtime.domain.training_evaluation import (
    CheckpointArtifact,
    StepEvaluation,
)
from pyromind_runtime.domain.versions import (
    ArtifactVersion,
    Iteration,
    ProjectContext,
    RunSource,
    RunSpec,
    SnapshotSource,
    VersionManifest,
    VersionProject,
    VersionRun,
)
from pyromind_runtime.infrastructure.file_artifact_snapshots import (
    FileArtifactSnapshots,
)
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.infrastructure.sqlite_version_store import SQLiteVersionStore
from pyromind_runtime.ports.harness import SessionSpec

from .fake_adapter import FakeAdapter
from .test_evaluation_cases import FakeStorage
from .test_product_api import _app
from .test_workflow_execution import FakeSubmission


@pytest.fixture
def evidence():
    project = VersionProject(
        id="p1",
        conversation_id="c",
        owner_user_id="42",
        context=ProjectContext(
            current_iteration_id="iteration", iterations=[Iteration(id="iteration")]
        ),
    )
    training = VersionRun(
        id="r1",
        project_id="p1",
        request_key="training",
        spec=RunSpec(
            kind="workflow",
            iteration_id="iteration",
            inputs={"training_data": "data"},
        ),
        source=RunSource(kind="workflow", source_id="training", task_id="train-task"),
    )
    pipeline_id = uuid4()
    steps = (
        StepEvaluation(
            role="baseline",
            step=0,
            run_id="baseline-step",
            output_dir="/eval/base",
            eligible=True,
            comparison_group="group-a",
            metrics={"pass_rate": 0.5},
        ),
        StepEvaluation(
            step=10,
            run_id="step-10",
            output_dir="/eval/10",
            eligible=True,
            comparison_group="group-a",
            metrics={
                "pass_rate": 0.75,
                "classification": {"accuracy": 0.75},
                "valid": True,
            },
            checkpoint=CheckpointArtifact(
                step=10, path="/models/checkpoint-10", kind="full", complete=True
            ),
        ),
        StepEvaluation(
            dataset_role="test",
            step=11,
            run_id="test-11",
            output_dir="/eval/11",
            eligible=True,
            comparison_group="test-group",
            metrics={"pass_rate": 0.99},
            checkpoint=CheckpointArtifact(
                step=11, path="/models/checkpoint-11", kind="full", complete=True
            ),
        ),
    )
    pipeline = PipelineRun(
        run_id=pipeline_id,
        conversation_id="c",
        definition={
            "kind": "prediction_evaluation",
            "source_manifest": {"task_id": "train-task", "training_node_id": "train"},
        },
        fingerprint="score",
        output_dir="/eval",
        task_id="eval-task",
        submission_status="submitted",
        result={"selected_run_id": "step-10", "selected_step": 10},
        evaluations=steps,
    )
    scoring = VersionRun(
        id="r2",
        project_id="p1",
        request_key="score",
        spec=RunSpec(kind="evaluate", iteration_id="iteration"),
        source=RunSource(
            kind="pipeline", source_id=str(pipeline_id), task_id="eval-task"
        ),
    )

    def artifact(identity, kind, path, producer=None):
        return ArtifactVersion(
            id=identity,
            project_id="p1",
            producer_run_id=producer,
            kind=kind,
            logical_key=identity,
            version_no=1,
            registration_key=identity,
            state="reference_only",
            manifest=VersionManifest(
                storage_ref=path,
                source=SnapshotSource(storage_ref=path),
                target="/snapshots/" + identity,
                request_digest=identity,
            ),
        )

    tasks = tuple(
        ExternalTaskState(
            task_id=identity,
            kind="workflow_run",
            status="succeeded",
            submitted_at="2026-11-12T00:00:00Z",
            updated_at="2026-11-12T01:00:00Z",
        )
        for identity in ("train-task", "eval-task")
    )
    return {
        "project": project,
        "runs": [training, scoring],
        "artifacts": [
            artifact("data", "dataset", "/data/train.jsonl"),
            artifact("model", "model", "/models/checkpoint-10", "r2"),
        ],
        "requests": {},
        "pipelines": {str(pipeline_id): pipeline},
        "tasks": tasks,
    }


def test_experiment_grouping_and_projection_are_stable(evidence):
    catalog = experiment_catalog(**evidence, project_id="business-project")
    assert experiment_catalog(**evidence, project_id="business-project") == catalog
    assert len(catalog.versions) == 1
    experiment = catalog.versions[0]
    assert experiment.version == "v1" and experiment.experiment_id == "r1"
    assert experiment.project_id == "business-project"
    assert experiment.artifact_project_id == "p1"
    assert "/projects/business-project/experiments/v1" in experiment.directory_ref
    assert experiment.evaluations[0].steps[0].role == "baseline"
    assert experiment.evaluations[0].selection["selected_step"] == 10
    assert experiment.outputs[0].id == "model"
    assert experiment.issues == ("reference_only_input:training_data",)
    assert catalog.unassigned_evaluations == ()


def test_selection_test_role_and_metric_groups_are_preserved(evidence):
    catalog = experiment_catalog(**evidence)
    rows = experiment_metrics(catalog)
    assert {row.metric for row in rows} == {
        "pass_rate",
        "classification.accuracy",
    }
    assert {row.step for row in rows if row.selected} == {10}
    assert not any(row.selected for row in rows if row.dataset_role == "test")
    assert not any(row.selected for row in rows if row.role == "baseline")
    assert {row.comparison_group for row in rows} == {"group-a", "test-group"}
    pipeline_id = next(iter(evidence["pipelines"]))
    pipeline = evidence["pipelines"][pipeline_id]
    evidence["pipelines"][pipeline_id] = pipeline.model_copy(
        update={"result": {"selected_run_id": "test-11", "selected_step": 11}}
    )
    assert not any(
        row.selected for row in experiment_metrics(experiment_catalog(**evidence))
    )


def test_unverified_or_failed_evaluations_do_not_supply_trend_metrics(evidence):
    identity = next(iter(evidence["pipelines"]))
    pipeline = evidence["pipelines"][identity]
    evidence["pipelines"][identity] = pipeline.model_copy(
        update={
            "evaluations": tuple(
                step.model_copy(update={"eligible": False})
                for step in pipeline.evaluations
            )
        }
    )
    assert experiment_metrics(experiment_catalog(**evidence)) == ()
    evidence["pipelines"][identity] = pipeline
    evidence["tasks"] = tuple(
        task.model_copy(update={"status": "failed"}) for task in evidence["tasks"]
    )
    assert experiment_metrics(experiment_catalog(**evidence)) == ()


def test_model_evidence_associates_final_test_without_guessing(evidence):
    selection_id = next(iter(evidence["pipelines"]))
    test = PipelineRun(
        run_id=uuid4(),
        conversation_id="c",
        definition={
            "inference": {
                "model_path": "/workspace/models/checkpoint-10",
                "selection_run_id": selection_id,
            },
            "dataset_role": "test",
        },
        fingerprint="test",
        output_dir="/test",
    )
    evidence["pipelines"][str(test.run_id)] = test
    evidence["runs"].append(
        VersionRun(
            id="r3",
            project_id="p1",
            request_key="test",
            spec=RunSpec(kind="evaluate", iteration_id="iteration"),
            source=RunSource(kind="pipeline", source_id=str(test.run_id)),
        )
    )
    catalog = experiment_catalog(**evidence)
    assert len(catalog.versions[0].evaluations) == 2
    assert catalog.versions[0].evaluations[-1].dataset_role == "test"
    assert catalog.versions[0].evaluations[-1].selection == {}
    evidence["pipelines"][str(test.run_id)] = test.model_copy(
        update={"definition": {"inference": {"model_path": "/models/unrelated"}}}
    )
    catalog = experiment_catalog(**evidence)
    assert len(catalog.unassigned_evaluations) == 1
    assert len(catalog.versions[0].evaluations) == 1


def test_branch_parent_uses_model_origin_not_previous_run(evidence):
    original = evidence["runs"][0]
    project = evidence["project"]
    project.context.iterations.append(Iteration(id="next", parent_id="iteration"))
    evidence["runs"].append(
        original.model_copy(
            update={
                "id": "r3",
                "request_key": "next-training",
                "spec": original.spec.model_copy(
                    update={"iteration_id": "next", "inputs": {"model": "model"}}
                ),
                "source": RunSource(kind="workflow", source_id="next-training"),
            }
        )
    )
    versions = experiment_catalog(**evidence).versions
    assert versions[0].parent_experiment_ids == ()
    assert versions[1].parent_experiment_ids == ("r1",)
    assert versions[1].version == "v2"


def test_retry_keeps_evaluation_linked_to_its_training_attempt(evidence):
    original = evidence["runs"][0]
    evidence["runs"].append(
        original.model_copy(
            update={
                "id": "retry",
                "request_key": "retry",
                "source": RunSource(
                    kind="workflow", source_id="retry", task_id="retry-task"
                ),
            }
        )
    )
    pipeline_id = next(iter(evidence["pipelines"]))
    pipeline = evidence["pipelines"][pipeline_id]
    evidence["pipelines"][pipeline_id] = pipeline.model_copy(
        update={
            "definition": {
                **pipeline.definition,
                "source_manifest": {"task_id": "retry-task"},
            }
        }
    )

    catalog = experiment_catalog(**evidence)
    (experiment,) = catalog.versions
    assert experiment.experiment_id == original.id
    assert [attempt.run_id for attempt in experiment.attempts] == [original.id, "retry"]
    assert experiment.evaluations[0].training_run_id == "retry"
    assert experiment.outputs[0].producer_run_id == "r2"
    assert {row.version for row in experiment_metrics(catalog)} == {"v1"}


def test_foreign_evidence_is_rejected_and_old_pipeline_shape_is_preserved(evidence):
    pipeline_id = next(iter(evidence["pipelines"]))
    pipeline = evidence["pipelines"][pipeline_id]
    empty = pipeline.model_copy(update={"evaluations": ()})
    assert "evaluations" not in empty.model_dump()
    assert PipelineRun.model_validate(empty.model_dump()).evaluations == ()
    evidence["pipelines"][pipeline_id] = pipeline.model_copy(
        update={"conversation_id": "another-conversation"}
    )
    with pytest.raises(ValueError, match="another conversation"):
        experiment_catalog(**evidence)


async def test_experiment_query_is_read_only_and_profile_isolated(
    tmp_path, monkeypatch
):
    adapter = FakeAdapter()
    service = Versions(
        SQLiteVersionStore(tmp_path / "versions.sqlite3"),
        FileArtifactSnapshots(tmp_path / "snapshots"),
    )
    submission = FakeSubmission()
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        adapter,
        versions=service,
        workflow_submission=submission,
    )
    context = RequestContext(user_id="42")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="c",
            user_id="42",
            workspace_root=str(tmp_path / "conversations" / "c"),
        ),
        context,
    )
    await runtime.submit_versioned_workflow(
        "c", "train", submission.workflow, "# training", context
    )
    await runtime.submit_versioned_workflow(
        "c", "retry", submission.workflow, "# training retry", context
    )
    attached = list(adapter.attached)
    sent = list(adapter.sent)
    before = service.store.list_runs("p1")

    async def must_not_attach(*args):
        pytest.fail("experiment queries must not attach a Harness")

    monkeypatch.setattr(runtime, "_ensure_active", must_not_attach)
    try:
        transport = httpx.ASGITransport(app=_app(tmp_path, runtime))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"x-pyromind-debug-user-id": "42"},
        ) as client:
            response = await client.get("/api/v2/pyromind/conversations/c/experiments")
            assert response.status_code == 200
            (row,) = response.json()["versions"]
            detail = await client.get(
                "/api/v2/pyromind/conversations/c/experiments/" + row["experiment_id"]
            )
            assert detail.status_code == 200 and detail.json() == row
            retry_id = row["attempts"][1]["run_id"]
            retry_detail = await client.get(
                "/api/v2/pyromind/conversations/c/experiments/" + retry_id
            )
            assert retry_detail.status_code == 200 and retry_detail.json() == row
            metrics = await client.get(
                "/api/v2/pyromind/conversations/c/experiment-metrics?metric=pass_rate"
            )
            assert metrics.status_code == 200 and metrics.json() == []
            assert (
                await client.get("/api/v2/pyromind/conversations/c/experiments/missing")
            ).status_code == 404
            assert (
                await client.get(
                    "/api/v2/pyromind/conversations/c/experiments",
                    headers={"x-pyromind-debug-user-id": "other"},
                )
            ).status_code == 404
        with pytest.raises(PermissionError):
            await runtime.get_experiments("c", replace(context, profile_id="other"))
        assert service.store.list_runs("p1") == before
        assert adapter.attached == attached and adapter.sent == sent
        assert len(submission.submissions) == 2
    finally:
        await runtime.close()


def test_batch_finalize_persists_steps_and_replay_does_not_change_revision(
    evidence, tmp_path
):
    pipeline = next(iter(evidence["pipelines"].values()))
    manifest = {"evaluations": [{"step": 0}], "datasets": {}}
    spec = {"business_rule": "Exact match"}
    pipeline = pipeline.model_copy(
        update={
            "evaluations": (),
            "result": {},
            "definition": {
                "kind": "prediction_evaluation",
                "source_manifest": manifest,
                "scoring_spec": spec,
                "scoring_script": "# frozen",
            },
        }
    )
    step = (
        next(iter(evidence["pipelines"].values()))
        .evaluations[0]
        .model_copy(
            update={
                "run_id": str(uuid5(pipeline.run_id, "step:0")),
                "output_dir": pipeline.output_dir + "/steps/0",
            }
        )
    )
    store = FileProductStore(tmp_path)
    store.create(
        ConversationSnapshot(conversation_id="c", capabilities=HarnessCapabilities()),
        user_id="42",
    )
    store.save_pipeline_run(pipeline, expected_revision=None)
    storage = FakeStorage(
        {
            "/eval/batch_result.json": json.dumps(
                {
                    "run_id": str(pipeline.run_id),
                    "source_manifest_sha256": digest(manifest),
                    "scoring_spec_sha256": digest(spec),
                    "script_sha256": hashlib.sha256(b"# frozen").hexdigest(),
                    "steps": [step.model_dump(mode="json")],
                }
            ).encode()
        }
    )
    service = TrainingEvaluations(store, storage)
    first = service.finalize(str(pipeline.run_id))
    saved = store.load_pipeline_runs()[str(pipeline.run_id)]
    assert saved.evaluations == (step,)
    assert saved.result == first and first["model"] is None
    assert service.finalize(str(pipeline.run_id)) == first
    assert store.load_pipeline_runs()[str(pipeline.run_id)].revision == saved.revision


async def test_catalog_binds_business_project_and_rejects_cluster_mismatch(tmp_path):
    policy = PreparationPolicy(
        allowed_roles=("worker",),
        workflow_nodes=(
            WorkflowNodeDefinition(id="work", name="Work", roles=("worker",)),
        ),
    )
    project = ProjectView(
        project_id="business-project",
        conversation_id="c",
        plan_sequence=1,
        plan=ExperimentContext(),
        creation_command_id="create",
        cluster="cluster-a",
    )
    preparation_store = Mock()
    preparation_store.project.return_value = project
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        FakeAdapter(),
        preparation_store=preparation_store,
    )
    workspace = tmp_path / "conversations" / "c"
    workspace.mkdir(parents=True)
    store = FileProductStore(workspace)
    store.create(
        ConversationSnapshot(conversation_id="c", capabilities=HarnessCapabilities()),
        user_id="42",
        profile_id="test-profile",
        preparation_policy=policy,
    )
    context = RequestContext(
        user_id="42", profile_id="test-profile", x_cluster="cluster-a"
    )
    try:
        result = await runtime.get_experiments("c", context)
        assert result.project_id == "business-project" and result.versions == ()
        with pytest.raises(PermissionError, match="another cluster"):
            await runtime.get_experiments("c", replace(context, x_cluster="cluster-b"))
        with pytest.raises(PermissionError):
            await runtime.get_experiments("c", replace(context, profile_id="other"))
        assert preparation_store.project.call_args.args[0].profile_id == "test-profile"
    finally:
        await runtime.close()


def test_lineage_cli_reads_the_authorized_api(monkeypatch, capsys):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, json=[], request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setenv("SESSION_API_KEY", "ephemeral-session")
    monkeypatch.setenv("PYROMIND_COOKIE", "ephemeral-cookie")
    assert (
        lineage_main(["c", "--metrics", "--metric", "pass_rate", "--user-id", "42"])
        == 0
    )
    url, kwargs = calls[0]
    assert url.endswith("/api/v2/pyromind/conversations/c/experiment-metrics")
    assert kwargs["params"] == {"metric": "pass_rate"}
    assert kwargs["headers"]["cookie"] == "ephemeral-cookie"
    assert kwargs["headers"]["x-session-api-key"] == "ephemeral-session"
    assert capsys.readouterr().out.strip() == "[]"


def test_lineage_cli_reports_http_failure_without_response_contents(
    monkeypatch, capsys
):
    def denied(url, **kwargs):
        return httpx.Response(
            403, text="sensitive response", request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", denied)
    assert lineage_main(["c"]) == 1
    output = capsys.readouterr()
    assert "403" in output.err and "sensitive" not in output.err
