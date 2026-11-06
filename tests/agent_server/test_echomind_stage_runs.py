import json
from hashlib import sha256
from uuid import UUID, uuid4

import pytest
from pyromind_runtime.domain.experiment_plans import (
    PlanOwner,
    PlanRevision,
    WorkflowPlan,
)
from pyromind_runtime.infrastructure.sql_experiment_execution import (
    SQLExperimentExecutionStore,
    metadata,
)
from sqlalchemy import create_engine

from openhands.agent_server.echomind_projects import ProjectStore
from openhands.agent_server.echomind_stage_runs import (
    ProjectRunStore,
    StageRun,
    project_node_config_digest,
    require_sample_validation,
    run_project_node,
)


def _plan() -> WorkflowPlan:
    return WorkflowPlan.model_validate(
        {
            "skill": "experiment-planning",
            "name": "AVI experiment",
            "goal": "Validate a Dataset -> Inference -> Evaluator -> Train graph.",
            "resources": [
                {
                    "id": "images",
                    "name": "AVI images",
                    "uri": "/datasets/pcb_avi_eval_3/",
                }
            ],
            "stages": [
                {"id": "dataset", "name": "Dataset"},
                {"id": "inference", "name": "Inference"},
                {"id": "evaluator", "name": "Evaluator"},
                {"id": "train", "name": "Train"},
            ],
            "edges": [
                {"source": "dataset", "target": "inference"},
                {"source": "inference", "target": "evaluator"},
                {"source": "evaluator", "target": "train"},
            ],
        }
    )


def _sample_details(task_run_id: str = "task-run-sample") -> dict:
    return {
        "sample_count": 3,
        "image_integrity": {"status": "passed"},
        "bbox_validation": {"status": "not_applicable"},
        "schema_validation": {"status": "passed"},
        "schema_differences": [],
        "pipeline": f"public_data/experiment-runs/{task_run_id}/pipeline.py",
        "sample_output": "public_data/avi/sample.jsonl",
        "submission_spec": {
            "pipeline_path": (f"public_data/experiment-runs/{task_run_id}/pipeline.py"),
            "input_path": "/datasets/pcb_avi_eval_3/",
            "output_schema": "vision",
            "model_profile": "none",
            "convert_format": "none",
        },
        "checks": {
            "formal_validation": {
                "tool": (
                    ".agents/skills/data-processing/scripts/preparation/"
                    "validate_prepared_data.py"
                ),
                "schema": "vision",
                "status": "passed",
                "rows": 3,
                "command": (
                    "python3 .agents/skills/data-processing/scripts/preparation/"
                    "validate_prepared_data.py public_data/avi/sample.jsonl "
                    "--schema vision --image-root "
                    "/target-workspace/datasets/pcb_avi_eval_3 "
                    "--report public_data/avi/validation.json"
                ),
                "report_uri": "public_data/avi/validation.json",
                "output_sha256": "a" * 64,
            }
        },
    }


def _approve_plan(
    run_store: ProjectRunStore,
    project,
    *,
    run_id: str,
    request_id: str,
):
    return run_store.approve_plan(
        project=project,
        run_id=run_id,
        request_id=request_id,
        summary="Dataset input, output and acceptance criteria confirmed.",
        decisions={"schema": "vision"},
    )


def test_stage_runs_and_artifacts_persist_under_canonical_project_directory(
    tmp_path,
):
    project_id = uuid4().hex
    canonical_project_id = str(UUID(project_id))
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=project_id,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )

    run = run_project_node(
        project=project,
        node=project.graph.nodes[1],
        run_store=ProjectRunStore(project_store.root),
    )

    assert run.status == "succeeded"
    assert run.project_id == project_id
    assert run.task_type == "inference.run"
    assert run.executor_kind == "contract_probe"
    assert len(run.artifacts) == 1
    assert run.output_refs == [run.artifacts[0].artifact_id]
    assert not (project_store.root / project_id).exists()
    assert (
        project_store.root / canonical_project_id / "runs" / f"{run.run_id}.json"
    ).is_file()

    reloaded_store = ProjectRunStore(project_store.root)
    assert [item.run_id for item in reloaded_store.list(project_id)] == [run.run_id]
    artifact = reloaded_store.load_artifact(
        project_id=project_id,
        artifact_id=run.artifacts[0].artifact_id,
    )
    assert artifact.artifact.project_id == project_id
    assert artifact.content["node_id"] == "inference"
    assert artifact.content["validation_mode"] == "contract_probe"


def test_all_four_nodes_create_persistent_validation_runs(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)

    runs = [
        run_project_node(
            project=project,
            node=node,
            run_store=run_store,
            request_id=f"validation-{node.id}",
        )
        for node in project.graph.nodes
    ]

    assert {run.node_type for run in runs} == {
        "dataset",
        "inference",
        "evaluator",
        "train",
    }
    assert all(run.mode == "node_validation" for run in runs)
    assert all(run.status == "succeeded" for run in runs)
    assert all(run.input_refs == ["/datasets/pcb_avi_eval_3/"] for run in runs)
    assert [artifact.kind for run in runs for artifact in run.artifacts] == [
        "node_validation_report"
    ] * 4
    assert len(run_store.list(project.project.project_id)) == 4


def test_node_validation_request_id_is_idempotent(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    node = project.graph.nodes[0]

    first = run_project_node(
        project=project,
        node=node,
        run_store=run_store,
        request_id="validation-dataset",
    )
    second = run_project_node(
        project=project,
        node=node,
        run_store=run_store,
        request_id="validation-dataset",
    )

    assert second.run_id == first.run_id
    assert len(run_store.list(project.project.project_id)) == 1
    assert first.node_config_digest == project_node_config_digest(node)


def test_stage_run_and_task_run_link_both_ways_with_the_same_project(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    stage_run, _ = run_store.start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )
    _approve_plan(
        run_store,
        project,
        run_id=stage_run.run_id,
        request_id="approve-dataset",
    )
    stage_run = run_store.get(
        project_id=project.project.project_id,
        run_id=stage_run.run_id,
    )
    engine = create_engine(f"sqlite:///{tmp_path / 'task-runs.sqlite'}")
    metadata.create_all(engine)
    task_store = SQLExperimentExecutionStore(engine)
    owner = PlanOwner(
        user_id="42",
        conversation_id="conversation-1",
        profile_id="echomind",
    )

    try:
        task_run = task_store.request(
            owner,
            project.project.project_id,
            None,
            "dataset.prepare",
            "eval",
            "dataset-task",
            task_spec={
                "stage_id": "dataset",
                "task": {"name": "Prepare dataset", "instruction": "Prepare it"},
            },
            stage_run_id=stage_run.run_id,
        )
        linked, linked_task, created = run_store.bind_task_run(
            stage_run,
            request_id="dataset-task",
            task_run_id=task_run["id"],
            role_id="dataset",
        )

        assert task_run["project_id"] == linked.project_id
        assert task_run["stage_run_id"] == linked.run_id
        assert created is True
        assert linked_task.role_id == "dataset"
        assert linked.task_run_id == task_run["id"]
        replayed, replayed_task, replayed_created = run_store.bind_task_run(
            linked,
            request_id="dataset-task",
            task_run_id=task_run["id"],
            role_id="dataset",
        )
        assert replayed_created is False
        assert replayed_task == linked_task
        assert replayed.task_run_id == task_run["id"]

        relinked, second_task, second_created = run_store.bind_task_run(
            linked,
            request_id="dataset-full-run",
            task_run_id="different-task-run",
            role_id="dataset",
        )
        assert second_created is True
        assert second_task.task_run_id == "different-task-run"
        assert [item.task_run_id for item in relinked.task_runs] == [
            task_run["id"],
            "different-task-run",
        ]
        with pytest.raises(ValueError, match="another request_id"):
            run_store.bind_task_run(
                relinked,
                request_id="dataset-reused-task",
                task_run_id=task_run["id"],
                role_id="dataset",
            )
    finally:
        engine.dispose()


def test_request_id_cannot_be_reused_for_another_node(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run_project_node(
        project=project,
        node=project.graph.nodes[0],
        run_store=run_store,
        request_id="shared-request",
    )

    with pytest.raises(ValueError, match="request_id"):
        run_project_node(
            project=project,
            node=project.graph.nodes[1],
            run_store=run_store,
            request_id="shared-request",
        )


def test_dataset_stage_cannot_succeed_without_platform_task(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run, _ = run_store.start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )
    _approve_plan(
        run_store,
        project,
        run_id=run.run_id,
        request_id="approve-dataset",
    )
    run_store.bind_task_run(
        run,
        request_id="dataset-check",
        task_run_id="task-run-1",
        role_id="dataset",
    )

    with pytest.raises(ValueError, match="waiting_external"):
        run_store.record_result(
            project=project,
            run_id=run.run_id,
            request_id="complete-dataset",
            outcome="succeeded",
            message="Dataset child completed without a platform task.",
            input_refs=["/datasets/source.jsonl"],
            details={"verified": True},
        )

    unchanged = run_store.get(
        project_id=project.project.project_id,
        run_id=run.run_id,
    )
    assert unchanged.status == "running"
    assert unchanged.task_runs[0].status == "running"


def test_failed_stage_can_retry_with_a_new_task_run(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run, _ = run_store.start_agent_development(
        project=project,
        node=project.graph.nodes[1],
        request_id="start-inference",
    )
    _approve_plan(
        run_store,
        project,
        run_id=run.run_id,
        request_id="approve-inference",
    )
    run_store.bind_task_run(
        run,
        request_id="inference-first-attempt",
        task_run_id="task-run-1",
        role_id="inference",
    )
    failed = run_store.record_result(
        project=project,
        run_id=run.run_id,
        request_id="failed-first-attempt",
        outcome="failed",
        message="Inference failed.",
        input_refs=["/datasets/source.jsonl"],
        details={"error": "bad schema"},
        task_run_id="task-run-1",
    )

    retrying, retry_task, created = run_store.bind_task_run(
        failed,
        request_id="inference-retry",
        task_run_id="task-run-2",
        role_id="inference",
    )
    completed = run_store.record_result(
        project=project,
        run_id=run.run_id,
        request_id="complete-retry",
        outcome="succeeded",
        message="Inference retry completed.",
        input_refs=["/datasets/source.jsonl"],
        details={"verified": True},
        task_run_id="task-run-2",
    )

    assert created is True
    assert retrying.status == "running"
    assert retrying.error is None
    assert retry_task.task_run_id == "task-run-2"
    assert [item.status for item in completed.task_runs] == ["failed", "succeeded"]
    assert completed.status == "succeeded"


def test_agent_development_tracks_sample_and_full_run_as_separate_tasks(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run, _ = run_store.start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )
    _approve_plan(
        run_store,
        project,
        run_id=run.run_id,
        request_id="approve-dataset",
    )
    run_store.bind_task_run(
        run,
        request_id="dataset-sample",
        task_run_id="task-run-sample",
        role_id="dataset",
    )
    sampled = run_store.record_result(
        project=project,
        run_id=run.run_id,
        request_id="sample-result",
        outcome="waiting_user",
        message="Sample passed.",
        input_refs=["/datasets/source.jsonl"],
        details=_sample_details("task-run-sample"),
        task_run_id="task-run-sample",
    )
    run_store.begin_continuation(
        project=project,
        run_id=run.run_id,
        request_id="confirm-sample",
        command_id="command-confirm-sample",
    )
    run_store.bind_task_run(
        sampled,
        request_id="dataset-full",
        task_run_id="task-run-full",
        role_id="dataset",
    )
    submitted = run_store.record_result(
        project=project,
        run_id=run.run_id,
        request_id="submit-full",
        outcome="waiting_external",
        message="Full platform task submitted.",
        input_refs=["/datasets/source.jsonl"],
        details={},
        external_task_id="external-task-1",
        task_run_id="task-run-full",
        output_uri="/runs/dataset-full",
    )

    assert [item.task_run_id for item in submitted.task_runs] == [
        "task-run-sample",
        "task-run-full",
    ]
    assert submitted.task_runs[0].status == "waiting_user"
    assert submitted.task_runs[1].status == "waiting_external"
    assert submitted.task_runs[1].external_task_id == "external-task-1"
    assert submitted.external_task_id == "external-task-1"


def test_executor_failure_is_persisted(monkeypatch, tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)

    def fail_executor(_: str):
        raise ValueError("executor unavailable")

    monkeypatch.setattr(
        "openhands.agent_server.echomind_stage_runs.get_project_node_executor",
        fail_executor,
    )
    run = run_project_node(
        project=project,
        node=project.graph.nodes[1],
        run_store=run_store,
        request_id="failing-validation",
    )

    assert run.status == "failed"
    assert "executor unavailable" in (run.error or "")
    assert run_store.list(project.project.project_id)[0].status == "failed"


def test_legacy_stage_run_defaults_validation_metadata():
    run = StageRun.model_validate(
        {
            "run_id": "legacy-run",
            "project_id": uuid4().hex,
            "node_id": "dataset",
            "node_type": "dataset",
            "executor": "contract_probe",
            "status": "succeeded",
            "message": "Legacy run",
            "task_run_id": "legacy-task-run",
            "external_task_id": "legacy-external-task",
            "output_uri": "/runs/legacy",
            "created_at": "2026-11-02T08:00:00Z",
        }
    )

    assert run.mode == "node_validation"
    assert run.request_id is None
    assert run.node_config_digest == ""
    assert run.command_id is None
    assert run.updates == []
    assert run.task_type == "dataset.prepare"
    assert run.executor_kind == "contract_probe"
    assert run.graph_run_id is None
    assert run.output_refs == []
    assert run.config_digest_version == 1
    assert run.task_runs[0].task_run_id == "legacy-task-run"
    assert run.task_runs[0].external_task_id == "legacy-external-task"
    assert run.task_runs[0].status == "succeeded"


def test_agent_development_run_records_reviewable_dataset_sample(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    node = project.graph.nodes[0]

    run, created = run_store.start_agent_development(
        project=project,
        node=node,
        request_id="start-dataset",
    )
    attached = run_store.attach_command(run, command_id="command-1")
    with pytest.raises(ValueError, match="must be running"):
        run_store.record_result(
            project=project,
            run_id=attached.run_id,
            request_id="premature-sample",
            outcome="waiting_user",
            message="Sample must not be recorded during clarification.",
            input_refs=[],
            details={},
        )
    approved = _approve_plan(
        run_store,
        project,
        run_id=attached.run_id,
        request_id="approve-dataset",
    )
    replayed_approval = _approve_plan(
        run_store,
        project,
        run_id=attached.run_id,
        request_id="approve-dataset",
    )
    run_store.bind_task_run(
        attached,
        request_id="dataset-sample",
        task_run_id="task-run-sample",
        role_id="dataset",
    )
    waiting = run_store.record_result(
        project=project,
        run_id=attached.run_id,
        request_id="sample-1",
        outcome="waiting_user",
        message="Three AVI samples passed structural validation.",
        input_refs=["/datasets/pcb_avi_eval_3/"],
        details=_sample_details("task-run-sample"),
        task_run_id="task-run-sample",
    )
    replayed = run_store.record_result(
        project=project,
        run_id=attached.run_id,
        request_id="sample-1",
        outcome="waiting_user",
        message="This replay must not duplicate artifacts.",
        input_refs=[],
        details={},
    )

    assert created is True
    assert run.status == "clarifying"
    assert attached.status == "clarifying"
    assert attached.command_id == "command-1"
    assert approved.status == "running"
    assert approved.updates[0].details == {
        "summary": "Dataset input, output and acceptance criteria confirmed.",
        "decisions": {"schema": "vision"},
    }
    assert replayed_approval == approved
    assert waiting.mode == "agent_development"
    assert waiting.status == "waiting_user"
    assert waiting.finished_at is None
    assert waiting.updates[1].request_id == "sample-1"
    assert [artifact.kind for artifact in waiting.artifacts] == ["stage_sample_report"]
    sample_document = run_store.load_artifact(
        project_id=project.project.project_id,
        artifact_id=waiting.artifacts[0].artifact_id,
    )
    assert (
        sample_document.content["details"]["checks"]["formal_validation"]["status"]
        == "passed"
    )
    assert replayed == waiting
    assert (
        run_store.get(
            project_id=project.project.project_id,
            run_id=waiting.run_id,
        )
        == waiting
    )


def test_record_result_rejects_request_id_reused_for_another_status(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run, _ = run_store.start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )
    request_id = f"project-created:{project.project.project_id}"
    _approve_plan(
        run_store,
        project,
        run_id=run.run_id,
        request_id=request_id,
    )
    run_store.bind_task_run(
        run,
        request_id="dataset-sample",
        task_run_id="task-run-sample",
        role_id="dataset",
    )

    with pytest.raises(ValueError, match="already used for status 'running'"):
        run_store.record_result(
            project=project,
            run_id=run.run_id,
            request_id=request_id,
            outcome="waiting_user",
            message="Sample passed.",
            input_refs=["/datasets/pcb_avi_eval_3/"],
            details=_sample_details("task-run-sample"),
            task_run_id="task-run-sample",
        )

    unchanged = run_store.get(
        project_id=project.project.project_id,
        run_id=run.run_id,
    )
    assert unchanged.status == "running"
    assert unchanged.artifacts == []
    assert unchanged.task_runs[0].status == "running"


def test_agent_development_result_rejects_changed_node_configuration(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    node = project.graph.nodes[0]
    run, _ = run_store.start_agent_development(
        project=project,
        node=node,
        request_id="start-dataset",
    )
    _approve_plan(
        run_store,
        project,
        run_id=run.run_id,
        request_id="approve-dataset",
    )
    changed_node = node.model_copy(update={"config": {"capability": "changed"}})
    changed_project = project.model_copy(
        update={
            "graph": project.graph.model_copy(
                update={
                    "nodes": [
                        changed_node if item.id == node.id else item
                        for item in project.graph.nodes
                    ]
                }
            )
        }
    )

    with pytest.raises(ValueError, match="configuration changed"):
        run_store.record_result(
            project=changed_project,
            run_id=run.run_id,
            request_id="sample-1",
            outcome="waiting_user",
            message="Stale result",
            input_refs=[],
            details={},
        )


def test_legacy_stage_run_uses_its_original_config_digest_version(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run, _ = run_store.start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )
    _approve_plan(
        run_store,
        project,
        run_id=run.run_id,
        request_id="approve-dataset",
    )

    node = project.graph.nodes[0]
    legacy_node = node.model_copy(
        update={"config": {**node.config, "execution_role": "dataset"}}
    )
    legacy_project = project.model_copy(
        update={
            "graph": project.graph.model_copy(
                update={
                    "nodes": [
                        legacy_node if item.id == node.id else item
                        for item in project.graph.nodes
                    ]
                }
            )
        }
    )
    run_path = (
        project_store.root
        / str(UUID(project.project.project_id))
        / "runs"
        / f"{run.run_id}.json"
    )
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["config_digest_version"] = 1
    legacy_payload = json.dumps(
        {
            "type": legacy_node.type,
            "executor": legacy_node.executor.kind,
            "config": legacy_node.config,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload["node_config_digest"] = (
        f"sha256:{sha256(legacy_payload.encode('utf-8')).hexdigest()}"
    )
    run_path.write_text(json.dumps(payload), encoding="utf-8")
    run_store.bind_task_run(
        run,
        request_id="dataset-sample",
        task_run_id="task-run-sample",
        role_id="dataset",
    )

    waiting = run_store.record_result(
        project=legacy_project,
        run_id=run.run_id,
        request_id="sample-1",
        outcome="waiting_user",
        message="Legacy run recorded after upgrade.",
        input_refs=["/datasets/pcb_avi_eval_3/"],
        details=_sample_details("task-run-sample"),
        task_run_id="task-run-sample",
    )

    assert waiting.status == "waiting_user"


def test_dataset_run_continues_into_platform_task_and_version(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run, _ = run_store.start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )
    _approve_plan(
        run_store,
        project,
        run_id=run.run_id,
        request_id="approve-dataset",
    )
    run_store.bind_task_run(
        run,
        request_id="dataset-sample",
        task_run_id="task-run-sample",
        role_id="dataset",
    )[0]
    run_store.record_result(
        project=project,
        run_id=run.run_id,
        request_id="sample-1",
        outcome="waiting_user",
        message="Sample passed.",
        input_refs=["/datasets/source.jsonl"],
        details=_sample_details("task-run-sample"),
        task_run_id="task-run-sample",
    )

    continued, created = run_store.begin_continuation(
        project=project,
        run_id=run.run_id,
        request_id="confirm-1",
        command_id="command-confirm-1",
    )
    replayed, replayed_created = run_store.begin_continuation(
        project=project,
        run_id=run.run_id,
        request_id="confirm-1",
        command_id="command-confirm-2",
    )
    run_store.bind_task_run(
        replayed,
        request_id="dataset-full",
        task_run_id="task-run-full",
        role_id="dataset",
    )
    waiting = run_store.record_result(
        project=project,
        run_id=run.run_id,
        request_id="submit-1",
        outcome="waiting_external",
        message="Platform task submitted.",
        input_refs=["/datasets/source.jsonl"],
        details={},
        external_task_id="task-1",
        task_run_id="task-run-full",
        output_uri="/runs/dataset-1",
    )
    completed = run_store.record_result(
        project=project,
        run_id=run.run_id,
        request_id="complete-1",
        outcome="succeeded",
        message="Dataset outputs verified.",
        input_refs=["/datasets/source.jsonl"],
        details={
            "dataset_version": {
                "source_fingerprint": "sha256:source",
                "source_refs": ["/datasets/source.jsonl"],
                "train_uri": "/runs/dataset-1/train.jsonl",
                "eval_uri": "/runs/dataset-1/eval.jsonl",
                "train_records": 3,
                "eval_records": 3,
                "report_uri": "/runs/dataset-1/report.json",
            }
        },
        external_task_id="task-1",
        task_run_id="task-run-full",
        output_uri="/runs/dataset-1",
    )

    assert created is True
    assert continued.status == "running"
    assert replayed_created is False
    assert replayed.updates[-1].command_id == "command-confirm-1"
    assert waiting.status == "waiting_external"
    assert waiting.external_task_id == "task-1"
    assert waiting.output_uri == "/runs/dataset-1"
    assert completed.status == "succeeded"
    assert completed.finished_at is not None
    assert completed.output_refs == [completed.artifacts[-1].artifact_id]
    assert completed.artifacts[-1].schema_ref == "dataset/v1"
    assert [artifact.kind for artifact in completed.artifacts] == [
        "stage_sample_report",
        "stage_submission_receipt",
        "dataset_version",
    ]


def test_agent_development_run_rejects_second_platform_submission(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run, _ = run_store.start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )
    _approve_plan(
        run_store,
        project,
        run_id=run.run_id,
        request_id="approve-dataset",
    )
    run_store.bind_task_run(
        run,
        request_id="dataset-sample",
        task_run_id="task-run-sample",
        role_id="dataset",
    )
    run_store.record_result(
        project=project,
        run_id=run.run_id,
        request_id="sample-1",
        outcome="waiting_user",
        message="Sample passed.",
        input_refs=["/datasets/source.jsonl"],
        details=_sample_details("task-run-sample"),
        task_run_id="task-run-sample",
    )
    run_store.begin_continuation(
        project=project,
        run_id=run.run_id,
        request_id="confirm-1",
        command_id="command-confirm-1",
    )
    run = run_store.get(
        project_id=project.project.project_id,
        run_id=run.run_id,
    )
    run_store.bind_task_run(
        run,
        request_id="dataset-full",
        task_run_id="task-run-full",
        role_id="dataset",
    )
    run_store.record_result(
        project=project,
        run_id=run.run_id,
        request_id="submit-1",
        outcome="waiting_external",
        message="First platform task submitted.",
        input_refs=["/datasets/source.jsonl"],
        details={},
        external_task_id="task-1",
        task_run_id="task-run-full",
        output_uri="/runs/dataset-1",
    )

    with pytest.raises(ValueError, match="must be running"):
        run_store.record_result(
            project=project,
            run_id=run.run_id,
            request_id="submit-2",
            outcome="waiting_external",
            message="Second platform task submitted.",
            input_refs=["/datasets/source.jsonl"],
            details={},
            external_task_id="task-2",
            output_uri="/runs/dataset-2",
        )


def test_agent_development_success_cannot_change_submitted_task(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run, _ = run_store.start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )
    _approve_plan(
        run_store,
        project,
        run_id=run.run_id,
        request_id="approve-dataset",
    )
    run_store.bind_task_run(
        run,
        request_id="dataset-sample",
        task_run_id="task-run-sample",
        role_id="dataset",
    )
    run_store.record_result(
        project=project,
        run_id=run.run_id,
        request_id="sample-1",
        outcome="waiting_user",
        message="Sample passed.",
        input_refs=["/datasets/source.jsonl"],
        details=_sample_details("task-run-sample"),
        task_run_id="task-run-sample",
    )
    run_store.begin_continuation(
        project=project,
        run_id=run.run_id,
        request_id="confirm-1",
        command_id="command-confirm-1",
    )
    run = run_store.get(
        project_id=project.project.project_id,
        run_id=run.run_id,
    )
    run_store.bind_task_run(
        run,
        request_id="dataset-full",
        task_run_id="task-run-full",
        role_id="dataset",
    )
    run_store.record_result(
        project=project,
        run_id=run.run_id,
        request_id="submit-1",
        outcome="waiting_external",
        message="Platform task submitted.",
        input_refs=["/datasets/source.jsonl"],
        details={},
        external_task_id="task-1",
        task_run_id="task-run-full",
        output_uri="/runs/dataset-1",
    )

    with pytest.raises(ValueError, match="cannot change"):
        run_store.record_result(
            project=project,
            run_id=run.run_id,
            request_id="complete-1",
            outcome="succeeded",
            message="Dataset outputs verified.",
            input_refs=["/datasets/source.jsonl"],
            details={"dataset_version": {}},
            external_task_id="task-2",
            task_run_id="task-run-full",
            output_uri="/runs/dataset-1",
        )


def test_reconcile_interrupted_agent_development_runs(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run, _ = run_store.start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="interrupted",
    )
    _approve_plan(
        run_store,
        project,
        run_id=run.run_id,
        request_id="approve-interrupted",
    )

    recovered = run_store.reconcile_interrupted()
    reloaded = run_store.get(
        project_id=project.project.project_id,
        run_id=run.run_id,
    )

    assert [item.run_id for item in recovered] == [run.run_id]
    assert reloaded.status == "failed"
    assert "RuntimeRestarted" in (reloaded.error or "")
    assert reloaded.finished_at is not None


def test_sample_validation_requires_formal_validator_evidence(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    node = project.graph.nodes[0].model_copy(
        update={
            "config": {
                **project.graph.nodes[0].config,
                "sample_validation": {
                    "required": True,
                    "required_details": [
                        "sample_count",
                        "image_integrity",
                        "bbox_validation",
                        "schema_validation",
                        "schema_differences",
                        "pipeline",
                        "sample_output",
                        "formal_validation",
                    ],
                    "formal_validation": {
                        "tool": (
                            ".agents/skills/data-processing/scripts/preparation/"
                            "validate_prepared_data.py"
                        ),
                        "schema": "vision",
                        "status": "passed",
                        "required_fields": [
                            "tool",
                            "schema",
                            "status",
                            "rows",
                            "command",
                            "report_uri",
                            "output_sha256",
                        ],
                        "command_contains": [
                            ".agents/skills/data-processing/scripts/preparation/"
                            "validate_prepared_data.py",
                            "--schema vision",
                            "--image-root",
                            "--report",
                        ],
                        "rows_detail": "sample_count",
                    },
                },
            }
        }
    )

    with pytest.raises(ValueError, match="required sample details"):
        require_sample_validation(node, {})

    without_checks = _sample_details()
    without_checks.pop("checks")
    with pytest.raises(ValueError, match="details.checks"):
        require_sample_validation(node, without_checks)

    invalid = _sample_details()
    invalid["checks"]["formal_validation"]["schema"] = "text"
    with pytest.raises(ValueError, match="schema"):
        require_sample_validation(node, invalid)

    mismatched = _sample_details()
    mismatched["checks"]["formal_validation"]["rows"] = 2
    with pytest.raises(ValueError, match="sample_count"):
        require_sample_validation(node, mismatched)

    require_sample_validation(node, _sample_details())


def test_dataset_sample_requires_a_frozen_submission_spec(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run, _ = run_store.start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )
    _approve_plan(
        run_store,
        project,
        run_id=run.run_id,
        request_id="approve-dataset",
    )
    run_store.bind_task_run(
        run,
        request_id="dataset-sample",
        task_run_id="task-run-sample",
        role_id="dataset",
    )

    missing = _sample_details("task-run-sample")
    missing.pop("submission_spec")
    with pytest.raises(ValueError, match="submission_spec"):
        run_store.record_result(
            project=project,
            run_id=run.run_id,
            request_id="sample-missing-spec",
            outcome="waiting_user",
            message="Missing frozen submission contract.",
            input_refs=["/datasets/pcb_avi_eval_3/"],
            details=missing,
            task_run_id="task-run-sample",
        )

    wrong_pipeline = _sample_details("task-run-sample")
    wrong_pipeline["submission_spec"]["pipeline_path"] = (
        "public_data/experiment-runs/another-task/pipeline.py"
    )
    with pytest.raises(ValueError, match="pipeline_path"):
        run_store.record_result(
            project=project,
            run_id=run.run_id,
            request_id="sample-wrong-pipeline",
            outcome="waiting_user",
            message="Pipeline belongs to another TaskRun.",
            input_refs=["/datasets/pcb_avi_eval_3/"],
            details=wrong_pipeline,
            task_run_id="task-run-sample",
        )
