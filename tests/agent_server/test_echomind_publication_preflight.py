from uuid import uuid4

import pytest

from pyromind_agent_server.echomind_projects import (
    ProjectEdge,
    ProjectGraph,
    ProjectInfo,
    ProjectNode,
    ProjectNodeInput,
    ProjectNodeInputBinding,
    ProjectNodeOutput,
    ProjectStore,
)
from pyromind_agent_server.echomind_publication import (
    PublicationValidationError,
    build_publication_preflight,
    compile_working_graph,
)
from pyromind_agent_server.echomind_storage import canonical_sha256
from pyromind_runtime.domain.experiment_plans import PlanRevision, WorkflowPlan
from scripts.echomind_prepare_runtime_validation import (
    _replace_wandb_credentials_with_references,
)


def test_compile_working_graph_is_stable_and_read_only(tmp_path):
    store, project = _publishable_project(tmp_path)
    before = _file_snapshot(store.root)

    preflight = build_publication_preflight(project)
    first = compile_working_graph(project)
    second = compile_working_graph(
        ProjectInfo(
            project=project.project,
            graph=project.graph.model_copy(
                update={
                    "nodes": [
                        node.model_copy(
                            update={
                                "status": (
                                    "succeeded" if node.id == "dataset" else node.status
                                )
                            }
                        )
                        for node in project.graph.nodes
                    ]
                }
            ),
        )
    )

    assert preflight.ready
    assert preflight.issues == []
    assert first.content_sha256 == first.compute_content_sha256()
    assert first == second
    assert _file_snapshot(store.root) == before
    assert project.project.lifecycle == "development"
    assert project.project.active_release_id is None
    assert not (store.root / project.project.project_id / "published_graphs").exists()


def test_production_preflight_rejects_debug_executors(tmp_path):
    store, project = _publishable_project(tmp_path)
    before = _file_snapshot(store.root)

    debug = build_publication_preflight(project)
    production = build_publication_preflight(
        project,
        publication_mode="production",
    )

    assert debug.ready
    assert not production.ready
    assert {issue.code for issue in production.issues} == {
        "debug_executor_in_production"
    }
    with pytest.raises(PublicationValidationError) as exc_info:
        compile_working_graph(project, publication_mode="production")
    assert exc_info.value.issues == production.issues
    assert _file_snapshot(store.root) == before


@pytest.mark.parametrize(
    ("executor", "issue_code"),
    [
        ("shell", "unknown_executor"),
        ("train_runtime", "executor_type_mismatch"),
        ("dataset_runtime", "executor_configuration_invalid"),
    ],
)
def test_production_preflight_requires_executable_dataset_contract(
    tmp_path, executor, issue_code
):
    _, project = _publishable_project(tmp_path)
    nodes = [
        node.model_copy(update={"executor": executor})
        if node.type == "dataset"
        else node
        for node in project.graph.nodes
    ]
    project = ProjectInfo(
        project=project.project,
        graph=project.graph.model_copy(update={"nodes": nodes}),
    )

    issues = build_publication_preflight(project, publication_mode="production").issues

    assert any(
        issue.node_id == "dataset" and issue.code == issue_code for issue in issues
    )


@pytest.mark.parametrize(
    "node_type", ["ModelTrainSFTNode", "ModelTrainDPONode", "ModelTrainGRPONode"]
)
def test_production_preflight_requires_train_workflow_wandb(tmp_path, node_type):
    _, project = _publishable_project(tmp_path)
    workflow = {
        "nodes": [
            {"id": "11", "data": {"nodeType": "PyroMindWandbConfigBuilderNode"}},
            {"id": "12", "data": {"nodeType": node_type, "config": {}}},
        ],
        "edges": [],
    }
    nodes = [
        node.model_copy(
            update={
                "executor": "train_runtime",
                "config": {"train_runtime": {"workflow": workflow}},
            }
        )
        if node.type == "train"
        else node
        for node in project.graph.nodes
    ]
    project = ProjectInfo(
        project=project.project,
        graph=project.graph.model_copy(update={"nodes": nodes}),
    )
    issues = build_publication_preflight(project, publication_mode="production").issues

    assert any(
        issue.node_id == "train"
        and issue.code == "executor_configuration_invalid"
        and "12" in issue.message
        and "PyroMindWandbConfigBuilderNode.wandb_config" in issue.message
        for issue in issues
    )

    workflow["edges"] = [
        {
            "source": "11",
            "sourceHandle": "wandb_config",
            "target": "12",
            "targetHandle": "wandb_config",
        }
    ]
    issues = build_publication_preflight(project, publication_mode="production").issues
    assert not any(issue.node_id == "train" for issue in issues)


def test_validation_recipe_preserves_wandb_topology_with_credential_reference():
    builder = {
        "id": "11",
        "data": {
            "nodeType": "WandbConfigBuilderNode",
            "config": {"wandb_api_key": "a" * 40, "wandb_project": "pcb"},
        },
    }
    train = {"id": "12", "data": {"nodeType": "ModelTrainSFTNode"}}
    edge = {
        "source": "11",
        "sourceHandle": "wandb_config",
        "target": "12",
        "targetHandle": "wandb_config",
    }
    workflow = {"nodes": [builder, train], "edges": [edge]}

    _replace_wandb_credentials_with_references(workflow)

    assert workflow["nodes"] == [builder, train]
    assert workflow["edges"] == [edge]
    assert builder["data"]["config"] == {
        "wandb_api_key": "WANDB_API_KEY",
        "wandb_project": "pcb",
    }


def test_preflight_reports_legacy_contract_gaps_without_mutation(tmp_path):
    store, project = _legacy_project(tmp_path)
    before = _file_snapshot(store.root)

    first = build_publication_preflight(project)
    second = build_publication_preflight(project)
    issue_codes = {issue.code for issue in first.issues}

    assert not first.ready
    assert first == second
    assert {
        "agent_launch_mode",
        "implementation_binding_missing",
        "required_input_missing",
        "required_output_missing",
        "candidate_inference_policy_missing",
        "promotion_policy_invalid",
    } <= issue_codes
    with pytest.raises(PublicationValidationError) as exc_info:
        compile_working_graph(project)
    assert exc_info.value.issues == first.issues
    assert _file_snapshot(store.root) == before


def test_preflight_rejects_port_contract_mismatch_and_topology(tmp_path):
    _, project = _publishable_project(tmp_path)
    train = next(node for node in project.graph.nodes if node.id == "train")
    train_input = train.inputs["train_dataset"].model_copy(
        update={"schema_ref": "echomind://schemas/incompatible/v1"}
    )
    invalid = ProjectInfo(
        project=project.project,
        graph=ProjectGraph(
            graph_id=project.graph.graph_id,
            project_id=project.project.project_id,
            template=project.graph.template,
            lifecycle="development",
            nodes=[
                node.model_copy(
                    update={
                        "inputs": {
                            **node.inputs,
                            "train_dataset": train_input,
                        }
                        if node.id == "train"
                        else node.inputs
                    }
                )
                for node in project.graph.nodes
            ],
            edges=[
                ProjectEdge(source="dataset", target="evaluator"),
                ProjectEdge(source="train", target="evaluator"),
            ],
        ),
    )

    issue_codes = {issue.code for issue in build_publication_preflight(invalid).issues}

    assert "port_contract_mismatch" in issue_codes
    assert "profile_topology_mismatch" in issue_codes


def _publishable_project(tmp_path) -> tuple[ProjectStore, ProjectInfo]:
    plan = WorkflowPlan.model_validate(
        {
            "skill": "experiment-planning",
            "name": "Publishable experiment",
            "goal": "Run Dataset -> Train -> Evaluator.",
            "stages": [
                {"id": "dataset", "name": "Dataset"},
                {"id": "train", "name": "Train"},
                {"id": "evaluator", "name": "Evaluator"},
            ],
            "edges": [
                {"source": "dataset", "target": "train"},
                {"source": "train", "target": "evaluator"},
            ],
        }
    )
    store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=plan,
            confirmed_by="42",
        ),
        conversation_id="conversation-1",
        cluster="us-west-1#pre",
        owner_user_id="42",
    )
    digest = canonical_sha256({"executor": "contract_probe", "version": 1})
    nodes = [
        ProjectNode(
            id="dataset",
            type="dataset",
            title="Dataset",
            launch_mode="executor",
            executor="contract_probe",
            implementation_ref="builtin://contract_probe/v1",
            implementation_digest=digest,
            inputs={
                "source_batch": ProjectNodeInput(
                    artifact_type="tracer.batch",
                    schema_ref="echomind://schemas/tracer-batch/v1",
                    binding=ProjectNodeInputBinding(
                        kind="run_input",
                        name="tracer_batch",
                    ),
                )
            },
            outputs={
                "train_messages": ProjectNodeOutput(
                    artifact_type="dataset.messages.train",
                    schema_ref="echomind://schemas/messages-dataset/v1",
                ),
                "val_messages": ProjectNodeOutput(
                    artifact_type="dataset.messages.validation",
                    schema_ref="echomind://schemas/messages-dataset/v1",
                ),
            },
        ),
        ProjectNode(
            id="train",
            type="train",
            title="Train",
            launch_mode="executor",
            executor="contract_probe",
            implementation_ref="builtin://contract_probe/v1",
            implementation_digest=digest,
            inputs={
                "train_dataset": ProjectNodeInput(
                    artifact_type="dataset.messages.train",
                    schema_ref="echomind://schemas/messages-dataset/v1",
                    binding=ProjectNodeInputBinding(
                        kind="node_output",
                        node_id="dataset",
                        port="train_messages",
                    ),
                ),
                "validation_dataset": ProjectNodeInput(
                    artifact_type="dataset.messages.validation",
                    schema_ref="echomind://schemas/messages-dataset/v1",
                    binding=ProjectNodeInputBinding(
                        kind="node_output",
                        node_id="dataset",
                        port="val_messages",
                    ),
                ),
                "base_model": ProjectNodeInput(
                    artifact_type="model.checkpoint",
                    schema_ref="echomind://schemas/model/v1",
                    binding=ProjectNodeInputBinding(
                        kind="run_input",
                        name="base_model",
                    ),
                ),
            },
            outputs={
                "trained_model": ProjectNodeOutput(
                    artifact_type="model.checkpoint",
                    schema_ref="echomind://schemas/model/v1",
                )
            },
        ),
        ProjectNode(
            id="evaluator",
            type="evaluator",
            title="Evaluator",
            launch_mode="executor",
            executor="contract_probe",
            implementation_ref="builtin://contract_probe/v1",
            implementation_digest=digest,
            inputs={
                "candidate_model": ProjectNodeInput(
                    artifact_type="model.checkpoint",
                    schema_ref="echomind://schemas/model/v1",
                    binding=ProjectNodeInputBinding(
                        kind="node_output",
                        node_id="train",
                        port="trained_model",
                    ),
                ),
                "validation_dataset": ProjectNodeInput(
                    artifact_type="dataset.messages.validation",
                    schema_ref="echomind://schemas/messages-dataset/v1",
                    binding=ProjectNodeInputBinding(
                        kind="node_output",
                        node_id="dataset",
                        port="val_messages",
                    ),
                ),
            },
            outputs={
                "eval_report": ProjectNodeOutput(
                    artifact_type="evaluation.report",
                    schema_ref="echomind://schemas/evaluation-report/v1",
                )
            },
            policies={
                "promotion": {"metric": "accuracy", "direction": "maximize"},
                "candidate_inference": {
                    "create": "evaluator",
                    "health_check": "required",
                    "cleanup": "finalizer",
                },
            },
        ),
    ]
    graph = ProjectGraph(
        graph_id=project.graph.graph_id,
        project_id=project.project.project_id,
        template=project.graph.template,
        lifecycle="development",
        nodes=nodes,
        edges=project.graph.edges,
    )
    saved = store.save(ProjectInfo(project=project.project, graph=graph))
    return store, saved


def _legacy_project(tmp_path):
    plan = WorkflowPlan.model_validate(
        {
            "skill": "experiment-planning",
            "name": "Legacy experiment",
            "goal": "Run Dataset -> Train -> Evaluator.",
            "stages": [
                {"id": "dataset", "name": "Dataset"},
                {"id": "train", "name": "Train"},
                {"id": "evaluator", "name": "Evaluator"},
            ],
            "edges": [
                {"source": "dataset", "target": "train"},
                {"source": "train", "target": "evaluator"},
            ],
        }
    )
    store = ProjectStore.for_conversations_dir(tmp_path / "legacy-conversations")
    project = store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=plan,
            confirmed_by="42",
        ),
        conversation_id="conversation-1",
        cluster="us-west-1#pre",
        owner_user_id="42",
    )
    return store, project


def _file_snapshot(root):
    return {
        path.relative_to(root): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
