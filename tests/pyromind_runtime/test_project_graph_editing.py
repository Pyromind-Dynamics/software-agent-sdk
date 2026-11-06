from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest
from pyromind_agent_server.echomind_project_graphs import EchoMindProjectGraphStore
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_plans import PlanRevision, WorkflowPlan
from pyromind_runtime.domain.project_graphs import (
    AddProjectEdge,
    AddProjectNode,
    GraphPatch,
    ProjectEdge,
    ProjectExecutorBinding,
    ProjectGraph,
    ProjectInputPort,
    ProjectNode,
    ProjectOutputPort,
    ProjectPortBinding,
    ProjectRequirement,
    RemoveProjectNode,
    UpdateProjectNode,
    UpdateProjectRequirements,
    canonical_execution_role,
)
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.harness import SessionSpec

from openhands.agent_server.echomind_projects import ProjectStore

from .fake_adapter import FakeAdapter


def _plan() -> WorkflowPlan:
    return WorkflowPlan.model_validate(
        {
            "skill": "experiment-planning",
            "name": "AVI experiment",
            "goal": "Build a four-node experiment.",
            "resources": [
                {
                    "id": "images",
                    "name": "AVI images",
                    "uri": "datasets/avi/train.jsonl",
                }
            ],
            "stages": [
                {"id": "dataset", "name": "Dataset"},
                {"id": "train", "name": "Train"},
            ],
            "edges": [{"source": "dataset", "target": "train"}],
        }
    )


def _project(tmp_path):
    store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    return store, project


def test_graph_patch_edits_nodes_edges_requirements_and_is_idempotent(tmp_path):
    store, project = _project(tmp_path)
    patch = GraphPatch(
        request_id="edit-1",
        operations=(
            UpdateProjectNode(
                node_id="train",
                title="Train detector",
                config={"epochs": 3},
            ),
            AddProjectNode(
                node=ProjectNode(
                    id="inference",
                    type="inference",
                    task_type="inference.run",
                    title="Inference",
                    executor=ProjectExecutorBinding(kind="contract_probe"),
                )
            ),
            AddProjectEdge(edge=ProjectEdge(source="train", target="inference")),
            UpdateProjectRequirements(
                requirements=[
                    ProjectRequirement(
                        key="images",
                        label="AVI images",
                        value="datasets/avi/train.jsonl",
                        status="confirmed",
                    ),
                    ProjectRequirement(
                        key="labels",
                        label="Labels",
                        value="",
                        status="missing",
                    ),
                ]
            ),
        ),
    )

    updated = store.apply_graph_patch(
        project.project.project_id,
        owner_user_id="42",
        patch=patch,
    )
    replayed = store.apply_graph_patch(
        project.project.project_id,
        owner_user_id="42",
        patch=patch,
    )

    assert updated == replayed
    assert next(node for node in updated.nodes if node.id == "train").title == (
        "Train detector"
    )
    assert {edge.source + ":" + edge.target for edge in updated.edges} == {
        "dataset:train",
        "train:inference",
    }
    assert [item.key for item in updated.requirements] == ["images", "labels"]


def test_graph_patch_rejects_invalid_edges_cycles_and_request_reuse(tmp_path):
    store, project = _project(tmp_path)

    with pytest.raises(ValueError, match="self edges"):
        store.apply_graph_patch(
            project.project.project_id,
            owner_user_id="42",
            patch=GraphPatch(
                request_id="self-edge",
                operations=(
                    AddProjectEdge(
                        edge=ProjectEdge(source="dataset", target="dataset")
                    ),
                ),
            ),
        )

    with pytest.raises(ValueError, match="cycles"):
        store.apply_graph_patch(
            project.project.project_id,
            owner_user_id="42",
            patch=GraphPatch(
                request_id="cycle",
                operations=(
                    AddProjectEdge(edge=ProjectEdge(source="train", target="dataset")),
                ),
            ),
        )

    first = GraphPatch(
        request_id="stable-retry",
        operations=(UpdateProjectNode(node_id="dataset", title="Prepared dataset"),),
    )
    store.apply_graph_patch(
        project.project.project_id,
        owner_user_id="42",
        patch=first,
    )
    with pytest.raises(ValueError, match="already used"):
        store.apply_graph_patch(
            project.project.project_id,
            owner_user_id="42",
            patch=GraphPatch(
                request_id="stable-retry",
                operations=(UpdateProjectNode(node_id="train", title="Changed"),),
            ),
        )


def test_graph_patch_removes_connected_edges_and_rejects_frozen_project(tmp_path):
    store, project = _project(tmp_path)

    updated = store.apply_graph_patch(
        project.project.project_id,
        owner_user_id="42",
        patch=GraphPatch(
            request_id="remove-dataset",
            operations=(RemoveProjectNode(node_id="dataset"),),
        ),
    )
    assert [node.id for node in updated.nodes] == ["train"]
    assert updated.edges == []

    project_dir = tmp_path / "projects" / str(UUID(project.project.project_id))
    graph_payload = json.loads((project_dir / "graph.json").read_text(encoding="utf-8"))
    graph_payload["lifecycle"] = "frozen"
    (project_dir / "graph.json").write_text(
        json.dumps(graph_payload),
        encoding="utf-8",
    )
    project_payload = json.loads(
        (project_dir / "project.json").read_text(encoding="utf-8")
    )
    project_payload["lifecycle"] = "frozen"
    (project_dir / "project.json").write_text(
        json.dumps(project_payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frozen"):
        store.apply_graph_patch(
            project.project.project_id,
            owner_user_id="42",
            patch=GraphPatch(
                request_id="frozen",
                operations=(UpdateProjectNode(node_id="train", title="Blocked"),),
            ),
        )


def test_legacy_project_graph_migrates_node_contract_fields():
    project_id = uuid4().hex
    graph = ProjectGraph.model_validate(
        {
            "graph_id": uuid4().hex,
            "project_id": project_id,
            "nodes": [
                {
                    "id": "dataset",
                    "type": "dataset",
                    "title": "Dataset",
                    "executor": "contract_probe",
                    "config": {},
                }
            ],
            "edges": [],
        }
    )

    node = graph.nodes[0]
    assert node.task_type == "dataset.prepare"
    assert node.executor.kind == "contract_probe"
    assert node.execution_role == "dataset"
    assert node.inputs == {}
    assert node.outputs == {}
    assert node.policies == {}


def test_legacy_development_template_loads_nodes_in_agent_launch_mode(tmp_path):
    store, project = _project(tmp_path)
    project_dir = tmp_path / "projects" / str(UUID(project.project.project_id))
    graph_path = project_dir / "graph.json"
    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    for node in graph_payload["nodes"]:
        node["launch_mode"] = "executor"
        node["executor"] = {"kind": "contract_probe"}
    graph_path.write_text(json.dumps(graph_payload), encoding="utf-8")

    loaded = store.load(
        project.project.project_id,
        owner_user_id="42",
    )

    assert {node.launch_mode for node in loaded.graph.nodes} == {"agent"}
    assert {node.id: node.execution_role for node in loaded.graph.nodes} == {
        "dataset": "dataset",
        "train": "train",
    }


def test_evaluator_is_the_canonical_alias_for_eval_role():
    graph = ProjectGraph.model_validate(
        {
            "graph_id": uuid4().hex,
            "project_id": uuid4().hex,
            "nodes": [
                {
                    "id": "evaluator",
                    "type": "evaluator",
                    "title": "Evaluator",
                    "executor": "contract_probe",
                    "config": {},
                }
            ],
            "edges": [],
        }
    )

    assert graph.nodes[0].execution_role == "eval"
    assert canonical_execution_role("evaluator") == "eval"
    assert canonical_execution_role("eval") == "eval"
    assert canonical_execution_role("dataset") == "dataset"


def test_graph_patch_persists_generic_ports_and_executor_binding(tmp_path):
    store, project = _project(tmp_path)
    patch = GraphPatch(
        request_id="generic-contract",
        operations=(
            UpdateProjectNode(
                node_id="dataset",
                type="quality_gate",
                task_type="vision.dataset_quality",
                execution_role="dataset",
                inputs={
                    "source": ProjectInputPort(
                        artifact_type="raw_dataset",
                        schema_ref="raw/v1",
                        binding=ProjectPortBinding(
                            kind="run_input",
                            name="images",
                        ),
                    )
                },
                outputs={
                    "dataset_version": ProjectOutputPort(
                        artifact_type="dataset_version",
                        schema_ref="dataset/v2",
                    )
                },
                executor=ProjectExecutorBinding(kind="agent_development"),
                policies={"timeout_seconds": 600},
            ),
            AddProjectNode(
                node=ProjectNode(
                    id="quality",
                    type="quality_gate",
                    task_type="vision.dataset_quality",
                    title="Dataset quality",
                    executor=ProjectExecutorBinding(kind="contract_probe"),
                )
            ),
            AddProjectEdge(
                edge=ProjectEdge(
                    source="dataset",
                    source_port="dataset_version",
                    target="quality",
                    target_port="dataset",
                )
            ),
        ),
    )

    updated = store.apply_graph_patch(
        project.project.project_id,
        owner_user_id="42",
        patch=patch,
    )

    dataset = next(node for node in updated.nodes if node.id == "dataset")
    assert dataset.type == "quality_gate"
    assert dataset.task_type == "vision.dataset_quality"
    assert dataset.executor.kind == "agent_development"
    assert dataset.execution_role == "dataset"
    assert dataset.inputs["source"].binding == ProjectPortBinding(
        kind="run_input",
        name="images",
    )
    assert dataset.outputs["dataset_version"].schema_ref == "dataset/v2"
    assert updated.edges[-1].source_port == "dataset_version"
    assert updated.edges[-1].target_port == "dataset"


async def test_graph_tools_use_runtime_permissions_and_emit_change_event(tmp_path):
    conversations = tmp_path / "conversations"
    _, project = _project(tmp_path)
    adapter = FakeAdapter("pi")
    runtime = ConversationRuntime(
        conversations,
        {"pi": adapter},
        default_harness_id="pi",
        project_graph_store=EchoMindProjectGraphStore(conversations),
    )
    context = RequestContext(user_id="42", profile_id="echomind")
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id="conversation-1",
                user_id="42",
                profile_id="echomind",
                workspace_root=str(conversations / "conversation-1"),
                tool_allowlist=("get_project_graph", "apply_graph_patch"),
            ),
            context,
        )
        before = await runtime.experiment_plan_call(
            "conversation-1",
            "get_project_graph",
            {"project_id": project.project.project_id},
            context,
        )
        after = await runtime.experiment_plan_call(
            "conversation-1",
            "apply_graph_patch",
            {
                "project_id": project.project.project_id,
                "request_id": "runtime-edit",
                "operations": [
                    {
                        "op": "update_node",
                        "node_id": "train",
                        "title": "Updated by Manager",
                    }
                ],
            },
            context,
        )

        assert before["graph_id"] == after["graph_id"]
        nodes = after["nodes"]
        assert isinstance(nodes, list)
        train = next(
            node
            for node in nodes
            if isinstance(node, dict) and node.get("id") == "train"
        )
        assert train["title"] == "Updated by Manager"
        events = FileProductStore(conversations / "conversation-1").replay()
        assert [event.type for event in events].count("project_graph.changed") == 1
    finally:
        await runtime.close()
