from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from openhands.agent_server.echomind_graph_runs import (
    ActiveGraphRunExistsError,
    GraphRun,
    GraphRunStore,
    TaskRun,
)
from openhands.agent_server.echomind_projects import (
    ProjectEdge,
    ProjectGraph,
    ProjectNode,
    ProjectNodeInput,
    ProjectNodeInputBinding,
    ProjectNodeOutput,
)
from openhands.agent_server.echomind_publication import (
    PromotionPolicy,
    PublishedGraph,
    RuntimeRequirements,
)
from openhands.agent_server.echomind_storage import canonical_sha256


def test_graph_run_request_and_batch_idempotency_and_single_active_run(tmp_path):
    project_id = str(uuid4())
    release_id = str(uuid4())
    release_digest = canonical_sha256({"release": release_id})
    store = GraphRunStore(tmp_path / "projects")

    first = store.create(
        project_id=project_id,
        release_id=release_id,
        release_digest=release_digest,
        request_id="request-1",
        trigger="tracer_batch",
        source_tracer_resource_id="tracer-1",
        source_batch_id="batch-1",
        inputs={"tracer_batch": "batch-1"},
    )
    assert (
        store.create(
            project_id=project_id,
            release_id=release_id,
            release_digest=release_digest,
            request_id="request-1",
            trigger="tracer_batch",
            source_tracer_resource_id="tracer-1",
            source_batch_id="batch-1",
            inputs={"tracer_batch": "batch-1"},
        )
        == first
    )
    assert (
        store.create(
            project_id=project_id,
            release_id=release_id,
            release_digest=release_digest,
            request_id="request-duplicate-batch",
            trigger="tracer_batch",
            source_tracer_resource_id="tracer-1",
            source_batch_id="batch-1",
            inputs={"tracer_batch": "batch-1"},
        )
        == first
    )

    with pytest.raises(ActiveGraphRunExistsError):
        store.create(
            project_id=project_id,
            release_id=release_id,
            release_digest=release_digest,
            request_id="request-2",
            trigger="tracer_batch",
            source_tracer_resource_id="tracer-1",
            source_batch_id="batch-2",
            inputs={"tracer_batch": "batch-2"},
        )

    now = datetime.now(UTC)
    completed = GraphRun.model_validate(
        first.model_copy(
            update={
                "status": "succeeded",
                "outcome": "baseline_retained",
                "started_at": now,
                "finished_at": now,
            }
        ).model_dump()
    )
    store.save(completed)
    second = store.create(
        project_id=project_id,
        release_id=release_id,
        release_digest=release_digest,
        request_id="request-2",
        trigger="tracer_batch",
        source_tracer_resource_id="tracer-1",
        source_batch_id="batch-2",
        inputs={"tracer_batch": "batch-2"},
    )

    assert second.round_no == 2
    assert store.active(project_id) == second
    assert store.load(project_id=project_id, run_id=first.run_id) == completed


def test_graph_run_and_task_run_status_invariants(tmp_path):
    project_id = str(uuid4())
    release_id = str(uuid4())
    release_digest = canonical_sha256({"release": release_id})
    store = GraphRunStore(tmp_path / "projects")
    graph_run = store.create(
        project_id=project_id,
        release_id=release_id,
        release_digest=release_digest,
        request_id="request-1",
        trigger="manual",
        inputs={},
    )
    now = datetime.now(UTC)

    with pytest.raises(ValidationError):
        GraphRun.model_validate(
            graph_run.model_copy(
                update={
                    "status": "succeeded",
                    "outcome": "baseline_retained",
                    "started_at": now,
                }
            ).model_dump()
        )

    task_run = TaskRun(
        task_run_id=str(uuid4()),
        stage_run_id=str(uuid4()),
        provider="platform",
        provider_run_id="provider-task-1",
        idempotency_key="train:run-1",
        status="submitted",
        request_digest=store.request_digest({"node": "train"}),
        submitted_at=now,
    )
    store.save_task_run(
        project_id=project_id,
        graph_run_id=graph_run.run_id,
        task_run=task_run,
    )
    assert (
        store.load_task_run(
            project_id=project_id,
            graph_run_id=graph_run.run_id,
            task_run_id=task_run.task_run_id,
        )
        == task_run
    )


def test_legacy_project_graph_loads_with_empty_runtime_contract():
    graph = ProjectGraph.model_validate(
        {
            "schema_version": 1,
            "graph_id": "legacy-graph",
            "project_id": "legacy-project",
            "template": "experiment-planning/v1",
            "lifecycle": "development",
            "nodes": [
                {
                    "id": "dataset",
                    "type": "dataset",
                    "title": "Dataset",
                    "launch_mode": "agent",
                    "executor": "contract_probe",
                    "config": {"capability": "data-processing"},
                }
            ],
            "edges": [],
        }
    )

    node = graph.nodes[0]
    assert node.inputs == {}
    assert node.outputs == {}
    assert node.policies == {}
    assert node.implementation_ref is None
    assert node.implementation_digest is None


def test_node_output_binding_and_published_graph_digest():
    dataset = ProjectNode(
        id="dataset",
        type="dataset",
        title="Dataset",
        launch_mode="executor",
        outputs={
            "train": ProjectNodeOutput(
                artifact_type="dataset.supervised",
                schema_ref="echomind://schemas/supervised-dataset/v1",
            )
        },
    )
    train = ProjectNode(
        id="train",
        type="train",
        title="Train",
        inputs={
            "dataset": ProjectNodeInput(
                artifact_type="dataset.supervised",
                schema_ref="echomind://schemas/supervised-dataset/v1",
                binding=ProjectNodeInputBinding(
                    kind="node_output",
                    node_id="dataset",
                    port="train",
                ),
            )
        },
    )
    graph = ProjectGraph(
        graph_id=str(uuid4()),
        project_id=str(uuid4()),
        nodes=[dataset, train],
        edges=[ProjectEdge(source="dataset", target="train")],
    )

    with pytest.raises(ValidationError, match="unknown source output"):
        ProjectGraph(
            graph_id=graph.graph_id,
            project_id=graph.project_id,
            nodes=[
                dataset,
                train.model_copy(
                    update={
                        "inputs": {
                            "dataset": train.inputs["dataset"].model_copy(
                                update={
                                    "binding": ProjectNodeInputBinding(
                                        kind="node_output",
                                        node_id="dataset",
                                        port="missing",
                                    )
                                }
                            )
                        }
                    }
                ),
            ],
            edges=graph.edges,
        )

    published = PublishedGraph.build(
        release_id=str(uuid4()),
        project_id=graph.project_id,
        source_graph_id=graph.graph_id,
        source_graph_digest=canonical_sha256(graph.model_dump(mode="json")),
        runtime_requirements=RuntimeRequirements(),
        promotion_policy=PromotionPolicy(metric="accuracy"),
        nodes=[dataset, train],
        edges=graph.edges,
        published_by="test",
    )
    assert published.content_sha256 == published.compute_content_sha256()
