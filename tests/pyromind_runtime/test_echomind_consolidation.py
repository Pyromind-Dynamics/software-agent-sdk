from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from pyromind_agent_server import echomind_consolidation, echomind_script_consolidation
from pyromind_agent_server.echomind_acquisition_runtime import platform_run_inputs
from pyromind_agent_server.echomind_consolidation import (
    ConsolidationContext,
    ConsolidationDatasetReplay,
    ConsolidationProposal,
    ConsolidationRecipe,
    apply_consolidation,
    compile_consolidation,
    consolidation_context,
)
from pyromind_agent_server.echomind_projects import (
    ProjectEdge,
    ProjectGraph,
    ProjectInfo,
    ProjectNode,
    ProjectRecord,
    ProjectStore,
)
from pyromind_agent_server.echomind_publication import (
    ProjectPublicationService,
    build_runtime_contract_draft,
)
from pyromind_agent_server.echomind_runtime_slots import RuntimeSlot
from pyromind_agent_server.echomind_script_consolidation import (
    ConsolidationCompilation,
    build_script_compilation,
    load_script_compilation,
    save_script_compilation,
)
from pyromind_agent_server.echomind_storage import canonical_sha256
from pyromind_runtime.application.consolidation_evidence import (
    ConsolidationEvidence,
    ConsolidationStageEvidence,
)
from pyromind_runtime.domain.context import RequestContext


@pytest.fixture
def project(tmp_path: Path) -> ProjectInfo:
    project_id = uuid4().hex
    resources = [
        {"id": "image", "uri": "/workspace/images"},
        {"id": "manifest", "uri": "/workspace/old/prelabel_manifest.jsonl"},
        {"id": "base", "uri": "/workspace/models/base"},
        {"id": "holdout", "uri": "/workspace/data/fixed.jsonl"},
    ]
    graph = ProjectGraph(
        graph_id=uuid4().hex,
        project_id=project_id,
        nodes=[
            ProjectNode(
                id="prepare",
                type="dataset",
                title="Dataset",
                config={
                    "resources": resources,
                    "tasks": [
                        {
                            "inputs": [
                                {"name": "raw_images", "resource_id": "image"},
                                {
                                    "name": "prelabel_manifest",
                                    "resource_id": "manifest",
                                },
                            ]
                        }
                    ],
                },
            ),
            ProjectNode(
                id="train",
                type="train",
                title="Train",
                config={
                    "resources": resources,
                    "tasks": [
                        {"inputs": [{"name": "base_model", "resource_id": "base"}]}
                    ],
                },
            ),
            ProjectNode(
                id="eval",
                type="evaluator",
                title="Eval",
                config={
                    "resources": resources,
                    "tasks": [
                        {"inputs": [{"name": "eval_data", "resource_id": "holdout"}]}
                    ],
                },
            ),
        ],
        edges=[
            ProjectEdge(source="prepare", target="train"),
            ProjectEdge(source="train", target="eval"),
        ],
    )
    return ProjectStore(tmp_path).save(
        ProjectInfo(
            project=ProjectRecord(
                project_id=project_id,
                name="PCB",
                description="",
                skill="echomind",
                owner_user_id="42",
                conversation_id=uuid4().hex,
                cluster="cn-east-1",
            ),
            graph=graph,
        )
    )


@pytest.fixture
def context(
    project: ProjectInfo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ConsolidationContext:
    def receipt(task_id: str) -> ConsolidationStageEvidence:
        return ConsolidationStageEvidence(
            task_id=task_id,
            source="workflow_execution_request",
            definition_sha256="sha256:" + "a" * 64,
            revision=2,
        )

    evidence = ConsolidationEvidence(
        conversation_id=project.project.conversation_id,
        dataset=receipt("1086"),
        train=receipt("1102"),
        eval=receipt("1107"),
    )
    monkeypatch.setattr(
        echomind_consolidation,
        "select_consolidation_evidence",
        lambda *_args, **_kwargs: evidence,
    )
    return consolidation_context(project, tmp_path)


def test_eval_accepts_approved_dataset_input(
    project: ProjectInfo, context: ConsolidationContext, tmp_path: Path
):
    evaluation = next(node for node in project.graph.nodes if node.type == "evaluator")
    evaluation.config["tasks"][0]["inputs"][0]["name"] = "dataset"
    assert consolidation_context(project, tmp_path).validation_dataset == (
        context.validation_dataset
    )


def _workflow(**params: str) -> dict:
    return {
        "nodes": [
            {"id": "1", "data": {"nodeType": "CustomCommandCPUNode", "config": params}}
        ],
        "edges": [],
    }


@pytest.fixture
def proposal(context: ConsolidationContext) -> ConsolidationProposal:
    path = "nodes[0].data.config.command"
    return ConsolidationProposal.model_validate(
        {
            "evidence_sha256": context.evidence_sha256,
            "dataset_runtime": {
                "workflow": _workflow(command=""),
                "command_binding_path": path,
                "command_template": "prepare {source_batch} {output_uri}",
                "output_uri": "/workspace/runs/{graph_run_id}/dataset",
            },
            "train_runtime": {
                "workflow": _workflow(command=""),
                "command_binding_path": path,
                "command_template": (
                    "train {base_model} {train_dataset} "
                    "{validation_dataset} {output_uri}"
                ),
                "output_uri": "/workspace/runs/{graph_run_id}/merged",
                "artifact_selection": "merged_model",
            },
            "evaluator_runtime": {
                "workflow": _workflow(command=""),
                "command_binding_path": path,
                "command_template": (
                    "evaluate {candidate_model} {validation_dataset} {output_uri}"
                ),
                "output_uri": "/workspace/runs/{graph_run_id}/eval/report.json",
                "artifact_selection": "evaluation_report",
            },
        }
    )


def test_train_only_plan_compiles_without_training_validation(
    project: ProjectInfo,
    context: ConsolidationContext,
    proposal: ConsolidationProposal,
) -> None:
    context = context.model_copy(update={"train_uses_validation": False})
    train_runtime = dict(proposal.train_runtime)
    train_runtime["command_template"] = (
        "train {base_model} {train_dataset} {output_uri}"
    )
    draft = compile_consolidation(
        project,
        context,
        proposal.model_copy(
            update={
                "train_runtime": train_runtime,
                "batch_validation_source": "none",
            }
        ),
    )
    dataset, train, evaluation = draft.nodes
    assert dataset.config["batch_validation_source"] == "none"
    assert set(train.inputs) == {"base_model", "train_dataset"}
    assert (
        evaluation.inputs["validation_dataset"].binding.value
        == context.validation_dataset
    )


@pytest.mark.parametrize("legacy_replay", [False, True])
def test_reviews_and_applies_original_project(
    legacy_replay,
    train_reader_probe,
    tmp_path: Path,
    project: ProjectInfo,
    context: ConsolidationContext,
    proposal: ConsolidationProposal,
) -> None:
    store = ProjectStore(tmp_path)
    policy = {
        "batch_validation_source": "fixed",
        "batch_label_source": "collection",
        "fixed_validation_path": "/workspace/training-validation.jsonl",
    }
    proposal = proposal.model_copy(update=policy)
    draft = compile_consolidation(project, context, proposal)
    dataset, train, evaluation = draft.nodes
    assert all(dataset.config.get(k) == v for k, v in policy.items())
    assert dataset.inputs.keys() == {"source_batch"}
    assert dataset.config["runtime_input_defaults"] == {
        "source_batch": "/workspace/images"
    }
    assert train.inputs["validation_dataset"].binding.port == "val_messages"
    assert evaluation.inputs["candidate_model"].binding.node_id == train.id
    assert (
        evaluation.inputs["validation_dataset"].binding.value
        == "/workspace/data/fixed.jsonl"
    )
    assert store.load(project.project.project_id, owner_user_id="42") == project

    reviewed = canonical_sha256(proposal.model_dump(mode="json"))
    with pytest.raises(ValueError, match="换批"):
        apply_consolidation(
            store,
            project,
            conversations_root=tmp_path,
            task_ids={},
            proposal=proposal,
            expected_proposal_sha256=reviewed,
        )
    save_script_compilation(
        store,
        project,
        context,
        ConsolidationCompilation(
            evidence_sha256=context.evidence_sha256,
            proposal=proposal,
            stages=(),
            blockers=(),
            dataset_replay=ConsolidationDatasetReplay(
                train_reader_probes=None
                if legacy_replay
                else (train_reader_probe, train_reader_probe),
                dataset_runtime_sha256=canonical_sha256(proposal.dataset_runtime),
                task_ids=("112", "113"),
                source_kinds=("jsonl", "jsonl"),
                source_fingerprints=("a" * 64, "b" * 64),
                train_fingerprints=("d" * 64, "e" * 64),
                train_rows=(1, 2),
                val_rows=(1, 1),
            ),
        ),
        tmp_path,
    )
    if legacy_replay:
        with pytest.raises(ValueError, match="换批"):
            apply_consolidation(
                store,
                project,
                conversations_root=tmp_path,
                task_ids={},
                proposal=proposal,
                expected_proposal_sha256=reviewed,
            )
        return
    updated = apply_consolidation(
        store,
        project,
        conversations_root=tmp_path,
        task_ids={"dataset": "1086", "train": "1102", "eval": "1107"},
        proposal=proposal,
        expected_proposal_sha256=reviewed,
    )
    assert updated.project == project.project
    assert updated.graph.graph_id == project.graph.graph_id
    assert build_runtime_contract_draft(updated).nodes[0].inputs.keys() == {
        "source_batch"
    }
    published = ProjectPublicationService(tmp_path).publish(
        project=updated, published_by="42", publication_mode="production"
    )
    assert platform_run_inputs(
        published, RuntimeSlot(project_id=project.project.project_id)
    ) == ("source_batch", {})


@pytest.mark.parametrize("authored_dataset", [False, True])
def test_script_compilation_reports_missing_recipe_or_accepts_agent_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    project: ProjectInfo,
    context: ConsolidationContext,
    proposal: ConsolidationProposal,
    authored_dataset: bool,
) -> None:
    dataset, *remaining = project.graph.nodes
    raw = "/workspace/text/records.jsonl"
    config = {
        **dataset.config,
        "resources": [{"id": "text", "uri": raw}],
        "tasks": [{"inputs": [{"name": "source_batch", "resource_id": "text"}]}],
    }
    project = project.model_copy(
        update={
            "graph": project.graph.model_copy(
                update={
                    "nodes": [dataset.model_copy(update={"config": config}), *remaining]
                }
            ),
        }
    )
    store = ProjectStore(tmp_path)
    store.save(project)
    revised = consolidation_context(project, tmp_path)
    assert context.evidence == revised.evidence
    assert revised.source_batch == raw

    monkeypatch.setattr(
        echomind_script_consolidation,
        "consolidation_references",
        Mock(return_value={"train": {"runtime_candidate": proposal.train_runtime}}),
    )
    monkeypatch.setattr(
        echomind_script_consolidation,
        "successful_dataset_recipe_runs",
        Mock(return_value=[]),
    )
    evidence_store = Mock()
    evidence_store.load_workflow_execution_requests.return_value = {}
    monkeypatch.setattr(
        echomind_script_consolidation,
        "FileProductStore",
        Mock(return_value=evidence_store),
    )
    monkeypatch.setattr(
        echomind_script_consolidation,
        "_workflow_receipt",
        Mock(return_value=Mock(workflow=_workflow(command="historical prepare"))),
    )
    recipe = ConsolidationRecipe(
        evidence_sha256=revised.evidence_sha256,
        dataset_runtime=proposal.dataset_runtime if authored_dataset else None,
        evaluator_runtime=proposal.evaluator_runtime,
    )
    compiled = build_script_compilation(
        project,
        revised,
        tmp_path,
        RequestContext(user_id="42"),
        versions=None,
        recipe=recipe,
    )
    review = save_script_compilation(store, project, revised, compiled, tmp_path)
    assert load_script_compilation(store, revised) == compiled
    assert store.load(project.project.project_id, owner_user_id="42") == project
    assert {stage.stage: stage.status for stage in review.stages} == {
        "dataset": "blocked",
        "train": "ready",
        "eval": "ready",
    }
    assert review.draft is None
    assert [item.stage for item in review.blockers] == ["dataset"]
    if authored_dataset:
        assert review.blockers[0].code == "dataset_replay_required"


@pytest.mark.parametrize(
    ("stage", "mutation", "message"),
    [
        (
            "dataset_runtime",
            "DF_API_KEY=fixture-value prepare {source_batch} {output_uri}",
            "credential value",
        ),
        (
            "train_runtime",
            "train {base_model} {train_dataset} /workspace/old/val {output_uri}",
            "must bind validation_dataset",
        ),
        (
            "evaluator_runtime",
            "eval {candidate_model} {validation_dataset} /workspace/old/report.json",
            "bind its run-specific output",
        ),
    ],
)
def test_rejects_unsafe_or_static_workflow(
    project: ProjectInfo,
    context: ConsolidationContext,
    proposal: ConsolidationProposal,
    stage: str,
    mutation: str,
    message: str,
) -> None:
    definition = proposal.model_dump(mode="json")
    definition[stage]["command_template"] = mutation
    with pytest.raises(ValueError, match=message):
        compile_consolidation(
            project, context, ConsolidationProposal.model_validate(definition)
        )


def test_rejects_stale_review(
    tmp_path: Path, project: ProjectInfo, proposal: ConsolidationProposal
) -> None:
    store = ProjectStore(tmp_path)
    with pytest.raises(ValueError, match="proposal changed"):
        apply_consolidation(
            store,
            project,
            conversations_root=tmp_path,
            task_ids={},
            proposal=proposal,
            expected_proposal_sha256="sha256:" + "0" * 64,
        )
    changed = project.graph.model_copy(update={"template": "revised"})
    store.save(project.model_copy(update={"graph": changed}))
    with pytest.raises(ValueError, match="different Project content or task evidence"):
        apply_consolidation(
            store,
            project,
            conversations_root=tmp_path,
            task_ids={},
            proposal=proposal,
            expected_proposal_sha256=canonical_sha256(proposal.model_dump(mode="json")),
        )


def test_plan_selects_explicit_source_and_external_training_validation(
    tmp_path: Path, project: ProjectInfo, context: ConsolidationContext
):
    dataset, train, _ = project.graph.nodes
    for name, uri in (
        ("dataset", "/workspace/other"),
        ("source_batch", "/workspace/approved"),
    ):
        dataset.config["resources"].append({"id": name, "uri": uri})
        dataset.config["tasks"][0]["inputs"].append({"name": name, "resource_id": name})
    train.config["tasks"][0]["inputs"].append(
        {
            "name": "training_validation",
            "task_id": "prepare",
            "output_name": "val_messages",
        }
    )
    reviewed = consolidation_context(project, tmp_path)
    assert reviewed.source_batch == "/workspace/approved"
    assert reviewed.training_validation is None
    train.config["resources"].append(
        {"id": "fixed", "uri": "/workspace/fixed-train.jsonl"}
    )
    train.config["tasks"][0]["inputs"].append(
        {"name": "fixed_train_validation", "resource_id": "fixed"}
    )
    reviewed = consolidation_context(project, tmp_path)
    assert reviewed.training_validation == "/workspace/fixed-train.jsonl"
    assert reviewed.validation_dataset != reviewed.training_validation
