from hashlib import sha256
from pathlib import Path
from typing import Literal
from unittest.mock import Mock
from uuid import uuid4

import pytest

from pyromind_agent_server.echomind_consolidation_references import (
    consolidation_evaluation_reference,
    consolidation_references,
)
from pyromind_common.tools.data_preparation.runner import runtime_bundle_fingerprint
from pyromind_runtime.application.consolidation_evidence import (
    ConsolidationEvidenceError,
    select_consolidation_evidence,
    successful_dataset_recipe_runs,
)
from pyromind_runtime.application.versions import Versions
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_execution import ExperimentExecutionView
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.preparation import PreparationPolicy, ProjectView
from pyromind_runtime.domain.snapshot import ConversationSnapshot, ExternalTaskState
from pyromind_runtime.domain.task_correlation import TaskCorrelation
from pyromind_runtime.domain.versions import (
    RunSource,
    RunSpec,
    VersionRun,
    canonical,
    fingerprint,
)
from pyromind_runtime.domain.workflow_execution import (
    DatasetRecipeReceipt,
    WorkflowExecutionRequest,
)
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.product_store import ProductStore

from .test_sdk_merge_evaluation_receipts import inference_graph as inference_graph


_CONVERSATION = "1d4ac155c318441b8ec8828f2d6a09c3"


_NOW = "2026-01-01T00:00:00Z"


def _task(
    task_id: str, kind: Literal["data_preparation", "workflow_run"]
) -> ExternalTaskState:
    return ExternalTaskState(
        task_id=task_id,
        kind=kind,
        status="succeeded",
        correlation=TaskCorrelation(conversation_id=_CONVERSATION, task_kind=kind),
        submitted_at=_NOW,
        updated_at=_NOW,
    )


def _workflow(task: ExternalTaskState) -> WorkflowExecutionRequest:
    return WorkflowExecutionRequest(
        request_id=f"receipt-{task.task_id}",
        conversation_id=_CONVERSATION,
        workflow_path="public_data/workflow.py",
        fingerprint="fingerprint",
        workflow={
            "nodes": [
                {
                    "id": "1",
                    "type": "default",
                    "data": {
                        "nodeType": "WandbConfigBuilderNode"
                        if task.kind == "workflow_run"
                        else "CustomCommandCPUNode",
                        "config": {
                            "command": "DF_API_KEY=example-test-value",
                            "wandb_api_key": "a" * 40,
                            "extra_credential": "credential-test-value",
                            "cpu": 4,
                        },
                    },
                }
            ],
            "edges": [],
        },
        summary={"has_training_node": task.kind == "workflow_run"},
        correlation=task.correlation,
        task_id=task.task_id,
        status="submitted",
        revision=2,
    )


@pytest.fixture
def evidence_store() -> Mock:
    dataset = _task("10", "data_preparation")
    train = _task("20", "workflow_run")
    failed = _task("21", "workflow_run").model_copy(update={"status": "failed"})
    evaluation = _task("30", "data_preparation").model_copy(
        update={"output_dir": "/workspace/outputs/eval"}
    )
    store = Mock(spec_set=ProductStore)
    store.load_snapshot.return_value = ConversationSnapshot(
        conversation_id=_CONVERSATION,
        capabilities=HarnessCapabilities(),
        external_tasks=(dataset, train, failed, evaluation),
    )
    store.load_workflow_execution_requests.return_value = {
        task.task_id: _workflow(task) for task in (dataset, train, failed)
    }
    store.load_pipeline_runs.return_value = {
        "eval": PipelineRun(
            run_id=uuid4(),
            conversation_id=_CONVERSATION,
            definition={
                "input_path": "/datasets/heldout.jsonl",
                "dataset_config": {"reference_field": "label"},
                "evaluation_config": {
                    "generation": {"max_tokens": 1024},
                    "metrics_url": "https://storage.invalid?X-Amz-Signature=test-value",
                    "nested": {"authorization": "arbitrary-test-value"},
                },
                "inference": {},
            },
            output_dir="/outputs/eval",
            fingerprint="fingerprint",
            task_id=evaluation.task_id,
            submission_status="submitted",
        )
    }
    return store


def test_selects_successful_receipts_without_serializing_payloads(evidence_store: Mock):
    evidence = select_consolidation_evidence(evidence_store)

    assert evidence.dataset.task_id == "10"
    assert evidence.train.task_id == "20"
    assert evidence.eval.task_id == "30"
    assert evidence.eval.source == "pipeline_run"
    assert evidence.dataset.revision == 2
    assert evidence.dataset.definition_sha256.startswith("sha256:")
    assert "example-test-value" not in evidence.model_dump_json()


@pytest.fixture
def approved_evidence(evidence_store, request, inference_graph):
    store = Mock(spec_set=FileProductStore)
    store.verify_task_correlation.side_effect = lambda value: (
        value.signature == "a" * 64
    )
    store.preparation_policy.return_value = PreparationPolicy.model_validate(
        {
            "workflow_nodes": [
                {
                    "id": "eval",
                    "name": "Eval",
                    "tasks": [
                        {
                            "id": "scoring",
                            "evaluation_target": "model",
                        }
                    ],
                }
            ]
        }
    )
    project = ProjectView.model_validate(
        {
            "conversation_id": _CONVERSATION,
            "project_id": "project",
            "creation_command_id": "create",
            "plan_sequence": 3,
            "plan": {
                "stages": [
                    {"id": stage, "name": stage, "kind": stage}
                    for stage in ("train", "eval")
                ],
                "tasks": [
                    {"id": "train", "stage_id": "train", "name": "Train"},
                    {
                        "id": "scoring",
                        "stage_id": "eval",
                        "name": "Eval",
                        "kind": "scoring",
                        "inputs": [
                            {"name": "model", "resource_id": "model"},
                            {"name": "dataset", "resource_id": "data"},
                        ],
                    },
                ],
                "resources": [
                    {
                        "id": "model",
                        "name": "Model",
                        "storage_path": "/workspace/models/merged",
                    },
                    {
                        "id": "data",
                        "name": "Val",
                        "storage_path": "/workspace/val.jsonl",
                    },
                ],
            },
        }
    )
    train_graph = {
        "nodes": [
            {"id": "t", "data": {"nodeType": "ModelTrainSFTNode"}},
            {
                "id": "w",
                "data": {
                    "nodeType": "PyroMindWandbConfigBuilderNode",
                    "config": {"wandb_api_key": "WANDB_API_KEY"},
                },
            },
        ],
        "edges": [],
    }
    snapshot = evidence_store.load_snapshot.return_value
    tasks = [snapshot.external_tasks[0]]
    receipts = {"10": _workflow(tasks[0])}
    runs = []
    for item, stage, workflow, identifier, kind in [
        ("train", "train", train_graph, "20", "workflow_run"),
        ("scoring", "eval", inference_graph, "30", request.param),
    ]:
        correlation = TaskCorrelation.model_validate(
            {
                "conversation_id": _CONVERSATION,
                "task_kind": kind,
                "signature": "a" * 64,
                "execution": {
                    "project_id": "project",
                    "task_id": item,
                    "attempt": 1,
                    "plan_sequence": 3,
                },
            }
        )
        task = _task(identifier, "workflow_run").model_copy(
            update={"correlation": correlation, "kind": kind}
        )
        tasks.append(task)
        receipt = _workflow(task).model_copy(
            update={
                "request_id": "sdk:" + identifier,
                "workflow_path": "version://v42",
                "summary": {"name": stage},
                "workflow": workflow,
            }
        )
        receipts[receipt.request_id] = receipt
        runs.append(
            ExperimentExecutionView(
                run_id=f"project:{item}:1",
                project_id="project",
                plan_sequence=3,
                plan_attempt=1,
                stage_id=stage,
                task_id=item,
                task_name=item,
                role_id="main",
                state="succeeded",
                report_state="ready",
                intent="execute",
                platform_task_id=identifier,
                platform_task_ids=(identifier,),
            )
        )
    store.load_snapshot.return_value = snapshot.model_copy(
        update={"external_tasks": tuple(tasks), "experiment_executions": tuple(runs)}
    )
    store.load_workflow_execution_requests.return_value = receipts
    store.load_pipeline_runs.return_value = {}
    return store, project


@pytest.mark.parametrize(
    "approved_evidence", ["workflow_run", "evaluation"], indirect=True
)
def test_sdk_consolidation_uses_frozen_graph_and_scoped_primary_receipt(
    approved_evidence,
):
    store, project = approved_evidence
    evidence = select_consolidation_evidence(store, approved_project=project)
    assert (evidence.train.task_id, evidence.eval.task_id, evidence.eval.source) == (
        "20",
        "30",
        "workflow_execution_request",
    )
    refs = consolidation_references(store, evidence, approved_project=project)
    train = refs["train"]
    evaluation = refs["eval"]
    assert isinstance(train, dict) and train["wandb_credential_task_id"] == "20"
    assert isinstance(evaluation, dict) and "workflow" in evaluation
    assert evaluation["workflow_path"] == "version://v42"
    assert "runtime_candidate" not in evaluation
    assert "runtime_blocker" in consolidation_evaluation_reference(
        Mock(spec_set=Versions), evidence, RequestContext(user_id="42")
    )

    store.load_workflow_execution_requests.return_value["sdk:30"].workflow["nodes"][0][
        "data"
    ]["config"]["model_path"] = "/workspace/wrong"
    with pytest.raises(ConsolidationEvidenceError, match="No verified successful eval"):
        select_consolidation_evidence(
            store, approved_project=project, task_ids={"eval": "30"}
        )
    store.verify_task_correlation.return_value = False
    store.verify_task_correlation.side_effect = None
    with pytest.raises(
        ConsolidationEvidenceError, match="No verified successful train"
    ):
        select_consolidation_evidence(
            store, approved_project=project, task_ids={"train": "20"}
        )


def test_replay_is_verification_evidence_for_original_dataset(evidence_store: Mock):
    task = _task("11", "data_preparation").model_copy(update={"output_dir": "/check"})
    snapshot = evidence_store.load_snapshot.return_value
    evidence_store.load_snapshot.return_value = snapshot.model_copy(
        update={"external_tasks": (*snapshot.external_tasks, task)}
    )
    replay = DatasetRecipeReceipt(
        source_path="/workspace/replay",
        output_directory="/workspace/check",
        command_template="{source_batch} {output_uri}",
        assets=(),
        replay_of_task_id="10",
        uses_labeling_gateway=False,
    )
    evidence_store.load_workflow_execution_requests.return_value["11"] = _workflow(
        task
    ).model_copy(update={"dataset_recipe": replay})
    evidence = select_consolidation_evidence(evidence_store)
    assert evidence.dataset.task_id == "10"
    assert [
        r.task_id for r in successful_dataset_recipe_runs(evidence_store, evidence)
    ] == ["11"]


def test_ambiguous_stage_requires_an_explicit_successful_task(evidence_store: Mock):
    another_train = _task("22", "workflow_run")
    snapshot = evidence_store.load_snapshot.return_value
    evidence_store.load_snapshot.return_value = snapshot.model_copy(
        update={"external_tasks": (*snapshot.external_tasks, another_train)}
    )
    evidence_store.load_workflow_execution_requests.return_value["22"] = _workflow(
        another_train
    )

    with pytest.raises(ConsolidationEvidenceError, match="Select one of 20, 22"):
        select_consolidation_evidence(evidence_store)
    with pytest.raises(
        ConsolidationEvidenceError, match="No verified successful train"
    ):
        select_consolidation_evidence(evidence_store, task_ids={"train": "21"})
    selected = select_consolidation_evidence(evidence_store, task_ids={"train": "22"})
    assert selected.train.task_id == "22"


def test_receipt_must_match_the_conversation_task_correlation(evidence_store: Mock):
    receipts = evidence_store.load_workflow_execution_requests.return_value
    receipts["20"] = receipts["20"].model_copy(
        update={
            "correlation": TaskCorrelation(
                conversation_id=uuid4().hex, task_kind="workflow_run"
            )
        }
    )

    with pytest.raises(
        ConsolidationEvidenceError, match="No verified successful train"
    ):
        select_consolidation_evidence(evidence_store)


def test_consolidation_references_retain_safe_configuration(evidence_store: Mock):
    evidence = select_consolidation_evidence(evidence_store)

    references = consolidation_references(evidence_store, evidence)

    assert references == {
        "dataset": {
            "task_id": "10",
            "workflow_path": "public_data/workflow.py",
            "source_nodes": [
                {"id": "1", "node_type": "CustomCommandCPUNode", "config": {"cpu": 4}}
            ],
            "command_omitted": True,
        },
        "train": {
            "wandb_credential_task_id": "20",
            "workflow": {
                "nodes": [
                    {
                        "id": "1",
                        "type": "default",
                        "data": {
                            "nodeType": "WandbConfigBuilderNode",
                            "config": {"cpu": 4, "wandb_api_key": "WANDB_API_KEY"},
                        },
                    }
                ],
                "edges": [],
            },
        },
        "eval": {
            "definition": {
                "input_path": "/datasets/heldout.jsonl",
                "dataset_config": {"reference_field": "label"},
                "evaluation_config": {
                    "generation": {"max_tokens": 1024},
                    "metrics_url": "[redacted]",
                    "nested": {},
                },
                "inference": {},
            },
            "output_dir": "/workspace/outputs/eval",
        },
    }


def test_consolidation_references_detect_changed_definition(evidence_store: Mock):
    evidence = select_consolidation_evidence(evidence_store)
    pipelines = evidence_store.load_pipeline_runs.return_value
    updated = {**pipelines["eval"].definition, "input_path": "/new/heldout.jsonl"}
    pipelines["eval"] = pipelines["eval"].model_copy(update={"definition": updated})

    with pytest.raises(ConsolidationEvidenceError, match="evidence changed"):
        consolidation_references(evidence_store, evidence)


def test_evaluation_references_include_only_ready_assets_of_selected_receipt(
    evidence_store: Mock,
) -> None:
    evidence = select_consolidation_evidence(evidence_store)
    context = RequestContext(user_id="owner")
    versions = Mock(spec=Versions)
    versions.store = Mock()
    versions.project.return_value.id = "owner-project"
    selected = VersionRun(
        id="r2",
        project_id="owner-project",
        request_key="eval",
        spec=RunSpec(
            kind="evaluate",
            iteration_id="iteration",
            inputs={"runtime_script": "v2", "evaluation_config": "v3", "code": "v4"},
            parameters={
                "definition": {
                    "workflow": {
                        "nodes": [
                            {
                                "id": "cpu",
                                "data": {
                                    "nodeType": "CustomCommandCPUNode",
                                    "config": {
                                        "command": "fixture private command",
                                        "cpu": 4,
                                    },
                                },
                            }
                        ]
                    }
                }
            },
        ),
        source=RunSource(
            kind="pipeline",
            source_id="eval",
            task_id=evidence.eval.task_id,
            execution_revision=evidence.eval.revision,
        ),
    )
    unrelated = selected.model_copy(
        update={
            "source": selected.source.model_copy(update={"execution_revision": 99}),
        }
    )
    versions.store.list_runs.return_value = [unrelated, selected]
    ready = Mock(state="ready")
    ready.manifest.storage_ref = "/outputs/evaluate.py"
    ready.manifest.digest = "a" * 64
    versions.store.get_version.side_effect = [ready, Mock(state="failed")]
    reference = consolidation_evaluation_reference(versions, evidence, context)
    assert reference["assets"] == {
        "runtime_script": {
            "path": "/workspace/outputs/evaluate.py",
            "sha256": "a" * 64,
        }
    }
    assert "fixture private command" not in str(reference["workflow"])
    assert "CustomCommandCPUNode" in str(reference["workflow"])
    versions.project.assert_called_once_with(_CONVERSATION, context, create=False)
    versions.store.list_runs.assert_called_once_with("owner-project")
    assert [call.args for call in versions.store.get_version.call_args_list] == [
        ("owner-project", "v2"),
        ("owner-project", "v3"),
    ]


@pytest.mark.parametrize(
    ("wandb_node_type", "extra_output"),
    [
        ("WandbConfigBuilderNode", None),
        ("PyroMindWandbConfigBuilderNode", None),
        ("PyroMindWandbConfigBuilderNode", "logs_dir"),
        ("PyroMindWandbConfigBuilderNode", "nested"),
    ],
)
def test_archived_train_recipe_rebinds_only_qualified_topology(
    evidence_store: Mock, wandb_node_type: str, extra_output: str | None
):
    node_specs: dict[str, tuple[str, JsonObject]] = {
        "t": ("ModelTrainSFTNode", {"output_path": "/workspace/old-train"}),
        "w": (wandb_node_type, {"wandb_api_key": "a" * 40}),
        "m": (
            "ModelMergeLoraNode",
            {
                "model_path": "/workspace/old-base",
                "output_path": "/workspace/old-merged",
            },
        ),
        "j1": ("PathJoinNode", {"base_path": "/workspace/old", "subpath": "train"}),
        "j2": ("PathJoinNode", {"base_path": "/workspace/old", "subpath": "val"}),
        "d": ("DatasetConfigBuilderNode", {}),
        "dm": ("DatasetConfigBuilderMessageNode", {}),
        "de": ("DatasetExtraConfigBuilderNode", {}),
        "b": ("ModelConfigBuilderNode", {"model_path": "/workspace/old-base"}),
        "l": ("LoraConfigBuilderNode", {}),
        "tc": ("TrainingConfigBuilderNode", {}),
        "a": ("AccelerateConfigBuilderNode", {}),
    }
    if extra_output == "logs_dir":
        node_specs["t"][1]["logs_dir"] = "/workspace/old-logs"
    if extra_output == "nested":
        node_specs["t"][1]["options"] = {"logLocation": "/workspace/old-logs"}
    links = [
        ("j1", "joined_path", "d", "train_data_path"),
        ("j2", "joined_path", "d", "val_data_path"),
        ("dm", "dataset_kind_config", "d", "dataset_kind_config"),
        ("de", "dataset_extra_config", "d", "dataset_extra_config"),
        ("d", "dataset_config", "t", "dataset_config"),
        ("b", "model_config", "t", "model_config"),
        ("l", "lora_config", "t", "lora_config"),
        ("tc", "training_config", "t", "training_config"),
        ("a", "accelerate_config", "t", "accelerate_config"),
        ("w", "wandb_config", "t", "wandb_config"),
        ("t", "model_output_path", "m", "lora_path"),
    ]
    workflow = {
        "nodes": [
            {"id": node_id, "data": {"nodeType": node_type, "config": config}}
            for node_id, (node_type, config) in node_specs.items()
        ],
        "edges": [
            {
                "source": source,
                "sourceHandle": source_port,
                "target": target,
                "targetHandle": target_port,
            }
            for source, source_port, target, target_port in links
        ],
    }
    receipts = evidence_store.load_workflow_execution_requests.return_value
    receipts["20"] = receipts["20"].model_copy(update={"workflow": workflow})

    evidence = select_consolidation_evidence(evidence_store)
    reference = consolidation_references(evidence_store, evidence)["train"]
    assert isinstance(reference, dict)
    archived = reference["workflow"]
    assert isinstance(archived, dict)
    archived_nodes = archived["nodes"]
    assert isinstance(archived_nodes, list) and len(archived_nodes) == 12
    if extra_output:
        assert "runtime_candidate" not in reference
        return

    recipe = reference["runtime_candidate"]
    assert isinstance(recipe, dict)
    assert recipe["artifact_selection"] == "merged_model"
    runtime_workflow = recipe["workflow"]
    assert isinstance(runtime_workflow, dict)
    runtime_nodes = runtime_workflow["nodes"]
    assert isinstance(runtime_nodes, list)
    assert {
        "id": "w",
        "data": {
            "nodeType": "PyroMindWandbConfigBuilderNode",
            "config": {"wandb_api_key": "WANDB_API_KEY"},
        },
    } in runtime_nodes
    assert workflow["nodes"][1]["data"]["nodeType"] == wandb_node_type
    assert recipe["output_node_id"] == "m"
    output_uri = recipe["output_uri"]
    assert isinstance(output_uri, str)
    assert output_uri.endswith("{graph_run_id}/{stage_run_id}/merged-model")
    assert recipe["workflow_templates"] == {
        "nodes[0].data.config.output_path": output_uri.replace("/merged-model", "/lora")
    }
    assert recipe["workflow_bindings"] == {
        "nodes[3].data.config.train_data_path": "train_dataset",
        "nodes[3].data.config.val_data_path": "validation_dataset",
        "nodes[2].data.config.model_path": "base_model",
        "nodes[6].data.config.model_path": "base_model",
    }
    assert "/workspace/old" not in str(recipe["workflow"])


@pytest.mark.parametrize("change", [None, "report_bundle", "rubric_version"])
def test_eval_candidate_pins_archived_scoring_and_binds_runtime_paths(
    evidence_store: Mock, tmp_path: Path, change: str | None
):
    files = {
        "evaluate_inference.py": "# frozen scoring runtime\n",
        "evaluation_report.css": "/* frozen report styles */\n",
        "evaluation_report.js": "// frozen report code\n",
    }
    for name, content in files.items():
        (tmp_path / name).write_text(content)
    runtime_digest = runtime_bundle_fingerprint(tmp_path, files)
    if change == "report_bundle":
        (tmp_path / "evaluation_report.css").write_text("changed\n")
    definition: JsonObject = {
        "runtime_sha256": runtime_digest,
        "input_path": "/datasets/heldout.jsonl",
        "inference": {"model_path": "/models/historical-merged"},
        "dataset_config": {"reference_field": "label"},
        "evaluation_config": {"mode": "agent_rubric", "rubrics": {"accuracy": 1}},
    }
    pipelines = evidence_store.load_pipeline_runs.return_value
    source_id = str(pipelines["eval"].run_id)
    old_run_dir = (
        f"/workspace/.pyromind-agent/{_CONVERSATION}/data_preparation/{source_id}"
    )
    pipelines["eval"] = pipelines["eval"].model_copy(
        update={"definition": definition, "output_dir": old_run_dir}
    )
    snapshot = evidence_store.load_snapshot.return_value
    evidence_store.load_snapshot.return_value = snapshot.model_copy(
        update={
            "external_tasks": tuple(
                task.model_copy(update={"output_dir": old_run_dir})
                if task.task_id == "30"
                else task
                for task in snapshot.external_tasks
            )
        }
    )
    evidence = select_consolidation_evidence(evidence_store)
    assets = {
        "runtime_script": ("evaluate_inference.py", files["evaluate_inference.py"]),
        "dataset_config": (
            "dataset_config.json",
            canonical(definition["dataset_config"]),
        ),
        "evaluation_config": (
            "evaluation_config.json",
            canonical(definition["evaluation_config"]),
        ),
        "code": ("workflow.py", "# frozen workflow DSL\n"),
    }
    asset_paths = {
        role: f"/workspace/versions/{role}/{name}" for role, (name, _) in assets.items()
    }
    command = (
        f"python3 {asset_paths['runtime_script']} --run-id {source_id} "
        '--endpoint "$param" --model default '
        "--model-reference /workspace/models/historical-merged "
        "--dataset-path /workspace/datasets/heldout.jsonl "
        f"--dataset-config {asset_paths['dataset_config']} "
        f"--evaluation-config {asset_paths['evaluation_config']} "
        f"--output-dir {old_run_dir} --report-assets-dir {old_run_dir}"
    )
    workflow: JsonObject = {
        "nodes": [
            {
                "id": "serving",
                "data": {
                    "nodeType": "VLLMInference",
                    "config": {
                        "model_path": "/workspace/models/historical-merged",
                        "gpu_count": 1,
                    },
                },
            },
            {
                "id": "scorer",
                "data": {
                    "nodeType": "CustomCommandCPUNode",
                    "config": {
                        "command": command,
                        "cpu": 4,
                    },
                },
            },
        ],
        "edges": [
            {
                "source": "serving",
                "sourceHandle": "endpoint",
                "target": "scorer",
                "targetHandle": "param",
            }
        ],
    }
    versions = Mock(spec=Versions)
    versions.project.return_value = Mock(id="owner-project")
    versions.store = Mock()
    versions.store.list_runs.return_value = [
        VersionRun(
            id="r2",
            project_id="owner-project",
            request_key="eval",
            spec=RunSpec(
                kind="evaluate",
                iteration_id="iteration",
                inputs={role: role for role in assets},
                parameters={
                    "definition": {
                        "definition": definition,
                        "workflow": workflow,
                        "runtime_script": files["evaluate_inference.py"],
                    },
                },
            ),
            source=RunSource(kind="pipeline", source_id=source_id, task_id="30"),
        )
    ]
    versions_by_role = {}
    for role, (name, content) in assets.items():
        manifest = Mock(
            storage_ref=asset_paths[role],
            digest=fingerprint({name: sha256(content.encode()).hexdigest()}),
            source=Mock(filename=name),
        )
        versions_by_role[role] = Mock(state="ready", manifest=manifest)
    if change == "rubric_version":
        versions_by_role["evaluation_config"].manifest.digest = "a" * 64
    versions.store.get_version.side_effect = lambda _, role: versions_by_role[role]

    ref = consolidation_evaluation_reference(
        versions, evidence, RequestContext(user_id="owner"), runtime_dir=tmp_path
    )

    if change:
        blocker = ref.get("runtime_blocker")
        assert isinstance(blocker, str)
        assert "report assets" in blocker or "evaluation_config" in blocker
        assert "runtime_candidate" not in ref
        return
    recipe = ref["runtime_candidate"]
    assert isinstance(recipe, dict)
    assert recipe["workflow_bindings"] == {
        "nodes[0].data.config.model_path": "candidate_model"
    }
    assert recipe["command_binding_path"] == "nodes[1].data.config.command"
    command_template = recipe["command_template"]
    assert isinstance(command_template, str)
    assert "--model-reference '{candidate_model}'" in command_template
    assert "--dataset-path '{validation_dataset}'" in command_template
    assert "--output-dir " + old_run_dir not in command_template
    assert "--report-assets-dir " + old_run_dir in command_template
    assert "historical-merged" not in command_template
    output_uri = recipe["output_uri"]
    assert isinstance(output_uri, str)
    assert output_uri.endswith("{graph_run_id}/{stage_run_id}/evaluation/report.json")
    registered = ref["assets"]
    assert isinstance(registered, dict)
    assert registered["evaluation_report.css"] == {
        "path": f"{old_run_dir}/evaluation_report.css",
        "sha256": sha256(files["evaluation_report.css"].encode()).hexdigest(),
    }
