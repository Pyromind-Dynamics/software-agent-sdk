from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from string import Formatter
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, computed_field

from pyromind_agent_server.echomind_dataset_runtime import DatasetWorkflowConfig
from pyromind_agent_server.echomind_graph_runs import GraphRunStore
from pyromind_agent_server.echomind_projects import (
    ProjectGraph,
    ProjectInfo,
    ProjectNode,
    ProjectNodeInput,
    ProjectNodeInputBinding,
    ProjectNodeOutput,
    ProjectStore,
)
from pyromind_agent_server.echomind_publication import (
    RuntimeContractUpgradeDraft,
    build_runtime_contract_draft,
    compile_working_graph,
)
from pyromind_agent_server.echomind_storage import (
    ProjectLease,
    canonical_sha256,
    project_directory,
    read_model,
)
from pyromind_agent_server.echomind_train_backend import _WANDB_KEY_MARKER
from pyromind_agent_server.echomind_train_runtime import (
    TrainRuntimeError,
    TrainWorkflowSpec,
    prepare_workflow_for_submission,
)
from pyromind_runtime.application.consolidation_evidence import (
    ConsolidationEvidence,
    select_consolidation_evidence,
)
from pyromind_runtime.application.model_evaluation_plan import (
    MODEL_EVALUATION_DATA_INPUTS,
)
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.domain.preparation import ProjectView
from pyromind_runtime.domain.sft_reader import sft_reader
from pyromind_runtime.domain.storage_paths import normalize_workspace_path
from pyromind_runtime.infrastructure.file_product_store import FileProductStore


class ConsolidationContext(BaseModel):
    approved_project: ProjectView | None = Field(default=None, exclude=True, repr=False)
    project_id: str
    evidence: ConsolidationEvidence
    source_graph_sha256: str
    evidence_sha256: str
    source_batch: str = Field(
        validation_alias=AliasChoices("source_batch", "image_directory")
    )
    base_model: str
    validation_dataset: str
    train_uses_validation: bool = Field(default=True, exclude_if=lambda value: value)
    training_validation: str | None = Field(
        default=None, exclude_if=lambda v: v is None
    )

    @computed_field
    @property
    def image_directory(self) -> str:
        return self.source_batch

    @property
    def proposal_path(self) -> str:
        return f"public_data/execution-plans/{self.evidence_sha256[7:]}.json"


class ConsolidationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    batch_label_source: Literal["dataset", "collection"] | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
    )
    batch_validation_source: Literal["fixed", "batch", "none"] | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
    )
    fixed_validation_path: str | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    evidence_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    dataset_runtime: JsonObject
    train_runtime: JsonObject
    evaluator_runtime: JsonObject


type ConsolidationStageName = Literal["dataset", "train", "eval"]


class ConsolidationStageStatus(BaseModel):
    stage: ConsolidationStageName
    status: Literal["ready", "blocked"]
    verification: Literal["static", "replayed"] | None = None
    verification_task_ids: tuple[str, ...] = ()


class TrainReaderProbe(BaseModel):
    method: Literal["datasets.Dataset.from_list"]
    datasets_version: str = Field(min_length=1, max_length=64)
    pyarrow_version: str = Field(min_length=1, max_length=64)


class ConsolidationDatasetReplay(BaseModel):
    train_reader_probes: tuple[TrainReaderProbe, TrainReaderProbe] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    batch_label_source: Literal["dataset", "collection"] | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
    )
    batch_validation_source: Literal["fixed", "batch", "none"] | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
    )
    fixed_validation_path: str | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    dataset_runtime_sha256: str
    task_ids: tuple[str, str]
    source_kinds: tuple[str, str]
    source_fingerprints: tuple[str, str]
    train_fingerprints: tuple[str, str]
    train_rows: tuple[int, int]
    val_rows: tuple[int, int]


class _ReviewedReplay(BaseModel):
    evidence_sha256: str
    proposal: ConsolidationProposal | None = None
    dataset_replay: ConsolidationDatasetReplay | None = None


def compilation_path(store: ProjectStore, context: ConsolidationContext) -> Path:
    return (
        project_directory(store.root, context.project_id)
        / "execution-plan-previews"
        / f"{context.evidence_sha256[7:]}.json"
    )


class ConsolidationBlocker(BaseModel):
    stage: ConsolidationStageName
    code: str
    message: str


class ConsolidationRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    evidence_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    dataset_runtime: JsonObject | None = None
    train_runtime: JsonObject | None = None
    evaluator_runtime: JsonObject | None = None


class ConsolidationReview(BaseModel):
    context: ConsolidationContext
    proposal_path: str
    proposal_sha256: str | None = None
    draft: RuntimeContractUpgradeDraft | None = None
    issue: str | None = None
    stages: tuple[ConsolidationStageStatus, ...] = ()
    blockers: tuple[ConsolidationBlocker, ...] = ()
    assets: JsonObject = Field(default_factory=dict)


_NODE_STAGES = {"dataset": "dataset", "train": "train", "evaluator": "eval"}
_EXECUTORS = {
    "dataset": "dataset_runtime",
    "train": "train_runtime",
    "evaluator": "evaluator_runtime",
}
_MESSAGES = "echomind://schemas/messages-dataset/v1"
_MODEL = "echomind://schemas/model/v1"
_TRAIN_ROWS = ProjectNodeOutput(
    artifact_type="dataset.messages.train", schema_ref=_MESSAGES
)
_VAL_ROWS = ProjectNodeOutput(
    artifact_type="dataset.messages.validation", schema_ref=_MESSAGES
)
_TRAINED_MODEL = ProjectNodeOutput(artifact_type="model.checkpoint", schema_ref=_MODEL)
_REPORT = ProjectNodeOutput(
    artifact_type="evaluation.report",
    schema_ref="echomind://schemas/evaluation-report/v1",
)
_SECRET_FIELD = re.compile(
    r"(?:api|access|private|secret|session)key|authorization|cookie|password|"
    r"clientsecret|credential|token$"
)
_SECRET_ASSIGNMENT = re.compile(
    r"\b(?:[\w.-]*(?:api[_-]?key|access[_-]?key|secret[_-]?key|password|token)|"
    r"authorization|cookie)[\"']?\s*[:=]\s*(?!\s)[\"']?([^\s;,'\"]+)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"\b(?:Bearer\s+[\w.+/=-]{12,}|wandb_[\w-]{20,}|sk-[\w-]{16,}|"
    r"eyJ[\w-]+\.[\w-]+\.[\w-]+)\b",
    re.IGNORECASE,
)


def _validate_credentials(value: object, *, wandb_reference: bool) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _SECRET_FIELD.search(re.sub(r"[^a-z0-9]", "", key.casefold())):
                if (
                    key == "dataflow_credential_task_id"
                    and isinstance(item, str)
                    and re.fullmatch(r"[0-9]{1,64}", item)
                ):
                    continue
                if key == "wandb_credential_task_id" and wandb_reference:
                    continue
                if key == "wandb_api_key" and (
                    item in (None, "")
                    or (wandb_reference and item == _WANDB_KEY_MARKER)
                ):
                    continue
                raise ValueError("Persisted workflow contains a credential field.")
            _validate_credentials(item, wandb_reference=wandb_reference)
    elif isinstance(value, list):
        for item in value:
            _validate_credentials(item, wandb_reference=wandb_reference)
    elif isinstance(value, str) and (
        _SECRET_VALUE.search(value)
        or any(
            not match.group(1).startswith("$")
            for match in _SECRET_ASSIGNMENT.finditer(value)
        )
    ):
        raise ValueError("Persisted workflow contains a credential value.")


def consolidation_context(
    project: ProjectInfo,
    conversations_root: Path,
    *,
    task_ids: Mapping[str, str] | None = None,
    approved_project: ProjectView | None = None,
) -> ConsolidationContext:
    if approved_project and (
        approved_project.project_id.replace("-", "")
        != project.project.project_id.replace("-", "")
        or approved_project.conversation_id != project.project.conversation_id
        or (
            approved_project.cluster
            and approved_project.cluster != project.project.cluster
        )
    ):
        raise ValueError("The approved plan does not belong to this Project.")
    by_type = _stages(project)
    store = FileProductStore(conversations_root / project.project.conversation_id)
    evidence = select_consolidation_evidence(
        store,
        task_ids=task_ids,
        approved_project=approved_project,
    )
    data = {
        "project_id": project.project.project_id,
        "evidence": evidence.model_dump(mode="json"),
        "source_graph_sha256": canonical_sha256(project.graph.model_dump(mode="json")),
        "source_batch": _approved_path(
            by_type["dataset"], names={"source_batch"}, optional=True
        )
        or _approved_path(
            by_type["dataset"],
            names={
                "raw_images",
                "image_directory",
                "source_dataset",
                "raw_dataset",
                "raw_data",
                "source_data",
            },
            optional=True,
        )
        or _approved_path(by_type["dataset"], names={"data", "dataset"}),
        "base_model": _approved_path(by_type["train"], names={"base_model"}),
        "validation_dataset": _approved_path(
            by_type["evaluator"],
            names=MODEL_EVALUATION_DATA_INPUTS,
        ),
    }
    fixed_train = _approved_path(
        by_type["train"],
        names={"training_validation", "fixed_train_validation"},
        optional=True,
    )
    if fixed_train:
        if not fixed_train.endswith(".jsonl"):
            raise ValueError("The fixed train-time validation input must be JSONL.")
        data["training_validation"] = fixed_train
    elif approved_project and _confirmed_train_only(store, evidence, approved_project):
        data["train_uses_validation"] = False
    if not str(data["validation_dataset"]).endswith(".jsonl"):
        raise ValueError("The approved fixed evaluation input must be JSONL.")
    return ConsolidationContext.model_validate(
        {
            **data,
            "evidence_sha256": canonical_sha256(data),
            "approved_project": approved_project,
        }
    )


def _confirmed_train_only(
    store: FileProductStore, evidence: ConsolidationEvidence, project: ProjectView
) -> bool:
    receipt = store.load_workflow_execution_requests().get(
        "sdk:" + evidence.train.task_id
    )
    if not receipt or not receipt.correlation or not receipt.correlation.execution:
        return False
    task_id = receipt.correlation.execution.task_id
    task = next((t for t in project.plan.tasks if t.id == task_id), None)
    if not task:
        return False
    dataset_inputs = {"train_dataset", "train_data", "train_messages"}
    names = {item.name for item in task.inputs}
    if not (names & dataset_inputs) or names - dataset_inputs - {"base_model", "model"}:
        return False
    nodes, edges = receipt.workflow.get("nodes"), receipt.workflow.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return False
    builders = [
        (node.get("id"), data.get("config"))
        for node in nodes
        if isinstance(node, dict)
        and isinstance(data := node.get("data"), dict)
        and data.get("nodeType") == "DatasetConfigBuilderNode"
    ]
    if len(builders) != 1:
        return False
    node_id, config = builders[0]
    if not isinstance(node_id, str) or not isinstance(config, dict):
        return False
    connected = [
        edge.get("targetHandle")
        for edge in edges
        if isinstance(edge, dict) and edge.get("target") == node_id
    ]
    return (
        bool(config.get("train_data_path") or "train_data_path" in connected)
        and not config.get("val_data_path")
        and "val_data_path" not in connected
    )


def _stages(project: ProjectInfo) -> dict[str, ProjectNode]:
    nodes = {node.type: node for node in project.graph.nodes}
    if project.project.lifecycle != "development":
        raise ValueError("Reopen the Project before preparing an execution plan.")
    if len(project.graph.nodes) != 3 or nodes.keys() != _NODE_STAGES.keys():
        raise ValueError("Consolidation requires one Dataset, Train and Eval stage.")
    return nodes


def _approved_path(
    node: ProjectNode, *, names: set[str], optional: bool = False
) -> str | None:
    resources = {
        resource["id"]: resource.get("uri")
        for resource in (node.config.get("resources") or ())
        if isinstance(resource, dict) and isinstance(resource.get("id"), str)
    }
    inputs = [
        item
        for task in (node.config.get("tasks") or ())
        if isinstance(task, dict)
        for item in (task.get("inputs") or ())
        if isinstance(item, dict)
        and item.get("name") in names
        and not item.get("task_id")
        and not item.get("output_name")
    ]
    if not inputs and optional:
        return None
    paths = {resources.get(item.get("resource_id")) for item in inputs}
    if len(paths) != 1:
        raise ValueError(
            f"Stage {node.id} needs one confirmed {' or '.join(sorted(names))} source."
        )
    path = paths.pop()
    if not isinstance(path, str):
        raise ValueError(f"Stage {node.id} has an unbound input resource.")
    return normalize_workspace_path(path)


def _input(contract: ProjectNodeOutput, **binding: str) -> ProjectNodeInput:
    return ProjectNodeInput(
        artifact_type=contract.artifact_type,
        schema_ref=contract.schema_ref,
        binding=ProjectNodeInputBinding.model_validate(binding),
    )


def _stage_inputs_outputs(
    kind: str, context: ConsolidationContext, stage_nodes: dict[str, ProjectNode]
) -> tuple[dict[str, ProjectNodeInput], dict[str, ProjectNodeOutput]]:
    dataset, train = stage_nodes["dataset"], stage_nodes["train"]
    if kind == "dataset":
        return {
            "source_batch": ProjectNodeInput(
                artifact_type="dataset.source",
                schema_ref="echomind://schemas/dataset-source/v1",
                binding=ProjectNodeInputBinding(kind="run_input", name="source_batch"),
            ),
        }, {"train_messages": _TRAIN_ROWS, "val_messages": _VAL_ROWS}
    if kind == "train":
        inputs = {
            "train_dataset": _input(
                _TRAIN_ROWS,
                kind="node_output",
                node_id=dataset.id,
                port="train_messages",
            ),
            "base_model": _input(
                _TRAINED_MODEL, kind="literal", value=context.base_model
            ),
        }
        if context.train_uses_validation:
            inputs["validation_dataset"] = _input(
                _VAL_ROWS, kind="node_output", node_id=dataset.id, port="val_messages"
            )
        return inputs, {"trained_model": _TRAINED_MODEL}
    return {
        "candidate_model": _input(
            _TRAINED_MODEL, kind="node_output", node_id=train.id, port="trained_model"
        ),
        "validation_dataset": _input(
            _VAL_ROWS, kind="literal", value=context.validation_dataset
        ),
    }, {"eval_report": _REPORT}


def compile_consolidation(
    project: ProjectInfo, context: ConsolidationContext, proposal: ConsolidationProposal
) -> RuntimeContractUpgradeDraft:
    if proposal.evidence_sha256 != context.evidence_sha256:
        raise ValueError(
            "The proposal is for different Project content or task evidence."
        )
    if not context.train_uses_validation and proposal.batch_validation_source != "none":
        raise ValueError(
            "The approved training plan requires all batch samples for training."
        )
    if context.train_uses_validation and proposal.batch_validation_source == "none":
        raise ValueError("The approved training plan requires a validation input.")
    stage_nodes = _stages(project)
    configs = {
        "dataset": (proposal.dataset_runtime, context.evidence.dataset),
        "train": (proposal.train_runtime, context.evidence.train),
        "evaluator": (proposal.evaluator_runtime, context.evidence.eval),
    }
    nodes: list[ProjectNode] = []
    for node in project.graph.nodes:
        executor = _EXECUTORS[node.type]
        config, provenance = configs[node.type]
        inputs, outputs = _stage_inputs_outputs(node.type, context, stage_nodes)
        _validate_stage_config(node, config, inputs, provenance.task_id)
        node_config = {
            **node.config,
            executor: config,
            "development_evidence": provenance.model_dump(mode="json"),
        }
        if node.type == "dataset":
            for field in (
                "batch_label_source",
                "batch_validation_source",
                "fixed_validation_path",
            ):
                if field in proposal.model_fields_set:
                    policy = proposal.model_dump(include={field}, exclude_none=True)
                    previous = node_config.get(field)
                    if policy and previous is not None and previous != policy[field]:
                        raise ValueError(
                            f"Dataset policy {field} conflicts with the Project"
                        )
                    node_config.update(policy)
            node_config["runtime_input_defaults"] = {
                "source_batch": context.source_batch
            }
        nodes.append(
            node.model_copy(
                update={
                    "launch_mode": "executor",
                    "executor": executor,
                    "config": node_config,
                    "inputs": inputs,
                    "outputs": outputs,
                }
            )
        )
    graph = ProjectGraph.model_validate(
        {**project.graph.model_dump(mode="json"), "nodes": nodes}
    )
    candidate = ProjectInfo(project=project.project, graph=graph)
    draft = build_runtime_contract_draft(candidate)
    compile_working_graph(
        ProjectInfo(
            project=project.project,
            graph=graph.model_copy(update={"nodes": draft.nodes}),
        ),
        publication_mode="production",
    )
    return draft


def validate_consolidation_stage(
    project: ProjectInfo,
    context: ConsolidationContext,
    stage: ConsolidationStageName,
    config: JsonObject,
) -> JsonObject | None:
    nodes = _stages(project)
    node = nodes["evaluator" if stage == "eval" else stage]
    receipts = {
        "dataset": context.evidence.dataset,
        "train": context.evidence.train,
        "eval": context.evidence.eval,
    }
    inputs, _ = _stage_inputs_outputs(node.type, context, nodes)
    return _validate_stage_config(node, config, inputs, receipts[stage].task_id)


def _validate_stage_config(
    node: ProjectNode,
    config: JsonObject,
    inputs: Mapping[str, ProjectNodeInput],
    task_id: str,
) -> JsonObject | None:
    if not isinstance(config.get("workflow"), dict):
        raise ValueError(f"Node {node.id} needs an inline frozen workflow.")
    validate_persisted_payload(config)
    _validate_credentials(
        config,
        wandb_reference=(
            node.type == "train" and config.get("wandb_credential_task_id") == task_id
        ),
    )
    if node.type == "dataset":
        DatasetWorkflowConfig.model_validate(config).validate_workflow_inputs(
            input_ports=inputs
        )
        return None
    return _validate_workflow_render(node, config=config, inputs=inputs)


def _validate_workflow_render(
    node: ProjectNode, *, config: JsonObject, inputs: Mapping[str, ProjectNodeInput]
) -> JsonObject:
    values = {
        "project_id": "probe",
        "release_id": "probe-release",
        "stage_run_id": "probe-stage",
        "node_id": node.id,
        "graph_run_id": "consolidation-probe-run",
    }
    output_uri = config.get("output_uri")
    if not isinstance(output_uri, str) or not any(
        field == "graph_run_id" for _, field, _, _ in Formatter().parse(output_uri)
    ):
        raise ValueError(f"Stage {node.id} needs a run-specific output_uri.")
    try:
        output_uri = normalize_workspace_path(output_uri.format_map(values))
        artifact_uri = config.get("artifact_uri")
        if artifact_uri is not None:
            if not isinstance(artifact_uri, str):
                raise ValueError("Invalid artifact_uri.")
            artifact_uri = normalize_workspace_path(artifact_uri.format_map(values))
            if (
                not artifact_uri.startswith(f"{output_uri}/")
                and artifact_uri != output_uri
            ):
                raise ValueError("Artifact must be inside the run-specific output.")
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Stage {node.id} has an invalid output or artifact path."
        ) from exc
    selection = "merged_model" if node.type == "train" else "evaluation_report"
    if config.get("artifact_selection") != selection:
        raise ValueError(f"Stage {node.id} must select {selection}.")
    if selection == "evaluation_report" and not output_uri.endswith(".json"):
        raise ValueError(
            f"Stage {node.id} output_uri must be the evaluation JSON report."
        )
    probes = {port: f"/workspace/consolidation-probe/{port}" for port in inputs}
    spec = TrainWorkflowSpec.model_validate(
        {
            "task_name": "consolidation-probe",
            "cluster": "probe",
            "conversation_id": "probe",
            "workflow": config["workflow"],
            "workflow_path": config.get("workflow_path"),
            "wandb_credential_task_id": config.get("wandb_credential_task_id"),
            "bindings": config.get("workflow_bindings", {}),
            "workflow_templates": config.get("workflow_templates", {}),
            "inputs": probes,
            "output_uri": output_uri,
            "artifact_uri": artifact_uri,
            "artifact_selection": selection,
            "output_node_id": config.get("output_node_id"),
            "output_parameter": config.get("output_parameter", "output_path"),
            "command_template": config.get("command_template"),
            "command_binding_path": config.get("command_binding_path"),
            "template_values": {**values, "output_uri": output_uri},
        }
    )
    assert spec.workflow is not None
    try:
        rendered = prepare_workflow_for_submission(spec.workflow, spec=spec)
    except TrainRuntimeError as exc:
        raise ValueError(
            f"Stage {node.id} could not render its workflow bindings."
        ) from exc
    if (
        node.type == "train"
        and (reader := sft_reader(rendered))
        and reader.field_mapping()
    ):
        expected = {"train_data_path": probes["train_dataset"]}
        if "validation_dataset" in probes:
            expected["val_data_path"] = probes["validation_dataset"]
        if reader.paths != expected:
            raise ValueError(
                "Train 读取的训练或验证文件与 Dataset 已校验的输出端口不匹配。"
            )
    rendered_nodes = json.dumps(rendered.get("nodes"))
    for port, uri in probes.items():
        if uri not in rendered_nodes:
            raise ValueError(f"Stage {node.id} workflow must bind {port}.")
    if output_uri not in rendered_nodes:
        raise ValueError(f"Stage {node.id} workflow must bind its run-specific output.")
    return rendered


def apply_consolidation(
    store: ProjectStore,
    project: ProjectInfo,
    *,
    conversations_root: Path,
    task_ids: Mapping[str, str],
    proposal: ConsolidationProposal,
    expected_proposal_sha256: str,
    approved_project: ProjectView | None = None,
    load_current_proposal: Callable[
        [ConsolidationContext], ConsolidationProposal | None
    ]
    | None = None,
) -> ProjectInfo:
    project_id = project.project.project_id
    with ProjectLease(store.root, project_id).acquire():
        current = store.load(project_id, owner_user_id=project.project.owner_user_id)
        context = consolidation_context(
            current,
            conversations_root,
            task_ids=task_ids,
            approved_project=approved_project,
        )
        if load_current_proposal is not None:
            latest = load_current_proposal(context)
            if latest is None:
                raise ValueError(
                    "The proposal changed. Review it again before applying."
                )
            proposal = latest
        if (
            canonical_sha256(proposal.model_dump(mode="json"))
            != expected_proposal_sha256
        ):
            raise ValueError("The proposal changed. Review it again before applying.")
        if GraphRunStore(store.root).active(project_id) is not None:
            raise ValueError(
                "Finish the active run before updating the execution plan."
            )
        draft = compile_consolidation(current, context, proposal)
        try:
            recorded = read_model(compilation_path(store, context), _ReviewedReplay)
        except FileNotFoundError:
            recorded = None
        if (
            recorded is None
            or recorded.evidence_sha256 != context.evidence_sha256
            or recorded.proposal != proposal
            or recorded.dataset_replay is None
            or recorded.dataset_replay.train_reader_probes is None
            or recorded.dataset_replay.dataset_runtime_sha256
            != canonical_sha256(proposal.dataset_runtime)
        ):
            raise ValueError(
                "此方案缺少与当前配方匹配的真实换批验证，请重新生成并验证。"
            )
        return store.save(
            ProjectInfo(
                project=current.project,
                graph=current.graph.model_copy(
                    update={"nodes": draft.nodes, "edges": draft.edges}
                ),
            )
        )
