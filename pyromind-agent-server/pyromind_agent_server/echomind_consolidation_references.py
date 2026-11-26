from __future__ import annotations

import re
import shlex
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path, PurePosixPath

from pyromind_agent_server.echomind_failure_evidence import redact_graph_failure_text
from pyromind_agent_server.echomind_train_backend import (
    _WANDB_KEY_MARKER,
    _WANDB_KEY_PATTERN,
)
from pyromind_agent_server.echomind_train_runtime import (
    TrainRuntimeError,
    TrainWorkflowSpec,
    prepare_workflow_for_submission,
)
from pyromind_common.tools.data_preparation.runner import runtime_bundle_fingerprint
from pyromind_runtime.application.consolidation_evidence import (
    ConsolidationEvidence,
    ConsolidationEvidenceError,
    ConsolidationStageEvidence,
    consolidation_definition_sha256,
    select_consolidation_evidence,
)
from pyromind_runtime.application.versions import Versions
from pyromind_runtime.domain.content import JsonObject, JsonValue
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.domain.storage_paths import (
    StoragePathError,
    from_storage_api_path,
)
from pyromind_runtime.domain.versions import VersionRun, canonical, fingerprint
from pyromind_runtime.domain.workflow_execution import WorkflowExecutionRequest
from pyromind_runtime.ports.product_store import ProductStore


_EMPTY_REQUEST = RequestContext(user_id="reference")
_EVALUATION_ASSET_ROLES = {
    "business_scoring",
    "dataset_config",
    "evaluation_config",
    "prediction_evaluation",
    "runtime_script",
    "scoring_script",
}
_REPORT_ASSET_NAMES = ("evaluation_report.css", "evaluation_report.js")
_EVAL_BUNDLE = ("evaluate_inference.py", *_REPORT_ASSET_NAMES)
_EVAL_RUNTIME_DIR = (
    Path(__file__).resolve().parents[2] / ".agents/skills/inference-evaluation/scripts"
)
_EVAL_ARGUMENTS = {
    "--run-id",
    "--endpoint",
    "--model",
    "--model-reference",
    "--dataset-path",
    "--dataset-config",
    "--evaluation-config",
    "--output-dir",
    "--report-assets-dir",
}
_SAFE_STORAGE_PATH = re.compile(r"/workspace/[a-zA-Z0-9._/-]+")
_CHECK_EVAL_ASSETS = (
    "import sys; from hashlib import sha256; from pathlib import Path; "
    "files = zip(sys.argv[1::2], sys.argv[2::2]); "
    "invalid = [p for p, digest in files if not Path(p).is_file() or "
    "sha256(Path(p).read_bytes()).hexdigest() != digest]; "
    "sys.exit('Eval assets missing or changed: ' + ', '.join(invalid) "
    "if invalid else 0)"
)
_SENSITIVE_FIELD = re.compile(
    r"(?:api|access|secret|private|session)key|authorization|cookie|password|"
    r"credential|signature|clientsecret|token$"
)
_CODE_FIELDS = {"command", "script", "environment", "env", "headers"}
_EVAL_FIELDS = {
    "cpu",
    "dataset_config",
    "evaluation_config",
    "inference",
    "input_path",
    "memory",
    "report_spec",
    "runtime_sha256",
}
_PATH_JOIN = "PathJoinNode"
_DATA = "DatasetConfigBuilderNode"
_BASE = "ModelConfigBuilderNode"
_TRAIN = "ModelTrainSFTNode"
_MERGE = "ModelMergeLoraNode"
_WANDB = "WandbConfigBuilderNode"
_TRAIN_EDGES = {
    (_PATH_JOIN, "joined_path", _DATA, "train_data_path"),
    (_PATH_JOIN, "joined_path", _DATA, "val_data_path"),
    (
        "DatasetConfigBuilderMessageNode",
        "dataset_kind_config",
        _DATA,
        "dataset_kind_config",
    ),
    (
        "DatasetExtraConfigBuilderNode",
        "dataset_extra_config",
        _DATA,
        "dataset_extra_config",
    ),
    (_DATA, "dataset_config", _TRAIN, "dataset_config"),
    (_BASE, "model_config", _TRAIN, "model_config"),
    ("LoraConfigBuilderNode", "lora_config", _TRAIN, "lora_config"),
    ("TrainingConfigBuilderNode", "training_config", _TRAIN, "training_config"),
    ("AccelerateConfigBuilderNode", "accelerate_config", _TRAIN, "accelerate_config"),
    (_WANDB, "wandb_config", _TRAIN, "wandb_config"),
    (_TRAIN, "model_output_path", _MERGE, "lora_path"),
}
_TRAIN_NODES = Counter({name: 1 for edge in _TRAIN_EDGES for name in edge[::2]})
_TRAIN_NODES[_PATH_JOIN] = 2
_PATH_FIELDS = {
    _PATH_JOIN: {"base_path", "subpath"},
    _DATA: set(),
    _BASE: {"model_path"},
    _TRAIN: {"output_path"},
    _MERGE: {"model_path", "output_path"},
}


def consolidation_references(
    store: ProductStore, evidence: ConsolidationEvidence
) -> JsonObject:
    current = select_consolidation_evidence(
        store,
        task_ids={
            "dataset": evidence.dataset.task_id,
            "train": evidence.train.task_id,
            "eval": evidence.eval.task_id,
        },
    )
    if current != evidence:
        raise ConsolidationEvidenceError("Development execution evidence changed")

    receipts = tuple(store.load_workflow_execution_requests().values())
    dataset = _workflow_receipt(receipts, evidence.dataset, evidence.conversation_id)
    train = _workflow_receipt(receipts, evidence.train, evidence.conversation_id)
    evaluation = [
        run
        for run in store.load_pipeline_runs().values()
        if run.task_id == evidence.eval.task_id
        and run.submission_status == "submitted"
        and run.conversation_id == evidence.conversation_id
        and run.execution_revision == evidence.eval.revision
        and consolidation_definition_sha256(run.definition)
        == evidence.eval.definition_sha256
    ]
    if len(evaluation) != 1:
        raise ConsolidationEvidenceError("Evaluation execution receipt changed")
    (evaluation_run,) = evaluation

    workflow, has_wandb_reference = _xyflow_reference(train.workflow)
    train_reference: JsonObject = {
        "workflow": workflow,
        "wandb_credential_task_id": train.task_id if has_wandb_reference else None,
    }
    if train.task_id and has_wandb_reference:
        candidate = _train_runtime_candidate(
            workflow, evidence.conversation_id, train.task_id
        )
        if candidate is not None:
            train_reference["runtime_candidate"] = candidate
    return {
        "dataset": {
            "task_id": dataset.task_id,
            "workflow_path": _source_path(dataset.workflow_path),
            "source_nodes": _dataset_nodes(dataset.workflow),
            "command_omitted": True,
        },
        "train": train_reference,
        "eval": {
            "definition": _project_fields(evaluation_run.definition, _EVAL_FIELDS),
            "output_dir": from_storage_api_path(evaluation_run.output_dir),
        },
    }


def consolidation_evaluation_reference(
    versions: Versions,
    evidence: ConsolidationEvidence,
    context: RequestContext,
    *,
    runtime_dir: Path = _EVAL_RUNTIME_DIR,
) -> JsonObject:
    project = versions.project(evidence.conversation_id, context, create=False)
    if project is None:
        return {"runtime_blocker": "No versioned Eval archive for this conversation."}
    records = [
        run
        for run in versions.store.list_runs(project.id)
        if run.source.kind == "pipeline"
        and run.source.task_id == evidence.eval.task_id
        and run.source.execution_revision == evidence.eval.revision
    ]
    if len(records) != 1:
        return {"runtime_blocker": "Could not identify a unique archived Eval run."}
    (record,) = records
    assets: JsonObject = {}
    for role, version_id in record.spec.inputs.items():
        if role not in _EVALUATION_ASSET_ROLES:
            continue
        try:
            version = versions.store.get_version(project.id, version_id)
        except LookupError:
            continue
        storage_ref = version.manifest.storage_ref
        digest = version.manifest.digest
        if (
            version.state != "ready"
            or not storage_ref
            or not digest
            or not re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", digest)
        ):
            continue
        try:
            path = from_storage_api_path(storage_ref)
        except StoragePathError:
            continue
        if _safe_text(path) == path:
            assets[role] = {"path": path, "sha256": digest.removeprefix("sha256:")}
    reference: JsonObject = {"assets": assets}
    definition = record.spec.parameters.get("definition")
    if isinstance(definition, dict) and isinstance(
        workflow := definition.get("workflow"), dict
    ):
        safe_workflow, _ = _xyflow_reference(workflow)
        reference["workflow"] = safe_workflow
        try:
            candidate, report_assets = _evaluation_runtime_candidate(
                record,
                definition,
                workflow,
                safe_workflow,
                assets,
                evidence,
                runtime_dir,
            )
            code_id = record.spec.inputs.get("code")
            if not code_id:
                raise ValueError("Eval workflow source version is missing.")
            code_version = versions.store.get_version(project.id, code_id)
            code_digest = code_version.manifest.digest
            if (
                code_version.state != "ready"
                or code_version.manifest.source.filename != "workflow.py"
                or not isinstance(code_digest, str)
                or not re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", code_digest)
            ):
                raise ValueError("Eval workflow source version is not verified.")
        except LookupError:
            reference["runtime_blocker"] = "Eval workflow source version is missing."
        except (OSError, ValueError, TrainRuntimeError) as exc:
            reference["runtime_blocker"] = _safe_text(str(exc))
        else:
            assets.update(report_assets)
            reference["workflow_version"] = code_id
            reference["runtime_candidate"] = candidate
    else:
        reference["runtime_blocker"] = "The archived Eval workflow is missing."
    return reference


def _evaluation_runtime_candidate(
    record: VersionRun,
    archive: JsonObject,
    workflow: JsonObject,
    candidate: JsonObject,
    assets: JsonObject,
    evidence: ConsolidationEvidence,
    runtime_dir: Path,
) -> tuple[JsonObject, JsonObject]:
    definition = archive.get("definition")
    script = archive.get("runtime_script")
    if (
        not isinstance(definition, dict)
        or consolidation_definition_sha256(definition)
        != evidence.eval.definition_sha256
        or not isinstance(script, str)
    ):
        raise ValueError("Archived Eval definition does not match the successful task.")
    if definition.get("report_spec") or definition.get("business_scoring"):
        raise ValueError(
            "Eval uses optional scoring or report assets without pinned bindings."
        )
    expected_bundle = definition.get("runtime_sha256")
    if (
        not isinstance(expected_bundle, str)
        or runtime_bundle_fingerprint(runtime_dir, _EVAL_BUNDLE) != expected_bundle
        or (runtime_dir / "evaluate_inference.py").read_bytes() != script.encode()
    ):
        raise ValueError(
            "Eval report assets are unavailable for its recorded runtime digest."
        )

    source_nodes = _objects(workflow.get("nodes"))
    safe_workflow = deepcopy(candidate)
    indexes, command = _evaluation_topology(
        source_nodes, _objects(workflow.get("edges"))
    )
    tokens = shlex.split(command)
    if (
        len(tokens) != 2 + 2 * len(_EVAL_ARGUMENTS)
        or tokens[0] != "python3"
        or set(tokens[2::2]) != _EVAL_ARGUMENTS
        or '--endpoint "$param"' not in command
    ):
        raise ValueError(
            "Archived Eval command does not match the qualified scorer CLI."
        )
    arguments = dict(zip(tokens[2::2], tokens[3::2], strict=True))
    model = arguments["--model"]
    run_path = _safe_eval_path(arguments["--report-assets-dir"])
    expected_path = (
        f"/workspace/.pyromind-agent/{evidence.conversation_id}/"
        f"data_preparation/{record.source.source_id}"
    )
    if (
        not re.fullmatch(r"[a-f0-9]{32}", evidence.conversation_id)
        or arguments["--run-id"] != record.source.source_id
        or arguments["--endpoint"] != "$param"
        or not re.fullmatch(r"[a-zA-Z0-9._-]{1,64}", model)
        or _safe_eval_path(arguments["--output-dir"]) != run_path
        or run_path != expected_path
        or _safe_eval_path(arguments["--dataset-path"])
        != _safe_eval_path(definition.get("input_path"))
    ):
        raise ValueError("Eval command does not match the recorded run and dataset.")

    dataset_config = definition.get("dataset_config")
    evaluation_config = definition.get("evaluation_config")
    if not isinstance(dataset_config, dict) or not isinstance(evaluation_config, dict):
        raise ValueError("Eval is missing its versioned dataset mapping or rubric.")
    source_artifacts = (
        ("runtime_script", tokens[1], script),
        ("dataset_config", arguments["--dataset-config"], canonical(dataset_config)),
        (
            "evaluation_config",
            arguments["--evaluation-config"],
            canonical(evaluation_config),
        ),
    )
    file_digests: dict[str, str] = {}
    file_paths: dict[str, str] = {}
    for role, raw_path, content in source_artifacts:
        path = _safe_eval_path(raw_path)
        digest = sha256(content.encode()).hexdigest()
        asset = assets.get(role)
        if (
            not isinstance(asset, dict)
            or asset.get("path") != path
            or asset.get("sha256") != fingerprint({PurePosixPath(path).name: digest})
        ):
            raise ValueError(f"Eval versioned asset {role} is missing or changed.")
        file_digests[path] = digest
        file_paths[role] = path
    report_assets: JsonObject = {}
    for name in _REPORT_ASSET_NAMES:
        path = f"{run_path}/{name}"
        digest = sha256((runtime_dir / name).read_bytes()).hexdigest()
        file_digests[path] = digest
        report_assets[name] = {"path": path, "sha256": digest}

    nodes = _objects(safe_workflow.get("nodes"))
    for node_type, index in indexes.items():
        config = _node_config(nodes[index])
        if config is None:
            raise ValueError("Eval workflow is missing a qualified node configuration.")
        if node_type == "VLLMInference":
            if _safe_eval_path(config.get("model_path")) != _safe_eval_path(
                arguments["--model-reference"]
            ):
                raise ValueError("Eval serving node and scorer use different models.")
            config["model_path"] = ""
        else:
            config["command"] = ""
        if _has_unbound_config(config):
            raise ValueError("Eval workflow contains an unbound Storage path.")

    run_output = (
        f"/workspace/.pyromind-agent/{evidence.conversation_id}/echomind-runs/"
        "{graph_run_id}/{stage_run_id}/evaluation"
    )
    output_uri = f"{run_output}/report.json"
    checks = " ".join(
        f"{shlex.quote(path)} {digest}" for path, digest in file_digests.items()
    )
    command_template = (
        f"python3 -c {shlex.quote(_CHECK_EVAL_ASSETS)} {checks} && "
        f"python3 {shlex.quote(file_paths['runtime_script'])} "
        "--run-id '{graph_run_id}' "
        f'--endpoint "$param" --model {shlex.quote(model)} '
        "--model-reference '{candidate_model}' --dataset-path '{validation_dataset}' "
        f"--dataset-config {shlex.quote(file_paths['dataset_config'])} "
        f"--evaluation-config {shlex.quote(file_paths['evaluation_config'])} "
        f"--output-dir '{run_output}' --report-assets-dir {shlex.quote(run_path)} "
        "&& test -s '{output_uri}'"
    )
    bindings = {
        f"nodes[{indexes['VLLMInference']}].data.config.model_path": "candidate_model"
    }
    command_path = f"nodes[{indexes['CustomCommandCPUNode']}].data.config.command"
    probe = {"graph_run_id": "probe-run", "stage_run_id": "probe-stage"}
    prepare_workflow_for_submission(
        safe_workflow,
        spec=TrainWorkflowSpec(
            task_name="consolidation-probe",
            conversation_id=evidence.conversation_id,
            cluster="probe",
            workflow=safe_workflow,
            bindings=bindings,
            inputs={
                "candidate_model": "/workspace/probe/merged-model",
                "validation_dataset": "/workspace/probe/heldout.jsonl",
            },
            output_uri=output_uri.format_map(probe),
            artifact_selection="evaluation_report",
            command_template=command_template,
            command_binding_path=command_path,
            template_values={**probe, "output_uri": output_uri.format_map(probe)},
        ),
    )
    result: JsonObject = {
        "workflow": safe_workflow,
        "workflow_bindings": dict(bindings),
        "command_template": command_template,
        "command_binding_path": command_path,
        "output_uri": output_uri,
        "artifact_selection": "evaluation_report",
        "runtime_sha256": expected_bundle,
    }
    return result, report_assets


def _safe_eval_path(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise ValueError("Eval requires a Storage path.")
    path = from_storage_api_path(value)
    if not _SAFE_STORAGE_PATH.fullmatch(path):
        raise ValueError("Eval uses a Storage path unsafe for its command template.")
    return path


def _node_config(node: JsonObject) -> JsonObject | None:
    data = node.get("data")
    config = data.get("config") if isinstance(data, dict) else None
    return config if isinstance(config, dict) else None


def _evaluation_topology(
    nodes: list[JsonObject], edges: list[JsonObject]
) -> tuple[dict[str, int], str]:
    indexes: dict[str, int] = {}
    for index, node in enumerate(nodes):
        data = node.get("data")
        if isinstance(data, dict) and isinstance(
            node_type := data.get("nodeType"), str
        ):
            indexes[node_type] = index
    if (
        len(nodes) != 2
        or indexes.keys() != {"VLLMInference", "CustomCommandCPUNode"}
        or len(edges) != 1
    ):
        raise ValueError(
            "Archived Eval topology is not the qualified Inference-to-CPU form."
        )
    serving = nodes[indexes["VLLMInference"]]
    scorer = nodes[indexes["CustomCommandCPUNode"]]
    edge = edges[0]
    config = _node_config(scorer)
    if (
        not isinstance(serving.get("id"), str)
        or not isinstance(scorer.get("id"), str)
        or serving["id"] == scorer["id"]
        or edge.get("source") != serving["id"]
        or edge.get("sourceHandle") != "endpoint"
        or edge.get("target") != scorer["id"]
        or edge.get("targetHandle") != "param"
        or config is None
        or not isinstance(command := config.get("command"), str)
    ):
        raise ValueError(
            "Archived Eval does not bind its scoring command to Inference."
        )
    return indexes, command


def _workflow_receipt(
    receipts: tuple[WorkflowExecutionRequest, ...],
    stage: ConsolidationStageEvidence,
    conversation_id: str,
) -> WorkflowExecutionRequest:
    matches = [
        receipt
        for receipt in receipts
        if receipt.task_id == stage.task_id
        and receipt.status == "submitted"
        and receipt.conversation_id == conversation_id
        and receipt.revision == stage.revision
        and consolidation_definition_sha256(receipt.workflow) == stage.definition_sha256
    ]
    if len(matches) != 1:
        raise ConsolidationEvidenceError("Workflow execution receipt changed")
    return matches[0]


def _source_path(value: str) -> str | None:
    if re.fullmatch(r"version://r[0-9]+", value):
        return value
    if not re.fullmatch(r"[\w./-]+", value) or ".." in PurePosixPath(value).parts:
        return None
    return value if _safe_text(value) == value else None


def _dataset_nodes(workflow: JsonObject) -> list[JsonValue]:
    result: list[JsonValue] = []
    for node in _objects(workflow.get("nodes")):
        data = node.get("data")
        if not isinstance(data, dict):
            continue
        config = data.get("config")
        result.append(
            {
                "id": _sanitize(node.get("id")),
                "node_type": _sanitize(data.get("nodeType")),
                "config": _project_fields(config, {"cpu", "memory"})
                if isinstance(config, dict)
                else {},
            }
        )
    return result


def _xyflow_reference(workflow: JsonObject) -> tuple[JsonObject, bool]:
    nodes: list[JsonValue] = []
    has_wandb_reference = False
    for node in _objects(workflow.get("nodes")):
        data = node.get("data")
        if not isinstance(data, dict):
            continue
        archived_config = data.get("config")
        config = (
            _safe_object(archived_config) if isinstance(archived_config, dict) else {}
        )
        if data.get("nodeType") == "WandbConfigBuilderNode" and isinstance(
            archived_config, dict
        ):
            key = archived_config.get("wandb_api_key")
            if isinstance(key, str) and _WANDB_KEY_PATTERN.fullmatch(key):
                config["wandb_api_key"] = _WANDB_KEY_MARKER
                has_wandb_reference = True
        nodes.append(
            {
                **_project_fields(node, {"id", "type", "position"}),
                "data": {
                    **_project_fields(data, {"nodeType", "display_name"}),
                    "config": config,
                },
            }
        )
    return (
        {
            **_project_fields(workflow, {"id", "name", "timestamp", "viewport"}),
            "nodes": nodes,
            "edges": [
                _project_fields(
                    edge, {"id", "source", "target", "sourceHandle", "targetHandle"}
                )
                for edge in _objects(workflow.get("edges"))
            ],
        },
        has_wandb_reference,
    )


def _train_runtime_candidate(
    workflow: JsonObject, conversation_id: str, task_id: str
) -> JsonObject | None:
    if not re.fullmatch(r"[a-f0-9]{32}", conversation_id):
        return None
    candidate = deepcopy(workflow)
    nodes = _objects(candidate.get("nodes"))
    edges = _objects(candidate.get("edges"))
    types: dict[str, str] = {}
    for node in nodes:
        node_id = node.get("id")
        data = node.get("data")
        if (
            not isinstance(node_id, str)
            or node_id in types
            or not isinstance(data, dict)
        ):
            return None
        node_type = data.get("nodeType")
        config = data.get("config")
        if not isinstance(node_type, str) or not isinstance(config, dict):
            return None
        path_fields = {key for key in config if key.endswith(("path", "_dir", "_uri"))}
        if path_fields != _PATH_FIELDS.get(node_type, set()):
            return None
        types[node_id] = node_type
    if Counter(types.values()) != _TRAIN_NODES or len(edges) != len(_TRAIN_EDGES):
        return None
    topology = set()
    path_joins = set()
    retained_edges = []
    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        source_port, target_port = edge.get("sourceHandle"), edge.get("targetHandle")
        if (
            not isinstance(source, str)
            or not isinstance(target, str)
            or not isinstance(source_port, str)
            or not isinstance(target_port, str)
        ):
            return None
        topology.add((types.get(source), source_port, types.get(target), target_port))
        if types.get(source) == _PATH_JOIN:
            path_joins.add(source)
        else:
            retained_edges.append(edge)
    if topology != _TRAIN_EDGES or len(path_joins) != 2:
        return None

    retained_nodes = [node for node in nodes if node["id"] not in path_joins]
    candidate["nodes"] = list(retained_nodes)
    candidate["edges"] = list(retained_edges)
    positions: dict[str, int] = {}
    for index, node in enumerate(retained_nodes):
        node_id, data = node["id"], node["data"]
        assert isinstance(node_id, str) and isinstance(data, dict)
        node_type = types[node_id]
        positions[node_type] = index
        config = data["config"]
        assert isinstance(config, dict)
        for key in _PATH_FIELDS.get(node_type, set()):
            config[key] = ""
        if node_type == _DATA:
            config.update(train_data_path="", val_data_path="")
        if _has_unbound_config(config):
            return None

    merge_id = retained_nodes[positions[_MERGE]]["id"]
    assert isinstance(merge_id, str)
    output_root = (
        f"/workspace/.pyromind-agent/{conversation_id}/echomind-runs/"
        "{graph_run_id}/{stage_run_id}"
    )
    output_uri = f"{output_root}/merged-model"
    bindings = {
        f"nodes[{positions[_DATA]}].data.config.train_data_path": "train_dataset",
        f"nodes[{positions[_DATA]}].data.config.val_data_path": "validation_dataset",
        f"nodes[{positions[_BASE]}].data.config.model_path": "base_model",
        f"nodes[{positions[_MERGE]}].data.config.model_path": "base_model",
    }
    templates = {
        f"nodes[{positions[_TRAIN]}].data.config.output_path": f"{output_root}/lora"
    }
    probe = {"graph_run_id": "probe-run", "stage_run_id": "probe-stage"}
    try:
        prepare_workflow_for_submission(
            candidate,
            spec=TrainWorkflowSpec(
                task_name="consolidation-probe",
                conversation_id=conversation_id,
                cluster="probe",
                workflow=candidate,
                wandb_credential_task_id=task_id,
                bindings=bindings,
                workflow_templates=templates,
                inputs={key: f"/workspace/probe/{key}" for key in bindings.values()},
                output_uri=output_uri.format_map(probe),
                artifact_selection="merged_model",
                output_node_id=merge_id,
                template_values=probe,
            ),
        )
    except (TrainRuntimeError, ValueError):
        return None
    result: JsonObject = {
        "workflow": candidate,
        "workflow_bindings": dict(bindings),
        "workflow_templates": dict(templates),
        "output_uri": output_uri,
        "output_node_id": merge_id,
        "artifact_selection": "merged_model",
        "wandb_credential_task_id": task_id,
    }
    return result


def _has_unbound_config(value: JsonValue) -> bool:
    if isinstance(value, dict):
        return any(_has_unbound_config(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_unbound_config(item) for item in value)
    return isinstance(value, str) and (
        value == "[redacted]" or value.startswith(("/", "file://", "storage://"))
    )


def _objects(value: JsonValue) -> list[JsonObject]:
    return (
        [entry for entry in value if isinstance(entry, dict)]
        if isinstance(value, list)
        else []
    )


def _project_fields(source: JsonObject, fields: set[str]) -> JsonObject:
    return _safe_object({key: value for key, value in source.items() if key in fields})


def _safe_object(source: JsonObject) -> JsonObject:
    return {
        key: _sanitize(value)
        for key, value in source.items()
        if (normalized := re.sub(r"[^a-z0-9]", "", key.lower())) not in _CODE_FIELDS
        and _SENSITIVE_FIELD.search(normalized) is None
        and _safe_text(key) == key
    }


def _sanitize(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return _safe_object(value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return _safe_text(value)
    return value


def _safe_text(value: str) -> str:
    try:
        validate_persisted_payload(value)
    except ValueError:
        return "[redacted]"
    return (
        value
        if redact_graph_failure_text(value, _EMPTY_REQUEST) == value
        else "[redacted]"
    )
