from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePosixPath
from string import Formatter
from typing import Literal

from pydantic import BaseModel, ConfigDict

from pyromind_agent_server.echomind_consolidation import (
    ConsolidationBlocker,
    _validate_credentials,
)
from pyromind_agent_server.echomind_dataset_runtime import DatasetWorkflowConfig
from pyromind_agent_server.echomind_train_runtime import TrainRuntimeError
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.domain.storage_paths import normalize_workspace_path


_STORAGE_LITERAL = re.compile(
    r"(?<![\w/.:\-])/(?:target-workspace|workspace)"
    r"(?:/[a-zA-Z0-9_.-]+)+(?=$|[\s\"'();])"
)
_DATAFLOW_STRUCTURED = re.compile(r"\bDF_OUTPUT_SCHEMA=['\"]?structured\b")
_DATAFLOW_GATEWAY = re.compile(r"\bDF_API_(?:KEY|BASE_URL|URL)=")
_ASSIGNMENT = re.compile(r"(?:^|[\s;])[a-zA-Z_][\w.-]*\s*=\s*['\"]?\Z")
_PROBE_IDS = {
    name: "dataset-recipe-probe"
    for name in ("project_id", "release_id", "graph_run_id", "stage_run_id", "node_id")
}


class DatasetAssetRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["input_manifest", "pipeline_source", "runtime_file", "annotations"]
    path: str


@dataclass(frozen=True)
class DatasetRecipeExtraction:
    runtime: JsonObject | None
    assets: tuple[DatasetAssetRef, ...] = ()
    blockers: tuple[ConsolidationBlocker, ...] = ()


def _archive_dataflow_commands(workflow: JsonObject) -> tuple[str, ...]:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return ()
    commands: list[str] = []
    for node in nodes:
        data = node.get("data") if isinstance(node, dict) else None
        config = data.get("config") if isinstance(data, dict) else None
        command = config.get("command") if isinstance(config, dict) else None
        if (
            isinstance(command, str)
            and "DF_STATE_DIR=" in command
            and _DATAFLOW_STRUCTURED.search(command)
        ):
            commands.append(command)
    return tuple(commands)


def _archive_assets(
    commands: tuple[str, ...], *, conversation_id: str | None
) -> tuple[DatasetAssetRef, ...]:
    assets: dict[str, DatasetAssetRef] = {}
    for command in commands:
        for match in _STORAGE_LITERAL.finditer(command):
            if _ASSIGNMENT.search(command, max(0, match.start() - 128), match.start()):
                continue
            literal = match[0].replace("/target-workspace/", "/workspace/", 1)
            try:
                path = normalize_workspace_path(literal)
                _validate_credentials(path, wandb_reference=False)
            except ValueError:
                continue
            parts = PurePosixPath(path).parts
            if parts[:3] == ("/", "workspace", ".pyromind-agent") and (
                len(parts) < 5 or parts[3] != conversation_id
            ):
                continue
            filename = parts[-1]
            role: Literal[
                "input_manifest", "pipeline_source", "runtime_file", "annotations"
            ]
            if filename == "processed.jsonl":
                role = "annotations"
            elif filename.endswith(".jsonl") and "manifest" in filename:
                role = "input_manifest"
            elif filename.endswith(".py") and "runtime-cache" in parts:
                role = "runtime_file"
            elif filename in {"requirements.txt", "requirements-cpu.txt"}:
                role = "runtime_file"
            elif filename == "pipeline.py":
                role = "pipeline_source"
            else:
                continue
            assets[path] = DatasetAssetRef(role=role, path=path)
    return tuple(assets[path] for path in sorted(assets))


def _dataset_blocker(code: str, message: str) -> ConsolidationBlocker:
    return ConsolidationBlocker(stage="dataset", code=code, message=message)


def _validated_runtime(runtime: JsonObject, source_batch: str) -> JsonObject:
    validate_persisted_payload(runtime)
    _validate_credentials(runtime, wandb_reference=False)
    config = DatasetWorkflowConfig.model_validate(runtime)
    nodes = config.workflow.get("nodes")
    edges = config.workflow.get("edges")
    if not isinstance(nodes, list) or not nodes or not isinstance(edges, list):
        raise ValueError("Dataset requires an inline Studio workflow")
    allowed_fields = {*_PROBE_IDS, "source_batch", "output_uri"}
    for template in (
        config.output_uri,
        config.command_template or "",
        *config.workflow_templates.values(),
    ):
        for _, field, spec, conversion in Formatter().parse(template):
            if field is not None and (
                field not in allowed_fields or spec or conversion
            ):
                raise ValueError("Unsupported Dataset workflow template field")
    config.validate_workflow_inputs(input_ports=("source_batch",))
    config.render_workflow(
        inputs={"source_batch": normalize_workspace_path(source_batch)},
        runtime_values=_PROBE_IDS,
    )
    return deepcopy(runtime)


def extract_dataset_recipe(
    *,
    archive_workflow: JsonObject,
    source_batch: str,
    reviewed_runtime: JsonObject | None = None,
    conversation_id: str | None = None,
) -> DatasetRecipeExtraction:
    """Project safe archive references; compile only an explicit, reviewed contract.

    Archive references attest to what an older command named. Storage existence,
    annotation review, and execution of the rendered contract need separate checks.
    """
    commands = _archive_dataflow_commands(archive_workflow)
    assets = _archive_assets(commands, conversation_id=conversation_id)
    if reviewed_runtime is not None:
        try:
            runtime = _validated_runtime(reviewed_runtime, source_batch)
        except (ValueError, TrainRuntimeError):
            return DatasetRecipeExtraction(
                runtime=None,
                assets=assets,
                blockers=(
                    _dataset_blocker(
                        "dataset_recipe_invalid",
                        "Dataset 配方须提供无内嵌凭据的 Studio 工作流，绑定 "
                        "source_batch 以及每轮独立的训练和验证产物目录。",
                    ),
                ),
            )
        return DatasetRecipeExtraction(runtime=runtime, assets=assets)

    if not commands:
        return DatasetRecipeExtraction(
            runtime=None,
            blockers=(
                _dataset_blocker(
                    "dataset_recipe_required",
                    "研发回执缺少从已确认输入到训练与验证 JSONL 的执行契约。"
                    "请提供已审阅的 Dataset 配方。",
                ),
            ),
        )

    blockers = [
        _dataset_blocker(
            "dataset_raw_input_binding_missing",
            "历史任务读取已有预标注输入，缺少从本轮 source_batch 生成输入的步骤。",
        ),
        _dataset_blocker(
            "dataset_label_confirmation_missing",
            "历史任务只证明产出了结构化标注结果。配方需绑定确认后的标签来源，"
            "或使用项目已审批且能执行的标签策略。",
        ),
        _dataset_blocker(
            "dataset_sft_export_missing",
            "历史预标注工作流缺少将已确认标签转换为本轮训练和验证 JSONL 的步骤。"
            "请提供已审阅的 Dataset 配方。",
        ),
    ]
    if any(_DATAFLOW_GATEWAY.search(command) for command in commands):
        blockers.append(
            _dataset_blocker(
                "dataset_labeling_gateway_unbound",
                "历史任务的标注网关由提交工具临时注入。需要已授权的运行时标注能力，"
                "或直接使用已确认的标签输入。",
            )
        )
    return DatasetRecipeExtraction(
        runtime=None, assets=assets, blockers=tuple(blockers)
    )
