from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from pyromind_agent_server.echomind_consolidation import (
    ConsolidationBlocker,
    ConsolidationContext,
    ConsolidationDatasetReplay,
)
from pyromind_agent_server.echomind_dataset_consolidation import _validated_runtime
from pyromind_agent_server.echomind_sandbox_storage import (
    SandboxEvaluationStorage,
    _require_sandbox,
)
from pyromind_agent_server.echomind_storage import canonical_sha256
from pyromind_common.platform.scene_core import DATAFLOW_RUNTIME_FILENAMES
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.storage_paths import (
    normalize_workspace_path,
    to_storage_api_path,
)
from pyromind_runtime.domain.workflow_execution import (
    DatasetRecipeReceipt,
    WorkflowExecutionRequest,
)
from pyromind_runtime.ports.storage import (
    EvaluationStorage,
    EvaluationStorageMissingError,
    EvaluationStorageUnavailableError,
)


_VERIFIER = (
    Path(__file__).resolve().parents[2]
    / ".agents/skills/data-processing/scripts/preparation/verify_training_splits.py"
)
Digest = Annotated[str, Field(pattern="^[0-9a-f]{64}$")]


class _Verification(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    source_kind: Literal["image_directory", "record_directory", "jsonl"]
    source_sha256: Digest
    train_identity_sha256: Digest
    train_sha256: Digest
    val_sha256: Digest
    train_rows: int = Field(gt=0)
    val_rows: int = Field(ge=0)


class _LiveVerification(_Verification):
    approved_source_member: bool


@dataclass(frozen=True)
class _Sample:
    task_id: str
    recipe: DatasetRecipeReceipt
    summary: _LiveVerification


@dataclass(frozen=True)
class DatasetReplayResult:
    runtime: JsonObject | None = None
    proof: ConsolidationDatasetReplay | None = None
    blocker: ConsolidationBlocker | None = None


def _blocked(code: str, message: str) -> DatasetReplayResult:
    return DatasetReplayResult(
        blocker=ConsolidationBlocker(stage="dataset", code=code, message=message)
    )


def _candidate(
    receipt: WorkflowExecutionRequest, context: ConsolidationContext
) -> JsonObject:
    recipe = receipt.dataset_recipe
    assert recipe is not None and receipt.task_id
    nodes = receipt.workflow.get("nodes")
    if (
        not isinstance(nodes, list)
        or len(nodes) != 1
        or receipt.workflow.get("edges") != []
    ):
        raise ValueError("Unsupported Dataset workflow")
    node = nodes[0]
    data = node.get("data") if isinstance(node, dict) else None
    if not isinstance(data, dict) or data.get("nodeType") != "CustomCommandCPUNode":
        raise ValueError("Dataset requires a CPU command node")
    config = data.get("config")
    if not isinstance(config, dict):
        raise ValueError("Invalid CPU node")
    parameters = {k: v for k, v in config.items() if k not in {"command", "param"}}
    parameters["command"] = ""
    if recipe.uses_labeling_gateway:
        parameters["param"] = "DF_API_KEY"
    runtime: JsonObject = {
        "workflow": {
            "nodes": [
                {
                    "id": "1",
                    "type": "default",
                    "position": {"x": 0, "y": 0},
                    "data": {"nodeType": "CustomCommandCPUNode", "config": parameters},
                }
            ],
            "edges": [],
        },
        "command_binding_path": "nodes[0].data.config.command",
        "command_template": recipe.command_template,
        "output_uri": f"/workspace/.pyromind-agent/{context.evidence.conversation_id}"
        "/echomind-runs/{graph_run_id}/{stage_run_id}/dataset",
    }
    if recipe.uses_labeling_gateway:
        runtime["dataflow_credential_task_id"] = receipt.task_id
    return _validated_runtime(runtime, context.source_batch)


_MAX_DECODE_IMAGES = 64
_MAX_IMAGE_BYTES = 4 * 1024 * 1024
_MAX_DECODE_BYTES = 12 * 1024 * 1024


def _finish_live_verification(output: str) -> bytes:
    # Account for base64 padding per file plus bounded path/receipt metadata.
    if len(output) > 4 * _MAX_DECODE_BYTES // 3 + _MAX_DECODE_IMAGES * 8192:
        raise ValueError("换批小样超出图片复核大小限制")
    document = json.loads(output)
    if not isinstance(document, dict):
        raise ValueError("Invalid Dataset verification response")
    proof = document.pop("deferred_media", {})
    if not isinstance(proof, dict) or len(proof) > _MAX_DECODE_IMAGES:
        raise ValueError("Invalid Dataset media proof")
    total = 0
    for sha, entry in proof.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Invalid Dataset media path")
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("Replay media is outside Storage")
        encoded = entry.get("data")
        if (
            not isinstance(encoded, str)
            or len(encoded) > ((_MAX_IMAGE_BYTES + 2) // 3) * 4
        ):
            raise ValueError("换批小样中的图片超出复核范围")
        raw = base64.b64decode(encoded, validate=True)
        total += len(raw)
        if len(raw) > _MAX_IMAGE_BYTES or total > _MAX_DECODE_BYTES:
            raise ValueError("换批小样超出图片复核大小限制")
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("换批小样图片摘要不匹配")
        try:
            with Image.open(BytesIO(raw)) as image:
                if image.width * image.height > 32_000_000:
                    raise ValueError("换批小样超出图片解码尺寸限制")
                image.load()
        except (Image.DecompressionBombError, OSError, SyntaxError) as exc:
            raise ValueError("换批小样图片无法解码") from exc
    return _LiveVerification.model_validate(document).model_dump_json().encode()


def recheck_dataset_replay(
    storage: EvaluationStorage, recipe: DatasetRecipeReceipt, *, approved_source: str
) -> bytes:
    if not isinstance(storage, SandboxEvaluationStorage):
        raise EvaluationStorageUnavailableError(
            "换批复核需要已绑定的 Storage Sandbox。"
        )
    config = storage.binding.config

    def path(uri: str) -> str:
        return config.mount_root + to_storage_api_path(uri)

    command = [
        "sh",
        "-c",
        "for py in /opt/pyromind/dataflow-venv/bin/python "
        "/home/openhands/dataflow-venv/bin/python "
        "/tmp/pyromind-dataflow-venv-1.0.13-cpu/bin/python "
        'python3; do if "$py" -I -c "from PIL import Image" >/dev/null 2>&1; '
        'then exec "$py" "$@"; fi; done; exec python3 "$@"',
        "verify-dataset",
        "-I",
        "-c",
        _VERIFIER.read_text(),
        "--read-only",
        "--defer-image-decode",
        str(_MAX_DECODE_IMAGES),
        str(_MAX_IMAGE_BYTES),
        str(_MAX_DECODE_BYTES),
        "--mount-root",
        config.mount_root,
        "--approved-source",
        path(approved_source),
        "--source-batch",
        path(recipe.source_path),
        "--output-dir",
        path(recipe.output_directory),
    ]
    if recipe.fixed_validation_path:
        command.extend(["--fixed-validation", path(recipe.fixed_validation_path)])
    try:
        _require_sandbox(storage.client, config, require_writable=False)
        response = storage.client.exec_command(config.sandbox_id, command, timeout=180)
    except Exception:
        raise EvaluationStorageUnavailableError(
            "暂时无法从绑定的 Sandbox 复核 Dataset 输入和导出文件。"
        ) from None
    if response.returncode != 0:
        if any(
            marker in (response.stderr or response.output)
            for marker in (
                "Replay image exceeds the file limit",
                "Replay images exceed the batch limit",
            )
        ):
            raise ValueError("换批图片小样超过回放上限，请选更小的样本批次")
        raise ValueError("复核失败，请检查小样、训练文件及验证集")
    return _finish_live_verification(response.output)


def _summary(
    recipe: DatasetRecipeReceipt,
    context: ConsolidationContext,
    read: Callable[[str], bytes],
    check: Callable[[DatasetRecipeReceipt], bytes],
) -> _LiveVerification:
    output = normalize_workspace_path(recipe.output_directory)
    verifier = [asset for asset in recipe.assets if asset.role == "verifier"]
    if (
        len(verifier) != 1
        or verifier[0].sha256 != hashlib.sha256(_VERIFIER.read_bytes()).hexdigest()
    ):
        raise ValueError("Dataset verification implementation changed")
    raw = read(f"{output}/verification.json")
    if len(raw) > 4096:
        raise ValueError("Unexpected validation artifact")
    result = _Verification.model_validate_json(raw)
    if context.train_uses_validation != bool(result.val_rows) or (
        not context.train_uses_validation and recipe.fixed_validation_path
    ):
        raise ValueError(
            "Replay validation rows do not match the approved Train inputs"
        )
    live = _LiveVerification.model_validate_json(check(recipe))
    if live.model_dump(exclude={"approved_source_member"}) != result.model_dump():
        raise ValueError("Original source or exported data changed since verification")
    return live


def _replay_key(
    recipe: DatasetRecipeReceipt,
    context: ConsolidationContext,
    read: Callable[[str], bytes],
    runtime: JsonObject,
) -> str:
    root = PurePosixPath("/workspace/.pyromind-agent", context.evidence.conversation_id)
    legacy = root / "data_preparation"
    output = PurePosixPath(normalize_workspace_path(recipe.output_directory))
    if not any(
        base in output.parents
        for base in (legacy, root / "public_data" / "data-preparation")
    ):
        raise ValueError("Dataset replay output is outside this conversation's staging")
    roles = {asset.role for asset in recipe.assets}
    if not {"pipeline", "export", "verifier"} <= roles or len(roles) != len(
        recipe.assets
    ):
        raise ValueError("Incomplete Dataset runtime assets")
    command = recipe.command_template
    normalized: dict[str, str] = {}

    def register(path: PurePosixPath, identity: str) -> None:
        normalized[str(path)] = identity
        normalized["/target-workspace" + to_storage_api_path(str(path))] = identity

    for asset in recipe.assets:
        path = PurePosixPath(normalize_workspace_path(asset.path))
        if path.parent not in (
            legacy / "recipe-assets" / asset.sha256,
            output / "recipe-assets" / asset.sha256,
        ):
            raise ValueError("Recipe assets must be pinned in the source conversation")
        if hashlib.sha256(read(path.as_posix())).hexdigest() != asset.sha256:
            raise ValueError("Frozen Dataset asset has changed")
        register(path, f"asset:{asset.role}:{asset.sha256}")
    # SDK staging keeps a copy of the DataFlow support bundle per run.
    for dirname in set(
        re.findall(
            re.escape(to_storage_api_path(str(output)))
            + r"/(runtime(?:-r[0-9]+)?)(?=[/'\"\s;]|$)",
            command,
        )
    ):
        directory = output / dirname
        digest = hashlib.sha256()
        for name in sorted(DATAFLOW_RUNTIME_FILENAMES):
            digest.update(name.encode() + b"\0" + read(str(directory / name)) + b"\0")
        fingerprint = digest.hexdigest()
        if f"DF_RUNTIME_FINGERPRINT={fingerprint}" not in command:
            raise ValueError("Frozen DataFlow runtime fingerprint has changed")
        register(directory, f"runtime:{fingerprint}")
        for name in DATAFLOW_RUNTIME_FILENAMES:
            register(directory / name, f"runtime:{fingerprint}/{name}")
    # A directory prefix alone must never normalize an unverified file inside it.
    pattern = "|".join(
        re.escape(path) for path in sorted(normalized, key=len, reverse=True)
    )
    command = re.sub(
        rf"(?<![\w/.-])(?:{pattern})(?![\w/.-])",
        lambda match: "\0" + normalized[match[0]] + "\0",
        command,
    )
    return canonical_sha256(
        {
            "runtime": {
                **{
                    k: v
                    for k, v in runtime.items()
                    if k != "dataflow_credential_task_id"
                },
                "command_template": command,
            },
            "assets": sorted((a.role, a.sha256) for a in recipe.assets),
            "fixed_validation": recipe.fixed_validation_path,
            "batch_label_source": recipe.batch_label_source,
        }
    )


def verify_dataset_replays(
    context: ConsolidationContext,
    receipts: list[WorkflowExecutionRequest],
    read: Callable[[str], bytes] | None,
    check: Callable[[DatasetRecipeReceipt], bytes] | None,
    *,
    train_fields: Mapping[str, str] | None = None,
) -> DatasetReplayResult:
    if not receipts:
        return DatasetReplayResult()
    if read is None or check is None:
        return _blocked(
            "dataset_replay_storage_unavailable",
            "读取换批验证需要当前用户的 Storage 访问权限。",
        )
    latest: _Sample | None = None
    latest_runtime: JsonObject = {}
    latest_key = ""
    mismatches = False
    for receipt in reversed(receipts):
        recipe = receipt.dataset_recipe
        assert recipe is not None and receipt.task_id
        try:
            runtime = _candidate(receipt, context)
            key = _replay_key(recipe, context, read, runtime)
            if latest and (
                key != latest_key
                or normalize_workspace_path(latest.recipe.source_path)
                == normalize_workspace_path(recipe.source_path)
                or latest.recipe.output_directory == recipe.output_directory
            ):
                continue
            proof = _summary(recipe, context, read, check)
            if train_fields and "image_field" in train_fields:
                splits = [("train", proof.train_sha256)]
                if proof.val_rows:
                    splits.append(("val", proof.val_sha256))
                for split, sha in splits:
                    issue = _vision_export_issue(
                        read(f"{recipe.output_directory}/{split}_messages.jsonl"),
                        sha,
                        train_fields,
                        split=split,
                    )
                    if issue:
                        return _blocked(
                            "dataset_train_fields_mismatch",
                            f"小样任务 {receipt.task_id} 的 {issue}。"
                            f"请按 Train {context.evidence.train.task_id} 的字段修正 "
                            "Dataset 导出，并用修正后的配方复核两批小样。",
                        )
        except EvaluationStorageUnavailableError:
            return _blocked(
                "dataset_replay_storage_unavailable",
                "读取换批验证产物暂不可用，请恢复当前用户的 Storage 访问。",
            )
        except (ValueError, EvaluationStorageMissingError, FileNotFoundError):
            if not latest:
                return _blocked(
                    "dataset_replay_invalid",
                    "最新成功任务的源数据、导出文件或校验脚本与保存的回放结果不匹配。",
                )
            mismatches = True
            continue
        if latest is None:
            latest = _Sample(receipt.task_id, recipe, proof)
            latest_key, latest_runtime = key, runtime
            continue
        if (
            not (latest.summary.approved_source_member or proof.approved_source_member)
            or latest.summary.source_sha256 == proof.source_sha256
            or latest.summary.train_identity_sha256 == proof.train_identity_sha256
            or (
                recipe.fixed_validation_path
                and latest.summary.val_sha256 != proof.val_sha256
            )
        ):
            continue
        if recipe.fixed_validation_path:
            mounted = "/target-workspace" + to_storage_api_path(
                recipe.fixed_validation_path
            )
            # Shell quoting happens at compilation; braces in filenames still
            # pass through the workflow's later format_map expansion.
            path = shlex.quote(mounted).replace("{", "{{").replace("}", "}}")
            latest_runtime["command_template"] = (
                f"{latest_runtime['command_template']}\n"
                f'test "$(sha256sum {path} | cut -c1-64)" = {proof.val_sha256}'
            )
        return DatasetReplayResult(
            runtime=latest_runtime,
            proof=ConsolidationDatasetReplay(
                batch_label_source=recipe.batch_label_source,
                batch_validation_source=(
                    "none"
                    if not context.train_uses_validation
                    else "fixed"
                    if recipe.fixed_validation_path
                    else "batch"
                ),
                fixed_validation_path=recipe.fixed_validation_path,
                dataset_runtime_sha256=canonical_sha256(latest_runtime),
                task_ids=(receipt.task_id, latest.task_id),
                source_kinds=(proof.source_kind, latest.summary.source_kind),
                source_fingerprints=(proof.source_sha256, latest.summary.source_sha256),
                train_fingerprints=(
                    proof.train_identity_sha256,
                    latest.summary.train_identity_sha256,
                ),
                train_rows=(proof.train_rows, latest.summary.train_rows),
                val_rows=(proof.val_rows, latest.summary.val_rows),
            ),
        )
    assert latest is not None
    message = ("另一次回放产物校验失败。" if mismatches else "") + (
        "请用相同配方及不同的小批输入重放。"
        "至少一批须能按图片相对路径和内容，或原始 JSONL 记录，"
        f"追溯到 Project 已确认的源数据。已核对平台任务 {latest.task_id}。"
    )
    return _blocked("dataset_replay_required", message)


def train_reader_fields(runtime: Mapping[str, object]) -> dict[str, str]:
    workflow = runtime.get("workflow")
    if not isinstance(workflow, dict):
        return {}
    nodes, edges = workflow.get("nodes"), workflow.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return {}
    graph = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            return {}
        identity, data = node["id"], node.get("data")
        if identity in graph or not isinstance(data, dict):
            return {}
        graph[identity] = data

    def upstream(target: str, port: str) -> str | None:
        incoming = [
            edge
            for edge in edges
            if isinstance(edge, dict)
            and edge.get("target") == target
            and edge.get("targetHandle") == port
        ]
        if len(incoming) != 1 or incoming[0].get("sourceHandle") != port:
            return None
        source = incoming[0].get("source")
        return source if isinstance(source, str) and source in graph else None

    trainers = [
        i for i, data in graph.items() if data.get("nodeType") == "ModelTrainSFTNode"
    ]
    if len(trainers) != 1:
        return {}
    dataset = upstream(trainers[0], "dataset_config")
    if not dataset or graph[dataset].get("nodeType") != "DatasetConfigBuilderNode":
        return {}
    reader = upstream(dataset, "dataset_kind_config")
    if not reader:
        return {}
    config = graph[reader].get("config")
    kind = graph[reader].get("nodeType")
    if not isinstance(config, dict):
        return {}
    if (
        kind == "DatasetConfigBuilderMessageNode"
        and config.get("messages_field") == "messages"
    ):
        return {"messages_field": "messages"}
    if kind != "DatasetConfigBuilderVisionNode":
        return {}
    roles = ["image_field", "user_prompt_field", "assistant_response_field"]
    if config.get("system_prompt_field"):
        roles.append("system_prompt_field")
    fields = {
        key: config[key]
        for key in roles
        if isinstance(config.get(key), str)
        and re.fullmatch(r"[\w.-]{1,128}", config[key])
    }
    return fields if len(set(fields.values())) == len(roles) else {}


def compatible_train_reader(runtime: Mapping[str, object]) -> bool:
    """Whether a verified messages export alone meets the Train reader contract."""
    return train_reader_fields(runtime) == {"messages_field": "messages"}


def _vision_export_issue(
    raw: bytes, expected_sha: str, fields: Mapping[str, str], *, split: str
) -> str | None:
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Dataset export changed after verification")
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Expected a verified SFT record")
        invalid: list[str] = []
        for role, name in fields.items():
            value = row.get(name)
            if role == "image_field":
                valid = (
                    isinstance(value, list)
                    and bool(value)
                    and value == row.get("images")
                )
            else:
                valid = isinstance(value, str) and (
                    role == "system_prompt_field" or bool(value.strip())
                )
            if not valid:
                invalid.append(name)
        if invalid:
            return (
                f"{split}_messages.jsonl 第 {line_number} 行缺少或无法读取 "
                + "、".join(invalid)
            )
    return None
