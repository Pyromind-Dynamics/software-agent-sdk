"""Resolve immutable scoring inputs for a signed SDK inference task."""

import json
import re
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from uuid import UUID

from pyromind_common.platform.evaluation import staged_evaluation_digest
from pyromind_common.platform.evaluation_core import INFERENCE_RUNTIME_FILENAMES
from pyromind_common.platform.scene_core import runtime_bundle_fingerprint
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.pipeline import pipeline_definition_fingerprint
from pyromind_runtime.domain.storage_paths import from_storage_api_path
from pyromind_runtime.domain.task_correlation import ProjectTaskReference
from pyromind_runtime.domain.versions import fingerprint
from pyromind_runtime.domain.workflow_execution import WorkflowExecutionRequest
from pyromind_runtime.infrastructure.file_product_store import FileProductStore


def sdk_evaluation_assets(
    store: FileProductStore,
    receipt: WorkflowExecutionRequest,
    arguments: dict[str, str],
    runtime_dir: Path,
    read: Callable[[str], bytes] | None,
) -> tuple[JsonObject, str, JsonObject]:
    correlation = receipt.correlation
    snapshot = store.load_snapshot()
    if (
        not correlation
        or correlation.type != "main"
        or correlation.task_kind not in {"workflow_run", "evaluation"}
        or not correlation.execution
        or correlation.conversation_id != snapshot.conversation_id
        or receipt.request_id != f"sdk:{receipt.task_id}"
        or not store.verify_task_correlation(correlation)
        or not any(
            task.task_id == receipt.task_id
            and task.correlation == correlation
            and task.status == "succeeded"
            for task in snapshot.external_tasks
        )
    ):
        raise ValueError("SDK 评测工作流缺少同一成功任务的签名回执。")
    root = from_storage_api_path(arguments["--output-dir"])
    run_id = arguments["--run-id"]
    path = PurePosixPath(root)
    if (
        path.name != str(UUID(run_id))
        or not path.is_relative_to(
            f"/workspace/.pyromind-agent/{snapshot.conversation_id}"
        )
        or any(
            from_storage_api_path(arguments[key]) != f"{root}/{filename}"
            for key, filename in (
                ("script", "evaluate_inference.py"),
                ("--dataset-config", "dataset_config.json"),
                ("--evaluation-config", "evaluation_config.json"),
            )
        )
    ):
        raise ValueError("SDK 评测目录、评分脚本或配置与本次运行不匹配。")
    definition = _registered_definition(
        store, receipt, correlation.execution, root, run_id
    )
    worker_digest = None
    if definition is None:
        if read is None:
            raise ValueError("缺少已关联的评测定义，请登录 Project 读取 SDK 评测产物。")
        definition, worker_digest = _read_verified_definition(
            read, root, run_id, arguments
        )
    inference = definition.get("inference")
    if (
        definition.get("run_id") != run_id
        or not isinstance(inference, dict)
        or not isinstance(model := inference.get("model_path"), str)
        or from_storage_api_path(model)
        != from_storage_api_path(arguments["--model-reference"])
        or definition.get("report_spec")
        or definition.get("business_scoring")
        or definition.get("frozen_reward")
    ):
        raise ValueError("SDK 评测运行、模型或附加评分资产与工作流不匹配。")
    expected = definition.get("runtime_sha256")
    digests = _bundle_digests(runtime_dir, expected, worker_digest)
    if digests is None:
        if read is None:
            raise ValueError(
                "本机评分包与旧版不同，请登录 Project 以核验原运行目录中的评分文件。"
            )
        if worker_digest is None:
            manifest = _read_document(read, root, "run_manifest.json")
            worker_digest = _manifest_digest(manifest, definition, run_id, arguments)
        # Keep the same streaming digest algorithms used by the stager and worker.
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in INFERENCE_RUNTIME_FILENAMES:
                try:
                    data = read(f"{root}/{name}")
                except Exception as exc:
                    raise ValueError(f"无法读取原评分资源 {name}。") from exc
                (directory / name).write_bytes(data)
            digests = _bundle_digests(directory, expected, worker_digest)
        if digests is None:
            raise ValueError(
                "Storage 中的评分脚本或报告资源与原运行定义的 SHA-256 不匹配。"
            )

    assets: JsonObject = {
        "runtime_script" if name == "evaluate_inference.py" else name: {
            "path": f"{root}/{name}",
            "sha256": digest,
        }
        for name, digest in digests.items()
    }
    for role in ("dataset_config", "evaluation_config"):
        value = definition.get(role)
        if not isinstance(value, dict):
            raise ValueError(f"SDK 评测缺少本次运行的 {role}。")
        content = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        assets[role] = {
            "path": f"{root}/{role}.json",
            "sha256": sha256(content).hexdigest(),
        }
    return definition, root, assets


def _registered_definition(
    store: FileProductStore,
    receipt: WorkflowExecutionRequest,
    scope: ProjectTaskReference,
    root: str,
    run_id: str,
) -> JsonObject | None:
    matches = [
        run
        for run in store.load_pipeline_runs().values()
        if run.task_id == receipt.task_id
        and run.conversation_id == receipt.conversation_id
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("同一 SDK 评测任务关联了多个运行定义。")
    run = matches[0]
    result = run.result
    source = result.get("receipt_source")
    if not isinstance(source, dict):
        return None
    identity = {
        "request_id": receipt.request_id,
        "external_task_id": receipt.task_id,
        "workflow_path": receipt.workflow_path,
        "fingerprint": receipt.fingerprint,
        "project_id": scope.project_id,
        "plan_sequence": scope.plan_sequence,
        "plan_task_id": scope.task_id,
        "attempt": scope.attempt,
    }
    paths = {
        "definition_path": "evaluation-definition.json",
        "report_path": "report.json",
        "metrics_path": "metrics.json",
    }
    if (
        str(run.run_id) != run_id
        or pipeline_definition_fingerprint(run.definition) != run.fingerprint
        or from_storage_api_path(run.output_dir) != root
        or run.submission_status != "submitted"
        or result.get("status") != "succeeded"
        or result.get("evaluation_status") != "validated"
        or result.get("report_status") != "ready"
        or any(source.get(key) != value for key, value in identity.items())
        or any(
            not isinstance(value := source.get(key), str)
            or from_storage_api_path(value) != f"{root}/{name}"
            for key, name in paths.items()
        )
    ):
        raise ValueError("SDK 评测定义、报告或指标的回执与选定成功任务不一致。")
    return run.definition


def _bundle_digests(
    directory: Path,
    expected: object,
    worker_digest: str | None,
) -> dict[str, str] | None:
    files = {name: directory / name for name in INFERENCE_RUNTIME_FILENAMES}
    if (
        not all(path.is_file() for path in files.values())
        or staged_evaluation_digest(files) != expected
    ):
        return None
    if (
        worker_digest
        and runtime_bundle_fingerprint(directory, INFERENCE_RUNTIME_FILENAMES)
        != worker_digest
    ):
        raise ValueError("评分包与原评测运行清单记录的 SHA-256 不匹配。")
    return {name: sha256(path.read_bytes()).hexdigest() for name, path in files.items()}


def _read_document(read: Callable[[str], bytes], root: str, name: str) -> JsonObject:
    try:
        value = json.loads(read(f"{root}/{name}"))
    except Exception as exc:
        raise ValueError(
            f"无法读取 SDK 评测产物 {name}，请检查 Storage 登录态和产物路径。"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"SDK 评测产物 {name} 格式错误。")
    return value


def _manifest_digest(
    manifest: JsonObject, definition: JsonObject, run_id: str, arguments: dict[str, str]
) -> str:
    if (
        manifest.get("run_id") != run_id
        or not isinstance(model := manifest.get("model_reference"), str)
        or from_storage_api_path(model)
        != from_storage_api_path(arguments["--model-reference"])
    ):
        raise ValueError("SDK 评测运行清单与工作流的运行 ID 或模型不符。")
    if manifest.get("dataset_config") != definition.get(
        "dataset_config"
    ) or manifest.get("evaluation_config_sha256") != fingerprint(
        definition.get("evaluation_config")
    ):
        raise ValueError("SDK 评测数据映射或评分标准与运行清单不符。")
    digest = manifest.get("runtime_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("SDK 评测运行清单缺少评分包 SHA-256。")
    return digest


def _read_verified_definition(
    read: Callable[[str], bytes],
    root: str,
    run_id: str,
    arguments: dict[str, str],
) -> tuple[JsonObject, str]:
    definition = _read_document(read, root, "evaluation-definition.json")
    manifest = _read_document(read, root, "run_manifest.json")
    worker_digest = _manifest_digest(manifest, definition, run_id, arguments)
    report = _read_document(read, root, "report.json")
    metrics = _read_document(read, root, "metrics.json")
    artifacts = report.get("artifacts")
    if report.get("metrics") != metrics or not metrics:
        raise ValueError("SDK 评测报告与 metrics.json 的指标不一致。")
    if (
        not isinstance(artifacts, dict)
        or any(
            not isinstance(path := artifacts.get(name), str)
            or from_storage_api_path(path) != f"{root}/{name}"
            for name in ("metrics.json", "predictions.jsonl")
        )
        or report.get("status") != "succeeded"
        or report.get("evaluation_status") != "validated"
        or report.get("report_status") != "ready"
    ):
        raise ValueError(
            "SDK 评测报告尚未验证通过，或报告引用的产物与本次任务目录不符。"
        )
    return definition, worker_digest
