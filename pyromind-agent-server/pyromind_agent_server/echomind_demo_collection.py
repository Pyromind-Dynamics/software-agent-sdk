"""Prepare a five-sample Shanghai PCB collection demonstration."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID

import httpx
from pydantic import BaseModel, Field, field_validator

from pyromind_agent_server.echomind_acquisition import (
    AcquisitionSource,
    AcquisitionSplit,
)
from pyromind_agent_server.echomind_acquisition_runtime import (
    TracerBatchMaterializer,
    _assistant_text,
    _extract_json_result,
)
from pyromind_agent_server.echomind_projects import ProjectInfo
from pyromind_agent_server.echomind_sandbox_storage import (
    SandboxEvaluationStorage,
    _require_sandbox,
)
from pyromind_common.platform.pcb_collection import (
    PCB_REPORT_IMAGES,
    SELECT_PCB_TRIPLETS,
)
from pyromind_runtime.domain.storage_paths import (
    normalize_workspace_path,
    to_storage_api_path,
)
from pyromind_runtime.ports.storage import (
    EvaluationStorage,
    EvaluationStorageMissingError,
)


_FILES = PCB_REPORT_IMAGES
_SELECT = SELECT_PCB_TRIPLETS
_SEED_DIGESTS = (
    "cf3de1d0ee55973f608fb462f3b40fad9e7ee5452dc7a1cb5e3aeef38e37c77a",
    "2bc79ff75ab905346b0e7983d4ec03e90f0f3d77d1e5b19e65c2232b8e13078d",
)
# Ordered media digests of the five labeled triplets used by Train task 1102.
_SEED_MEDIA = (
    "f1ea983d7ff97ddfc8cad20b20bc02018c33878d8886b4c75cc4f4f9e036d6be",
    "44b591e4f081acead01bb3784fb402dbc40f574990949bf4ce5f34e92bb83f75",
    "f31fd646f5d3fdaddb1fc896861f5f2c2c71849ca419ada46b9532abb09c3611",
    "4d58556de07750204046c00a35d6905ed8879641f78ca9d75bf436f39e76ea7e",
    "128a322fbfa2bd23f3e46eb1c53f6ea0000e5908dfc48ad83a2c97a4c63876a5",
)


class PcbCollectionConfig(BaseModel):
    release_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    image_directory: str = Field(max_length=4000)
    system_prompt: str = Field(min_length=1, max_length=32000)
    user_prompt: str = Field(min_length=1, max_length=32000)

    @field_validator("image_directory")
    @classmethod
    def dataset_directory(cls, value: str) -> str:
        directory = normalize_workspace_path(value.rstrip("/"))
        if not directory.startswith("/workspace/datasets/"):
            raise ValueError("请选择 Storage datasets 下的图片目录")
        return directory


def prepare_pcb_source(
    storage: EvaluationStorage,
    source: AcquisitionSource,
    selected: list[dict[str, Any]],
    config: PcbCollectionConfig,
) -> None:
    if len(selected) != 5:
        raise ValueError("需选取 5 组完整的 PCB 图片")
    rows = [
        {
            **candidate,
            "user_prompt": config.user_prompt,
            "system_prompt": config.system_prompt,
            "label_source": "dataset",
        }
        for candidate in selected
    ]
    TracerBatchMaterializer(storage).write(
        directory=str(PurePosixPath(source.train_uri).parent),
        rows={"train": rows, "val": []},
    )


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def demo_source_directory(project: ProjectInfo, request_id: str) -> str:
    conversation = UUID(project.project.conversation_id).hex
    return (
        f"/workspace/.pyromind-agent/{conversation}/project-batches/demo-collection/"
        + _digest(request_id.encode())
    )


def demo_expected_source(
    project: ProjectInfo, request_id: str, raw_directory: str
) -> AcquisitionSource:
    directory = demo_source_directory(project, request_id)
    return AcquisitionSource(
        image_root=normalize_workspace_path(raw_directory),
        train_uri=directory + "/train.jsonl",
        val_uri=directory + "/val.jsonl",
    )


def make_inference_labeler(
    endpoint: str, key: str, model: str
) -> Callable[[list[dict[str, Any]]], str]:
    base = endpoint.rstrip("/").removesuffix("/v1")

    def label(messages: list[dict[str, Any]]) -> str:
        try:
            response = httpx.post(
                base + "/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": messages, "temperature": 0},
                timeout=300,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Inference 预标注返回了无效结果")
            return _assistant_text(payload)
        except httpx.HTTPStatusError as exc:
            raise ValueError(
                f"Inference 预标注失败（HTTP {exc.response.status_code}）"
            ) from None
        except (httpx.HTTPError, ValueError):
            raise ValueError("Inference 预标注未返回可读的结果") from None

    return label


def _seed_rows(
    storage: SandboxEvaluationStorage, source: AcquisitionSource
) -> dict[str, dict[str, Any]]:
    seeds = {}
    for split, path, expected in zip(
        ("train", "val"), (source.train_uri, source.val_uri), _SEED_DIGESTS, strict=True
    ):
        data = storage.read(path)
        if _digest(data) != expected:
            raise ValueError("演示的已核验标注清单发生变化")
        for line in data.splitlines():
            row = json.loads(line)
            seeds[row["id"]] = {**row, "reference_split": split}
    for row, digest in zip(seeds.values(), _SEED_MEDIA, strict=True):
        row["triplet_sha256"] = digest
    return seeds


def _select(
    project: ProjectInfo,
    storage: SandboxEvaluationStorage,
    source: AcquisitionSource,
    seeds: dict[str, dict[str, Any]],
    offset: int,
    fixed_validation: bool = False,
) -> list[dict[str, Any]]:
    binding = storage.binding
    if (
        UUID(binding.project_id) != UUID(project.project.project_id)
        or binding.cluster != project.project.cluster
        or binding.owner_user_id != project.project.owner_user_id
    ):
        raise ValueError("Storage 所属项目、集群或用户不匹配")
    config = binding.config
    _require_sandbox(storage.client, config)
    template = next(iter(seeds.values()))
    seed_root = str(PurePosixPath(template["images"][0]).parents[1])
    heldout = (
        next(n for n in project.graph.nodes if n.type == "evaluator")
        .inputs["validation_dataset"]
        .binding
    )
    if heldout.kind != "literal" or not isinstance(heldout.value, str):
        raise ValueError("演示需要 Project 固定的 Eval 数据集")
    response = storage.client.exec_command(
        config.sandbox_id,
        [
            "python3",
            "-c",
            _SELECT,
            config.mount_root,
            to_storage_api_path(source.image_root),
            to_storage_api_path(seed_root),
            json.dumps(list(seeds)),
            str(offset),
            json.dumps(_FILES),
            to_storage_api_path(heldout.value),
            json.dumps(
                [
                    {k: row[k] for k in ("id", "images", "triplet_sha256")}
                    for row in seeds.values()
                    if row["reference_split"] == "val"
                ]
                if fixed_validation
                else []
            ),
        ],
        timeout=150,
    )
    if response.returncode:
        if "overlaps the fixed Eval set" in response.output:
            raise ValueError("所选图片与固定 Eval 数据集重叠，请更换范围")
        raise ValueError(
            "请检查图片目录、选择范围、三张配套图片以及固定 Eval 数据集的可读性"
        )
    rows = json.loads(response.output)
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("所选区间没有返回 5 组三图")
    return rows


def _predict(
    storage: SandboxEvaluationStorage,
    row: dict[str, Any],
    prompts: dict[str, str],
    infer: Callable[[list[dict[str, Any]]], str],
) -> str:
    content: list[dict[str, Any]] = []
    for path, expected in zip(row["images"], row["image_sha256"], strict=True):
        image = storage.read(path)
        if not image.startswith(b"\xff\xd8") or _digest(image) != expected:
            raise ValueError("选中的 JPEG 在预标注前发生了变化或图片不可读")
        url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": prompts["user_prompt"]})
    answer = infer(
        [
            {"role": "system", "content": prompts["system_prompt"]},
            {"role": "user", "content": content},
        ]
    )
    label = _extract_json_result(answer.rsplit("</think>", 1)[-1])
    if (
        not label
        or label.get("result") not in ("真点", "假点")
        or not isinstance(label.get("category"), str)
        or not label["category"].strip()
    ):
        raise ValueError(
            f"样本 {row['id']} 的 Inference 结果缺少有效的真点/假点或 category"
        )
    return json.dumps(label, ensure_ascii=False, allow_nan=False)


def prepare_demo_source(
    *,
    project: ProjectInfo,
    storage: SandboxEvaluationStorage,
    seed_source: AcquisitionSource,
    raw_directory: str,
    offset: int,
    limit: int,
    request_id: str,
    model: str | None,
    infer: Callable[[list[dict[str, Any]]], str] | None,
    fixed_validation: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> AcquisitionSource:
    if offset < 0 or limit != 5:
        raise ValueError("起始位置需大于等于零，每轮固定选取 5 条")
    source = demo_expected_source(project, request_id, raw_directory)
    directory = demo_source_directory(project, request_id)
    seeds = _seed_rows(storage, seed_source)
    selected = _select(project, storage, source, seeds, offset, fixed_validation)
    identity = {
        "raw_directory": source.image_root,
        "offset": offset,
        "limit": limit,
        "model": model,
        "selection": selected,
    }
    if fixed_validation:
        identity["validation"] = "fixed"
    receipt = directory + "/selection.json"
    try:
        saved = json.loads(storage.read(receipt))
    except EvaluationStorageMissingError:
        saved = None
    if saved is not None:
        if saved["request"] != identity:
            raise ValueError("同一请求选中的图片、范围或模型已变更")
        rows = saved["rows"]
    else:
        rows = []
        for candidate in selected:
            name = candidate["id"]
            seed = seeds.get(name)
            if seed is not None and seed["triplet_sha256"] != _digest(
                "".join(candidate["image_sha256"]).encode()
            ):
                seed = None
            example = seed or next(iter(seeds.values()))
            prompts = {key: example[key] for key in ("system_prompt", "user_prompt")}
            row = {**candidate, **prompts, "label_source": "dataset"}
            if infer is not None:
                gt = (
                    seed["gt"] if seed else _predict(storage, candidate, prompts, infer)
                )
                row.update(
                    gt=gt,
                    label_source="archived_training_label"
                    if seed
                    else "inference_pseudo_label",
                    label_model=None if seed else model,
                )
            rows.append(row)
            if progress:
                progress(len(rows), limit)
        storage.write(
            receipt,
            json.dumps(
                {"request": identity, "rows": rows}, ensure_ascii=False
            ).encode(),
        )
    splits: dict[AcquisitionSplit, list[dict[str, Any]]] = {
        "train": rows if fixed_validation else rows[:3],
        "val": [] if fixed_validation else rows[3:],
    }
    TracerBatchMaterializer(storage).write(directory=directory, rows=splits)
    return source
