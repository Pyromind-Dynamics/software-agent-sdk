#!/usr/bin/env python3
"""Normalize verified PCB triplets and confirmed labels for the AVI Dataset stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_ROLES = ("待检实拍图", "差分图", "同位置参考图")
_FILENAMES = ("defect.jpg", "diff.jpg", "gt.jpg")
_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_BINARY_LABELS = {"真点", "假点"}
_TRACER_LABELS = {"pass", "fail", "defect"}


@dataclass(frozen=True)
class _Sample:
    id: str
    source_images: tuple[str, ...]
    local_images: tuple[Path, ...]
    system_prompt: str
    user_prompt: str
    answer: str
    label: str
    label_source: str


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    handle = path.open("x", encoding="utf-8")
    try:
        with handle:
            for row in rows:
                line = json.dumps(row, ensure_ascii=False, allow_nan=False)
                handle.write(line + "\n")
    except (OSError, ValueError):
        path.unlink()
        raise


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} is not JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number} is not an object")
            identity = _required_text(row.get("id"), f"{path}:{number} id")
            if identity in rows:
                raise ValueError(f"{path}:{number} duplicates sample {identity!r}")
            rows[identity] = row
    if not rows:
        raise ValueError(f"{path} contains no samples")
    return rows


def _require_triplet(paths: tuple[Path, ...], sample: str, root: Path) -> None:
    if (
        tuple(path.name for path in paths) != _FILENAMES
        or len({path.parent for path in paths}) != 1
    ):
        raise ValueError(f"sample {sample!r} must have one ordered, co-located triplet")
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or resolved != path:
            raise ValueError(f"sample {sample!r} has an unsafe image path: {path}")
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"sample {sample!r} is missing a usable image: {path}")


def write_prelabel_manifest(
    image_root: Path,
    output: Path,
    *,
    inspection_rules: str = "",
    stage_images: bool = False,
    storage_root: Path = Path("/workspace"),
) -> None:
    """Write input for the existing image prelabel pipeline without reading labels."""
    image_root = image_root.resolve(strict=True)
    manifest_root = output.parent.resolve(strict=True)
    if not image_root.is_dir():
        raise ValueError("the image root must be a directory")
    if stage_images:
        storage_root = storage_root.resolve(strict=True)
        if not (
            image_root.is_relative_to(storage_root)
            and manifest_root.is_relative_to(storage_root)
            and not manifest_root.is_relative_to(image_root)
        ):
            raise ValueError("stage into a run directory in Storage outside the images")
    elif not image_root.is_relative_to(manifest_root):
        raise ValueError("the manifest directory must contain the image root")
    rows: list[dict[str, Any]] = []
    for directory in sorted(image_root.iterdir()):
        if directory.name.startswith("."):
            continue
        if not directory.is_dir():
            if directory.suffix.lower() in _IMAGE_EXTENSIONS:
                raise ValueError(f"ungrouped image in the image root: {directory}")
            continue
        paths = tuple(directory / filename for filename in _FILENAMES)
        _require_triplet(paths, directory.name, image_root)
        for extra in directory.iterdir():
            if extra.is_dir() or (
                extra.suffix.lower() in _IMAGE_EXTENSIONS and extra not in paths
            ):
                raise ValueError(
                    f"unexpected image or directory in {directory}: {extra}"
                )
        row: dict[str, Any] = {
            "id": directory.name,
            "image_labels": list(_ROLES),
            "inspection_rules": inspection_rules,
        }
        if stage_images:
            row["images"] = [f"images/{directory.name}/{name}" for name in _FILENAMES]
            row["original_source_images"] = [
                (Path("/workspace") / path.relative_to(storage_root)).as_posix()
                for path in paths
            ]
        else:
            row["images"] = [
                path.relative_to(manifest_root).as_posix() for path in paths
            ]
        rows.append(row)
    if not rows:
        raise ValueError(f"{image_root} contains no PCB triplets")
    if stage_images:
        _write_staged_manifest(image_root, output, rows)
    else:
        _write_jsonl(output, rows)


def _write_staged_manifest(
    image_root: Path, output: Path, rows: list[dict[str, Any]]
) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    destination = output.parent / "images"
    destination.mkdir()
    try:
        for row in rows:
            directory = destination / row["id"]
            directory.mkdir()
            for name in _FILENAMES:
                shutil.copyfile(image_root / row["id"] / name, directory / name)
        _write_jsonl(output, rows)
    except (OSError, ValueError):
        shutil.rmtree(destination)
        raise


def _local_image(value: str, image_root: Path, storage_root: Path) -> Path:
    path = Path(value)
    if ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"invalid image reference: {value!r}")
    if not path.is_absolute():
        return image_root / path
    for mount in (Path("/workspace"), Path("/target-workspace")):
        if path.is_relative_to(mount):
            return storage_root / path.relative_to(mount)
    if path.is_relative_to(storage_root):
        return path
    raise ValueError(f"image reference is outside Storage: {value!r}")


def _source_images(row: dict[str, Any]) -> tuple[str, ...]:
    images = row.get("images")
    roles = row.get("image_labels")
    if roles is not None and roles != list(_ROLES):
        raise ValueError("image_labels must name the verified triplet roles in order")
    if not isinstance(images, list) or len(images) != len(_ROLES):
        raise ValueError("images must contain exactly three ordered image paths")
    return tuple(_required_text(value, "image path") for value in images)


def _original_images(identity: str, raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or len(raw) != len(_ROLES):
        raise ValueError("original_source_images must contain three Storage paths")
    images = tuple(_required_text(value, "original image path") for value in raw)
    paths = tuple(Path(value) for value in images)
    if (
        tuple(path.name for path in paths) != _FILENAMES
        or len({path.parent for path in paths}) != 1
        or any(
            not path.is_relative_to("/workspace")
            or ".." in path.parts
            or path.parent.name != identity
            for path in paths
        )
    ):
        raise ValueError("original_source_images must identify the ordered raw triplet")
    return images


def _answer_and_label(gt: Any) -> tuple[str, str]:
    if isinstance(gt, dict):
        answer = json.dumps(gt, ensure_ascii=False, allow_nan=False)
        value = gt
    else:
        answer = _required_text(gt, "gt")
        data = answer.rsplit("</think>", 1)[-1].strip()
        if data.startswith("<answer>") and data.endswith("</answer>"):
            data = data.removeprefix("<answer>").removesuffix("</answer>")
        try:
            value = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError("gt must contain a JSON annotation object") from exc
    label = value.get("result") if isinstance(value, dict) else None
    if not isinstance(label, str) or label not in _BINARY_LABELS | _TRACER_LABELS:
        raise ValueError("gt must declare a supported PCB result")
    return answer, label


def _samples(
    rows: dict[str, dict[str, Any]],
    *,
    image_root: Path,
    storage_root: Path,
    system_prompt: str,
    user_prompt: str,
    label_sources: dict[str, str],
) -> list[_Sample]:
    samples: list[_Sample] = []
    image_owners: dict[Path, str] = {}
    for identity, row in rows.items():
        try:
            if "confirmed" in row and row["confirmed"] is not True:
                raise ValueError("labeled input has not been confirmed")
            source = _source_images(row)
            local = tuple(
                _local_image(path, image_root, storage_root) for path in source
            )
            _require_triplet(local, identity, storage_root)
            for path in local:
                if path in image_owners:
                    raise ValueError(f"image also belongs to {image_owners[path]!r}")
                image_owners[path] = identity
            answer, label = _answer_and_label(row.get("gt"))
            samples.append(
                _Sample(
                    identity,
                    _original_images(identity, row["original_source_images"])
                    if identity in label_sources and "original_source_images" in row
                    else source,
                    local,
                    _required_text(
                        row.get("system_prompt", system_prompt), "system_prompt"
                    ),
                    _required_text(row.get("user_prompt", user_prompt), "user_prompt"),
                    answer,
                    label,
                    label_sources.get(identity, "labeled_input"),
                )
            )
        except ValueError as exc:
            raise ValueError(f"sample {identity!r}: {exc}") from exc
    return samples


def _prelabel_answer(row: dict[str, Any]) -> dict[str, Any]:
    label = row.get("result")
    if not isinstance(label, str) or label not in _BINARY_LABELS:
        raise ValueError("prelabel result must be 真点 or 假点")
    confidence = row.get("confidence")
    if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
        raise ValueError("prelabel confidence must be numeric and between 0 and 1")
    boxes = row.get("boxes")
    if not isinstance(boxes, list):
        raise ValueError("prelabel boxes must be a list")
    for box in boxes:
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(value) not in (int, float) for value in box)
            or not 0 <= box[0] < box[2] <= 1000
            or not 0 <= box[1] < box[3] <= 1000
        ):
            raise ValueError("prelabel boxes must use nonempty 0–1000 xyxy bounds")
    answer = {"result": label, "confidence": confidence, "boxes": boxes}
    for field in ("category", "reason", "note"):
        if value := row.get(field):
            answer[field] = _required_text(value, field)
    if not {"category", "reason"} & answer.keys():
        raise ValueError("prelabel must include a category or reason")
    return answer


def _join_prelabels(
    labels: dict[str, dict[str, Any]],
    manifest: Path,
    *,
    storage_root: Path,
    allow_pseudo_labels: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    input_rows = _read_rows(manifest)
    manifest_root = manifest.parent.resolve(strict=True)
    missing = input_rows.keys() - labels.keys()
    extra = labels.keys() - input_rows.keys()
    if missing or extra:
        raise ValueError(
            "prelabel IDs must match the source manifest exactly "
            f"(missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]})"
        )
    joined: dict[str, dict[str, Any]] = {}
    provenance: dict[str, str] = {}
    for identity, input_row in input_rows.items():
        row = labels[identity]
        try:
            if row.get("confirmed") is not True and (
                "confirmed" in row or not allow_pseudo_labels
            ):
                raise ValueError(
                    "prelabel needs confirmation or an approved pseudo-label policy"
                )
            observed = row.get("source_images")
            if not isinstance(observed, dict) or observed.keys() != set(_ROLES):
                raise ValueError(
                    "prelabel source_images must name the three source roles"
                )
            original = _source_images(input_row)
            for role, value in zip(_ROLES, original, strict=True):
                captured = _required_text(observed[role], f"source_images.{role}")
                if _local_image(captured, manifest_root, storage_root) != _local_image(
                    value, manifest_root, storage_root
                ):
                    raise ValueError(
                        f"prelabel image for {role} differs from the source manifest"
                    )
            confirmed = row.get("confirmed") is True
            answer = row["gt"] if confirmed and "gt" in row else _prelabel_answer(row)
        except ValueError as exc:
            raise ValueError(f"sample {identity!r}: {exc}") from exc
        joined[identity] = {**input_row, "gt": answer}
        provenance[identity] = (
            "human_confirmed" if confirmed else "approved_pseudo_label"
        )
    return joined, provenance


def _validation_ids(samples: list[_Sample], count: int) -> set[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for sample in sorted(samples, key=lambda sample: sample.id):
        groups[sample.label].append(sample.id)
    if not 1 <= count <= len(samples) - len(groups):
        raise ValueError(
            "validation size must leave at least one training row per class"
        )
    eligible = [groups[label][:-1] for label in sorted(groups)]
    selected: set[str] = set()
    for index in range(max(map(len, eligible))):
        for group in eligible:
            if index < len(group):
                selected.add(group[index])
                if len(selected) == count:
                    return selected
    raise AssertionError("validation count was already checked")


def _write_messages(
    samples: list[_Sample], validation_ids: set[str], output: Path
) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise ValueError(f"output directory must not already exist: {output}")
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=".pcb-dataset-", dir=output.parent) as tmp:
        stage = Path(tmp) / "output"
        for sample in samples:
            digest = hashlib.sha256(sample.id.encode("utf-8")).hexdigest()
            images = Path("images") / digest
            (stage / images).mkdir(parents=True)
            for local in sample.local_images:
                shutil.copyfile(local, stage / images / local.name)
            copied = [(images / name).as_posix() for name in _FILENAMES]
            is_validation = sample.id in validation_ids
            (validation if is_validation else train).append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "value": sample.system_prompt}
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                *(
                                    {"type": "image_url", "value": path}
                                    for path in copied
                                ),
                                {"type": "text", "value": sample.user_prompt},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "value": sample.answer}],
                        },
                    ]
                }
            )
            audit.append(
                {
                    "id": sample.id,
                    "split": "val" if is_validation else "train",
                    "label_source": sample.label_source,
                    "source_images": dict(
                        zip(_ROLES, sample.source_images, strict=True)
                    ),
                    "images": dict(zip(_ROLES, copied, strict=True)),
                }
            )
        _write_jsonl(stage / "train_messages.jsonl", train)
        _write_jsonl(stage / "val_messages.jsonl", validation)
        _write_jsonl(stage / "source_manifest.jsonl", audit)
        os.rename(stage, output)
    return {"train": len(train), "val": len(validation)}


def materialize_messages(
    source: Path,
    output: Path,
    *,
    storage_root: Path,
    image_root: Path | None = None,
    val_size: int = 2,
    system_prompt: str = "",
    user_prompt: str = "",
    source_manifest: Path | None = None,
    allow_pseudo_labels: bool = False,
) -> dict[str, int]:
    """Convert labeled rows. A materialized batch contributes only train.jsonl."""
    if source.is_dir():
        source = source / "train.jsonl"
    root = storage_root.resolve(strict=True)
    rows = _read_rows(source)
    provenance: dict[str, str] = {}
    if source_manifest is not None:
        if image_root is not None:
            raise ValueError(
                "source-manifest mode resolves images from the manifest directory"
            )
        rows, provenance = _join_prelabels(
            rows,
            source_manifest,
            storage_root=root,
            allow_pseudo_labels=allow_pseudo_labels,
        )
        image_root = source_manifest.parent
    elif allow_pseudo_labels:
        raise ValueError("a source manifest is required for pseudo-labels")
    images = (image_root or source.parent).resolve(strict=True)
    samples = _samples(
        rows,
        image_root=images,
        storage_root=root,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        label_sources=provenance,
    )
    labels = {sample.label for sample in samples}
    if labels & _BINARY_LABELS and labels & _TRACER_LABELS:
        raise ValueError(
            "dataset mixes PCB labeling schemes without an approved mapping"
        )
    return _write_messages(samples, _validation_ids(samples, val_size), output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("image_root", type=Path)
    manifest.add_argument("output", type=Path)
    manifest.add_argument("--inspection-rules-file", type=Path)
    manifest.add_argument("--stage-images", action="store_true")
    manifest.add_argument("--storage-root", type=Path, default=Path("/workspace"))
    messages = commands.add_parser("messages")
    messages.add_argument("source", type=Path)
    messages.add_argument("output", type=Path)
    messages.add_argument("--storage-root", type=Path, default=Path("/workspace"))
    messages.add_argument("--image-root", type=Path)
    messages.add_argument("--val-size", type=int, default=2)
    messages.add_argument("--system-prompt-file", type=Path)
    messages.add_argument("--user-prompt-file", type=Path)
    messages.add_argument("--source-manifest", type=Path)
    messages.add_argument("--allow-pseudo-labels", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "manifest":
            inspection_rules = (
                args.inspection_rules_file.read_text(encoding="utf-8")
                if args.inspection_rules_file
                else ""
            )
            write_prelabel_manifest(
                args.image_root,
                args.output,
                inspection_rules=inspection_rules,
                stage_images=args.stage_images,
                storage_root=args.storage_root,
            )
        else:
            counts = materialize_messages(
                args.source,
                args.output,
                storage_root=args.storage_root,
                image_root=args.image_root,
                val_size=args.val_size,
                source_manifest=args.source_manifest,
                allow_pseudo_labels=args.allow_pseudo_labels,
                system_prompt=(
                    args.system_prompt_file.read_text(encoding="utf-8")
                    if args.system_prompt_file
                    else ""
                ),
                user_prompt=(
                    args.user_prompt_file.read_text(encoding="utf-8")
                    if args.user_prompt_file
                    else ""
                ),
            )
            print(json.dumps(counts))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"PCB dataset preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
