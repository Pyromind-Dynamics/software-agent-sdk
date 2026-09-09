"""Self-contained AVI/PCB synthesis runtime staged onto Pyromind workers."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from PIL import Image


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SUPPORTED = frozenset({"scratch", "dot", "small_object", "pad", "hole"})


def run_avi_pcb_plan(input_dir: Path, output_file: Path, plan_file: Path) -> int:
    plan = _load_object(plan_file)
    if plan.get("adapter") != "avi_pcb":
        raise ValueError("AVI runtime requires adapter='avi_pcb'")
    output_dir = output_file.parent
    if output_dir.resolve().is_relative_to(input_dir.resolve()):
        raise ValueError("output must not be inside the source directory")
    all_hosts = _hosts(input_dir)
    known_splits = {"train", "validation", "val", "test", "eval"}
    if not all_hosts or any(
        item[1].get("split") not in known_splits for item in all_hosts
    ):
        raise ValueError("all AVI samples must declare a known split")
    hosts = [item for item in all_hosts if item[1].get("split") == "train"]
    if not hosts:
        raise ValueError("no AVI training hosts were found")
    selector = plan.get("host_selector") or {}
    selector_labels = {str(item) for item in selector.get("labels", [])}
    if selector_labels:
        hosts = [
            item
            for item in hosts
            if selector_labels
            & _labels(item[1].get("labels", item[1].get("label", [])))
        ]
        if not hosts:
            raise ValueError("host selector matched no training samples")
    rng = random.Random(int(plan.get("seed", 42)))
    rng.shuffle(hosts)
    max_reuse = int(plan.get("max_children_per_source", 0))
    requests = plan.get("requests")
    if not isinstance(requests, list):
        raise ValueError("augmentation requests must be a list")
    requested = sum(int(item.get("count", 0)) for item in requests)
    if requested > len(hosts) * max_reuse:
        raise ValueError("requested children exceed bounded host capacity")

    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_file = output_dir / "provenance.jsonl"
    failure_file = output_dir / "failures.jsonl"
    failure_file.touch(exist_ok=True)
    uses: Counter[str] = Counter()
    generated_count = 0
    with (
        output_file.open("w", encoding="utf-8") as synthesized,
        provenance_file.open("w", encoding="utf-8") as provenance,
    ):
        for request in requests:
            strategy_id = str(request.get("strategy_id", ""))
            prefix, _, operation = strategy_id.partition(".")
            if prefix != "avi_pcb" or operation not in SUPPORTED:
                raise ValueError(f"missing synthesis strategy: {strategy_id}")
            for _ in range(int(request.get("count", 0))):
                host_dir, meta = next(
                    item for item in hosts if uses[str(item[1]["id"])] < max_reuse
                )
                host_id = str(meta["id"])
                seed = _child_seed(int(plan.get("seed", 42)), host_id, generated_count)
                try:
                    annotation, validation = _generate(
                        host_dir=host_dir,
                        host_id=host_id,
                        operation=operation,
                        label=str(request.get("label", operation)),
                        parameters=dict(request.get("parameters") or {}),
                        output_dir=output_dir,
                        seed=seed,
                        index=generated_count,
                    )
                except Exception as exc:
                    _append_jsonl(
                        failure_file,
                        {
                            "schema_version": 1,
                            "sample_id": host_id,
                            "stage": "synthesis",
                            "error_code": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                    raise
                uses[host_id] += 1
                generated_count += 1
                synthesized.write(json.dumps(annotation, ensure_ascii=False) + "\n")
                provenance.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "sample_id": annotation["sample_id"],
                            "parent_sample_id": host_id,
                            "strategy_id": strategy_id,
                            "parameters": request.get("parameters") or {},
                            "seed": seed,
                            "annotation": annotation,
                            "validation": validation,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return generated_count


def _hosts(source: Path) -> list[tuple[Path, dict[str, Any]]]:
    candidates = (
        [source] if (source / "meta.json").is_file() else sorted(source.iterdir())
    )
    hosts = []
    for candidate in candidates:
        required = ("defect.jpg", "diff.jpg", "gt.jpg", "meta.json")
        if not candidate.is_dir() or not all(
            (candidate / name).is_file() for name in required
        ):
            continue
        meta = _load_object(candidate / "meta.json")
        meta["id"] = str(meta.get("id") or candidate.name)
        hosts.append((candidate, meta))
    return hosts


def _generate(
    *,
    host_dir: Path,
    host_id: str,
    operation: str,
    label: str,
    parameters: dict[str, Any],
    output_dir: Path,
    seed: int,
    index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = cv2.imread(str(host_dir / "defect.jpg"), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError("unable to decode host image")
    generated = original.copy()
    bbox = _inject(generated, operation, random.Random(seed), parameters)
    height, width = generated.shape[:2]
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("generated bbox is invalid")
    difference = cv2.absdiff(original, generated)
    changed = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    nonzero = int(np.count_nonzero(changed))
    if nonzero == 0:
        raise ValueError("synthesis produced no pixel change")
    sample_id = f"{host_id}__{operation}_{index:06d}"
    asset_dir = output_dir / "assets" / sample_id
    asset_dir.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(asset_dir / "defect.jpg"), generated)
    cv2.imwrite(str(asset_dir / "diff.jpg"), difference)
    cv2.imwrite(str(asset_dir / "gt.jpg"), changed)
    # Pillow performs a second decoder check before the artifact is committed.
    Image.open(asset_dir / "defect.jpg").verify()
    annotation = {
        "sample_id": sample_id,
        "category": label,
        "bbox_xyxy": list(bbox),
        "image": f"assets/{sample_id}/defect.jpg",
        "diff": f"assets/{sample_id}/diff.jpg",
        "gt": f"assets/{sample_id}/gt.jpg",
    }
    _review(original, generated, difference, bbox, asset_dir / "review.png")
    return annotation, {
        "pixel_change_count": nonzero,
        "diff_nonempty": True,
        "bbox_valid": True,
        "label_consistent": annotation["category"] == label,
    }


def _inject(
    image: Any, operation: str, rng: random.Random, parameters: dict[str, Any]
) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    cx, cy = (
        rng.randint(width // 8, width * 7 // 8),
        rng.randint(height // 8, height * 7 // 8),
    )
    color = tuple(int(value) for value in parameters.get("bgr", (20, 20, 20)))
    if operation == "scratch":
        length = min(int(parameters.get("length", max(8, width // 6))), width // 2)
        thickness = max(1, int(parameters.get("thickness", 2)))
        points = np.asarray(
            [
                (cx - length // 2, cy + rng.randint(-3, 3)),
                (cx, cy + rng.randint(-3, 3)),
                (cx + length // 2, cy + rng.randint(-3, 3)),
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [points], False, color, thickness)
        return (
            max(0, int(points[:, 0].min()) - thickness),
            max(0, int(points[:, 1].min()) - thickness),
            min(width, int(points[:, 0].max()) + thickness + 1),
            min(height, int(points[:, 1].max()) + thickness + 1),
        )
    if operation in {"dot", "small_object"}:
        radius = max(2, int(parameters.get("radius", min(width, height) // 80)))
        cv2.circle(image, (cx, cy), radius, color, -1)
        return cx - radius, cy - radius, cx + radius + 1, cy + radius + 1
    box_width = max(4, int(parameters.get("width", width // 20)))
    box_height = max(4, int(parameters.get("height", height // 20)))
    x1, y1 = cx - box_width // 2, cy - box_height // 2
    x2, y2 = x1 + box_width, y1 + box_height
    if operation == "hole":
        radius = max(2, min(box_width, box_height) // 2)
        cv2.circle(image, (cx, cy), radius, color, -1)
        return cx - radius, cy - radius, cx + radius + 1, cy + radius + 1
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), color, -1)
    return x1, y1, x2, y2


def _review(before: Any, after: Any, diff: Any, bbox: Any, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, title in zip(
        axes, (before, after, diff), ("before", "after", "diff"), strict=True
    ):
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    x1, y1, x2, y2 = bbox
    axes[1].add_patch(
        plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red")
    )
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _child_seed(seed: int, sample_id: str, index: int) -> int:
    value = hashlib.sha256(f"{seed}:{sample_id}:{index}".encode()).digest()
    return int.from_bytes(value[:8], "big")


def _labels(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
