"""Scene-specific synthesis strategies.

The server imports this module without importing heavyweight image packages.
Those dependencies are loaded only inside the AVI/PCB runtime profile.
"""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openhands.tools.dataset_ops.adapters import AviPcbAdapter, DatasetRecord
from openhands.tools.dataset_ops.contracts import AugmentationPlan, SynthesisRequest
from openhands.tools.dataset_ops.labeling import FailureLedger


@dataclass(frozen=True, slots=True)
class SynthesisArtifact:
    sample_id: str
    parent_sample_id: str
    strategy_id: str
    asset_dir: Path
    annotation: dict[str, Any]
    parameters: dict[str, Any]
    seed: int
    validation: dict[str, Any]


class SynthesisStrategy(ABC):
    """A strategy atomically creates pixels and their annotation."""

    strategy_prefix: str

    @abstractmethod
    def generate(
        self,
        *,
        source: DatasetRecord,
        request: SynthesisRequest,
        output_dir: Path,
        seed: int,
        child_index: int,
    ) -> SynthesisArtifact: ...


class SynthesisStrategyRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, SynthesisStrategy] = {}

    def register(self, strategy: SynthesisStrategy) -> None:
        prefix = strategy.strategy_prefix.rstrip(".")
        if not prefix or prefix in self._strategies:
            raise ValueError(f"duplicate or empty synthesis strategy: {prefix!r}")
        self._strategies[prefix] = strategy

    def resolve(self, strategy_id: str) -> SynthesisStrategy:
        prefix = strategy_id.partition(".")[0]
        try:
            return self._strategies[prefix]
        except KeyError as exc:
            raise ValueError(
                f"missing synthesis strategy for {strategy_id!r}; "
                "generic pixel synthesis is not allowed"
            ) from exc


class AviPcbSynthesisStrategy(SynthesisStrategy):
    """Deterministic PCB defect injection for scratch/dot/pad/hole defects."""

    strategy_prefix = "avi_pcb"
    _SUPPORTED = frozenset({"scratch", "dot", "pad", "hole", "small_object"})

    def generate(
        self,
        *,
        source: DatasetRecord,
        request: SynthesisRequest,
        output_dir: Path,
        seed: int,
        child_index: int,
    ) -> SynthesisArtifact:
        cv2, np, plt = _image_dependencies()
        if source.is_evaluation:
            raise ValueError("validation/test/eval records cannot be synthesis hosts")
        operation = request.strategy_id.partition(".")[2]
        if operation not in self._SUPPORTED:
            raise ValueError(f"unsupported AVI/PCB operation: {operation!r}")

        image_path = source.source_path / "defect.jpg"
        if not image_path.is_file():
            raise ValueError(f"missing AVI/PCB host image: {image_path}")
        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original is None:
            raise ValueError(f"unable to decode AVI/PCB host image: {image_path}")

        effective_seed = _child_seed(seed, source.sample_id, child_index)
        rng = random.Random(effective_seed)
        generated = original.copy()
        height, width = generated.shape[:2]
        if min(height, width) < 8:
            raise ValueError("AVI/PCB host image is too small (minimum 8x8)")
        params = dict(request.parameters)
        bbox = _inject_defect(
            cv2=cv2,
            image=generated,
            operation=operation,
            rng=rng,
            parameters=params,
        )
        difference = cv2.absdiff(original, generated)
        changed = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        nonzero = int(np.count_nonzero(changed))
        if nonzero == 0:
            raise ValueError("AVI/PCB synthesis produced no pixel change")

        x1, y1, x2, y2 = bbox
        bbox_valid = 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        if not bbox_valid:
            raise ValueError(f"generated invalid bbox: {bbox}")

        sample_id = f"{source.sample_id}__{operation}_{child_index:04d}"
        asset_dir = output_dir / "assets" / sample_id
        asset_dir.mkdir(parents=True, exist_ok=False)
        if not cv2.imwrite(str(asset_dir / "defect.jpg"), generated):
            raise OSError("failed to write synthesized defect image")
        if not cv2.imwrite(str(asset_dir / "diff.jpg"), difference):
            raise OSError("failed to write synthesized diff image")
        cv2.imwrite(str(asset_dir / "gt.jpg"), changed)

        annotation = {
            "sample_id": sample_id,
            "category": request.label,
            "bbox_xyxy": [x1, y1, x2, y2],
            "image": f"assets/{sample_id}/defect.jpg",
            "diff": f"assets/{sample_id}/diff.jpg",
            "gt": f"assets/{sample_id}/gt.jpg",
        }
        provenance = {
            "schema_version": 1,
            "sample_id": sample_id,
            "parent_sample_id": source.sample_id,
            "strategy_id": request.strategy_id,
            "parameters": params,
            "seed": effective_seed,
            "annotation": annotation,
        }
        (asset_dir / "meta.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_review_figure(
            plt, cv2, original, generated, difference, annotation, asset_dir
        )
        validation = {
            "pixel_change_count": nonzero,
            "diff_nonempty": True,
            "bbox_valid": True,
            "label_consistent": annotation["category"] == request.label,
        }
        return SynthesisArtifact(
            sample_id=sample_id,
            parent_sample_id=source.sample_id,
            strategy_id=request.strategy_id,
            asset_dir=asset_dir,
            annotation=annotation,
            parameters=params,
            seed=effective_seed,
            validation=validation,
        )


def default_strategy_registry() -> SynthesisStrategyRegistry:
    registry = SynthesisStrategyRegistry()
    registry.register(AviPcbSynthesisStrategy())
    return registry


def execute_avi_plan(
    plan: AugmentationPlan,
    *,
    source_dir: Path,
    output_dir: Path,
    registry: SynthesisStrategyRegistry | None = None,
) -> int:
    """Execute an approved AVI plan with deterministic bounded host reuse."""

    if plan.adapter != "avi_pcb":
        raise ValueError("execute_avi_plan only accepts adapter='avi_pcb'")
    if output_dir.resolve().is_relative_to(source_dir.resolve()):
        raise ValueError("output_dir must not be inside the read-only source directory")
    all_hosts = list(AviPcbAdapter().iter_records(source_dir))
    if not all_hosts:
        raise ValueError("no AVI/PCB hosts were found")
    known_splits = {"train", "validation", "val", "test", "eval"}
    if any(host.split not in known_splits for host in all_hosts):
        raise ValueError("every AVI/PCB sample must declare a known split")
    hosts = [host for host in all_hosts if host.split == "train"]
    if not hosts:
        raise ValueError("no AVI/PCB training hosts were found")
    if plan.host_selector.labels:
        expected = set(plan.host_selector.labels)
        hosts = [host for host in hosts if expected & _record_labels(host)]
    if not hosts:
        raise ValueError("host selector matched no training samples")

    rng = random.Random(plan.seed)
    rng.shuffle(hosts)
    requested = sum(item.count for item in plan.requests)
    capacity = len(hosts) * plan.max_children_per_source
    if requested > capacity:
        raise ValueError(
            f"requested {requested} children exceeds bounded host capacity {capacity}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "synthesized.jsonl"
    provenance_file = output_dir / "provenance.jsonl"
    failures = FailureLedger(output_dir / "failures.jsonl")
    strategy_registry = registry or default_strategy_registry()
    host_uses = {host.sample_id: 0 for host in hosts}
    generated_count = 0
    with (
        output_file.open("w", encoding="utf-8") as synthesized,
        provenance_file.open("w", encoding="utf-8") as provenance,
    ):
        for request in plan.requests:
            strategy = strategy_registry.resolve(request.strategy_id)
            for child_index in range(request.count):
                host = next(
                    item
                    for item in hosts
                    if host_uses[item.sample_id] < plan.max_children_per_source
                )
                try:
                    artifact = strategy.generate(
                        source=host,
                        request=request,
                        output_dir=output_dir,
                        seed=plan.seed,
                        child_index=generated_count,
                    )
                except Exception as exc:
                    failures.record(
                        sample_id=host.sample_id,
                        stage="synthesis",
                        error_code=type(exc).__name__,
                        message=str(exc),
                    )
                    raise
                host_uses[host.sample_id] += 1
                generated_count += 1
                synthesized.write(
                    json.dumps(artifact.annotation, ensure_ascii=False) + "\n"
                )
                provenance.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "sample_id": artifact.sample_id,
                            "parent_sample_id": artifact.parent_sample_id,
                            "strategy_id": artifact.strategy_id,
                            "parameters": artifact.parameters,
                            "seed": artifact.seed,
                            "annotation": artifact.annotation,
                            "validation": artifact.validation,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return generated_count


def _record_labels(record: DatasetRecord) -> set[str]:
    raw = record.value.get("labels", record.value.get("label", ()))
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return {str(item) for item in raw}
    return set()


def _child_seed(seed: int, sample_id: str, child_index: int) -> int:
    import hashlib

    digest = hashlib.sha256(f"{seed}:{sample_id}:{child_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _image_dependencies() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import matplotlib
        import numpy as np

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "avi_pcb_cpu requires opencv-python-headless, numpy, Pillow, and Matplotlib"
        ) from exc
    return cv2, np, plt


def _inject_defect(
    *,
    cv2: Any,
    image: Any,
    operation: str,
    rng: random.Random,
    parameters: dict[str, Any],
) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    cx = rng.randint(max(2, width // 8), min(width - 3, width * 7 // 8))
    cy = rng.randint(max(2, height // 8), min(height - 3, height * 7 // 8))
    color = tuple(int(value) for value in parameters.get("bgr", (20, 20, 20)))
    if operation == "scratch":
        length = min(int(parameters.get("length", max(8, width // 6))), width // 2)
        thickness = max(1, int(parameters.get("thickness", 2)))
        x1, x2 = max(0, cx - length // 2), min(width - 1, cx + length // 2)
        jitter = max(1, int(parameters.get("jitter", 3)))
        points = []
        steps = max(3, length // 4)
        for index in range(steps + 1):
            x = x1 + round((x2 - x1) * index / steps)
            y = max(0, min(height - 1, cy + rng.randint(-jitter, jitter)))
            points.append((x, y))
        import numpy as np

        cv2.polylines(
            image, [np.asarray(points, dtype=np.int32)], False, color, thickness
        )
        ys = [point[1] for point in points]
        return (
            x1,
            max(0, min(ys) - thickness),
            x2 + 1,
            min(height, max(ys) + thickness + 1),
        )
    if operation in {"dot", "small_object"}:
        radius = max(1, int(parameters.get("radius", max(2, min(width, height) // 80))))
        cv2.circle(image, (cx, cy), radius, color, -1)
        return (
            max(0, cx - radius),
            max(0, cy - radius),
            min(width, cx + radius + 1),
            min(height, cy + radius + 1),
        )

    box_width = max(3, int(parameters.get("width", max(6, width // 20))))
    box_height = max(3, int(parameters.get("height", max(6, height // 20))))
    x1, y1 = max(0, cx - box_width // 2), max(0, cy - box_height // 2)
    x2, y2 = min(width, x1 + box_width), min(height, y1 + box_height)
    if operation == "hole":
        radius = max(2, min(x2 - x1, y2 - y1) // 2)
        cv2.circle(image, (cx, cy), radius, color, -1)
        return (
            max(0, cx - radius),
            max(0, cy - radius),
            min(width, cx + radius + 1),
            min(height, cy + radius + 1),
        )
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), color, -1)
    return x1, y1, x2, y2


def _write_review_figure(
    plt: Any,
    cv2: Any,
    original: Any,
    generated: Any,
    difference: Any,
    annotation: dict[str, Any],
    asset_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, title in zip(
        axes,
        (original, generated, difference),
        ("before", "after", "diff"),
        strict=True,
    ):
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    x1, y1, x2, y2 = annotation["bbox_xyxy"]
    rectangle = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red")
    axes[1].add_patch(rectangle)
    figure.tight_layout()
    figure.savefig(asset_dir / "review.png", dpi=120)
    plt.close(figure)
