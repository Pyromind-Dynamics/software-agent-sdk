from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

import openhands.tools.dataset_ops.synthesis as synthesis_module
from openhands.tools.dataset_ops.adapters import AviPcbAdapter
from openhands.tools.dataset_ops.contracts import (
    AugmentationPlan,
    GapItem,
    GapPlan,
    SynthesisRequest,
)
from openhands.tools.dataset_ops.synthesis import (
    AviPcbSynthesisStrategy,
    execute_avi_plan,
)


class FakeCv2:
    IMREAD_COLOR = 1
    COLOR_BGR2GRAY = 2
    COLOR_BGR2RGB = 3

    @staticmethod
    def imread(path: str, mode: int) -> Any:  # noqa: ARG004
        return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()

    @staticmethod
    def imwrite(path: str, value: Any) -> bool:
        image = value if value.ndim == 2 else value[:, :, ::-1]
        Image.fromarray(image.astype(np.uint8)).save(path)
        return True

    @staticmethod
    def absdiff(left: Any, right: Any) -> Any:
        return np.abs(left.astype(np.int16) - right.astype(np.int16)).astype(np.uint8)

    @staticmethod
    def cvtColor(value: Any, code: int) -> Any:
        if code == FakeCv2.COLOR_BGR2GRAY:
            return value.max(axis=2)
        return value[:, :, ::-1]

    @staticmethod
    def _draw(value: Any, operation: Any) -> None:
        rgb = Image.fromarray(value[:, :, ::-1])
        operation(ImageDraw.Draw(rgb))
        value[:] = np.asarray(rgb)[:, :, ::-1]

    @classmethod
    def polylines(
        cls, value: Any, points: list[Any], closed: bool, color: Any, thickness: int
    ) -> None:  # noqa: ARG003
        cls._draw(
            value,
            lambda draw: draw.line(
                [tuple(item) for item in points[0]],
                fill=tuple(reversed(color)),
                width=thickness,
            ),
        )

    @classmethod
    def circle(
        cls, value: Any, center: tuple[int, int], radius: int, color: Any, fill: int
    ) -> None:  # noqa: ARG003
        x, y = center
        cls._draw(
            value,
            lambda draw: draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=tuple(reversed(color)),
            ),
        )

    @classmethod
    def rectangle(
        cls,
        value: Any,
        left: tuple[int, int],
        right: tuple[int, int],
        color: Any,
        fill: int,
    ) -> None:  # noqa: ARG003
        cls._draw(
            value,
            lambda draw: draw.rectangle((*left, *right), fill=tuple(reversed(color))),
        )


class FakeAxis:
    def imshow(self, value: Any) -> None:  # noqa: ARG002
        pass

    def set_title(self, value: str) -> None:  # noqa: ARG002
        pass

    def axis(self, value: str) -> None:  # noqa: ARG002
        pass

    def add_patch(self, value: Any) -> None:  # noqa: ARG002
        pass


class FakeFigure:
    def tight_layout(self) -> None:
        pass

    def savefig(self, path: Path, dpi: int) -> None:  # noqa: ARG002
        Image.new("RGB", (8, 8)).save(path)


class FakePlot:
    @staticmethod
    def Rectangle(*args: Any, **kwargs: Any) -> object:  # noqa: N802, ARG004
        return object()

    @staticmethod
    def subplots(rows: int, columns: int, figsize: Any) -> tuple[Any, list[Any]]:  # noqa: ARG004
        return FakeFigure(), [FakeAxis() for _ in range(columns)]

    @staticmethod
    def close(figure: Any) -> None:  # noqa: ARG004
        pass


@pytest.fixture
def pcb_record(tmp_path: Path) -> Any:
    sample = tmp_path / "source" / "host"
    sample.mkdir(parents=True)
    image = Image.new("RGB", (96, 96), (220, 210, 200))
    for name in ("defect.jpg", "diff.jpg", "gt.jpg"):
        image.save(sample / name)
    (sample / "meta.json").write_text(
        json.dumps({"id": "host", "split": "train"}), encoding="utf-8"
    )
    return next(AviPcbAdapter().iter_records(sample))


@pytest.mark.parametrize("operation", ["scratch", "dot", "pad", "hole"])
def test_avi_operations_are_valid_and_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pcb_record: Any, operation: str
) -> None:
    monkeypatch.setattr(
        synthesis_module,
        "_image_dependencies",
        lambda: (FakeCv2, np, FakePlot),
    )
    request = SynthesisRequest(
        dimension="defect",
        label=operation,
        strategy_id=f"avi_pcb.{operation}",
        count=1,
    )
    strategy = AviPcbSynthesisStrategy()
    first = strategy.generate(
        source=pcb_record,
        request=request,
        output_dir=tmp_path / "first",
        seed=7,
        child_index=0,
    )
    second = strategy.generate(
        source=pcb_record,
        request=request,
        output_dir=tmp_path / "second",
        seed=7,
        child_index=0,
    )
    assert first.validation == {
        "pixel_change_count": first.validation["pixel_change_count"],
        "diff_nonempty": True,
        "bbox_valid": True,
        "label_consistent": True,
    }
    assert first.validation["pixel_change_count"] > 0
    assert first.annotation["bbox_xyxy"] == second.annotation["bbox_xyxy"]
    assert (first.asset_dir / "defect.jpg").read_bytes() == (
        second.asset_dir / "defect.jpg"
    ).read_bytes()
    assert (first.asset_dir / "review.png").is_file()


def test_avi_strategy_rejects_validation_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pcb_record: Any
) -> None:
    monkeypatch.setattr(
        synthesis_module, "_image_dependencies", lambda: (FakeCv2, np, FakePlot)
    )
    invalid = type(pcb_record)(
        sample_id=pcb_record.sample_id,
        source_path=pcb_record.source_path,
        value=pcb_record.value,
        media_paths=pcb_record.media_paths,
        split="validation",
    )
    with pytest.raises(ValueError, match="cannot be synthesis hosts"):
        AviPcbSynthesisStrategy().generate(
            source=invalid,
            request=SynthesisRequest(
                dimension="defect",
                label="dot",
                strategy_id="avi_pcb.dot",
                count=1,
            ),
            output_dir=tmp_path / "output",
            seed=1,
            child_index=0,
        )


def test_avi_plan_enforces_host_reuse_capacity(tmp_path: Path, pcb_record: Any) -> None:
    request = SynthesisRequest(
        dimension="defect",
        label="dot",
        strategy_id="avi_pcb.dot",
        count=2,
    )
    plan = AugmentationPlan(
        source_path=str(pcb_record.source_path.parent),
        adapter="avi_pcb",
        gap_plan=GapPlan(
            status="approved",
            analysis_run_id="analysis",
            gaps=(
                GapItem(
                    dimension="defect",
                    label="dot",
                    current_count=0,
                    current_ratio=0,
                    target_count=2,
                    reason="business priority",
                    recommended_strategy="avi_pcb.dot",
                ),
            ),
        ),
        requests=(request,),
        max_children_per_source=1,
        output_schema="avi_pcb_v1",
    )
    with pytest.raises(ValueError, match="bounded host capacity"):
        execute_avi_plan(
            plan,
            source_dir=pcb_record.source_path.parent,
            output_dir=tmp_path / "output",
        )
