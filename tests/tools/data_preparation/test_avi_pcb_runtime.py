from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw


class _Axis:
    def imshow(self, value: Any) -> None:  # noqa: ARG002
        pass

    def set_title(self, value: str) -> None:  # noqa: ARG002
        pass

    def axis(self, value: str) -> None:  # noqa: ARG002
        pass

    def add_patch(self, value: Any) -> None:  # noqa: ARG002
        pass


class _Figure:
    def tight_layout(self) -> None:
        pass

    def savefig(self, path: Path, dpi: int) -> None:  # noqa: ARG002
        Image.new("RGB", (8, 8)).save(path)


def _runtime_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    cv2.COLOR_BGR2GRAY = 2
    cv2.COLOR_BGR2RGB = 3
    cv2.imread = lambda path, mode: np.asarray(  # noqa: ARG005
        Image.open(path).convert("RGB")
    )[:, :, ::-1].copy()

    def imwrite(path: str, value: Any) -> bool:
        image = value if value.ndim == 2 else value[:, :, ::-1]
        Image.fromarray(image.astype(np.uint8)).save(path)
        return True

    def draw(value: Any, operation: Any) -> None:
        rgb = Image.fromarray(value[:, :, ::-1])
        operation(ImageDraw.Draw(rgb))
        value[:] = np.asarray(rgb)[:, :, ::-1]

    cv2.imwrite = imwrite
    cv2.absdiff = lambda left, right: np.abs(
        left.astype(np.int16) - right.astype(np.int16)
    ).astype(np.uint8)
    cv2.cvtColor = lambda value, code: (
        value.max(axis=2) if code == cv2.COLOR_BGR2GRAY else value[:, :, ::-1]
    )
    cv2.polylines = lambda value, points, closed, color, thickness: draw(  # noqa: ARG005
        value,
        lambda canvas: canvas.line(
            [tuple(item) for item in points[0]],
            fill=tuple(reversed(color)),
            width=thickness,
        ),
    )
    cv2.circle = lambda value, center, radius, color, fill: draw(  # noqa: ARG005
        value,
        lambda canvas: canvas.ellipse(
            (
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ),
            fill=tuple(reversed(color)),
        ),
    )
    cv2.rectangle = lambda value, left, right, color, fill: draw(  # noqa: ARG005
        value,
        lambda canvas: canvas.rectangle((*left, *right), fill=tuple(reversed(color))),
    )
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.use = lambda value: None
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.subplots = lambda rows, columns, figsize: (  # noqa: ARG005
        _Figure(),
        [_Axis() for _ in range(columns)],
    )
    pyplot.Rectangle = lambda *args, **kwargs: object()  # noqa: ARG005, N806
    pyplot.close = lambda figure: None  # noqa: ARG005
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    path = (
        Path(__file__).parents[3]
        / ".agents/skills/data-processing/scripts/preparation/avi_pcb_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("test_avi_pcb_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(root: Path, *, split: str = "train") -> Path:
    sample = root / "nested" / "host"
    sample.mkdir(parents=True)
    image = Image.new("RGB", (96, 96), (220, 210, 200))
    for name in ("defect.jpg", "diff.jpg", "gt.jpg"):
        image.save(sample / name)
    (sample / "meta.json").write_text(
        json.dumps({"id": "host", "split": split}), encoding="utf-8"
    )
    return root


@pytest.mark.parametrize("operation", ["scratch", "dot", "pad", "hole"])
def test_execute_plan_generates_valid_reproducible_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    runtime = _runtime_module(monkeypatch)
    source = _source(tmp_path / "source")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "seed": 7,
                "max_children_per_source": 1,
                "requests": [
                    {
                        "strategy_id": f"avi_pcb.{operation}",
                        "label": operation,
                        "count": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    first = tmp_path / "first" / "processed.jsonl"
    second = tmp_path / "second" / "processed.jsonl"

    assert runtime.execute_plan(source, first, plan) == 1
    assert runtime.execute_plan(source, second, plan) == 1

    first_row = json.loads(first.read_text(encoding="utf-8"))
    second_row = json.loads(second.read_text(encoding="utf-8"))
    assert first_row["bbox_xyxy"] == second_row["bbox_xyxy"]
    assert first_row["artifacts"]
    assert (first.parent / first_row["artifacts"][0]["path"]).is_file()
    validation = json.loads((first.parent / "validation.json").read_text())
    assert validation["status"] == "passed"
    assert validation["diff_nonempty"] is True
    assert (first.parent / "provenance.jsonl").is_file()


def test_execute_plan_rejects_non_train_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime_module(monkeypatch)
    source = _source(tmp_path / "source", split="validation")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "seed": 7,
                "max_children_per_source": 1,
                "requests": [
                    {"strategy_id": "avi_pcb.dot", "label": "dot", "count": 1}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no AVI training hosts"):
        runtime.execute_plan(source, tmp_path / "output/processed.jsonl", plan)
