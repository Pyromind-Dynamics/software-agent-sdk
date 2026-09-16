from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


SCRIPTS = (
    Path(__file__).parents[3] / ".agents/skills/data-processing/scripts/preparation"
)


def _load(name: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def synthesis(monkeypatch: pytest.MonkeyPatch) -> Any:
    return _load("template_synthesis", monkeypatch)


@pytest.fixture
def sample() -> tuple[Any, Any, Any]:
    normal = np.full((18, 24, 3), 180, dtype=np.uint8)
    mask = np.zeros((18, 24), dtype=bool)
    mask[3:6, 4:8] = True
    donor = normal.copy()
    donor[mask] = (20, 30, 40)
    return normal, donor, mask


def _absolute_difference(left: Any, right: Any) -> Any:
    return np.abs(left.astype(np.int16) - right.astype(np.int16)).astype(np.uint8)


def test_extract_filters_noise_and_respects_box(synthesis: Any, sample: Any) -> None:
    _, donor, evidence = sample
    evidence[0, 0] = True
    evidence[12:15, 12:15] = True
    template = synthesis.extract_template(
        evidence, bbox_xyxy=(0, 0, 10, 10), min_area=2, source_rgb=donor
    )
    assert synthesis.bbox_from_mask(template.mask) == (4, 3, 8, 6)
    assert template.mask.sum() == 12
    template.texture[:] = 0
    assert donor[0, 0].tolist() == [180, 180, 180]
    with pytest.raises(ValueError, match="empty defect"):
        synthesis.extract_template(evidence, min_area=100)


def test_transform_keeps_texture_registered_and_clips(
    synthesis: Any, sample: Any
) -> None:
    _, donor, mask = sample
    template = synthesis.extract_template(mask, source_rgb=donor)
    for angle in (0, 17, 90):
        placed = synthesis.transform_template(
            template,
            target_shape=(12, 16),
            center_xy=(1, 1),
            scale_xy=(1.5, 2),
            angle_deg=angle,
        )
        assert placed.mask.any()
        assert np.all(placed.texture[placed.mask] == [20, 30, 40])
        assert synthesis.bbox_from_mask(placed.mask)[:2] == (0, 0)
    with pytest.raises(ValueError, match="outside"):
        synthesis.transform_template(
            template, target_shape=(12, 16), center_xy=(100, 100)
        )


def test_constraints_and_custom_connectivity_rules(synthesis: Any) -> None:
    copper = np.zeros((12, 16), dtype=bool)
    copper[4:8, 2:14] = True
    cut = np.zeros_like(copper)
    cut[2:10, 7:9] = True
    constrained = synthesis.constrain_template(synthesis.Template(cut), copper)
    assert constrained.mask.sum() == 8

    def remains_connected(mask: Any) -> Any:
        count = len(synthesis.mask_components(copper & ~mask))
        return synthesis.Check("edge_defect", count == 1, f"components={count}")

    checks = synthesis.validate_rules(constrained.mask, [remains_connected])
    assert checks[0].passed is False
    assert checks[0].reason == "components=2"
    notch = constrained.mask.copy()
    notch[6:] = False
    assert synthesis.validate_rules(notch, [remains_connected])[0].passed
    with pytest.raises(ValueError, match="empty defect"):
        synthesis.constrain_template(constrained, np.zeros_like(copper))


@pytest.mark.parametrize("feather", [0.0, 0.6])
def test_inject_uses_normal_base_without_donor_defect(
    synthesis: Any, sample: Any, feather: float
) -> None:
    normal, donor, original_mask = sample
    before = normal.copy()
    placed = synthesis.transform_template(
        synthesis.extract_template(original_mask, source_rgb=donor),
        target_shape=original_mask.shape,
        center_xy=(17, 13),
    )
    synthetic = synthesis.inject_anomaly(
        normal, placed, base_role="normal_image", feather=feather
    )
    assert np.array_equal(normal, before)
    assert np.array_equal(synthetic[original_mask], normal[original_mask])
    assert np.array_equal(np.any(synthetic != normal, axis=2), placed.mask)
    material = np.full_like(normal, 45)
    custom = synthesis.inject_anomaly(
        normal, placed, base_role="normal_image", fill_rgb=material
    )
    assert np.all(custom[placed.mask] == 45)


@pytest.mark.parametrize("role", ["label_mask", "structure_mask", "unknown"])
def test_rejects_non_normal_gt_role(synthesis: Any, sample: Any, role: str) -> None:
    normal, donor, mask = sample
    with pytest.raises(ValueError, match="normal_image"):
        synthesis.inject_anomaly(
            normal, synthesis.extract_template(mask, source_rgb=donor), base_role=role
        )


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        ("identity", (4, 3, 8, 6)),
        ("flip_lr", (16, 3, 20, 6)),
        ("flip_ud", (4, 12, 8, 15)),
        ("rotate_180", (16, 12, 20, 15)),
    ],
)
def test_pair_transform_recomputes_bbox_and_difference(
    synthesis: Any, sample: Any, transform: str, expected: Any
) -> None:
    normal, donor, mask = sample
    synthetic, reference, moved = synthesis.apply_pair_transform(
        donor, normal, mask, transform
    )
    result = synthesis.compute_artifacts(
        synthetic,
        reference,
        moved,
        difference_fn=_absolute_difference,
        difference_kind="pixel_change_diff",
        visibility_threshold=64,
        min_visible_pixels=6,
    )
    assert result.bbox_xyxy == expected
    assert result.visible_pixels == 12
    assert np.array_equal(result.difference.max(axis=2) > 0, moved)


def test_visibility_checks_actual_changes_inside_defect(
    synthesis: Any, sample: Any
) -> None:
    normal, donor, mask = sample
    kwargs = dict(
        difference_kind="custom_diff", visibility_threshold=64, min_visible_pixels=1
    )
    with pytest.raises(ValueError, match="no pixel change"):
        synthesis.compute_artifacts(
            normal, normal, mask, difference_fn=_absolute_difference, **kwargs
        )
    outside = np.full_like(normal, 255)
    outside[mask] = 0
    with pytest.raises(ValueError, match="insufficient visible"):
        synthesis.compute_artifacts(
            donor, normal, mask, difference_fn=lambda a, b: outside, **kwargs
        )


def test_single_file_pipeline_runs_in_isolation_and_reproduces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample: Any
) -> None:
    bundler = _load("bundle_template_pipeline", monkeypatch)
    driver = tmp_path / "driver.py"
    driver.write_text(
        """from __future__ import annotations
import json
import random
import sys
import numpy as np
from template_synthesis import (
    Check, extract_template, transform_template, constrain_template,
    validate_rules, inject_anomaly, apply_pair_transform, compute_artifacts,
)

data = np.load(sys.argv[1])
normal, donor, mask = data["normal"], data["donor"], data["mask"]
rng = random.Random(73)
placed = transform_template(
    extract_template(mask, source_rgb=donor), target_shape=mask.shape,
    center_xy=(17 + rng.randint(-1, 1), 13), angle_deg=rng.uniform(-8, 8),
)
placed = constrain_template(placed, np.ones_like(mask))
checks = validate_rules(placed.mask, [lambda m: Check("area", m.sum() >= 4, "area")])
assert all(c.passed for c in checks)
image = inject_anomaly(normal, placed, base_role="normal_image")
image, reference, moved = apply_pair_transform(image, normal, placed.mask, "flip_lr")
result = compute_artifacts(
    image, reference, moved,
    difference_fn=lambda a, b: np.abs(
        a.astype(np.int16) - b.astype(np.int16)
    ).astype(np.uint8),
    difference_kind="pixel_change_diff", visibility_threshold=32, min_visible_pixels=4,
)
np.savez(sys.argv[2], image=image, reference=reference, mask=moved,
         diff=result.difference, bbox=result.bbox_xyxy)
""",
        encoding="utf-8",
    )
    stage = tmp_path / "isolated"
    bundle = stage / "pipeline.py"
    fingerprint = bundler.bundle_pipeline(driver, bundle)
    assert bundler.bundle_pipeline(driver, bundle) == fingerprint
    for path in (bundle, SCRIPTS / "template_synthesis.py"):
        ast.parse(path.read_text(), feature_version=(3, 10))
    driver.unlink()
    normal, donor, mask = sample
    source = tmp_path / "input.npz"
    np.savez(source, normal=normal, donor=donor, mask=mask)
    for index in (1, 2):
        subprocess.run(
            [
                sys.executable,
                "-I",
                str(bundle),
                str(source),
                str(stage / f"{index}.npz"),
            ],
            cwd=stage,
            check=True,
            capture_output=True,
            text=True,
        )
    with np.load(stage / "1.npz") as first, np.load(stage / "2.npz") as second:
        for name in first.files:
            assert np.array_equal(first[name], second[name])
        assert np.array_equal(first["reference"], np.fliplr(normal))
        assert np.array_equal(
            np.any(first["image"] != first["reference"], axis=2), first["mask"]
        )
