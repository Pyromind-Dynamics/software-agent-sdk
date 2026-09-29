from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageFilter


SCRIPTS = (
    Path(__file__).parents[3] / ".agents/skills/data-processing/scripts/preparation"
)


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Any:
    for name in (
        "template_synthesis",
        "reference_synthesis",
        "bundle_template_pipeline",
    ):
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return sys.modules["reference_synthesis"]


@pytest.fixture
def paired(runtime: Any) -> Any:
    material = np.zeros((48, 64), dtype=bool)
    material[8:40, 8:56] = True
    gt = np.zeros((48, 64, 3), dtype=np.uint8)
    gt[material] = [204, 153, 0]
    normal = np.empty_like(gt)
    normal[:] = [30, 20, 10]
    normal[material] = [200, 180, 160]
    source = normal.copy()
    source[18:22, 28:32] = [60, 30, 110]
    box = (24, 14, 36, 26)
    pair = runtime.align_pair(
        source,
        gt,
        source[:, :, 0] > 100,
        material,
        runtime.region_mask(material.shape, box),
        radius=3,
        max_mismatch=0.01,
        min_pixels=400,
    )
    reference = runtime.build_normal_reference(
        pair,
        source_id="paired",
        seed=42,
        min_material_pixels=100,
        boundary_margin=2,
        texture_std_cap=0,
    )
    defect = runtime.extract_defect(
        pair,
        reference,
        source_id="paired",
        annotation_box=box,
        method="appearance",
        min_area=3,
        max_box_fraction=0.5,
        boundary_margin=1,
        appearance_threshold=50,
    )
    return pair, reference, defect, normal


def test_alignment_moves_source_not_gt(runtime: Any, paired: Any) -> None:
    pair, _, _, _ = paired
    displaced = runtime._shift(pair.source, -2, 1)
    material = runtime._shift(pair.source_material, -2, 1)
    excluded = runtime._shift(pair.excluded, -2, 1)
    aligned = runtime.align_pair(
        displaced,
        pair.gt,
        material,
        pair.gt_material,
        excluded,
        radius=3,
        max_mismatch=0.01,
        min_pixels=400,
    )
    assert aligned.shift_xy == (2, -1)
    assert np.array_equal(aligned.gt, pair.gt)
    assert np.array_equal(aligned.source[aligned.valid], pair.source[aligned.valid])
    with pytest.raises(ValueError, match="alignment_failed"):
        runtime.align_pair(
            displaced,
            pair.gt,
            ~material,
            pair.gt_material,
            excluded,
            radius=0,
            max_mismatch=0.01,
            min_pixels=400,
        )


def test_recolor_uses_gt_structure_and_excludes_original_defect(
    runtime: Any, paired: Any
) -> None:
    pair, reference, _, expected = paired
    assert np.array_equal(reference.image, expected)
    assert np.array_equal(reference.material, pair.gt_material)
    assert not np.array_equal(reference.image[18:22, 28:32], pair.source[18:22, 28:32])
    with pytest.raises(ValueError, match="insufficient_material_support"):
        runtime.build_normal_reference(
            replace(pair, excluded=np.ones_like(pair.excluded)),
            source_id="x",
            seed=1,
            min_material_pixels=10,
            boundary_margin=1,
            texture_std_cap=0,
        )


def test_extracts_only_anomaly_not_normal_copper_or_whole_box(
    runtime: Any, paired: Any
) -> None:
    pair, reference, defect, _ = paired
    expected = np.zeros_like(pair.gt_material)
    expected[18:22, 28:32] = True
    assert np.array_equal(defect.template.mask, expected)
    assert np.all(defect.template.texture[expected] == [60, 30, 110])
    assert runtime.normalized_box_to_pixels([375, 250, 625, 750], (48, 64)) == (
        24,
        12,
        40,
        36,
    )
    with pytest.raises(ValueError, match="reviewed mask must lie inside"):
        runtime.extract_defect(
            pair,
            reference,
            source_id="x",
            annotation_box=(24, 14, 36, 26),
            method="reviewed",
            min_area=3,
            max_box_fraction=0.5,
            boundary_margin=1,
            reviewed_mask=pair.gt_material,
        )
    with pytest.raises(ValueError, match="empty defect"):
        runtime.extract_defect(
            replace(pair, source=reference.image),
            reference,
            source_id="x",
            annotation_box=(24, 14, 36, 26),
            method="appearance",
            min_area=3,
            max_box_fraction=0.5,
            boundary_margin=1,
            appearance_threshold=50,
        )


@pytest.mark.parametrize("method", ["extra", "missing"])
def test_structural_evidence_comes_from_pair_difference(
    runtime: Any, paired: Any, method: str
) -> None:
    pair, reference, _, _ = paired
    source_material = pair.gt_material.copy()
    box = (0, 10, 6, 20) if method == "extra" else (24, 14, 36, 26)
    mask = np.zeros_like(source_material)
    if method == "extra":
        mask[12:15, 2:5] = True
    else:
        mask[18:21, 28:31] = True
    source_material[mask] = method == "extra"
    template = runtime.extract_defect(
        replace(pair, source_material=source_material),
        reference,
        source_id="x",
        annotation_box=box,
        method=method,
        min_area=3,
        max_box_fraction=0.5,
        boundary_margin=0,
    )
    assert np.array_equal(template.template.mask, mask)


def test_structure_detects_bridge_cut_and_wrong_contact(runtime: Any) -> None:
    material = np.zeros((20, 24), dtype=bool)
    material[4:8, 2:22] = True
    material[12:16, 2:22] = True
    bridge = np.zeros_like(material)
    bridge[8:12, 10:12] = True
    assert not runtime.structure_check(
        material,
        bridge,
        operation="add",
        relation="edge",
        target_foreground=False,
        component_delta=0,
    ).passed
    assert runtime.structure_check(
        material,
        bridge,
        operation="add",
        relation="edge",
        target_foreground=False,
        component_delta=-1,
    ).passed
    cut = np.zeros_like(material)
    cut[4:8, 10:12] = True
    assert not runtime.structure_check(
        material,
        cut,
        operation="remove",
        relation="edge",
        target_foreground=True,
        component_delta=0,
    ).passed
    cut[:] = False
    cut[18:19, 10:12] = True
    assert not runtime.structure_check(
        material,
        cut,
        operation="add",
        relation="edge",
        target_foreground=False,
        component_delta=None,
    ).passed


@pytest.fixture
def synthesis_options(runtime: Any, paired: Any) -> dict[str, Any]:
    _, reference, _, _ = paired
    centers = np.zeros_like(reference.material)
    centers[30, 44] = True
    return dict(
        candidate_centers=centers,
        allowed_mask=reference.material,
        rules=[
            lambda m: runtime.structure_check(
                reference.material,
                m,
                operation="appearance",
                relation="interior",
                target_foreground=True,
                component_delta=0,
            )
        ],
        seed=7,
        scale_xy=(1.0, 1.0),
        angle_deg=0.0,
        pair_transform="flip_lr",
        min_area=4,
        max_area=24,
        max_clip_fraction=0.05,
        max_attempts=4,
        feather=0,
        highpass_radius=2,
        highpass_gain=3,
        visibility_threshold=5,
        min_visible_pixels=4,
    )


def test_sample_diff_export_and_reproducibility(
    runtime: Any, paired: Any, synthesis_options: Any, tmp_path: Path
) -> None:
    pair, reference, defect, expected = paired
    first = runtime.synthesize_sample(reference, defect, **synthesis_options)
    second = runtime.synthesize_sample(reference, defect, **synthesis_options)
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(reference.image, expected)
    assert np.array_equal(first.gt, np.fliplr(pair.gt))
    assert np.array_equal(np.any(first.image != first.reference, axis=2), first.mask)
    gray = Image.fromarray(first.image).convert("L")
    residual = np.asarray(gray, dtype=np.float32) - np.asarray(
        gray.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32
    )
    assert residual.min() < 0 < residual.max()
    assert np.array_equal(
        first.artifacts.difference, np.clip(abs(residual) * 3, 0, 255).astype(np.uint8)
    )
    record = runtime.export_sample(first, tmp_path, "one", "appearance")
    assert record["visual_status"] == "not_reviewed"
    assert record["training_ready"] is False
    assert np.array_equal(
        np.load(tmp_path / "assets/one/highpass_signed.npy"), residual
    )
    review = np.asarray(Image.open(tmp_path / "assets/one/review.png"))
    x1, y1, x2, y2 = first.artifacts.bbox_xyxy
    assert np.array_equal(
        review[y1 + 1 : y2 - 1, x1 + 1 : x2 - 1],
        first.image[y1 + 1 : y2 - 1, x1 + 1 : x2 - 1],
    )
    assert np.array_equal(review[:, 64:128], first.reference)


def test_severe_clipping_is_rejected(
    runtime: Any, paired: Any, synthesis_options: Any
) -> None:
    _, reference, defect, _ = paired
    centers = np.zeros_like(reference.material)
    centers[0, 0] = True
    synthesis_options.update(
        candidate_centers=centers, allowed_mask=np.ones_like(centers)
    )
    with pytest.raises(ValueError, match="lost_fraction"):
        runtime.synthesize_sample(reference, defect, **synthesis_options)


def test_review_failures_and_unknown_cannot_pass(runtime: Any) -> None:
    kwargs = dict(expected_category="a", expected_box=(1, 2, 5, 6), min_iou=0.5)
    assert runtime.review_status(None, **kwargs) == "review_failed"
    result = dict(
        category="a",
        bbox_xyxy=[1, 2, 5, 6],
        realistic=True,
        extra_anomalies=False,
        uncertain=False,
        context_consistent=True,
    )
    assert runtime.review_status(result, **kwargs, failed=True) == "review_failed"
    assert runtime.review_status(result, **kwargs) == "passed"
    for change in (
        dict(uncertain=True),
        dict(category="unknown"),
        dict(realistic=False),
        dict(extra_anomalies=True),
        dict(context_consistent=False),
        dict(bbox_xyxy=[20, 20, 24, 24]),
    ):
        assert runtime.review_status(result | change, **kwargs) == "needs_review"


def test_review_merge_matches_images_and_preserves_failure_status(runtime: Any) -> None:
    samples = [
        dict(
            sample_id=str(i),
            category="a",
            bbox_xyxy=(1, 2, 5, 6),
            artifacts=[dict(role="image", path=f"{i}.png")],
        )
        for i in range(4)
    ]
    answer = dict(
        category="a",
        bbox_xyxy=[1, 2, 5, 6],
        realistic=True,
        extra_anomalies=False,
        uncertain=False,
        context_consistent=True,
    )
    output = {
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "value": "2.png"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "value": "<think>checked</think><answer>"
                        + json.dumps(answer)
                        + "</answer>",
                    }
                ],
            },
        ]
    }
    merged = runtime.merge_review_outputs(
        samples, [output], selected_ids={"0", "2"}, min_iou=0.5
    )
    assert [r["visual_status"] for r in merged] == [
        "review_failed",
        "not_reviewed",
        "passed",
        "not_reviewed",
    ]
    duplicated = runtime.merge_review_outputs(
        samples, [output, output], selected_ids={"2"}, min_iou=0.5
    )
    assert duplicated[2]["visual_status"] == "review_failed"
    assert all(r["training_ready"] is False for r in merged)


def test_context_rejects_opposite_region_despite_permissive_strategy(
    runtime: Any, paired: Any, synthesis_options: Any
) -> None:
    _, reference, defect, _ = paired
    centers = np.zeros_like(reference.material)
    centers[20, 4] = True
    options = synthesis_options | dict(
        candidate_centers=centers,
        allowed_mask=~reference.material,
        rules=[lambda m: runtime.Check("custom", True, "allowed")],
    )
    with pytest.raises(ValueError, match="source_context"):
        runtime.synthesize_sample(reference, defect, **options)
    assert not runtime.context_candidates(reference, defect)[20, 4]


def test_context_does_not_depend_on_foreground_naming(
    runtime: Any, paired: Any, synthesis_options: Any
) -> None:
    _, reference, defect, _ = paired
    expected = runtime.synthesize_sample(reference, defect, **synthesis_options)
    inverted_reference = replace(reference, material=~reference.material)
    inverted_defect = replace(defect, source_material=~defect.source_material)
    actual = runtime.synthesize_sample(
        inverted_reference, inverted_defect, **synthesis_options
    )
    assert np.array_equal(actual.image, expected.image)
    assert actual.parameters["source_context"]["foreground_fraction"] == 0
    assert expected.parameters["source_context"]["foreground_fraction"] == 1


def test_context_detects_edge_to_interior_move(runtime: Any, paired: Any) -> None:
    _, reference, _, _ = paired
    near = np.zeros_like(reference.material)
    near[18:22, 8:12] = True
    far = np.zeros_like(near)
    far[18:22, 28:32] = True
    policy = runtime.ContextPolicy()
    source = runtime.context_profile(
        reference.material, near, boundary_band_ratio=policy.boundary_band_ratio
    )
    target = runtime.context_profile(
        reference.material, far, boundary_band_ratio=policy.boundary_band_ratio
    )
    assert source["foreground_fraction"] == target["foreground_fraction"] == 1
    assert not runtime.context_check(source, target, policy=policy).passed


def test_cross_reference_mapping_requires_evidence(
    runtime: Any, paired: Any, synthesis_options: Any
) -> None:
    _, reference, defect, _ = paired
    other = replace(reference, source_id="other")
    with pytest.raises(ValueError, match="material_mapping_required"):
        runtime.synthesize_sample(other, defect, **synthesis_options)
    with pytest.raises(ValueError, match="supporting evidence"):
        runtime.context_candidates(other, defect, material_mapping="same")
    sample = runtime.synthesize_sample(
        other,
        defect,
        material_mapping="same",
        mapping_basis="matched material regions in paired reference crops",
        **synthesis_options,
    )
    assert sample.parameters["mapping_basis"]


def test_review_bbox_must_be_within_image(runtime: Any) -> None:
    result = dict(
        category="a",
        bbox_xyxy=[-1, 2, 5, 6],
        realistic=True,
        extra_anomalies=False,
        uncertain=False,
        context_consistent=True,
    )
    kwargs = dict(expected_category="a", expected_box=(1, 2, 5, 6), min_iou=0.1)
    assert runtime.review_status(result, image_size=(8, 8), **kwargs) == "needs_review"
    result["bbox_xyxy"] = [1, 2, 9, 6]
    assert runtime.review_status(result, image_size=(8, 8), **kwargs) == "needs_review"
    del result["context_consistent"]
    assert runtime.review_status(result, **kwargs) == "review_failed"


def test_finalization_binds_inputs_and_cannot_promote_missing_checks(
    runtime: Any, paired: Any, synthesis_options: Any, tmp_path: Path
) -> None:
    pair, reference, defect, _ = paired

    def appearance(evidence: Any) -> list[Any]:
        changed = np.any(evidence.image != evidence.reference, axis=2)
        return [runtime.Check("appearance", bool(changed.any()), "visible appearance")]

    sample = runtime.synthesize_sample(
        reference,
        defect,
        **synthesis_options,
        source_evidence=runtime.SourceEvidence(
            pair.source,
            reference.image,
            runtime.region_mask(defect.template.mask.shape, defect.annotation_box),
            defect.template,
        ),
        validate_source=appearance,
        validate_rendered=appearance,
    )
    record = runtime.export_sample(sample, tmp_path, "a", "a")
    (tmp_path / "processed.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "plan.json").write_text('{"strategy":"test","requested":1}')
    (tmp_path / "strategy.py").write_text("# frozen strategy fixture")
    (tmp_path / "review_input.jsonl").write_text(
        json.dumps({"images": ["assets/a/image.png"]}) + "\n"
    )
    binding = runtime.freeze_review_binding(
        tmp_path,
        samples_path="processed.jsonl",
        review_input_path="review_input.jsonl",
        plan_path="plan.json",
        selected_ids={"a"},
        min_iou=0.5,
        strategy_path="strategy.py",
    )
    answer = dict(
        category="a",
        bbox_xyxy=list(sample.artifacts.bbox_xyxy),
        realistic=True,
        extra_anomalies=False,
        uncertain=False,
        context_consistent=True,
    )
    output = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "value": "assets/a/image.png"},
                    {
                        "type": "text",
                        "value": "Review evidence ID: " + binding["review_evidence_id"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "value": "<answer>" + json.dumps(answer) + "</answer>",
                    }
                ],
            },
        ]
    }
    rows, summary = runtime.finalize_review(tmp_path, binding, [output])
    assert summary["quality_status"] == "passed"
    assert rows[0]["training_ready"] is False
    unbound_rows, _ = runtime.finalize_review(
        tmp_path, binding | {"strategy_path": None}, [output]
    )
    assert unbound_rows[0]["quality_status"] == "needs_review"
    assert "unbound_strategy" in unbound_rows[0]["quality_reasons"]
    (tmp_path / "strategy.py").write_text("# changed strategy")
    stale_rows, stale_summary = runtime.finalize_review(tmp_path, binding, [output])
    assert stale_summary["stale_evidence"] == ["strategy.py"]
    assert "stale_evidence" in stale_rows[0]["quality_reasons"]
    (tmp_path / "strategy.py").write_text("# frozen strategy fixture")

    review_dir, quality_dir = tmp_path / "review", tmp_path / "quality"
    review_dir.mkdir()
    quality_dir.mkdir()
    (tmp_path / "binding.json").write_text(json.dumps(binding))
    (review_dir / "review_output.jsonl").write_text(json.dumps(output) + "\n")
    (quality_dir / "review_job.json").write_text(
        json.dumps(
            {
                "binding_path": "../binding.json",
                "review_output_path": "../review/review_output.jsonl",
            }
        )
    )
    for folder in (tmp_path, review_dir, quality_dir):
        (folder / "report.json").write_text('{"status":"executor_only"}')
    finalizer = tmp_path / "finalize.py"
    sys.modules["bundle_template_pipeline"].bundle_pipeline(
        SCRIPTS / "finalize_synthesis_review.py", finalizer
    )
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(finalizer),
            str(quality_dir / "review_job.json"),
            str(quality_dir / "reviewed.jsonl"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        json.loads((quality_dir / "reviewed.quality.json").read_text())[
            "quality_passed"
        ]
        == 1
    )
    assert (quality_dir / "assets/a/image.png").read_bytes() == (
        tmp_path / "assets/a/image.png"
    ).read_bytes()
    assert all(
        json.loads((folder / "report.json").read_text())["status"] == "executor_only"
        for folder in (tmp_path, review_dir, quality_dir)
    )
    unbound_output = json.loads(json.dumps(output))
    unbound_output["messages"][0]["content"].pop()
    unbound_rows, _ = runtime.finalize_review(tmp_path, binding, [unbound_output])
    assert unbound_rows[0]["visual_status"] == "review_failed"
    unchecked = runtime.merge_review_outputs(
        [record | {"checks": []}], [output], selected_ids={"a"}, min_iou=0.5
    )
    assert unchecked[0]["quality_status"] == "needs_review"
    (tmp_path / "plan.json").write_text('{"strategy":"changed"}')
    rows, summary = runtime.finalize_review(tmp_path, binding, [output])
    assert summary["stale_evidence"] == ["plan.json"]
    assert rows[0]["visual_status"] == "review_failed"
    assert summary["quality_status"] == "needs_review"


def test_bundle_contains_reference_module_and_review_matches_managed_contract(
    runtime: Any,
    tmp_path: Path,
) -> None:
    from openhands.tools.data_preparation.runner import (
        runtime_public_names,
        validate_managed_image_pipeline,
    )

    driver = tmp_path / "driver.py"
    driver.write_text(
        "import reference_synthesis as rs\nprint(rs.review_status(None, "
        "expected_category='x', expected_box=(0,0,1,1), min_iou=.5))\n"
    )
    bundled = tmp_path / "bundle.py"
    sys.modules["bundle_template_pipeline"].bundle_pipeline(driver, bundled)
    driver.unlink()
    output = subprocess.run(
        [sys.executable, "-I", str(bundled)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert output.stdout.strip() == "review_failed"
    validate_managed_image_pipeline(
        SCRIPTS / "image_synthesis_review.py",
        runtime_public_names(SCRIPTS / "image_utils.py"),
    )
    for name in ("reference_synthesis.py", "paired_reference_example.py"):
        ast.parse((SCRIPTS / name).read_text(), feature_version=(3, 10))


def test_review_dimensions_and_explicit_aliases(runtime: Any) -> None:
    answer = dict(
        category="extra (material)",
        bbox_xyxy=[40, 40, 50, 50],
        realistic=True,
        extra_anomalies=False,
        uncertain=False,
        context_consistent=True,
    )
    kwargs = dict(expected_category="extra", expected_box=(1, 2, 5, 6), min_iou=0.5)
    assessment = runtime.review_assessment(answer, **kwargs)
    assert set(assessment["reasons"]) == {"category", "localization"}
    assessment = runtime.review_assessment(
        answer, **kwargs, category_aliases={"extra (material)": "extra"}
    )
    assert assessment["reasons"] == ["localization"]
    assert assessment["checks"]["category"] is True
    assert runtime.review_status(answer | {"category": " extra"}, **kwargs) != "passed"


def test_real_history_cannot_be_promoted_by_successful_visual_review(
    runtime: Any,
) -> None:
    fixture = Path(__file__).parents[2] / "fixtures/synthesis/c94c4aa1"
    history = json.loads((fixture / "history.json").read_text())
    for name, digest in history["sha256"].items():
        assert hashlib.sha256((fixture / name).read_bytes()).hexdigest() == digest
    outputs = []
    for record in history["records"]:
        answer = dict(
            category=record["category"],
            bbox_xyxy=record["bbox_xyxy"],
            realistic=True,
            extra_anomalies=False,
            uncertain=False,
            context_consistent=True,
        )
        outputs.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "value": next(
                                    a["path"]
                                    for a in record["artifacts"]
                                    if a["role"] == "image"
                                ),
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "value": "<answer>" + json.dumps(answer) + "</answer>",
                            }
                        ],
                    },
                ]
            }
        )
    merged = runtime.merge_review_outputs(
        history["records"],
        outputs,
        selected_ids={r["sample_id"] for r in history["records"]},
        min_iou=0.5,
    )
    assert all(r["visual_status"] == "passed" for r in merged)
    assert all(r["quality_status"] == "needs_review" for r in merged)
    assert all("incomplete_validation" in r["quality_reasons"] for r in merged)
    assert all(r["training_ready"] is False for r in merged)
    observed = [
        runtime.review_assessment(
            review,
            expected_category=record["category"],
            expected_box=record["bbox_xyxy"],
            min_iou=0.5,
        )
        for record, review in zip(history["records"], history["reviews"], strict=True)
    ]
    assert all(
        "category" in r["reasons"] and "context" in r["reasons"] for r in observed
    )


def test_bundled_paired_driver_preserves_stage_outputs_and_rejects_unproven_source(
    runtime: Any, tmp_path: Path
) -> None:
    fixture = Path(__file__).parents[2] / "fixtures/synthesis/c94c4aa1"
    history = json.loads((fixture / "history.json").read_text())
    plan = history["plan"]
    for key in ("donor_rgb", "donor_cam"):
        plan[key] = str(fixture / plan[key])
    pipeline = tmp_path / "pipeline.py"
    sys.modules["bundle_template_pipeline"].bundle_pipeline(
        SCRIPTS / "paired_reference_example.py", pipeline
    )
    for version in (1, 2):
        folder = tmp_path / str(version) / "generation"
        folder.mkdir(parents=True)
        sentinel = folder / "report.json"
        sentinel.write_text('{"status":"runtime_report"}')
        plan_file = tmp_path / f"plan_{version}.json"
        plan_file.write_text(json.dumps(plan | {"validation_version": version}))
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                str(pipeline),
                str(plan_file),
                str(folder / "processed.jsonl"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert json.loads(sentinel.read_text()) == {"status": "runtime_report"}
        report = json.loads((folder / "synthesis_report.json").read_text())
        assert report["requested"] == 3
        if version == 2:
            assert result.returncode != 0
            assert report["generated"] == 0
            assert all(
                f["attempts"][0]["stage"] == "source" for f in report["failures"]
            )
        else:
            assert result.returncode == 0, result.stderr
            records = list(
                map(json.loads, (folder / "processed.jsonl").read_text().splitlines())
            )
            assert all(r["program_status"] == "needs_review" for r in records)
        original = (folder / "augmentation_plan.json").read_bytes()
        repeated = subprocess.run(
            [
                sys.executable,
                "-I",
                str(pipeline),
                str(plan_file),
                str(folder / "processed.jsonl"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert repeated.returncode != 0
        assert (folder / "augmentation_plan.json").read_bytes() == original


def test_legacy_template_can_move_to_different_sized_reference(
    runtime: Any, paired: Any, synthesis_options: Any
) -> None:
    _, reference, defect, _ = paired
    other = replace(
        reference,
        source_id="larger",
        image=np.pad(reference.image, ((8, 8), (8, 8), (0, 0))),
        gt=np.pad(reference.gt, ((8, 8), (8, 8), (0, 0))),
        material=np.pad(reference.material, 8),
    )
    centers = np.pad(synthesis_options["candidate_centers"], 8)
    result = runtime.synthesize_sample(
        other,
        defect,
        **(
            synthesis_options
            | {
                "candidate_centers": centers,
                "allowed_mask": other.material,
                "rules": [lambda m: runtime.Check("area", bool(m.any()), "nonempty")],
            }
        ),
        material_mapping="same",
        mapping_basis="same reference padded by 8 pixels",
    )
    assert result.image.shape == other.image.shape
    assert result.parameters["validation_coverage"]["source"] is False
    assert result.parameters["validation_coverage"]["rendered"] is False


def test_generic_export_binds_source_and_reports_shortfall(
    runtime: Any, paired: Any, synthesis_options: Any, tmp_path: Path
) -> None:
    pair, reference, defect, _ = paired

    def appearance(e: Any) -> list[Any]:
        delta = np.abs(e.image.astype(float) - e.reference)
        return [runtime.Check("contrast", bool((delta > 40).any()), "actual contrast")]

    result = runtime.synthesize_template(
        runtime.SourceEvidence(
            pair.source,
            reference.image,
            runtime.region_mask(defect.template.mask.shape, defect.annotation_box),
            defect.template,
            {"support": reference.material},
        ),
        reference.image,
        candidate_centers=synthesis_options["candidate_centers"],
        allowed_mask=reference.material,
        acceptance=runtime.Acceptance(1, 40, 0.05, 10, 4),
        seed=2,
        max_attempts=10,
        regions={"support": reference.material},
        validate_source=appearance,
        validate_candidate=lambda e: [
            runtime.Check(
                "inside",
                bool(np.all(e.regions["support"][e.constrained.mask])),
                "support",
            )
        ],
        validate_rendered=appearance,
    )
    record = runtime.export_sample(result, tmp_path, "one", "appearance")
    assert record["program_status"] == "passed"
    assert (tmp_path / "assets/one/source_image.png").is_file()
    assert (tmp_path / "assets/one/source_region_0.png").is_file()
    (tmp_path / "samples.jsonl").write_text(json.dumps(record) + "\n")
    plan = {
        "requested": 2,
        "review_min_iou": 0.5,
        "category_aliases": {"appearance (texture)": "appearance"},
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "strategy.py").write_text("# source, candidate, rendered strategy")
    (tmp_path / "review_input.jsonl").write_text(
        json.dumps({"images": ["assets/one/image.png"]}) + "\n"
    )
    args = dict(
        samples_path="samples.jsonl",
        review_input_path="review_input.jsonl",
        plan_path="plan.json",
        selected_ids={"one"},
        strategy_path="strategy.py",
    )
    with pytest.raises(ValueError, match="threshold"):
        runtime.freeze_review_binding(tmp_path, min_iou=0.1, **args)
    binding = runtime.freeze_review_binding(tmp_path, min_iou=0.5, **args)
    answer = dict(
        category="appearance (texture)",
        bbox_xyxy=list(result.artifacts.bbox_xyxy),
        realistic=True,
        extra_anomalies=False,
        uncertain=False,
        context_consistent=True,
    )
    output = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "value": "assets/one/image.png"},
                    {
                        "type": "text",
                        "value": "Review evidence ID: " + binding["review_evidence_id"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "value": "<answer>" + json.dumps(answer) + "</answer>",
                    },
                ],
            },
        ]
    }
    rows, summary = runtime.finalize_review(tmp_path, binding, [output])
    assert rows[0]["quality_status"] == "passed"
    assert summary["quality_status"] == "needs_review"
    assert summary["requested"] == 2 and summary["generated"] == 1
    assert summary["generation_shortfall"] == 1 and summary["quality_passed"] == 1
    assert "generation_shortfall" in summary["summary_reasons"]
    (tmp_path / "assets/one/source_image.png").write_bytes(b"changed")
    rows, summary = runtime.finalize_review(tmp_path, binding, [output])
    assert summary["stale_evidence"] == ["assets/one/source_image.png"]
    assert rows[0]["quality_status"] == "needs_review"
