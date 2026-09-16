"""GT geometry with measured appearance, evidence-bound defects and sample QA."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter
from template_synthesis import (
    Artifacts,
    Box,
    Check,
    Mask,
    PairTransform,
    Pixels,
    Template,
    apply_pair_transform,
    compute_artifacts,
    constrain_template,
    extract_template,
    inject_anomaly,
    mask_components,
    transform_template,
    validate_rules,
)


@dataclass(frozen=True)
class AlignedPair:
    source: Pixels
    source_material: Mask
    gt: Pixels
    gt_material: Mask
    excluded: Mask
    valid: Mask
    shift_xy: tuple[int, int]
    mismatch_ratio: float


@dataclass(frozen=True)
class NormalReference:
    image: Pixels
    gt: Pixels
    material: Mask
    source_id: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class DefectTemplate:
    template: Template
    source_id: str
    annotation_box: Box
    extraction_method: str
    source_material: Mask


@dataclass(frozen=True)
class ContextPolicy:
    """Dimensionless tolerances fixed before generating candidates."""

    material_fraction_tolerance: float = 0.05
    boundary_fraction_tolerance: float = 0.25
    clearance_tolerance: float = 1.0
    boundary_band_ratio: float = 0.25

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if not np.isfinite(value) or value < 0:
                raise ValueError("context tolerances must be finite and nonnegative")
        if self.material_fraction_tolerance >= 0.5:
            raise ValueError("material tolerance must distinguish opposite regions")
        if self.boundary_fraction_tolerance > 1 or self.boundary_band_ratio <= 0:
            raise ValueError("invalid boundary context tolerance")


@dataclass(frozen=True)
class Sample:
    image: Pixels
    reference: Pixels
    gt: Pixels
    mask: Mask
    artifacts: Artifacts
    checks: list[Check]
    parameters: dict[str, Any]


def region_mask(shape: tuple[int, int], box: Box) -> Mask:
    height, width = shape
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("annotation box must be inside the image")
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def normalized_box_to_pixels(box: Sequence[float], shape: tuple[int, int]) -> Box:
    """Convert explicitly declared 0..1000 xyxy annotations, without guessing units."""
    if len(box) != 4 or any(not 0 <= v <= 1000 for v in box):
        raise ValueError("expected four finite 0..1000 box coordinates")
    height, width = shape
    result = (
        int(np.floor(box[0] * width / 1000)),
        int(np.floor(box[1] * height / 1000)),
        int(np.ceil(box[2] * width / 1000)),
        int(np.ceil(box[3] * height / 1000)),
    )
    region_mask(shape, result)
    return result


def morph(mask: Mask, radius: int, *, expand: bool) -> Mask:
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if radius == 0:
        return mask.copy()
    padded = np.pad(mask, radius, constant_values=False)
    operation = ImageFilter.MaxFilter if expand else ImageFilter.MinFilter
    filtered = Image.fromarray(padded.astype(np.uint8) * 255).filter(
        operation(2 * radius + 1)
    )
    return np.asarray(filtered)[radius:-radius, radius:-radius] > 0


def _shift(array: Any, dx: int, dy: int) -> Any:
    out = np.zeros_like(array)
    height, width = array.shape[:2]
    if abs(dx) >= width or abs(dy) >= height:
        return out
    sx, sy = max(0, -dx), max(0, -dy)
    tx, ty = max(0, dx), max(0, dy)
    w, h = width - abs(dx), height - abs(dy)
    out[ty : ty + h, tx : tx + w] = array[sy : sy + h, sx : sx + w]
    return out


def align_pair(
    source: Pixels,
    gt: Pixels,
    source_material: Mask,
    gt_material: Mask,
    excluded: Mask,
    *,
    radius: int,
    max_mismatch: float,
    min_pixels: int,
) -> AlignedPair:
    """Translate source into GT coordinates; never move GT geometry to fit defects."""
    shape = gt_material.shape
    if (
        gt_material.ndim != 2
        or gt_material.dtype != bool
        or source.shape != (*shape, 3)
        or gt.shape != source.shape
        or source.dtype != np.uint8
        or gt.dtype != np.uint8
        or source_material.shape != shape
        or source_material.dtype != bool
        or excluded.shape != shape
        or excluded.dtype != bool
    ):
        raise ValueError("pair requires same-sized RGB images and boolean masks")
    if radius < 0 or not 0 <= max_mismatch < 1 or min_pixels < 1:
        raise ValueError("invalid alignment limits")
    best: tuple[float, int, int, int] | None = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            valid = _shift(np.ones(shape, dtype=bool), dx, dy)
            use = valid & ~_shift(excluded, dx, dy)
            if int(use.sum()) < min_pixels:
                continue
            mismatch = float(
                (_shift(source_material, dx, dy) != gt_material)[use].mean()
            )
            score = (mismatch, abs(dx) + abs(dy), dy, dx)
            if best is None or score < best:
                best = score
    if best is None or best[0] > max_mismatch:
        raise ValueError(
            "alignment_failed: insufficient support or structural mismatch"
        )
    error, _, dy, dx = best
    return AlignedPair(
        _shift(source, dx, dy),
        _shift(source_material, dx, dy),
        gt.copy(),
        gt_material.copy(),
        _shift(excluded, dx, dy),
        _shift(np.ones(shape, dtype=bool), dx, dy),
        (dx, dy),
        error,
    )


def build_normal_reference(
    pair: AlignedPair,
    *,
    source_id: str,
    seed: int,
    min_material_pixels: int,
    boundary_margin: int,
    texture_std_cap: float,
) -> NormalReference:
    """Estimate robust material statistics outside defects; render ONLY GT geometry."""
    if (
        min_material_pixels < 1
        or texture_std_cap < 0
        or not np.isfinite(texture_std_cap)
    ):
        raise ValueError("invalid material sampling limits")
    rng = np.random.default_rng(seed)
    image = np.zeros_like(pair.source)
    parameters: dict[str, Any] = {
        "seed": seed,
        "shift_xy": pair.shift_xy,
        "mismatch_ratio": pair.mismatch_ratio,
        "boundary_margin": boundary_margin,
        "texture_std_cap": texture_std_cap,
        "materials": {},
    }
    for name, foreground in (("foreground", True), ("background", False)):
        target = pair.gt_material == foreground
        observed = pair.source_material == foreground
        support = (
            morph(target, boundary_margin, expand=False)
            & morph(observed, boundary_margin, expand=False)
            & pair.valid
            & ~pair.excluded
        )
        pool = pair.source[support].astype(np.float32)
        if len(pool) < min_material_pixels:
            raise ValueError(f"insufficient_material_support: {name}")
        color = np.median(pool, axis=0)
        std = np.minimum(
            1.4826 * np.median(np.abs(pool - color), axis=0), texture_std_cap
        )
        texture = rng.normal(0, 1, size=(int(target.sum()), 3)) * std + color
        image[target] = np.clip(np.rint(texture), 0, 255).astype(np.uint8)
        parameters["materials"][name] = {
            "pixels": len(pool),
            "median_rgb": color.tolist(),
            "std_rgb": std.tolist(),
        }
    return NormalReference(
        image, pair.gt.copy(), pair.gt_material.copy(), source_id, parameters
    )


def extract_defect(
    pair: AlignedPair,
    reference: NormalReference,
    *,
    source_id: str,
    annotation_box: Box,
    method: Literal["appearance", "extra", "missing", "reviewed"],
    min_area: int,
    max_box_fraction: float,
    boundary_margin: int,
    appearance_threshold: float = 0,
    reviewed_mask: Mask | None = None,
) -> DefectTemplate:
    """Derive local anomaly evidence within an annotation, not from material alone."""
    roi = region_mask(pair.gt_material.shape, annotation_box) & pair.valid
    if not 0 < max_box_fraction <= 1:
        raise ValueError("max_box_fraction must be in (0, 1]")
    if not np.array_equal(reference.material, pair.gt_material):
        raise ValueError("reference geometry does not match the aligned pair")
    if method == "reviewed":
        if (
            reviewed_mask is None
            or reviewed_mask.dtype != bool
            or reviewed_mask.shape != roi.shape
            or np.any(reviewed_mask & ~roi)
        ):
            raise ValueError(
                "reviewed mask must lie inside the annotation and valid image"
            )
        evidence = reviewed_mask
    elif method == "appearance":
        if appearance_threshold <= 0:
            raise ValueError("appearance extraction requires a positive threshold")
        interior = morph(pair.gt_material, boundary_margin, expand=False) | morph(
            ~pair.gt_material, boundary_margin, expand=False
        )
        distance = np.linalg.norm(
            pair.source.astype(np.float32) - reference.image.astype(np.float32), axis=2
        )
        evidence = (distance > appearance_threshold) & interior
    elif method == "extra":
        evidence = pair.source_material & ~morph(
            pair.gt_material, boundary_margin, expand=True
        )
    elif method == "missing":
        evidence = pair.gt_material & ~morph(
            pair.source_material, boundary_margin, expand=True
        )
    else:
        raise ValueError(f"unknown extraction method: {method}")
    template = extract_template(
        evidence & roi, min_area=min_area, source_rgb=pair.source
    )
    if int(template.mask.sum()) > int(roi.sum()) * max_box_fraction:
        raise ValueError("ambiguous_template: mask occupies too much of the annotation")
    return DefectTemplate(
        template, source_id, annotation_box, method, pair.gt_material.copy()
    )


def material_distance(material: Mask) -> NDArray[np.float32]:
    """Chebyshev distance to material boundaries, excluding image edges."""
    if material.ndim != 2 or material.dtype != bool:
        raise ValueError("material must be a boolean image")
    boundary = np.zeros_like(material)
    vertical = material[1:] != material[:-1]
    horizontal = material[:, 1:] != material[:, :-1]
    boundary[1:] |= vertical
    boundary[:-1] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    distance = np.full(material.shape, np.inf, dtype=np.float32)
    visited = boundary.copy()
    distance[visited] = 0
    if not visited.any():
        return distance
    step = 0
    while not visited.all():
        step += 1
        expanded = morph(visited, 1, expand=True)
        distance[expanded & ~visited] = step
        visited = expanded
    return distance


def context_profile(
    material: Mask,
    mask: Mask,
    *,
    boundary_band_ratio: float,
    distances: NDArray[np.float32] | None = None,
) -> dict[str, Any]:
    """Measure support from normal structure, never from anomalous pixel colors."""
    if mask.dtype != bool or mask.shape != material.shape or not mask.any():
        raise ValueError("context requires a nonempty mask matching the material map")
    if distances is None:
        distances = material_distance(material)
    scale = float(np.sqrt(mask.sum()))
    values = distances[mask] / scale
    clearance = float(values.min())
    return {
        "foreground_fraction": float(material[mask].mean()),
        "boundary_fraction": float((values <= boundary_band_ratio).mean()),
        "clearance": clearance if np.isfinite(clearance) else None,
    }


def mapped_source_material(
    reference: NormalReference,
    defect: DefectTemplate,
    *,
    material_mapping: Literal["same", "inverted"] | None = None,
    mapping_basis: str = "",
) -> Mask:
    """Same-pair identity is automatic; other mappings require explicit evidence."""
    same_pair = reference.source_id == defect.source_id and np.array_equal(
        reference.material, defect.source_material
    )
    if material_mapping is None:
        if not same_pair:
            raise ValueError(
                "material_mapping_required: verify cross-reference regions"
            )
        return defect.source_material
    if material_mapping not in {"same", "inverted"} or not mapping_basis.strip():
        raise ValueError("explicit material mapping requires its supporting evidence")
    return (
        defect.source_material
        if material_mapping == "same"
        else ~defect.source_material
    )


def context_candidates(
    reference: NormalReference,
    defect: DefectTemplate,
    *,
    policy: ContextPolicy | None = None,
    material_mapping: Literal["same", "inverted"] | None = None,
    mapping_basis: str = "",
) -> Mask:
    """Coarse center filter; full transformed support is checked independently later."""
    policy = policy or ContextPolicy()
    source = mapped_source_material(
        reference,
        defect,
        material_mapping=material_mapping,
        mapping_basis=mapping_basis,
    )
    fraction = float(source[defect.template.mask].mean())
    if fraction >= 1 - policy.material_fraction_tolerance:
        return reference.material.copy()
    if fraction <= policy.material_fraction_tolerance:
        return ~reference.material
    return np.ones_like(reference.material)


def context_check(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    policy: ContextPolicy,
    name: str = "source_context",
) -> Check:
    a, b = source["clearance"], target["clearance"]
    clearance_ok = (
        a is None and b is None
        if a is None or b is None
        else abs(a - b) <= policy.clearance_tolerance
    )
    passed = (
        abs(source["foreground_fraction"] - target["foreground_fraction"])
        <= policy.material_fraction_tolerance
        and abs(source["boundary_fraction"] - target["boundary_fraction"])
        <= policy.boundary_fraction_tolerance
        and clearance_ok
    )
    return Check(
        name, passed, json.dumps({"source": dict(source), "target": dict(target)})
    )


def placement_region(
    material: Mask,
    *,
    relation: Literal["interior", "exterior", "edge"],
    target_foreground: bool,
    margin: int,
) -> Mask:
    target = material if target_foreground else ~material
    if relation == "interior":
        return morph(target, margin, expand=False)
    if relation == "exterior":
        return morph(~target, margin, expand=False)
    if relation == "edge":
        return target & ~morph(target, max(1, margin), expand=False)
    raise ValueError(f"unknown placement relation: {relation}")


def structure_check(
    material: Mask,
    mask: Mask,
    *,
    operation: Literal["add", "remove", "appearance"],
    relation: Literal["edge", "interior", "any"],
    target_foreground: bool,
    component_delta: int | None,
    margin: int = 1,
) -> Check:
    target = material if target_foreground else ~material
    inside = bool(np.all(target[mask]))
    boundary = morph(material, margin, expand=True) ^ morph(
        material, margin, expand=False
    )
    touches = bool(np.any(morph(mask, 1, expand=True) & boundary))
    new = material.copy()
    if operation == "add":
        new[mask] = True
    elif operation == "remove":
        new[mask] = False
    elif operation != "appearance":
        raise ValueError(f"unknown structural operation: {operation}")
    delta = len(mask_components(new)) - len(mask_components(material))
    contact_ok = relation == "any" or (touches if relation == "edge" else not touches)
    passed = (
        inside and contact_ok and (component_delta is None or delta == component_delta)
    )
    return Check(
        "structure", passed, f"inside={inside}, touches={touches}, cc_delta={delta}"
    )


def highpass_residual(image: Pixels, *, radius: float) -> NDArray[np.float32]:
    """Signed gray(defect) minus Gaussian(gray(defect)); no reference subtraction."""
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Gaussian radius must be finite and positive")
    gray = Image.fromarray(image).convert("L")
    return np.asarray(gray, dtype=np.float32) - np.asarray(
        gray.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32
    )


def highpass_display(residual: NDArray[np.float32], *, gain: float) -> Pixels:
    """Black is zero response; brightness encodes magnitude of the signed residual."""
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("display gain must be finite and positive")
    return np.clip(np.abs(residual) * gain, 0, 255).astype(np.uint8)


def synthesize_sample(
    reference: NormalReference,
    defect: DefectTemplate,
    *,
    candidate_centers: Mask,
    allowed_mask: Mask,
    rules: Sequence[Callable[[Mask], Check]],
    seed: int,
    scale_xy: tuple[float, float],
    angle_deg: float,
    pair_transform: PairTransform,
    min_area: int,
    max_area: int,
    max_clip_fraction: float,
    max_attempts: int,
    feather: float,
    highpass_radius: float,
    highpass_gain: float,
    visibility_threshold: int,
    min_visible_pixels: int,
    fill_rgb: Pixels | None = None,
    context_policy: ContextPolicy | None = None,
    material_mapping: Literal["same", "inverted"] | None = None,
    mapping_basis: str = "",
) -> Sample:
    context_policy = context_policy or ContextPolicy()
    if (
        candidate_centers.dtype != bool
        or candidate_centers.shape != reference.material.shape
        or not candidate_centers.any()
    ):
        raise ValueError("no valid candidate centers")
    if (
        max_attempts < 1
        or not 0 <= max_clip_fraction < 1
        or not 1 <= min_area <= max_area
    ):
        raise ValueError("invalid candidate limits")
    if not rules:
        raise ValueError("explicit strategy rules are required")
    source_material = mapped_source_material(
        reference,
        defect,
        material_mapping=material_mapping,
        mapping_basis=mapping_basis,
    )
    source_context = context_profile(
        source_material,
        defect.template.mask,
        boundary_band_ratio=context_policy.boundary_band_ratio,
    )
    distances = material_distance(reference.material)

    def check_context(mask: Mask, name: str) -> Check:
        target = context_profile(
            reference.material,
            mask,
            boundary_band_ratio=context_policy.boundary_band_ratio,
            distances=distances,
        )
        return context_check(source_context, target, policy=context_policy, name=name)

    centers = np.argwhere(candidate_centers)
    rng = np.random.default_rng(seed)
    attempts: list[dict[str, Any]] = []
    for index in rng.permutation(len(centers))[:max_attempts]:
        cy, cx = (int(v) for v in centers[index])
        try:
            placed = transform_template(
                defect.template,
                target_shape=reference.material.shape,
                center_xy=(cx, cy),
                scale_xy=scale_xy,
                angle_deg=angle_deg,
            )
            placed = constrain_template(placed, allowed_mask)
            assert placed.uncropped_area is not None
            area = int(placed.mask.sum())
            clip = 1 - area / placed.uncropped_area
            checks = [
                Check("area", min_area <= area <= max_area, f"pixels={area}"),
                Check("clipping", clip <= max_clip_fraction, f"lost_fraction={clip}"),
                check_context(placed.mask, "source_context"),
            ]
            checks.extend(validate_rules(placed.mask, rules))
            if not all(c.passed for c in checks):
                raise ValueError(
                    "; ".join(f"{c.name}: {c.reason}" for c in checks if not c.passed)
                )
            image = inject_anomaly(
                reference.image,
                placed,
                base_role="gt_recolored_reference",
                fill_rgb=fill_rgb,
                feather=feather,
            )
            modified = np.any(image != reference.image, axis=2)
            if np.any(modified & ~placed.mask):
                raise ValueError("pixel changes outside defect mask")
            changed_context = check_context(modified, "modified_context")
            checks.append(changed_context)
            if not changed_context.passed:
                raise ValueError(f"{changed_context.name}: {changed_context.reason}")
            image, normal, mask = apply_pair_transform(
                image, reference.image, placed.mask, pair_transform
            )
            gt, _, _ = apply_pair_transform(
                reference.gt, reference.gt, placed.mask, pair_transform
            )
            artifacts = compute_artifacts(
                image,
                normal,
                mask,
                difference_fn=lambda s, _r: highpass_display(
                    highpass_residual(s, radius=highpass_radius), gain=highpass_gain
                ),
                difference_kind="defect_gray_gaussian_highpass",
                visibility_threshold=visibility_threshold,
                min_visible_pixels=min_visible_pixels,
            )
            return Sample(
                image,
                normal,
                gt,
                mask,
                artifacts,
                checks,
                {
                    "seed": seed,
                    "center_xy": [cx, cy],
                    "scale_xy": scale_xy,
                    "angle_deg": angle_deg,
                    "pair_transform": pair_transform,
                    "clipped_fraction": clip,
                    "rejected_attempts": attempts,
                    "highpass_radius": highpass_radius,
                    "highpass_gain": highpass_gain,
                    "defect_source": defect.source_id,
                    "reference_source": reference.source_id,
                    "extraction_method": defect.extraction_method,
                    "annotation_box": defect.annotation_box,
                    "reference_parameters": reference.parameters,
                    "source_context": source_context,
                    "context_policy": asdict(context_policy),
                    "material_mapping": material_mapping or "same_pair",
                    "mapping_basis": mapping_basis,
                },
            )
        except ValueError as exc:
            attempts.append({"center_xy": [cx, cy], "reason": str(exc)})
    raise ValueError(f"candidate_exhausted: {attempts}")


def review_status(
    result: Mapping[str, Any] | None,
    *,
    expected_category: str,
    expected_box: Box,
    min_iou: float,
    failed: bool = False,
    image_size: Sequence[int] | None = None,
) -> str:
    """Merge independently inferred category/location with expected annotation."""
    if failed or result is None:
        return "review_failed"
    required = {
        "category",
        "bbox_xyxy",
        "realistic",
        "extra_anomalies",
        "uncertain",
        "context_consistent",
    }
    if not required <= result.keys() or not 0 < min_iou <= 1:
        return "review_failed"
    box = result["bbox_xyxy"]
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(type(v) is not int for v in box)
    ):
        return "review_failed"
    if any(
        type(result[k]) is not bool
        for k in ("realistic", "extra_anomalies", "uncertain", "context_consistent")
    ):
        return "review_failed"
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        return "needs_review"
    if min(x1, y1) < 0 or (
        image_size is not None and (x2 > image_size[0] or y2 > image_size[1])
    ):
        return "needs_review"
    a, b, c, d = expected_box
    intersection = max(0, min(x2, c) - max(x1, a)) * max(0, min(y2, d) - max(y1, b))
    union = (x2 - x1) * (y2 - y1) + (c - a) * (d - b) - intersection
    passed = (
        not result["uncertain"]
        and result["realistic"]
        and not result["extra_anomalies"]
        and result["context_consistent"]
        and result["category"] == expected_category
        and intersection / union >= min_iou
    )
    return "passed" if passed else "needs_review"


def merge_review_outputs(
    samples: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    *,
    selected_ids: set[str],
    min_iou: float,
) -> list[dict[str, Any]]:
    """Read the managed vision wire format; match by image path, never row order."""
    answers: dict[str, Mapping[str, Any] | None] = {}
    for row in outputs:
        image_path = None
        try:
            messages = row["messages"]
            user = next(m for m in messages if m["role"] == "user")
            image_path = next(
                c["value"] for c in user["content"] if c["type"] == "image_url"
            )
            assistant = next(m for m in messages if m["role"] == "assistant")
            value = next(
                c["value"] for c in assistant["content"] if c["type"] == "text"
            )
            if not isinstance(value, str) or not isinstance(image_path, str):
                raise ValueError("invalid managed message content")
            _, separator, answer = value.rpartition("<answer>")
            if not separator or not answer.endswith("</answer>"):
                raise ValueError("missing managed answer envelope")
            parsed = json.loads(answer[: -len("</answer>")])
            if not isinstance(parsed, dict) or image_path in answers:
                raise ValueError("invalid or duplicate review")
            answers[image_path] = parsed
        except (KeyError, TypeError, ValueError, StopIteration):
            if isinstance(image_path, str):
                answers[image_path] = None
    merged = []
    for sample in samples:
        image_path = next(
            a["path"] for a in sample["artifacts"] if a["role"] == "image"
        )
        status = "not_reviewed"
        if sample["sample_id"] in selected_ids:
            status = review_status(
                answers.get(image_path),
                expected_category=sample["category"],
                expected_box=sample["bbox_xyxy"],
                min_iou=min_iou,
                image_size=sample.get("image_size"),
            )
        checks = sample.get("checks", [])
        program_passed = (
            sample.get("program_status") == "passed"
            and {"source_context", "modified_context"}
            <= {c.get("name") for c in checks}
            and all(c.get("passed") is True for c in checks)
        )
        merged.append(
            {
                **sample,
                "visual_status": status,
                "quality_status": "passed"
                if program_passed and status == "passed"
                else "needs_review",
                "training_ready": False,
            }
        )
    return merged


def freeze_review_binding(
    root: Path,
    *,
    samples_path: str,
    review_input_path: str,
    plan_path: str,
    selected_ids: set[str],
    min_iou: float,
) -> dict[str, Any]:
    """Bind review to the exact input, strategy and assets before model execution."""
    if not 0 < min_iou <= 1:
        raise ValueError("review requires positive min_iou")
    samples = [
        json.loads(line) for line in (root / samples_path).read_text().splitlines()
    ]
    if not selected_ids <= {s["sample_id"] for s in samples}:
        raise ValueError("selected review IDs must exist")
    paths = {samples_path, review_input_path, plan_path}
    for sample in samples:
        paths.update(a["path"] for a in sample["artifacts"])
    review_rows = [
        json.loads(line) for line in (root / review_input_path).read_text().splitlines()
    ]
    if any("review_evidence_id" in row for row in review_rows):
        raise ValueError("review input is already bound; create a new review input")
    for row in review_rows:
        paths.update(row["images"])
    files = {}
    for name in sorted(paths):
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
            raise ValueError("review evidence must be within output root")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence_id = hashlib.sha256(
        json.dumps(
            {
                "files": files,
                "selected_ids": sorted(selected_ids),
                "min_iou": min_iou,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    for row in review_rows:
        row["review_evidence_id"] = evidence_id
        row["user_prompt"] = (
            row.get("user_prompt", "") + f"\nReview evidence ID: {evidence_id}"
        )
    (root / review_input_path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review_rows)
    )
    files[review_input_path] = hashlib.sha256(
        (root / review_input_path).read_bytes()
    ).hexdigest()
    return {
        "files": files,
        "samples_path": samples_path,
        "review_input_path": review_input_path,
        "selected_ids": sorted(selected_ids),
        "min_iou": min_iou,
        "review_evidence_id": evidence_id,
    }


def finalize_review(
    root: Path, binding: Mapping[str, Any], outputs: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derive final status; stale assets invalidate model approval."""
    stale = []
    for name, digest in binding["files"].items():
        path = (root / name).resolve()
        if (
            Path(name).is_absolute()
            or not path.is_relative_to(root.resolve())
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            stale.append(name)
    sample_path = (root / binding["samples_path"]).resolve()
    if not sample_path.is_relative_to(root.resolve()):
        raise ValueError("samples path escapes output root")
    samples = [json.loads(line) for line in sample_path.read_text().splitlines()]
    bound_outputs = []
    for row in outputs:
        try:
            user = next(m for m in row["messages"] if m["role"] == "user")
            expected = f"Review evidence ID: {binding['review_evidence_id']}"
            if any(
                isinstance(c, dict)
                and c.get("type") == "text"
                and isinstance(c.get("value"), str)
                and expected in c["value"]
                for c in user["content"]
            ):
                bound_outputs.append(row)
        except (KeyError, TypeError, StopIteration):
            continue
    merged = merge_review_outputs(
        samples,
        [] if stale else bound_outputs,
        selected_ids=set(binding["selected_ids"]),
        min_iou=binding["min_iou"],
    )
    counts: dict[str, int] = {}
    for row in merged:
        counts[row["visual_status"]] = counts.get(row["visual_status"], 0) + 1
    summary = {
        "quality_status": "passed"
        if merged and not stale and all(r["quality_status"] == "passed" for r in merged)
        else "needs_review",
        "visual_status_counts": counts,
        "stale_evidence": stale,
        "training_ready": False,
    }
    return merged, summary


def review_grid(images: Sequence[Pixels], *, box: Box | None = None) -> Image.Image:
    """Draw an outline on the first (synthetic) panel; never paint over its interior."""
    panels = [Image.fromarray(a).convert("RGB") for a in images]
    if box is not None:
        x1, y1, x2, y2 = box
        ImageDraw.Draw(panels[0]).rectangle(
            (x1, y1, x2 - 1, y2 - 1), outline="cyan", width=1
        )
    canvas = Image.new(
        "RGB", (sum(p.width for p in panels), max(p.height for p in panels))
    )
    left = 0
    for panel in panels:
        canvas.paste(panel, (left, 0))
        left += panel.width
    return canvas


def export_sample(
    sample: Sample, output_dir: Path, sample_id: str, category: str
) -> dict[str, Any]:
    if not sample_id or Path(sample_id).name != sample_id or sample_id in {".", ".."}:
        raise ValueError("sample_id must be a plain name")
    target = output_dir / "assets" / sample_id
    target.mkdir(parents=True, exist_ok=False)
    arrays = {
        "image": sample.image,
        "reference": sample.reference,
        "original_gt": sample.gt,
        "mask": sample.mask.astype(np.uint8) * 255,
        "diff": sample.artifacts.difference,
    }
    artifacts = []
    roles = {
        "image": "image",
        "reference": "gt",
        "original_gt": "other",
        "mask": "annotation",
        "diff": "diff",
    }
    for name, array in arrays.items():
        Image.fromarray(array).save(target / f"{name}.png")
        artifacts.append(
            {
                "role": roles[name],
                "path": f"assets/{sample_id}/{name}.png",
                "semantics": name,
            }
        )
    residual = highpass_residual(
        sample.image, radius=sample.parameters["highpass_radius"]
    )
    np.save(target / "highpass_signed.npy", residual)
    artifacts.append(
        {
            "role": "other",
            "path": f"assets/{sample_id}/highpass_signed.npy",
            "semantics": "signed_gray_minus_gaussian",
        }
    )
    review_grid(
        [sample.image, sample.reference, sample.artifacts.difference],
        box=sample.artifacts.bbox_xyxy,
    ).save(target / "review.png")
    artifacts.append({"role": "review", "path": f"assets/{sample_id}/review.png"})
    return {
        "sample_id": sample_id,
        "category": category,
        "bbox_xyxy": sample.artifacts.bbox_xyxy,
        "image_size": [sample.image.shape[1], sample.image.shape[0]],
        "artifacts": artifacts,
        "program_status": "passed",
        "visual_status": "not_reviewed",
        "training_ready": False,
        "difference_kind": sample.artifacts.difference_kind,
        "parameters": sample.parameters,
        "checks": [
            {"name": c.name, "passed": c.passed, "reason": c.reason}
            for c in sample.checks
        ],
    }
