"""Domain-neutral mask/texture helpers; callers own semantics and orchestration."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter


Mask = NDArray[np.bool_]
Pixels = NDArray[np.uint8]
Box = tuple[int, int, int, int]
PairTransform = Literal["identity", "flip_lr", "flip_ud", "rotate_180"]


@dataclass(frozen=True)
class Template:
    mask: Mask
    texture: Pixels | None = None
    uncropped_area: int | None = None


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool | None
    reason: str
    measurements: dict[str, Any] = field(default_factory=dict)
    stage: str = "legacy"


@dataclass(frozen=True)
class Artifacts:
    difference: Pixels
    difference_kind: str
    bbox_xyxy: Box
    visible_pixels: int


@dataclass(frozen=True)
class SourceEvidence:
    image: Pixels | None
    reference: Pixels | None
    annotation: Mask
    template: Template
    regions: Mapping[str, Mask] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateEvidence:
    reference: Pixels
    transformed: Template
    constrained: Template
    allowed_mask: Mask
    regions: Mapping[str, Mask]
    clipped_fraction: float


@dataclass(frozen=True)
class RenderedEvidence:
    image: Pixels
    reference: Pixels
    mask: Mask
    changed: Mask
    regions: Mapping[str, Mask]
    bbox_xyxy: Box


@dataclass(frozen=True)
class Acceptance:
    min_area: int
    max_area: int
    max_clip_fraction: float
    visibility_threshold: int
    min_visible_pixels: int

    def __post_init__(self) -> None:
        if not (
            1 <= self.min_area <= self.max_area
            and 0 <= self.max_clip_fraction < 1
            and 0 <= self.visibility_threshold < 255
            and self.min_visible_pixels >= 1
        ):
            raise ValueError("invalid acceptance limits")


@dataclass(frozen=True)
class GeneratedSample:
    image: Pixels
    reference: Pixels
    mask: Mask
    artifacts: Artifacts
    checks: list[Check]
    parameters: dict[str, Any]
    regions: Mapping[str, Mask]
    source: SourceEvidence


class SynthesisFailure(ValueError):
    def __init__(self, code: str, attempts: list[dict[str, Any]]) -> None:
        self.code = code
        self.attempts = attempts
        super().__init__(f"{code}: {attempts}")


def validation_coverage(checks: Sequence[Check]) -> dict[str, bool]:
    return {
        stage: bool(selected := [c for c in checks if c.stage == stage])
        and all(c.passed is True for c in selected)
        for stage in ("source", "candidate", "rendered")
    }


def _stage_checks(checks: Sequence[Check], stage: str) -> list[Check]:
    return [replace(c, stage=stage) for c in checks] or [
        Check("missing_verdict", None, "strategy returned no checks", stage=stage)
    ]


def transform_region(mask: Mask, transform: PairTransform) -> Mask:
    """Transform target-space evidence, including empty regions, with the image."""
    _check_mask(mask)
    if transform == "identity":
        return mask.copy()
    if transform == "flip_lr":
        return np.fliplr(mask).copy()
    if transform == "flip_ud":
        return np.flipud(mask).copy()
    if transform == "rotate_180":
        return np.rot90(mask, 2).copy()
    raise ValueError(f"unsupported pair transform: {transform}")


def synthesize_template(
    source: SourceEvidence,
    reference: Pixels,
    *,
    candidate_centers: Mask,
    allowed_mask: Mask,
    acceptance: Acceptance,
    seed: int,
    max_attempts: int,
    scale_xy: tuple[float, float] = (1.0, 1.0),
    angle_deg: float = 0,
    pair_transform: PairTransform = "identity",
    feather: float = 0,
    fill_rgb: Pixels | None = None,
    regions: Mapping[str, Mask] | None = None,
    validate_source: Callable[[SourceEvidence], Sequence[Check]] | None = None,
    validate_candidate: Callable[[CandidateEvidence], Sequence[Check]] | None = None,
    validate_rendered: Callable[[RenderedEvidence], Sequence[Check]] | None = None,
) -> GeneratedSample:
    """Search with fixed acceptance; strategies judge meaning, including final pixels.

    Omitted callbacks support legacy exploration but leave validation incomplete.
    Callbacks receive copies; they do not control acceptance or rendering inputs.
    """
    source, reference = deepcopy(source), reference.copy()
    candidate_centers, allowed_mask = candidate_centers.copy(), allowed_mask.copy()
    regions = deepcopy(dict(regions or {}))
    fill_rgb = None if fill_rgb is None else fill_rgb.copy()
    _check_mask(source.annotation)
    if source.image is not None:
        _check_rgb(source.image, source.annotation.shape)
    if source.reference is not None:
        _check_rgb(source.reference, source.annotation.shape)
    bbox_from_mask(source.template.mask)
    if source.template.mask.shape != source.annotation.shape or np.any(
        source.template.mask & ~source.annotation
    ):
        raise ValueError("source template must lie inside its annotation")
    _check_mask(allowed_mask)
    _check_mask(candidate_centers)
    _check_rgb(reference, allowed_mask.shape)
    if candidate_centers.shape != allowed_mask.shape or max_attempts < 1:
        raise ValueError("invalid candidate search")
    for masks, shape in (
        (source.regions, source.annotation.shape),
        (regions, allowed_mask.shape),
    ):
        for mask in masks.values():
            _check_mask(mask)
            if mask.shape != shape:
                raise ValueError("region must match its image coordinates")
    source_checks = (
        _stage_checks(validate_source(deepcopy(source)), "source")
        if validate_source is not None and source.image is not None
        else []
    )
    if validate_source is not None and source.image is None:
        source_checks.append(
            Check(
                "source_image",
                None,
                "original source image is unavailable",
                stage="source",
            )
        )
    if any(c.passed is not True for c in source_checks):
        raise SynthesisFailure(
            "source_rejected",
            [{"stage": "source", "checks": [asdict(c) for c in source_checks]}],
        )
    centers = np.argwhere(candidate_centers)
    if not len(centers):
        raise SynthesisFailure(
            "candidate_exhausted", [{"stage": "search", "reason": "no centers"}]
        )
    attempts: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for index in rng.permutation(len(centers))[:max_attempts]:
        cy, cx = (int(v) for v in centers[index])
        checks = list(source_checks)
        stage = "candidate"
        try:
            transformed = transform_template(
                source.template,
                target_shape=allowed_mask.shape,
                center_xy=(cx, cy),
                scale_xy=scale_xy,
                angle_deg=angle_deg,
            )
            placed = constrain_template(transformed, allowed_mask)
            assert placed.uncropped_area is not None
            area = int(placed.mask.sum())
            clip = 1 - area / placed.uncropped_area
            checks.extend(
                [
                    Check(
                        "area",
                        acceptance.min_area <= area <= acceptance.max_area,
                        f"pixels={area}",
                        {"pixels": area},
                        "geometry",
                    ),
                    Check(
                        "clipping",
                        clip <= acceptance.max_clip_fraction,
                        f"lost_fraction={clip}",
                        {"lost_fraction": clip},
                        "geometry",
                    ),
                ]
            )
            candidate = CandidateEvidence(
                reference, transformed, placed, allowed_mask, regions, clip
            )
            if validate_candidate is not None:
                checks.extend(
                    _stage_checks(validate_candidate(deepcopy(candidate)), "candidate")
                )
            if any(c.passed is not True for c in checks):
                raise ValueError("candidate validation failed")
            stage = "rendered"
            image = inject_anomaly(
                reference,
                placed,
                base_role="normal_image",
                fill_rgb=fill_rgb,
                feather=feather,
            )
            image, normal, mask = apply_pair_transform(
                image, reference, placed.mask, pair_transform
            )
            final_regions = {
                name: transform_region(region, pair_transform)
                for name, region in regions.items()
            }
            changed = np.any(image != normal, axis=2)
            if np.any(changed & ~mask):
                raise ValueError("pixel changes outside defect mask")
            artifacts = compute_artifacts(
                image,
                normal,
                mask,
                difference_fn=lambda a, b: np.abs(
                    a.astype(np.int16) - b.astype(np.int16)
                ).astype(np.uint8),
                difference_kind="absolute_reference_difference",
                visibility_threshold=acceptance.visibility_threshold,
                min_visible_pixels=acceptance.min_visible_pixels,
            )
            checks.append(
                Check(
                    "pixel_change",
                    True,
                    "visible change confined to template",
                    {"visible_pixels": artifacts.visible_pixels},
                    "pixels",
                )
            )
            rendered = RenderedEvidence(
                image, normal, mask, changed, final_regions, artifacts.bbox_xyxy
            )
            if validate_rendered is not None:
                checks.extend(
                    _stage_checks(validate_rendered(deepcopy(rendered)), "rendered")
                )
            if any(c.passed is not True for c in checks):
                raise ValueError("rendered validation failed")
            return GeneratedSample(
                image,
                normal,
                mask,
                artifacts,
                checks,
                {
                    "validation_version": 2,
                    "validation_coverage": validation_coverage(checks),
                    "acceptance": asdict(acceptance),
                    "search": {
                        "seed": seed,
                        "max_attempts": max_attempts,
                        "scale_xy": scale_xy,
                        "angle_deg": angle_deg,
                    },
                    "pair_transform": pair_transform,
                    "feather": feather,
                    "center_xy": [cx, cy],
                    "clipped_fraction": clip,
                    "rejected_attempts": attempts,
                },
                final_regions,
                source,
            )
        except ValueError as exc:
            attempts.append(
                {
                    "center_xy": [cx, cy],
                    "stage": stage,
                    "reason": str(exc),
                    "checks": [asdict(c) for c in checks],
                }
            )
    raise SynthesisFailure("candidate_exhausted", attempts)


def _check_mask(mask: Mask) -> None:
    if mask.ndim != 2 or mask.dtype != np.bool_ or 0 in mask.shape:
        raise ValueError("mask must be a nonempty 2D boolean array")


def _check_rgb(image: Pixels, shape: tuple[int, ...]) -> None:
    if image.dtype != np.uint8 or image.shape != (*shape, 3):
        raise ValueError("image must be uint8 RGB matching the mask shape")


def bbox_from_mask(mask: Mask) -> Box:
    """Return pixel xyxy coordinates with exclusive right/bottom edges."""
    _check_mask(mask)
    ys, xs = np.where(mask)
    if not xs.size:
        raise ValueError("empty defect mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_components(mask: Mask, min_area: int = 1) -> list[Mask]:
    """Return eight-connected regions; callers decide which regions are defects."""
    _check_mask(mask)
    if min_area < 1:
        raise ValueError("min_area must be positive")
    height, width = mask.shape
    seen = np.zeros_like(mask)
    result = []
    for y, x in zip(*np.where(mask), strict=True):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        pixels = []
        while stack:
            yy, xx = stack.pop()
            pixels.append((yy, xx))
            for ny in range(max(0, yy - 1), min(height, yy + 2)):
                for nx in range(max(0, xx - 1), min(width, xx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if len(pixels) >= min_area:
            component = np.zeros_like(mask)
            yy, xx = zip(*pixels, strict=True)
            component[yy, xx] = True
            result.append(component)
    return result


def extract_template(
    evidence_mask: Mask,
    *,
    bbox_xyxy: Box | None = None,
    min_area: int = 1,
    source_rgb: Pixels | None = None,
) -> Template:
    """Accept a labeled mask or explicitly decoded difference; do not guess channels."""
    _check_mask(evidence_mask)
    mask = evidence_mask.copy()
    height, width = mask.shape
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("bbox must be inside the evidence image")
        mask[:] = False
        mask[y1:y2, x1:x2] = evidence_mask[y1:y2, x1:x2]
    components = mask_components(mask, min_area)
    mask[:] = False
    for component in components:
        mask |= component
    bbox_from_mask(mask)
    if source_rgb is not None:
        _check_rgb(source_rgb, mask.shape)
    return Template(mask, None if source_rgb is None else source_rgb.copy())


def transform_template(
    template: Template,
    *,
    target_shape: tuple[int, int],
    center_xy: tuple[int, int],
    scale_xy: tuple[float, float] = (1.0, 1.0),
    angle_deg: float = 0.0,
) -> Template:
    """Move mask and donor texture together using explicit transform parameters."""
    x1, y1, x2, y2 = bbox_from_mask(template.mask)
    if len(target_shape) != 2 or min(target_shape) <= 0:
        raise ValueError("target_shape must contain positive height and width")
    if any(not math.isfinite(s) or s <= 0 for s in scale_xy):
        raise ValueError("scales must be finite and positive")
    if not math.isfinite(angle_deg):
        raise ValueError("angle must be finite")
    size = (
        max(1, round((x2 - x1) * scale_xy[0])),
        max(1, round((y2 - y1) * scale_xy[1])),
    )

    def warp(array: Pixels) -> Pixels:
        return np.asarray(
            Image.fromarray(array)
            .resize(size, Image.Resampling.NEAREST)
            .rotate(
                angle_deg, resample=Image.Resampling.NEAREST, expand=True, fillcolor=0
            ),
            dtype=np.uint8,
        )

    crop = warp(template.mask[y1:y2, x1:x2].astype(np.uint8) * 255)
    height, width = target_shape
    left = center_xy[0] - crop.shape[1] // 2
    top = center_xy[1] - crop.shape[0] // 2
    dx1, dy1 = max(0, left), max(0, top)
    dx2, dy2 = min(width, left + crop.shape[1]), min(height, top + crop.shape[0])
    if dx2 <= dx1 or dy2 <= dy1:
        raise ValueError("transformed template is outside the target")
    source_slice = (slice(dy1 - top, dy2 - top), slice(dx1 - left, dx2 - left))
    target_slice = (slice(dy1, dy2), slice(dx1, dx2))
    mask = np.zeros(target_shape, dtype=bool)
    mask[target_slice] = crop[source_slice] > 0
    bbox_from_mask(mask)
    texture = None
    if template.texture is not None:
        _check_rgb(template.texture, template.mask.shape)
        texture = np.zeros((*target_shape, 3), dtype=np.uint8)
        texture[target_slice] = warp(template.texture[y1:y2, x1:x2])[source_slice]
    return Template(mask, texture, int(np.count_nonzero(crop)))


def constrain_template(template: Template, allowed_mask: Mask) -> Template:
    _check_mask(template.mask)
    _check_mask(allowed_mask)
    if allowed_mask.shape != template.mask.shape:
        raise ValueError("allowed mask must match the target shape")
    mask = template.mask & allowed_mask
    bbox_from_mask(mask)
    return Template(mask, template.texture, template.uncropped_area)


def validate_rules(mask: Mask, rules: Sequence[Callable[[Mask], Check]]) -> list[Check]:
    """Evaluate domain callbacks before rendering, separately from semantic review."""
    bbox_from_mask(mask)
    if not rules:
        raise ValueError("provide explicit strategy validation rules")
    return [rule(mask.copy()) for rule in rules]


def inject_anomaly(
    normal_rgb: Pixels,
    template: Template,
    *,
    base_role: str,
    fill_rgb: Pixels | None = None,
    feather: float = 0.0,
) -> Pixels:
    """Render on a COPY of a verified normal image, using donor or supplied material."""
    if base_role not in {"normal_image", "gt_recolored_reference"}:
        raise ValueError("base_role must be normal_image or gt_recolored_reference")
    bbox_from_mask(template.mask)
    _check_rgb(normal_rgb, template.mask.shape)
    texture = template.texture if fill_rgb is None else fill_rgb
    if texture is None:
        raise ValueError("provide donor texture or a target-sized material image")
    _check_rgb(texture, template.mask.shape)
    if not math.isfinite(feather) or feather < 0:
        raise ValueError("feather must be finite and nonnegative")
    alpha = template.mask.astype(np.float32)
    if feather:
        blurred = Image.fromarray(template.mask.astype(np.uint8) * 255).filter(
            ImageFilter.GaussianBlur(radius=feather)
        )
        # Keep all edits inside the annotated support, including feathered edges.
        alpha *= np.asarray(blurred, dtype=np.float32) / 255.0
    alpha = alpha[:, :, None]
    return np.rint(normal_rgb * (1.0 - alpha) + texture * alpha).astype(np.uint8)


def apply_pair_transform(
    synthetic: Pixels,
    reference: Pixels,
    mask: Mask,
    transform: PairTransform,
) -> tuple[Pixels, Pixels, Mask]:
    bbox_from_mask(mask)
    _check_rgb(synthetic, mask.shape)
    _check_rgb(reference, mask.shape)
    arrays = (synthetic, reference, mask)
    if transform == "identity":
        return synthetic.copy(), reference.copy(), mask.copy()
    if transform == "flip_lr":
        return tuple(np.fliplr(a).copy() for a in arrays)
    if transform == "flip_ud":
        return tuple(np.flipud(a).copy() for a in arrays)
    if transform == "rotate_180":
        return tuple(np.rot90(a, 2).copy() for a in arrays)
    raise ValueError(f"unsupported pair transform: {transform}")


def compute_artifacts(
    synthetic: Pixels,
    reference: Pixels,
    injected_mask: Mask,
    *,
    difference_fn: Callable[[Pixels, Pixels], Pixels],
    difference_kind: str,
    visibility_threshold: int,
    min_visible_pixels: int,
) -> Artifacts:
    bbox = bbox_from_mask(injected_mask)
    _check_rgb(synthetic, injected_mask.shape)
    _check_rgb(reference, injected_mask.shape)
    if not difference_kind.strip():
        raise ValueError("name the difference definition explicitly")
    if not 0 <= visibility_threshold < 255 or min_visible_pixels < 1:
        raise ValueError("invalid visibility limits")
    changed = np.any(synthetic != reference, axis=2)
    if not np.any(changed & injected_mask):
        raise ValueError("injection produced no pixel change inside the mask")
    difference = difference_fn(synthetic.copy(), reference.copy())
    if difference.dtype != np.uint8 or difference.shape not in (
        injected_mask.shape,
        (*injected_mask.shape, 3),
    ):
        raise ValueError("difference must be matching uint8 grayscale or RGB")
    intensity = difference if difference.ndim == 2 else difference.max(axis=2)
    visible = int(
        np.count_nonzero((intensity > visibility_threshold) & injected_mask & changed)
    )
    if visible < min_visible_pixels:
        raise ValueError(f"insufficient visible defect pixels: {visible}")
    return Artifacts(difference, difference_kind, bbox, visible)
