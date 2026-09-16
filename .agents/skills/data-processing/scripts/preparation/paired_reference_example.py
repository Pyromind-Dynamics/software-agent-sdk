"""Configurable paired-reference pilot: pipeline.py plan.json processed.jsonl."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import reference_synthesis as rs
from PIL import Image
from template_synthesis import bbox_from_mask


def main(input_path: str, output_path: str) -> None:
    source_plan = json.loads(Path(input_path).read_text())
    policy = rs.ContextPolicy(**source_plan.get("context_policy", {}))
    source_plan["context_policy"] = asdict(policy)
    source_plan["pipeline_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    source_plan.setdefault("review_min_iou", 0.5)
    for key in ("donor_rgb", "donor_cam"):
        source_plan[key] = str(
            (Path(input_path).resolve().parent / source_plan[key]).resolve()
        )
    out = Path(output_path).parent
    if (
        source_plan["source_split"] != "train"
        and source_plan["purpose"] != "method_validation"
    ):
        raise ValueError("training candidates require an explicit train split")
    if (out / "assets").exists() or Path(output_path).exists():
        raise ValueError("use a new output directory for each pilot")
    if not source_plan["variants"]:
        raise ValueError("at least one variant is required")
    out.mkdir(parents=True, exist_ok=True)
    (out / "augmentation_plan.json").write_text(
        json.dumps(source_plan, ensure_ascii=False, indent=2)
    )
    real = np.asarray(Image.open(source_plan["donor_rgb"]).convert("RGB"))
    gt = np.asarray(Image.open(source_plan["donor_cam"]).convert("RGB"))
    gray = np.asarray(Image.fromarray(real).convert("L"))
    source_material = gray > source_plan["source_material_threshold"]
    gt_material = np.all(
        (gt >= source_plan["gt_foreground_min"])
        & (gt <= source_plan["gt_foreground_max"]),
        axis=2,
    )
    box = rs.normalized_box_to_pixels(
        source_plan["annotation_box_1000"], gt_material.shape
    )
    exclusion = rs.morph(
        rs.region_mask(gt_material.shape, box),
        source_plan["exclusion_margin"],
        expand=True,
    )
    pair = rs.align_pair(
        real,
        gt,
        source_material,
        gt_material,
        exclusion,
        **source_plan["alignment"],
    )
    dx, dy = pair.shift_xy
    aligned_box = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
    reference = rs.build_normal_reference(
        pair,
        source_id=source_plan["donor_id"],
        **source_plan["reference"],
    )
    defect = rs.extract_defect(
        pair,
        reference,
        source_id=source_plan["donor_id"],
        annotation_box=aligned_box,
        **source_plan["extraction"],
    )
    mask = defect.template.mask
    masked = np.zeros_like(pair.source)
    masked[mask] = pair.source[mask]
    rs.review_grid(
        [pair.source, pair.gt, masked, mask.astype(np.uint8) * 255], box=aligned_box
    ).save(out / "template_review.png")
    rs.review_grid([gt, reference.image, pair.source]).save(
        out / "reference_review.png"
    )
    Image.fromarray(mask.astype(np.uint8) * 255).save(out / "template_mask.png")
    Image.fromarray(pair.source).save(out / "source_aligned.png")
    Image.fromarray(pair.gt).save(out / "source_gt.png")
    x1, y1, x2, y2 = aligned_box
    rs.review_grid(
        [
            pair.source[y1:y2, x1:x2],
            pair.gt[y1:y2, x1:x2],
            masked[y1:y2, x1:x2],
            mask[y1:y2, x1:x2].astype(np.uint8) * 255,
        ]
    ).save(out / "template_crop_review.png")

    def rule(injected: rs.Mask) -> rs.Check:
        return rs.structure_check(
            reference.material,
            injected,
            **source_plan["structure_rule"],
        )

    records = []
    reviews = []
    panels = []
    failures = []
    target_material = (
        reference.material
        if source_plan["structure_rule"]["target_foreground"]
        else ~reference.material
    )
    for i, variant in enumerate(source_plan["variants"]):
        sid = f"candidate_{i + 1:02d}"
        try:
            result = rs.synthesize_sample(
                reference,
                defect,
                candidate_centers=rs.placement_region(
                    reference.material,
                    **source_plan["placement"],
                )
                & rs.context_candidates(reference, defect, policy=policy),
                allowed_mask=rs.morph(
                    target_material, source_plan["allowed_margin"], expand=False
                ),
                rules=[rule],
                context_policy=policy,
                **source_plan["synthesis"],
                **variant,
            )
        except ValueError as exc:
            failures.append(
                {"sample_id": sid, "stage": "synthesis", "reason": str(exc)}
            )
            continue
        record = rs.export_sample(result, out, sid, source_plan["category"])
        record["source_annotation_note"] = source_plan.get("annotation_note", "")
        record["use"] = source_plan["purpose"]
        record["source_split"] = source_plan["source_split"]
        records.append(record)
        panels.append(
            rs.review_grid(
                [result.image, result.reference, result.artifacts.difference],
                box=result.artifacts.bbox_xyxy,
            )
        )
        reviews.append(
            {
                "id": sid,
                "images": [
                    f"assets/{sid}/{name}.png"
                    for name in ("image", "reference", "diff")
                ]
                + ["source_aligned.png", "source_gt.png", "template_mask.png"],
                "image_labels": [
                    "candidate",
                    "normal reference",
                    "grayscale high-pass",
                    "source defect image (untransformed)",
                    "source normal structure (same coordinates as source)",
                    "extracted source defect mask",
                ],
                "user_prompt": f"Inspect image 1 ({gt.shape[1]}x{gt.shape[0]} pixels). "
                + source_plan["category_definitions"]
                + " Compare its material support and boundary relationship with "
                "the source defect indicated in image 6 on images 4 and 5. "
                "Rotation or translation alone is not a violation. The category "
                "name and color alone do not establish the supporting material.",
            }
        )
    if panels:
        grid = Image.new("RGB", (panels[0].width, sum(p.height for p in panels)))
        for i, panel in enumerate(panels):
            grid.paste(panel, (0, i * panel.height))
        grid.save(out / "review_grid.png")
    for name, rows in [
        (Path(output_path).name, records),
        ("review_input.jsonl", reviews),
        ("provenance.jsonl", records),
        ("failures.jsonl", failures),
    ]:
        (out / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    binding = rs.freeze_review_binding(
        out,
        samples_path=Path(output_path).name,
        review_input_path="review_input.jsonl",
        plan_path="augmentation_plan.json",
        selected_ids={r["sample_id"] for r in records},
        min_iou=source_plan["review_min_iou"],
    )
    (out / "review_binding.json").write_text(json.dumps(binding, indent=2))
    report = {
        "generated": len(records),
        "requested": len(source_plan["variants"]),
        "failures": failures,
        "training_ready": False,
        "status": "candidate_review_only",
        "visual_status": "not_reviewed",
        "source_split": source_plan["source_split"],
        "use": source_plan["purpose"],
        "alignment": reference.parameters,
        "template_pixels": int(mask.sum()),
        "template_bbox": bbox_from_mask(mask),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            k: hashlib.sha256(Path(source_plan[k]).read_bytes()).hexdigest()
            for k in ("donor_rgb", "donor_cam")
        },
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out / "validation.json").write_text(
        json.dumps(
            {
                "program_status": "passed" if records and not failures else "failed",
                "visual_status": "not_reviewed",
                "training_ready": False,
            }
        )
    )
    (out / "progress.json").write_text(
        json.dumps(
            {
                "processed": len(source_plan["variants"]),
                "generated": len(records),
                "failed": len(failures),
                "status": "finished" if records else "failed",
            }
        )
    )
    (out / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Template synthesis review</title>'
        "<h2>Method-validation candidates; visual review pending</h2>"
        "<p>Template: source with annotation | GT | extracted pixels | mask</p>"
        '<img src="template_review.png"><p>GT | recolored normal | real capture</p>'
        '<img src="reference_review.png"><p>Synthetic with box | normal | high-pass</p>'
        '<img src="review_grid.png">'
    )
    print(json.dumps(report, ensure_ascii=False))
    if not records:
        raise ValueError("no candidates passed; see failures.jsonl")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
