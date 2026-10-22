"""Independent CPU jobs for distribution, review import, and report rendering.

The review job uses the existing annotation schema and UI assets. It neither
loads a model nor executes a scoring script. User extraction code is executed
only by the profile job in this CPU worker.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

from evaluate_inference import (
    _atomic_json,
    _build_html,
    _config_digest,
    _file_digest,
    _report_capabilities,
    _resolve_report_spec,
    _write_initial_annotation_products,
    _write_jsonl,
)
from evaluate_predictions import dataset_evidence, local_path


def read_frozen(reference: dict, mount: Path) -> bytes:
    path = local_path(reference["path"], mount)
    if _file_digest(path) != reference["sha256"]:
        raise ValueError(f"source fingerprint changed: {reference['path']}")
    return path.read_bytes()


def create_profile(manifest: dict, script: Path, output: Path, mount: Path) -> dict:
    module = types.ModuleType("dataset_extraction")
    exec(compile(script.read_bytes(), str(script), "exec"), module.__dict__)
    distribution, _, issues = dataset_evidence(manifest, module, mount, manifest)
    profile = {
        "schema_version": 1,
        "status": "incomplete" if issues else "ready",
        "datasets": manifest.get("datasets", {}),
        "distribution": distribution,
        "extraction_sha256": _file_digest(script),
        "label_mapping": manifest.get("label_mapping", {}),
        "issues": issues,
    }
    _atomic_json(output / "dataset_profile.json", profile)
    return profile


def require_profile(manifest: dict, mount: Path) -> dict:
    profile = json.loads(read_frozen(manifest["profile"], mount))
    if profile.get("status") != "ready" or profile.get("issues"):
        raise ValueError("complete dataset distribution before opening review")
    expected = manifest["datasets"]
    if profile["datasets"] != expected:
        raise ValueError("profile belongs to a different dataset version")
    for role in ("train", "validation"):
        if role not in expected or role not in profile["distribution"]:
            raise ValueError(f"{role} distribution is required before review")
    for entry in expected.values():
        # Verify the data still match the frozen profile, including an optional
        # test dataset. Do not recompute labels or execute user code here.
        read_frozen(entry, mount)
    if profile.get("label_mapping", {}) != manifest.get("label_mapping", {}):
        raise ValueError("profile label mapping does not match this review")
    return profile


def imported_cases(rows: list[dict], mapping: dict) -> list[dict]:
    """Map an existing badcase list to the shared annotation/UI row contract."""
    result = []
    for raw in rows:

        def field(name, default=None):
            return raw.get(mapping.get(name, name), default)

        if field("id") in (None, ""):
            raise ValueError("imported cases require stable IDs")
        if field("input") is None and not field("media"):
            raise ValueError("imported cases require input or media")
        if not field("failure_reason"):
            raise ValueError("imported cases require a problem description")
        result.append(
            {
                "id": str(field("id")),
                "sample": {"prompt": field("input", "")},
                "reference": field("reference"),
                "prediction": field("prediction"),
                "original_label": field("original_label", "unknown"),
                "media": field("media", []),
                "passed": False,
                "overall_score": 0.0,
                "latency_seconds": None,
                "evaluation_latency_seconds": 0.0,
                "error": None,
                "metrics": {},
                "details": {"imported": True},
                "rubric_results": [
                    {
                        "name": "imported_issue",
                        "criterion": (
                            "Imported issue, not an automatically scored verdict"
                        ),
                        "weight": 1,
                        "required": True,
                        "passed": False,
                        "score": 0.0,
                        "evidence": [],
                        "failure_reason": str(field("failure_reason")),
                    }
                ],
                "reference_label": raw.get(
                    mapping.get("reference_label", "reference_label")
                ),
                "prediction_label": raw.get(
                    mapping.get("prediction_label", "prediction_label")
                ),
                "category": raw.get(mapping.get("category", "category")),
            }
        )
    return result


def source_cases(manifest: dict, mount: Path) -> tuple[list[dict], dict]:
    source = manifest["source"]
    rows = [
        json.loads(line)
        for line in read_frozen(source, mount).splitlines()
        if line.strip()
    ]
    metrics = {}
    if source["kind"] == "import":
        rows = imported_cases(rows, manifest.get("field_mapping", {}))
    else:
        binding = json.loads(read_frozen(manifest["scoring_manifest"], mount))
        if (
            binding["run_id"] != source["scoring_run_id"]
            or binding["evaluation_config_sha256"] != source["scoring_spec_sha256"]
        ):
            raise ValueError("scoring source binding mismatch")
        if manifest.get("metrics"):
            metrics = json.loads(read_frozen(manifest["metrics"], mount))
        if any(not isinstance(row.get("passed"), bool) for row in rows):
            raise ValueError("scoring results require boolean passed values")
    ids = [str(row.get("id", "")) for row in rows]
    if any(not identity for identity in ids) or len(ids) != len(set(ids)):
        raise ValueError("case IDs must be present and unique")
    return rows, metrics


def render_existing_ui(
    rows: list[dict],
    metrics: dict,
    manifest: dict,
    output: Path,
    run_id: str,
    config_hash: str,
    mount: Path,
    *,
    review: bool,
) -> None:
    # Imported badcases have no full-dataset metric denominator. The renderer
    # explicitly suppresses aggregate quality statistics in this case.
    metrics = {
        "dataset_path": manifest["source"]["path"],
        "total": len(rows),
        "passed": 0,
        "failed": len(rows),
        "pass_rate": None,
        "aggregate_metrics": {},
        "rubric_metrics": {},
        **metrics,
        "review_only": review,
        "metrics_available": metrics.get("metrics_available", "pass_rate" in metrics)
        and manifest["source"]["kind"] == "scoring",
    }
    spec, _ = _resolve_report_spec(
        json.dumps(manifest["report_spec"]) if manifest.get("report_spec") else None,
        _report_capabilities(rows),
    )
    if review:
        spec["title"] = manifest.get("title", "Badcase 人工复核")
    else:
        spec["enabled_panels"] = [
            panel for panel in spec["enabled_panels"] if panel != "annotation"
        ]
    _atomic_json(output / "resolved_report_spec.json", spec)
    _build_html(
        rows,
        metrics,
        local_path(manifest["source"]["path"], mount),
        {},
        output,
        spec,
        "validated",
        {},
        run_id,
        config_hash,
        Path(__file__).parent,
    )


def create_review(manifest: dict, output: Path, run_id: str, mount: Path) -> dict:
    profile = require_profile(manifest, mount)
    all_rows, metrics = source_cases(manifest, mount)
    rows = [row for row in all_rows if not row["passed"]]
    source = manifest["source"]
    config_hash = source.get("scoring_spec_sha256") or _config_digest(source)
    task = {
        "schema_version": 1,
        "review_id": run_id,
        "source_kind": source["kind"],
        "source_path": source["path"],
        "source_sha256": source["sha256"],
        "scoring_run_id": source.get("scoring_run_id"),
        "scoring_spec_sha256": config_hash,
        "profile_sha256": manifest["profile"]["sha256"],
        "case_count": len(rows),
        "status": "ready",
    }
    for row in rows:
        label = str(row.get("original_label", "unknown"))
        row.setdefault("details", {})["distribution_context"] = {
            "label": label,
            **{
                role: values.get("labels", {}).get(label)
                for role, values in profile["distribution"].items()
            },
            "performance": metrics.get("original_label_metrics", {}).get(label),
        }
    # The binding is independent of the scoring run. No old human fields may
    # enter a new task, even if the imported file was a labeled export.
    for row in rows:
        for key in ("human_review", "human_annotation", "review_status"):
            row.pop(key, None)
    _write_jsonl(output / "predictions.jsonl", rows)
    _atomic_json(output / "dataset_profile.json", profile)
    _atomic_json(
        output / "run_manifest.json",
        {
            "schema_version": 2,
            "run_id": run_id,
            "evaluation_config_sha256": config_hash,
            "products_manifest_sha256": _config_digest(manifest),
            "kind": "badcase_review",
        },
    )
    _write_initial_annotation_products(output, rows, run_id, config_hash)
    # Ready is durable before optional rendering; rendering may be retried
    # without rescoring or resetting annotations.
    _atomic_json(output / "review_task.json", task)
    render_existing_ui(
        rows,
        {**metrics, "dataset_distribution": profile["distribution"]},
        manifest,
        output,
        run_id,
        config_hash,
        mount,
        review=True,
    )
    return task


def run_products(definition: dict, output: Path, run_id: str, mount: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifest = definition["products_manifest"]
    identity = _config_digest(definition)
    frozen = output / "products_definition.json"
    if frozen.exists() and _config_digest(json.loads(frozen.read_text())) != identity:
        raise ValueError("frozen inputs changed; create a new task")
    _atomic_json(frozen, definition)
    operation = definition["products_operation"]
    if operation == "profile":
        result = create_profile(manifest, output / "extract_dataset.py", output, mount)
    elif operation == "review":
        result = create_review(manifest, output, run_id, mount)
    else:
        rows, metrics = source_cases(manifest, mount)
        config_hash = manifest["source"].get("scoring_spec_sha256", identity)
        _atomic_json(
            output / "run_manifest.json",
            {
                "run_id": run_id,
                "evaluation_config_sha256": config_hash,
            },
        )
        render_existing_ui(
            rows, metrics, manifest, output, run_id, config_hash, mount, review=False
        )
        result = {"status": "ready", "source": manifest["source"]}
    _atomic_json(
        output / "report.json",
        {
            "status": "succeeded",
            "operation": operation,
            "result": result,
            "request_count": 0,
            **(
                {
                    "evaluation_status": "validated",
                    "report_spec": json.loads(
                        (output / "resolved_report_spec.json").read_text()
                    ),
                    "artifacts": {
                        "evaluation_report.html": str(
                            output / "evaluation_report.html"
                        ).removeprefix(str(mount)),
                    },
                }
                if (output / "evaluation_report.html").exists()
                else {}
            ),
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    run_products(
        json.loads((output / "products_definition.json").read_text()),
        output,
        args.run_id,
        Path("/workspace"),
    )


if __name__ == "__main__":
    main()
