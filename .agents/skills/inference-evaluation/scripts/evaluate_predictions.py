"""CPU-only scoring of saved predictions; user code executes only in this worker."""

from __future__ import annotations

import argparse
import json
import types
from collections import Counter, defaultdict
from pathlib import Path
from uuid import UUID, uuid5

from evaluate_inference import (
    _atomic_json,
    _classification_metrics,
    _config_digest,
    _file_digest,
    _metrics,
    _validate_evaluation,
    write_evaluation_products,
)


def local_path(reference: str, mount: Path) -> Path:
    from training_artifacts import storage_path

    return mount / storage_path(reference).lstrip("/")


def records(path: Path):
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                yield number, line


def stable_case(raw: dict, module, context: dict) -> dict:
    case = module.parse_record(raw, context)
    if not isinstance(case, dict) or case.get("reference") is None:
        raise ValueError("parse_record must return an object with a reference")
    if case.get("id") is None or case["id"] == "":
        # A stable input key is explicit evidence; line numbers are never IDs.
        if not case.get("input_key"):
            raise ValueError("case requires id or a stable input_key")
        case["id"] = "input:" + _config_digest(case["input_key"])
    case["id"] = str(case["id"])
    return case


def score(case: dict, module, spec: dict) -> dict:
    scored = module.score_case(case, spec)
    if not isinstance(scored, dict) or not isinstance(scored.get("passed"), bool):
        raise ValueError("score_case requires a boolean passed")
    scored.setdefault("overall_score", float(scored["passed"]))
    scored.setdefault(
        "rubric_results",
        [
            {
                "name": "business",
                "criterion": spec["business_rule"],
                "weight": 1,
                "required": True,
                "score": scored["overall_score"],
                "passed": scored["passed"],
                "evidence": scored.get("evidence", []),
                "failure_reason": None
                if scored["passed"]
                else scored.get("failure_reason", "Business criterion failed"),
            }
        ],
    )
    evaluation = _validate_evaluation(scored)
    # Reject non-JSON/nonfinite script output before producing any reports.
    json.dumps(evaluation, allow_nan=False)
    prediction = case.get("prediction")
    if prediction is None or prediction == "":
        evaluation.update(passed=False, overall_score=0.0)
        for item in evaluation["rubric_results"]:
            item.update(passed=False, score=0.0, failure_reason="Missing prediction")
    return {
        **evaluation,
        "id": case["id"],
        "reference": case["reference"],
        "prediction": prediction,
        "media": case.get("media", []),
        "sample": case.get("sample", {"prompt": case.get("input", "")}),
        "latency_seconds": None,
        "evaluation_latency_seconds": 0.0,
        "error": None,
        "reference_label": scored.get("reference_label"),
        "prediction_label": scored.get("prediction_label")
        if prediction not in (None, "")
        else None,
        "original_label": case.get("original_label"),
        "original_prediction_label": scored.get("original_prediction_label"),
        "category": case.get("category"),
    }


def validate_script(module, spec: dict) -> None:
    examples = spec.get("validation_examples", [])
    if not examples or {item.get("passed") for item in examples} != {True, False}:
        raise ValueError("validation_examples must include passing and failing cases")
    for example in examples:
        case = stable_case(
            example["record"],
            module,
            {"source": "evaluation", "validation_example": True},
        )
        result = score(case, module, spec)
        if result["passed"] is not example["passed"]:
            raise ValueError(f"scoring self-check failed for {case['id']}")


def dataset_evidence(manifest: dict, module, mount: Path, spec: dict):
    evidence, expected, warnings = {}, {}, []
    for source in ("train", "validation", "test"):
        entry = manifest.get("datasets", {}).get(source)
        if not entry:
            if source != "test":
                warnings.append(f"{source}_dataset_missing")
            continue
        try:
            distribution, identities, issues = _dataset_evidence(
                source, entry, module, mount, spec
            )
        except Exception as exc:
            warnings.append(
                f"{source}_dataset_profile_error:{type(exc).__name__}:{exc}"
            )
            continue
        evidence[source] = distribution
        warnings.extend(issues)
        if source == "validation":
            expected = identities
    return evidence, expected, warnings


def _dataset_evidence(source: str, entry: dict, module, mount: Path, spec: dict):
    expected, warnings = {}, []
    path = local_path(entry["path"], mount)
    if not entry.get("sha256") or _file_digest(path) != entry["sha256"]:
        raise ValueError(f"{source} dataset fingerprint changed or missing")
    counts, categories, business, identities = Counter(), Counter(), Counter(), set()
    total = 0
    for _, line in records(path):
        case = stable_case(json.loads(line), module, {"source": source})
        if case["id"] in identities and source == "validation":
            raise ValueError(f"duplicate {source} sample: {case['id']}")
        identities.add(case["id"])
        total += 1
        label = str(case.get("original_label") or "unknown")
        counts[label] += 1
        business_label = case.get(
            "business_label", spec.get("label_mapping", {}).get(label)
        )
        if business_label is not None:
            business[str(business_label)] += 1
        if case.get("category") is not None:
            categories[str(case["category"])] += 1
        if source == "validation":
            expected[case["id"]] = _config_digest(case["reference"])
    if not total:
        warnings.append(f"{source}_dataset_empty")
    if "unknown" in counts:
        warnings.append(f"{source}_labels_missing")
    return (
        {
            "path": entry["path"],
            "sha256": entry["sha256"],
            "total": total,
            "unique_ids": len(identities),
            "labels": {
                label: {"count": count, "ratio": count / total}
                for label, count in counts.items()
            },
            "business_labels": {
                label: {"count": count, "ratio": count / total}
                for label, count in business.items()
            },
            "categories": {
                label: {"count": count, "ratio": count / total}
                for label, count in categories.items()
            },
        },
        expected,
        warnings,
    )


def evaluate_batch(
    manifest: dict,
    spec: dict,
    script: Path,
    output: Path,
    run_id: str,
    mount: Path = Path("/workspace"),
) -> dict:
    from training_artifacts import inspect_checkpoint

    output.mkdir(parents=True, exist_ok=True)
    script_hash = _file_digest(script)
    spec_hash = _config_digest(spec)
    frozen = {
        "schema_version": 1,
        "run_id": run_id,
        "source_manifest_sha256": _config_digest(manifest),
        "evaluation_config_sha256": spec_hash,
        "script_sha256": script_hash,
    }
    manifest_file = output / "run_manifest.json"
    if manifest_file.exists() and json.loads(manifest_file.read_text()) != frozen:
        raise ValueError("frozen inputs changed; create a new full run")
    _atomic_json(manifest_file, frozen)
    module = types.ModuleType("frozen_business_scoring")
    # Execute precisely the hashed source, never timestamp-based .pyc caches.
    exec(compile(script.read_bytes(), str(script), "exec"), module.__dict__)
    validate_script(module, spec)
    distribution, expected, evidence_warnings = dataset_evidence(
        manifest, module, mount, spec
    )
    _atomic_json(
        output / "dataset_profile.json",
        {
            "schema_version": 1,
            "status": "incomplete" if evidence_warnings else "ready",
            "datasets": manifest.get("datasets", {}),
            "distribution": distribution,
            "extraction_sha256": script_hash,
            "label_mapping": spec.get("label_mapping", {}),
            "issues": evidence_warnings,
        },
    )
    checkpoints = {item["step"]: item for item in manifest.get("checkpoints", [])}
    steps, all_issues = [], []
    for entry in sorted(manifest["evaluations"], key=lambda value: value["step"]):
        step = entry["step"]
        child_id = str(uuid5(UUID(run_id), f"step:{step}"))
        child_dir = output / "steps" / str(step)
        child_dir.mkdir(parents=True, exist_ok=True)
        issues, results, identities = [], [], {}
        path = local_path(entry["path"], mount)
        try:
            if _file_digest(path) != entry["sha256"]:
                raise ValueError("evaluation source fingerprint changed")
            for line_no, line in records(path):
                try:
                    case = stable_case(
                        json.loads(line), module, {"source": "evaluation", "step": step}
                    )
                    if case["id"] in identities:
                        raise ValueError("duplicate sample ID")
                    identities[case["id"]] = _config_digest(case["reference"])
                except Exception as exc:
                    issues.append(
                        f"line {line_no}: invalid source record: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                try:
                    result = score(case, module, spec)
                except Exception as exc:
                    result = {
                        "id": case["id"],
                        "passed": False,
                        "overall_score": 0.0,
                        "reference": case["reference"],
                        "prediction": case.get("prediction"),
                        "sample": case.get("sample", {}),
                        "media": case.get("media", []),
                        "metrics": {},
                        "details": {},
                        "rubric_results": [],
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_seconds": None,
                        "original_label": case.get("original_label"),
                    }
                    # A script crash is not evidence that only the prediction is bad.
                    issues.append(f"scoring_error:{case['id']}")
                results.append(result)
        except Exception as exc:
            issues.append(f"source_unavailable:{type(exc).__name__}:{exc}")
        dataset_role = entry.get("dataset_role", "validation")
        if dataset_role != "validation":
            issues.append("not_validation_selection_evidence")
        elif not expected:
            issues.append("validation_membership_unverified")
        elif identities != expected:
            issues.append("validation_membership_or_ground_truth_mismatch")
        generation = entry.get("generation", manifest.get("generation"))
        if not isinstance(generation, dict) or not generation:
            issues.append("generation_configuration_unverified")
        comparison_group = _config_digest(
            {
                "cases": identities,
                "scoring": spec_hash,
                "script": script_hash,
                "generation": generation,
            }
        )
        checkpoint = None
        if step in checkpoints:
            prior = checkpoints[step]
            try:
                checkpoint = inspect_checkpoint(
                    local_path(prior["path"], mount),
                    step,
                    prior["path"],
                    prior.get("base_model"),
                )
            except Exception as exc:
                checkpoint = {**prior, "complete": False, "issues": [str(exc)]}
        metrics = _metrics(
            results,
            {
                "dataset_path": entry["path"],
                "evaluation_mode": "saved_predictions",
                "model_reference": checkpoint["path"] if checkpoint else None,
            },
            [],
        )
        metrics.update(
            evaluation_mode="saved_predictions",
            expected_samples=len(expected),
            completed_scores=sum(row["error"] is None for row in results),
            integrity_issues=list(issues),
            integrity={
                "parse_errors": sum(
                    "invalid source record" in issue for issue in issues
                ),
                "duplicate_ids": sum(
                    "duplicate sample ID" in issue for issue in issues
                ),
                "scoring_errors": sum(
                    issue.startswith("scoring_error:") for issue in issues
                ),
                "missing_predictions": sum(
                    row.get("prediction") in (None, "") for row in results
                ),
            },
            source_sha256=entry["sha256"],
            script_sha256=script_hash,
            request_count=0,
            current_request_count=0,
            reused_predictions=len(results),
            successful_predictions=sum(
                row.get("prediction") not in (None, "") for row in results
            ),
            evaluated_cases=sum(row["error"] is None for row in results),
            dataset_distribution=distribution,
        )
        metrics["classification"] = _classification_metrics(
            [
                (row.get("prediction_label"), row.get("reference_label"))
                for row in results
            ]
        )
        metrics["macro_f1"] = metrics["classification"]["macro_f1"]
        if any(row.get("original_prediction_label") is not None for row in results):
            metrics["original_classification"] = _classification_metrics(
                [
                    (row.get("original_prediction_label"), row.get("original_label"))
                    for row in results
                ]
            )
        for field in ("original_label", "reference_label", "category"):
            groups = defaultdict(list)
            for row in results:
                if row.get(field) is not None:
                    groups[str(row[field])].append(row)
            metrics[field + "_metrics"] = {
                label: {
                    "support": len(rows),
                    "passed": sum(r["passed"] for r in rows),
                    "pass_rate": sum(r["passed"] for r in rows) / len(rows),
                    **(
                        metrics.get(
                            "original_classification"
                            if field == "original_label"
                            else "classification",
                            {},
                        )
                        .get("per_class", {})
                        .get(label, {})
                        if field != "category"
                        else {
                            "classification": _classification_metrics(
                                [
                                    (
                                        r.get("prediction_label"),
                                        r.get("reference_label"),
                                    )
                                    for r in rows
                                ]
                            )
                        }
                    ),
                }
                for label, rows in groups.items()
            }
        for row in results:
            label = str(row.get("original_label", "unknown"))
            row["details"]["distribution_context"] = {
                "label": label,
                "train": distribution.get("train", {}).get("labels", {}).get(label),
                "validation": distribution.get("validation", {})
                .get("labels", {})
                .get(label),
                "performance": metrics["original_label_metrics"].get(label),
            }
        child_manifest = {
            **frozen,
            "run_id": child_id,
            "parent_run_id": run_id,
            "step": step,
            "source_sha256": entry["sha256"],
        }
        _atomic_json(child_dir / "run_manifest.json", child_manifest)
        if not results:
            issues.append("no_evaluated_cases")
        try:
            write_evaluation_products(
                results,
                metrics,
                path,
                {},
                child_dir,
                child_id,
                spec_hash,
                [],
                report_assets_dir=str(Path(__file__).parent),
                integrity_issues=issues + evidence_warnings,
                render_invalid=True,
                render_report=manifest.get("render_reports", True),
            )
        except RuntimeError as exc:
            issues.append(str(exc))
        steps.append(
            {
                "step": step,
                "role": entry.get("role", "checkpoint"),
                "dataset_role": dataset_role,
                "run_id": child_id,
                "output_dir": str(child_dir).removeprefix(str(mount)),
                "metrics": metrics,
                "comparison_group": comparison_group,
                "eligible": not issues,
                "issues": issues,
                "checkpoint": checkpoint,
            }
        )
        all_issues.extend(issues)
        _atomic_json(
            output / "progress.json",
            {
                "total": len(manifest["evaluations"]),
                "processed": len(steps),
                "failed": sum(not x["eligible"] for x in steps),
            },
        )
    batch = {
        "schema_version": 1,
        "run_id": run_id,
        "steps": steps,
        "dataset_distribution": distribution,
        "warnings": evidence_warnings,
        "scoring_spec_sha256": spec_hash,
        "script_sha256": script_hash,
        "source_manifest_sha256": frozen["source_manifest_sha256"],
    }
    _atomic_json(output / "batch_result.json", batch)
    _atomic_json(output / "metrics.json", {"request_count": 0, "steps": len(steps)})
    _atomic_json(
        output / "report.json",
        {
            "status": "succeeded",
            "evaluation_status": "pending_selection",
            "batch_result": "batch_result.json",
            "step_count": len(steps),
            "warnings": evidence_warnings,
            "issues": all_issues,
        },
    )
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    definition = json.loads((output / "prediction_evaluation.json").read_text())
    evaluate_batch(
        definition["source_manifest"],
        definition["scoring_spec"],
        output / "score_eval.py",
        output,
        args.run_id,
    )


if __name__ == "__main__":
    main()
