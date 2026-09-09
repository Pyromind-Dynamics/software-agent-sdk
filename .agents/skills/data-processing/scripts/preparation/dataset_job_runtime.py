"""Finalize and validate versioned dataset-analysis/synthesis artifacts."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def _load_object(path: Path) -> dict[str, Any]:
    value = _load_any_object(path)
    if value.get("schema_version") != 1:
        raise ValueError(f"{path.name} must use schema_version=1")
    return value


def _load_any_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} must be an object")
            values.append(value)
    return values


def _source_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    files = (
        [path]
        if path.is_file()
        else sorted(item for item in path.rglob("*") if item.is_file())
    )
    for item in files:
        if path.is_dir():
            digest.update(item.relative_to(path).as_posix().encode())
            digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _source_records(source: Path, adapter: str) -> list[dict[str, Any]]:
    if adapter in {"jsonl_text", "vision_manifest"}:
        values = _jsonl_objects(source)
        records = []
        for index, value in enumerate(values):
            records.append(
                {
                    "sample_id": str(
                        value.get("id") or value.get("sample_id") or index
                    ),
                    "split": value.get("split"),
                    "value": value,
                }
            )
        return records
    if adapter == "avi_pcb":
        records = []
        candidates = (
            [source] if (source / "meta.json").is_file() else sorted(source.iterdir())
        )
        for candidate in candidates:
            meta_path = candidate / "meta.json"
            if not candidate.is_dir() or not meta_path.is_file():
                continue
            value = _load_any_object(meta_path)
            records.append(
                {
                    "sample_id": str(value.get("id") or candidate.name),
                    "split": value.get("split"),
                    "value": value,
                }
            )
        return records
    raise ValueError(f"unsupported adapter: {adapter}")


def _training_records(source: Path, adapter: str) -> list[dict[str, Any]]:
    excluded = {"validation", "val", "test", "eval"}
    return [
        item
        for item in _source_records(source, adapter)
        if str(item.get("split") or "").lower() not in excluded
    ]


def _expected_analysis_ids(
    records: list[dict[str, Any]], spec: dict[str, Any]
) -> tuple[set[str], bool]:
    ids = [str(item["sample_id"]) for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("source contains duplicate sample_id values")
    if spec.get("label_source") == "existing_labels" or len(ids) <= 200:
        return set(ids), False
    sampling = spec.get("sampling")
    if not isinstance(sampling, dict) or sampling.get("max_samples") != 200:
        raise ValueError("inferred analysis must use max_samples=200")
    seed = int(sampling.get("seed", 42))
    ranked = sorted(
        ids,
        key=lambda sample_id: (
            hashlib.sha256(f"{seed}:{sample_id}".encode()).digest(),
            sample_id,
        ),
    )
    return set(ranked[:200]), True


def _validate_assignments(value: dict[str, Any], taxonomy: dict[str, Any]) -> None:
    dimensions_raw = taxonomy.get("dimensions")
    if not isinstance(dimensions_raw, list):
        raise ValueError("taxonomy dimensions must be a list")
    dimensions = {str(item["id"]): item for item in dimensions_raw}
    assignments = value.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(dimensions):
        raise ValueError("each label record must assign every taxonomy dimension")
    seen: set[str] = set()
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("assignment must be an object")
        dimension_id = str(assignment.get("dimension", ""))
        if dimension_id not in dimensions or dimension_id in seen:
            raise ValueError("unknown or duplicate taxonomy dimension")
        seen.add(dimension_id)
        labels = assignment.get("labels")
        if (
            not isinstance(labels, list)
            or not labels
            or len(labels) != len(set(labels))
        ):
            raise ValueError("labels must be a non-empty unique list")
        dimension = dimensions[dimension_id]
        known = {str(item["id"]) for item in dimension.get("labels", [])}
        if not {str(item) for item in labels}.issubset(known):
            raise ValueError("label record contains an unknown label")
        if dimension.get("assignment") == "single" and len(labels) != 1:
            raise ValueError("single-label dimension has multiple labels")


def _validate_analysis(
    *,
    source: Path,
    primary_values: list[dict[str, Any]],
    output_dir: Path,
    spec: dict[str, Any],
) -> dict[str, Any]:
    taxonomy = spec.get("taxonomy")
    if not isinstance(taxonomy, dict):
        raise ValueError("analysis_spec.json must contain taxonomy")
    taxonomy_bytes = json.dumps(
        taxonomy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if spec.get("taxonomy_fingerprint") != hashlib.sha256(taxonomy_bytes).hexdigest():
        raise ValueError("taxonomy_fingerprint does not match taxonomy")
    records = _training_records(source, str(spec.get("adapter")))
    expected, sampled = _expected_analysis_ids(records, spec)
    succeeded: set[str] = set()
    for value in primary_values:
        sample_id = str(value.get("sample_id", ""))
        if not sample_id or sample_id in succeeded:
            raise ValueError("labels.jsonl contains missing or duplicate sample_id")
        _validate_assignments(value, taxonomy)
        succeeded.add(sample_id)
    failure_values = _jsonl_objects(output_dir / "failures.jsonl")
    failed = {str(item.get("sample_id", "")) for item in failure_values}
    if (
        "" in failed
        or len(failed) != len(failure_values)
        or succeeded & failed
        or succeeded | failed != expected
    ):
        raise ValueError(
            "labels and failures do not exactly cover the selected manifest"
        )
    report_path = output_dir / "distribution_report.json"
    if not report_path.is_file():
        raise ValueError("analysis pipeline did not write distribution_report.json")
    report = _load_object(report_path)
    expected_mode = "sampled_estimate" if sampled else "exact"
    if report.get("analysis_mode") != expected_mode:
        raise ValueError(f"distribution report must use {expected_mode}")
    if report.get("total_records") != len(records):
        raise ValueError("distribution report total_records does not match source")
    if report.get("analyzed_records") != len(primary_values):
        raise ValueError("distribution report analyzed_records does not match labels")
    if report.get("failed_records") != len(failure_values):
        raise ValueError("distribution report failed_records does not match failures")
    distributions = report.get("distributions")
    if not isinstance(distributions, list):
        raise ValueError("distribution report must contain distributions")
    for item in distributions:
        if not isinstance(item, dict):
            raise ValueError("distribution entries must be objects")
        if sampled and (
            item.get("estimated_total_count") is None
            or not isinstance(item.get("confidence_95"), dict)
        ):
            raise ValueError(
                "sampled distributions require estimates and Wilson bounds"
            )
        if not sampled and (
            item.get("estimated_total_count") is not None
            or item.get("confidence_95") is not None
        ):
            raise ValueError("exact distributions must not claim sampled estimates")
    if spec.get("label_source") == "existing_labels":
        calls = output_dir / "llm_calls.jsonl"
        if calls.is_file() and calls.stat().st_size:
            raise ValueError("existing-label analysis must make zero model calls")
    (output_dir / "taxonomy.json").write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _validate_synthesis(
    *,
    source: Path,
    primary_values: list[dict[str, Any]],
    output_dir: Path,
    spec: dict[str, Any],
) -> dict[str, Any]:
    provenance_path = output_dir / "provenance.jsonl"
    if not provenance_path.is_file():
        raise ValueError("synthesis pipeline did not write provenance.jsonl")
    provenance = _jsonl_objects(provenance_path)
    output_ids = {str(item.get("sample_id", "")) for item in primary_values}
    provenance_ids = {str(item.get("sample_id", "")) for item in provenance}
    if "" in output_ids or len(output_ids) != len(primary_values):
        raise ValueError("synthesized.jsonl sample_id values must be unique")
    if output_ids != provenance_ids or len(provenance) != len(primary_values):
        raise ValueError("provenance must match synthesized records one-to-one")
    source_records = _source_records(source, str(spec.get("adapter")))
    known_splits = {"train", "validation", "val", "test", "eval"}
    if any(item.get("split") not in known_splits for item in source_records):
        raise ValueError("synthesis source contains an unknown or missing split")
    train_hosts = {
        str(item["sample_id"])
        for item in source_records
        if str(item.get("split") or "").lower() == "train"
    }
    max_reuse = int(spec.get("max_children_per_source", 0))
    uses: dict[str, int] = {}
    request_strategies = {
        str(item.get("strategy_id")) for item in spec.get("requests", [])
    }
    expected_strategy_counts: Counter[str] = Counter()
    for item in spec.get("requests", []):
        expected_strategy_counts[str(item.get("strategy_id"))] += int(
            item.get("count", 0)
        )
    actual_strategy_counts: Counter[str] = Counter()
    for item in provenance:
        parent = str(item.get("parent_sample_id", ""))
        if parent not in train_hosts:
            raise ValueError("provenance references a non-training or unknown host")
        uses[parent] = uses.get(parent, 0) + 1
        if uses[parent] > max_reuse:
            raise ValueError("max_children_per_source was exceeded")
        if item.get("strategy_id") not in request_strategies:
            raise ValueError("provenance contains an unplanned synthesis strategy")
        actual_strategy_counts[str(item.get("strategy_id"))] += 1
        if spec.get("adapter") == "avi_pcb":
            validation = item.get("validation")
            if not isinstance(validation, dict) or not all(
                validation.get(key) is True
                for key in ("diff_nonempty", "bbox_valid", "label_consistent")
            ):
                raise ValueError(
                    "synthesized record failed image/annotation validation"
                )
    if actual_strategy_counts != expected_strategy_counts:
        raise ValueError("synthesized counts do not match augmentation requests")
    if spec.get("adapter") == "avi_pcb":
        for value in primary_values:
            for key in ("image", "diff", "gt"):
                raw_path = value.get(key)
                if not isinstance(raw_path, str):
                    raise ValueError(f"AVI output is missing {key}")
                path = (output_dir / raw_path).resolve()
                if not path.is_relative_to(output_dir.resolve()) or not path.is_file():
                    raise ValueError(f"AVI output contains invalid {key} path")
    if spec.get("adapter") != "avi_pcb":
        (output_dir / "assets").mkdir(exist_ok=True)
    if not (output_dir / "assets").is_dir():
        raise ValueError("synthesis pipeline did not write assets/")
    return {
        "schema_version": 1,
        "kind": "data_synthesis",
        "synthesized_records": len(primary_values),
    }


def finalize(
    kind: str, output_dir: Path, primary: Path, spec_path: Path, source: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = _load_object(spec_path)
    declared_fingerprint = spec.get("source_fingerprint")
    actual_fingerprint = _source_fingerprint(source)
    if declared_fingerprint is not None and declared_fingerprint != actual_fingerprint:
        raise ValueError("source_fingerprint does not match the mounted input")
    primary_values = _jsonl_objects(primary)
    count = len(primary_values)
    failures = output_dir / "failures.jsonl"
    failures.touch(exist_ok=True)

    if kind == "dataset_analysis":
        canonical_spec = output_dir / "analysis_spec.json"
        report = _validate_analysis(
            source=source,
            primary_values=primary_values,
            output_dir=output_dir,
            spec=spec,
        )
        title = "Dataset distribution analysis"
    elif kind == "data_synthesis":
        canonical_spec = output_dir / "augmentation_plan.json"
        report = _validate_synthesis(
            source=source,
            primary_values=primary_values,
            output_dir=output_dir,
            spec=spec,
        )
        title = "Gap-driven data synthesis"
    else:
        raise ValueError(f"unsupported dataset job kind: {kind}")

    report["source_fingerprint"] = actual_fingerprint

    if canonical_spec.resolve() != spec_path.resolve():
        shutil.copyfile(spec_path, canonical_spec)
    validation = {
        "schema_version": 1,
        "kind": kind,
        "valid": True,
        "primary_output": primary.name,
        "records": count,
        "source_fingerprint": actual_fingerprint,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "progress.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "total": count,
                "processed": count,
                "succeeded": count,
                "failed": 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    body = html.escape(json.dumps(report, ensure_ascii=False, indent=2))
    (output_dir / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><h1>{html.escape(title)}</h1><pre>{body}</pre>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind", choices=("dataset_analysis", "data_synthesis"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.kind, args.output_dir, args.primary, args.spec, args.input)


if __name__ == "__main__":
    main()
