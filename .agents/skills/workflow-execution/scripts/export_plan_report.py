"""Write a versioned report from explicit metric and field mappings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def lookup(source, path):
    if not isinstance(path, list):
        raise ValueError("a source path must be a list of JSON keys or array indexes")
    for key in path:
        if isinstance(source, dict) and isinstance(key, str):
            source = source[key]
        elif isinstance(source, list) and type(key) is int and key >= 0:
            source = source[key]
        else:
            raise ValueError(f"invalid source path {path!r}")
    return source


def normalize_task_ids(values):
    ids = []
    for value in values:
        if type(value) not in (str, int) or not str(value).strip():
            raise ValueError("platform task IDs must be non-empty strings or integers")
        ids.append(str(value).strip())
    return list(dict.fromkeys(ids))


def declared_task_ids(source):
    multiple = source.get("platform_task_ids")
    if multiple is None:
        multiple = []
    if isinstance(multiple, dict):
        multiple = list(multiple.values())
    if not isinstance(multiple, list):
        raise ValueError("platform_task_ids must contain a list of IDs")
    single = source.get("platform_task_id")
    return multiple if single is None else [single, *multiple]


def export_report(source, mapping, *, platform_task_ids=()):
    report = {"schema_version": 1, "metrics": []}
    for field in ("summary", "findings", "badcases", "files", "checks"):
        if field in mapping:
            report[field] = lookup(source, mapping[field])
    raw_source = source if isinstance(source, dict) else {}
    provenance = lookup(source, mapping["source"]) if "source" in mapping else {}
    if not isinstance(provenance, dict):
        raise ValueError("report source must be a JSON object")
    provenance = dict(provenance)
    nested_source = raw_source.get("source")
    declared = normalize_task_ids(
        [
            *declared_task_ids(raw_source),
            *declared_task_ids(
                nested_source if isinstance(nested_source, dict) else {}
            ),
            *declared_task_ids(provenance),
        ]
    )
    explicit = normalize_task_ids(platform_task_ids)
    if explicit and set(declared) - set(explicit):
        raise ValueError(
            "source contains a platform task ID outside --platform-task-id"
        )
    ids = explicit or declared
    if ids:
        provenance.pop("platform_task_id", None)
        provenance.pop("platform_task_ids", None)
        if len(ids) == 1:
            provenance["platform_task_id"] = ids[0]
        else:
            provenance["platform_task_ids"] = ids
    if provenance or "source" in mapping:
        report["source"] = provenance
    metrics = mapping.get("metrics", [])
    if not isinstance(metrics, list) or len(metrics) > 24:
        raise ValueError("provide up to 24 explicit metric mappings")
    for metric in metrics:
        value = lookup(source, metric["path"])
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"expected a finite numeric metric at {metric['path']!r}")
        label, unit = metric["label"], metric["unit"]
        if not isinstance(label, str) or not label.strip():
            raise ValueError("a metric label is required")
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError("a metric unit is required")
        report["metrics"].append({"label": label, "value": value, "unit": unit})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument(
        "--platform-task-id",
        action="append",
        default=[],
        help="A verified platform task ID. Repeat for all tasks used by the report.",
    )
    arguments = parser.parse_args()
    try:
        with arguments.source.open(encoding="utf-8") as stream:
            if arguments.source.suffix == ".jsonl":
                source = [json.loads(line) for line in stream if line.strip()]
            else:
                source = json.load(stream)
        mapping = json.loads(arguments.mapping.read_text(encoding="utf-8"))
        report = export_report(
            source, mapping, platform_task_ids=arguments.platform_task_id
        )
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    except (OSError, OverflowError, TypeError, ValueError, LookupError) as exc:
        parser.exit(2, f"report export failed: {exc}\n")


if __name__ == "__main__":
    main()
