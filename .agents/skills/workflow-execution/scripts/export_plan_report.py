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


def export_report(source, mapping):
    report = {"schema_version": 1, "metrics": []}
    for field in ("summary", "findings", "badcases", "files", "checks", "source"):
        if field in mapping:
            report[field] = lookup(source, mapping[field])
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
    arguments = parser.parse_args()
    try:
        with arguments.source.open(encoding="utf-8") as stream:
            if arguments.source.suffix == ".jsonl":
                source = [json.loads(line) for line in stream if line.strip()]
            else:
                source = json.load(stream)
        mapping = json.loads(arguments.mapping.read_text(encoding="utf-8"))
        report = export_report(source, mapping)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    except (OSError, OverflowError, TypeError, ValueError, LookupError) as exc:
        parser.exit(2, f"report export failed: {exc}\n")


if __name__ == "__main__":
    main()
