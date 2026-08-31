"""Validate Pyromind messages or DPO preference JSONL and write a report."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from cleaning_utils import (
    ERROR_EMPTY_DATASET,
    ERROR_INVALID_FORMAT,
    ERROR_JSON_DECODE,
    ValidationError,
    detect_output_format,
    iter_jsonl,
    validate_record,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="training JSONL to validate")
    parser.add_argument("--report", required=True, help="report.json path")
    parser.add_argument(
        "--max-errors",
        type=int,
        default=50,
        help="Maximum detailed errors retained in the report",
    )
    return parser


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _merge_validation(
    report: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    stats = report.get("stats")
    cleaning_status = stats.get("status") if isinstance(stats, dict) else None
    passed = validation["status"] == "passed" and cleaning_status in {
        None,
        "completed",
    }
    return {
        **report,
        "status": "passed" if passed else "failed",
        "format": validation["format"],
        "validation": validation,
    }


def _append_error(
    errors: list[dict[str, Any]],
    reasons: Counter[str],
    error: ValidationError,
    max_errors: int,
) -> None:
    reasons[error.code] += 1
    if len(errors) < max_errors:
        errors.append(error.to_dict())


def validate(input_path: Path, *, max_errors: int) -> dict[str, Any]:
    total = 0
    invalid = 0
    reasons: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    detected_format: str | None = None
    records_with_assistant = 0

    if not input_path.is_file():
        error = ValidationError(
            ERROR_JSON_DECODE,
            f"input file does not exist: {input_path}",
        )
        _append_error(errors, reasons, error, max_errors)
        return {
            "status": "failed",
            "format": "unknown",
            "input": str(input_path),
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "dataset_errors": 1,
            "reasons": dict(reasons),
            "role_counts": {},
            "records_with_assistant": 0,
            "errors": errors,
        }

    for parsed in iter_jsonl(input_path, unwrap_huggingface=False):
        total += 1
        row_errors: list[ValidationError]
        if not parsed.ok:
            row_errors = [
                ValidationError(
                    parsed.error_type or ERROR_JSON_DECODE,
                    parsed.error or "parse failed",
                    line_number=parsed.line_number,
                )
            ]
        else:
            row_format = detect_output_format(parsed.data)
            if row_format is not None:
                if detected_format is None:
                    detected_format = row_format
                elif row_format != detected_format:
                    detected_format = "mixed"
                    row_errors = [
                        ValidationError(
                            ERROR_INVALID_FORMAT,
                            "dataset mixes messages and DPO preference records",
                            line_number=parsed.line_number,
                        )
                    ]
                    invalid += 1
                    for error in row_errors:
                        _append_error(errors, reasons, error, max_errors)
                    continue
            row_errors = validate_record(
                parsed.data,
                line_number=parsed.line_number,
            )
            if row_format == "messages" and isinstance(parsed.data, dict):
                roles = [
                    message.get("role")
                    for message in parsed.data.get("messages", [])
                    if isinstance(message, dict)
                    and isinstance(message.get("role"), str)
                ]
                role_counts.update(roles)
                if "assistant" in roles:
                    records_with_assistant += 1
        if not row_errors:
            continue
        invalid += 1
        for error in row_errors:
            _append_error(errors, reasons, error, max_errors)

    dataset_errors = 0
    if total == 0:
        dataset_errors = 1
        _append_error(
            errors,
            reasons,
            ValidationError(ERROR_EMPTY_DATASET, "input contains no records"),
            max_errors,
        )

    passed = invalid == 0 and dataset_errors == 0
    return {
        "status": "passed" if passed else "failed",
        "format": detected_format or "unknown",
        "input": str(input_path),
        "total": total,
        "valid": total - invalid,
        "invalid": invalid,
        "dataset_errors": dataset_errors,
        "reasons": dict(reasons),
        "role_counts": dict(role_counts),
        "records_with_assistant": records_with_assistant,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_errors < 1:
        raise SystemExit("--max-errors must be greater than 0")
    validation = validate(Path(args.input), max_errors=args.max_errors)
    report_path = Path(args.report)
    report = _merge_validation(_read_report(report_path), validation)
    _atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
