"""Validate cleaned JSON/JSONL/CSV files against target training formats."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from cleaning_utils import (
    ERROR_EMPTY_DATASET,
    ERROR_JSON_DECODE,
    ValidationError,
    iter_records,
    validate_record,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the validate_format.py command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Cleaned input file to validate")
    parser.add_argument(
        "--format",
        required=True,
        choices=("alpaca", "sharegpt", "messages"),
        help="Target format to validate",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=50,
        help="Maximum per-row errors to print before summary-only mode",
    )
    return parser


def print_error(error: ValidationError) -> None:
    """Print one validation error as JSON."""
    print(json.dumps({"error": error.to_dict()}, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    """Run validation and return a process exit code."""
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"input file does not exist: {input_path}", file=sys.stderr)
        return 2

    total = 0
    invalid = 0
    printed = 0
    reasons: Counter[str] = Counter()
    for parsed in iter_records(input_path):
        total += 1
        if not parsed.ok:
            invalid += 1
            reasons[parsed.error_type or ERROR_JSON_DECODE] += 1
            if printed < args.max_errors:
                print_error(
                    ValidationError(
                        parsed.error_type or ERROR_JSON_DECODE,
                        parsed.error or "parse failed",
                        "$",
                        parsed.line_number,
                    )
                )
                printed += 1
            continue
        errors = validate_record(
            parsed.data,
            args.format,
            line_number=parsed.line_number,
        )
        if not errors:
            continue
        invalid += 1
        for error in errors:
            reasons[error.code] += 1
            if printed < args.max_errors:
                print_error(error)
                printed += 1

    dataset_errors = 0
    if total == 0:
        dataset_errors = 1
        reasons[ERROR_EMPTY_DATASET] += 1
        if printed < args.max_errors:
            print_error(
                ValidationError(
                    ERROR_EMPTY_DATASET,
                    "input contains no records",
                    "$",
                )
            )

    summary: dict[str, Any] = {
        "input": str(input_path),
        "format": args.format,
        "total": total,
        "valid": total - invalid,
        "invalid": invalid,
        "dataset_errors": dataset_errors,
        "reasons": dict(reasons),
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    return 0 if invalid == 0 and dataset_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
