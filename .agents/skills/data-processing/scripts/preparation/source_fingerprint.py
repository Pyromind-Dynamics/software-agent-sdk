#!/usr/bin/env python3
"""Compute a deterministic byte fingerprint for a source file or directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def source_fingerprint(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = (
        [path]
        if path.is_file()
        else sorted(item for item in path.rglob("*") if item.is_file())
    )
    for item in files:
        if path.is_dir():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-report", type=Path)
    args = parser.parse_args()
    actual = source_fingerprint(args.path)
    if args.expected_report is not None:
        if not args.expected_report.is_file():
            print("prior source fingerprint report is missing", file=sys.stderr)
            return 92
        try:
            expected_payload = json.loads(
                args.expected_report.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            print("prior source fingerprint report is invalid", file=sys.stderr)
            return 92
        expected = (
            expected_payload.get("before")
            if isinstance(expected_payload, dict)
            else None
        )
        if not isinstance(expected, str) or not expected:
            print("prior source fingerprint is unavailable", file=sys.stderr)
            return 92
        if isinstance(expected, str) and expected != actual:
            print(
                "source fingerprint changed since the previous execution",
                file=sys.stderr,
            )
            return 91
    print(actual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
