#!/usr/bin/env python3
"""Compute a deterministic byte fingerprint for a source file or directory."""

from __future__ import annotations

import argparse
import hashlib
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
    args = parser.parse_args()
    print(source_fingerprint(args.path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
