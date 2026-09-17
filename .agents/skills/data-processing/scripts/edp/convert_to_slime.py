#!/usr/bin/env python3
"""Convert verified records into slime RL training format.

Reads the manifest consumed by ``sandbox_runner.py`` plus the verdicts the
runner produced, keeps only records judged usable, and emits one slime
record (``prompt`` / ``label`` / ``metadata``) per line. Reward is not
embedded: slime judges reward at rollout time from ``metadata.test_sh``,
so usable-but-zero-reward records stay in (RL needs the 0-signal too).

Usage:
    python convert_to_slime.py \\
        --manifest run-dir/manifest.jsonl \\
        --verdicts run-dir/verdicts.jsonl \\
        --out slime.jsonl \\
        [--protocol tmax]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
        if isinstance(data, dict):
            records.append(data)
    return records


def to_slime_record(rec: dict, protocol: str) -> dict | None:
    """Build one slime record from a manifest entry.

    Returns None when the entry is unusable for slime (missing task_id /
    prompt / test_sh) — the caller decides whether that is a skip.
    """
    task_id = str(rec.get("task_id", ""))
    prompt = rec.get("prompt")
    test_sh = rec.get("test_sh")
    if not task_id or not prompt or not test_sh:
        return None
    return {
        "prompt": [{"role": "user", "content": str(prompt)}],
        "label": task_id,
        "metadata": {
            "protocol": protocol,
            "instance_id": task_id,
            "image": str(rec.get("image", "")),
            "workdir": str(rec.get("workdir", "")),
            "problem_statement": str(prompt),
            "test_sh": str(test_sh),
        },
    }


def convert(
    manifest: list[dict], verdicts: list[dict], protocol: str
) -> tuple[list[dict], int]:
    """Return slime records for usable manifest entries plus a skip count."""
    usable = {
        str(v.get("task_id"))
        for v in verdicts
        if v.get("verdict") == "usable" and v.get("task_id") is not None
    }
    records: list[dict] = []
    skipped = 0
    for rec in manifest:
        task_id = str(rec.get("task_id", ""))
        if task_id not in usable:
            skipped += 1
            continue
        record = to_slime_record(rec, protocol)
        if record is None:
            skipped += 1
            continue
        records.append(record)
    return records, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--verdicts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--protocol", default="tmax")
    args = parser.parse_args(argv)

    manifest = _load_jsonl(args.manifest)
    verdicts = _load_jsonl(args.verdicts)
    records, skipped = convert(manifest, verdicts, args.protocol)

    with open(args.out, "w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"manifest={len(manifest)} usable-converted={len(records)} skipped={skipped} "
        f"out={args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
