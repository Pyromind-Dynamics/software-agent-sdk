#!/usr/bin/env python3
"""Merge validated shard runs into training data, on the platform node.

This is the final aggregation step: it reads the per-shard run directories
produced by edp_submit (each holding ``run/verdicts.jsonl`` and
``run/traces/`` next to its shard manifest), merges all verdicts into one
dataset, and in the same pass converts usable records to the slime RL
format and high-reward traces to the SFT format. It runs inside a
CustomCommandCPUNode with pure standard-library code plus the sibling
convert_to_slime / convert_to_sft scripts staged next to this file.

Checkpoint semantics: the merged ``verdicts.jsonl``
under --out-dir doubles as the checkpoint — task_ids already present are
skipped, so rerunning after an interrupted aggregation, or appending more
shards later, resumes instead of duplicating. progress.json is rewritten
atomically after every record; report.json lands at the end.

Usage (run on the node):
    python3 aggregate_results.py \
        --run-dirs <storage run dir> [<storage run dir> ...] \
        --out-dir <storage output dir> \
        [--protocol tmax] [--min-reward 1.0] \
        [--system-prompt ...] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import convert_to_sft
import convert_to_slime


MOUNT_PREFIX = "/target-workspace"
RUN_SUBDIR = "run"
MANIFEST_FILENAME = "manifest.jsonl"
VERDICTS_FILENAME = "verdicts.jsonl"
TRACES_DIRNAME = "traces"
SLIME_FILENAME = "slime.jsonl"
SFT_FILENAME = "sft.jsonl"
PROGRESS_FILENAME = "progress.json"
REPORT_FILENAME = "report.json"


def _resolve_mount_path(path: str) -> str:
    """Resolve a Storage path against the node mount (or a local file)."""
    if Path(path).exists():
        return path
    mounted = f"{MOUNT_PREFIX}/{path.lstrip('/')}"
    if Path(mounted).exists():
        return mounted
    raise ValueError(f"cannot resolve {path!r}: no local file, no mount at {mounted}")


def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield dict lines; a torn trailing line is skipped, not fatal."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            yield data


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _progress_payload(
    total: int, processed: int, succeeded: int, failed: int, started_at: float
) -> dict[str, Any]:
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    elapsed_s = max(elapsed_ms / 1000.0, 1e-9)
    rate = processed / elapsed_s if processed else 0.0
    remaining = max(total - processed, 0)
    return {
        "total": total,
        "processed": processed,
        "succeeded": succeeded,
        "failed": failed,
        "elapsed_ms": elapsed_ms,
        "records_per_second": rate,
        "eta_ms": int(remaining / rate * 1000) if rate > 0 else None,
        "updated_at": datetime.now(UTC).isoformat(),
    }


def _shard_inputs(run_dir: Path) -> tuple[Path, Path, Path]:
    """Locate (manifest, verdicts, traces_dir) for one edp_submit run dir."""
    verdicts = run_dir / RUN_SUBDIR / VERDICTS_FILENAME
    if not verdicts.is_file():
        raise ValueError(
            f"run dir {run_dir} has no {RUN_SUBDIR}/{VERDICTS_FILENAME} — was "
            "the batch actually submitted and finished?"
        )
    # edp_submit nests the run output under the shard manifest's directory
    # (<manifest_dir>/<run_id>/run/...), so the manifest sits one level up.
    manifest = run_dir.parent / MANIFEST_FILENAME
    if not manifest.is_file():
        raise ValueError(f"no shard manifest at {manifest} for run dir {run_dir}")
    return manifest, verdicts, run_dir / RUN_SUBDIR / TRACES_DIRNAME


def aggregate(
    run_dirs: list[str],
    out_dir: str,
    *,
    protocol: str = "tmax",
    min_reward: float = 1.0,
    system_prompt: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Merge shard runs into one dataset + training formats.

    Returns the report payload (the caller writes it to report.json).
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_verdicts = out_root / VERDICTS_FILENAME
    out_slime = out_root / SLIME_FILENAME
    out_sft = out_root / SFT_FILENAME
    progress_path = out_root / PROGRESS_FILENAME

    shards = [_shard_inputs(Path(_resolve_mount_path(d))) for d in run_dirs]
    total = sum(_count_lines(verdicts) for _, verdicts, _ in shards)

    # Resume state comes from the merged verdicts checkpoint itself.
    done: set[str] = set()
    verdict_counts: Counter[str] = Counter()
    error_categories: Counter[str] = Counter()
    for entry in _iter_jsonl(out_verdicts) if out_verdicts.exists() else []:
        task_id = str(entry.get("task_id", ""))
        if task_id:
            done.add(task_id)
        verdict_value = str(entry.get("verdict", "unknown"))
        verdict_counts[verdict_value] += 1
        if verdict_value != "usable" and entry.get("error_category"):
            error_categories[str(entry["error_category"])] += 1
    slime_count = _count_lines(out_slime)
    sft_count = _count_lines(out_sft)

    processed = len(done)  # includes checkpointed records
    new_processed = 0
    skipped_duplicate = 0
    skipped_incomplete = 0
    skipped_low_reward = 0
    skipped_no_trace = 0
    skipped_invalid_trace = 0
    started_at = time.monotonic()

    def _checkpoint() -> None:
        succeeded = verdict_counts.get("usable", 0)
        failed = sum(
            count for value, count in verdict_counts.items() if value != "usable"
        )
        _write_progress(
            progress_path,
            _progress_payload(total, processed, succeeded, failed, started_at),
        )

    # A torn trailing line (process died mid-write) must not glue itself to
    # the next appended record; cap it with a newline before appending.
    for append_target in (out_verdicts, out_slime, out_sft):
        if append_target.exists() and append_target.stat().st_size > 0:
            with append_target.open("rb") as existing:
                existing.seek(-1, 2)
                if existing.read(1) != b"\n":
                    with append_target.open("ab") as fixup:
                        fixup.write(b"\n")

    limit_reached = False
    with (
        out_verdicts.open("a", encoding="utf-8") as vout,
        out_slime.open("a", encoding="utf-8") as sout,
        out_sft.open("a", encoding="utf-8") as fout,
    ):
        for manifest_path, verdicts_path, traces_dir in shards:
            if limit_reached:
                break
            manifest_index = {
                str(rec.get("task_id", "")): rec for rec in _iter_jsonl(manifest_path)
            }
            for entry in _iter_jsonl(verdicts_path):
                if limit is not None and new_processed >= limit:
                    limit_reached = True
                    break
                task_id = str(entry.get("task_id", ""))
                if not task_id or task_id in done:
                    skipped_duplicate += 1
                    continue

                vout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                vout.flush()
                done.add(task_id)
                processed += 1
                new_processed += 1
                verdict_value = str(entry.get("verdict", "unknown"))
                verdict_counts[verdict_value] += 1
                if verdict_value != "usable" and entry.get("error_category"):
                    error_categories[str(entry["error_category"])] += 1

                if verdict_value == "usable":
                    record = manifest_index.get(task_id)
                    slime_record = (
                        convert_to_slime.to_slime_record(record, protocol)
                        if record is not None
                        else None
                    )
                    if slime_record is None:
                        skipped_incomplete += 1
                    else:
                        sout.write(json.dumps(slime_record, ensure_ascii=False) + "\n")
                        sout.flush()
                        slime_count += 1

                    reward = entry.get("reward")
                    reward_value = float(reward) if reward is not None else None
                    if reward_value is None or reward_value < min_reward:
                        skipped_low_reward += 1
                    else:
                        trace_path = convert_to_sft.find_trace_path(traces_dir, task_id)
                        if trace_path is None:
                            skipped_no_trace += 1
                        else:
                            prompt = str(record.get("prompt", "")) if record else None
                            messages = convert_to_sft.build_sft_messages(
                                convert_to_sft.load_trace_events(trace_path),
                                prompt,
                                system_prompt,
                            )
                            if messages is None:
                                skipped_invalid_trace += 1
                            else:
                                fout.write(
                                    json.dumps(
                                        {"messages": messages}, ensure_ascii=False
                                    )
                                    + "\n"
                                )
                                fout.flush()
                                sft_count += 1

                print(
                    f"[{processed}/{total}] task_id={task_id} "
                    f"verdict={verdict_value} reward={entry.get('reward')}",
                    flush=True,
                )
                _checkpoint()

    _checkpoint()
    return {
        "total": total,
        "processed": processed,
        "new_processed": new_processed,
        "skipped_duplicate": skipped_duplicate,
        "verdicts": dict(verdict_counts),
        "error_categories": dict(error_categories),
        # Broken-out because verifier_env_missing records are usually
        # recoverable by fixing the image rather than discarding the data.
        "verifier_env_missing": error_categories.get("verifier_env_missing", 0),
        "slime_records": slime_count,
        "sft_samples": sft_count,
        "skipped_incomplete": skipped_incomplete,
        "skipped_low_reward": skipped_low_reward,
        "skipped_no_trace": skipped_no_trace,
        "skipped_invalid_trace": skipped_invalid_trace,
        "run_dirs": list(run_dirs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dirs",
        required=True,
        nargs="+",
        help="edp_submit output_dir per shard (Storage path or local)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Storage dir where verdicts/slime/sft/progress/report land",
    )
    parser.add_argument("--protocol", default="tmax")
    parser.add_argument("--min-reward", type=float, default=1.0)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap new records this pass"
    )
    args = parser.parse_args(argv)

    try:
        report = aggregate(
            args.run_dirs,
            _resolve_mount_path(args.out_dir),
            protocol=args.protocol,
            min_reward=args.min_reward,
            system_prompt=args.system_prompt,
            limit=args.limit,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"aggregate failed: {exc}", file=sys.stderr)
        return 1

    out_root = Path(_resolve_mount_path(args.out_dir))
    (out_root / REPORT_FILENAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
