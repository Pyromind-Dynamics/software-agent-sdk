#!/usr/bin/env python3
"""Template-driven manifest renderer that runs on the platform node.

This is the "build the manifest" step: the agent only writes/confirms a render
template (field mapping + join sources + shard size) and points it at the
Storage data source; this script runs inside a CustomCommandCPUNode, reads the
parquet in bounded batches (pyarrow iter_batches, memory ~ one shard), renders
each record into the runner manifest shape, writes each shard straight to
Storage, and finally emits a ``shards.json`` index the agent can consume to
submit validation batches. The agent never holds the full dataset or a
full-size manifest.

Template shape (JSON, Storage or local):
    {
      "schema_version": 1,
      "name": "tmax-render",
      "data_source": "datasets/tmax/data/train-00000-of-00001.parquet",
      "fields": {
        "task_id": "task_id",
        "image": {"join": {"source": "datasets/tmax/processed/manifest.jsonl",
                            "on": "task_id", "column": "image"},
                  "on_missing": "fail"},
        "workdir": {"fixed": "/home/user"},
        "prompt": "description",
        "test_sh": {"kind": "pytest_wrapper", "source_field": "test_final_state",
                    "target_path": "/workspace/test_final_state.py"}
      },
      "shard_size": 500
    }

Every ``fields.*`` entry accepts the same spec forms: a bare column name
(dotted paths like ``env_config.task_id`` descend into struct columns),
``{"field": col}``, ``{"fixed": v}``, ``{"join": {...}}`` (Storage join
table keyed by task_id), ``{"kind": "message", "source_field": "messages",
"role": "user"}`` (extract a message from a chat-format list column —
open-instruct style datasets keep the prompt there) and ``{"kind":
"storage_file", "path_template": "task-data/{task_id}/tests/test.sh"}``
(read a per-row file from Storage; placeholders reference already-rendered
record fields such as ``{task_id}``). ``test_sh`` additionally accepts
``{"kind": "pytest_wrapper", "source": <any spec>, "target_path": ...}``
which wraps the resolved python source into a pytest verifier script
(``source_field`` is the legacy spelling of ``source``).

Row-level problems (join misses, missing storage files, missing chat
messages) skip the record into ``render_failures.jsonl`` — values are never
guessed and one bad row never aborts the batch.

Outputs under --output-root (Storage):
    batch-001/manifest.jsonl
    batch-002/manifest.jsonl
    ...
    shards.json            {"shards": [...], "rendered": N}
    render_failures.jsonl  skipped records (join misses)
    progress.json          live snapshot (total / rendered / failed / ETA)
    report.json            final summary

Usage (run on the node; pandas/pyarrow installed by the render task):
    python3 render_manifest.py \
        --template <local or /target-workspace<storage> template path> \
        --data-source <storage data path> \
        --output-root <storage output root> \
        [--limit N] [--local-out <dir>]
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


MOUNT_PREFIX = "/target-workspace"
SHARDS_FILENAME = "shards.json"
FAILURES_FILENAME = "render_failures.jsonl"
PROGRESS_FILENAME = "progress.json"
REPORT_FILENAME = "report.json"

# Scattered data problems (join misses, a few missing files) never line up
# this long; only a systematically wrong template does. Aborting there turns
# a full-dataset wasted render into a few-seconds failure.
MAX_CONSECUTIVE_SKIPS = 200


class RenderSkip(Exception):
    """A row-level data problem; the record goes to render_failures.jsonl."""

    def __init__(self, task_id: str, field: str, reason: str) -> None:
        super().__init__(f"task_id={task_id} field={field}: {reason}")
        self.task_id = task_id
        self.field = field
        self.reason = reason


class JoinMiss(RenderSkip):
    """A join lookup found no row for the rendered task_id."""

    def __init__(self, task_id: str, field: str) -> None:
        super().__init__(task_id, field, "join miss")


def _pytest_wrapper(script_source: str, target_path: str) -> str:
    return (
        "#!/bin/bash\n"
        f"cat > {target_path} <<'PYEOF'\n"
        f"{script_source}\n"
        "PYEOF\n"
        f"python3 -m pytest {target_path} -q\n"
        "rc=$?\n"
        'if [ "$rc" -eq 0 ]; then\n'
        '  echo "1.0" > /logs/verifier/reward.txt\n'
        "else\n"
        '  echo "0.0" > /logs/verifier/reward.txt\n'
        "fi\n"
        "exit 0\n"
    )


def _row_value(row: Any, column: str, row_index: int) -> Any:
    """Read one parquet column; ``a.b`` descends into struct columns."""
    value: Any = row
    for part in column.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(
                f"row {row_index}: template field references missing column "
                f"{column!r} (columns: {list(row.keys())})"
            )
        value = value[part]
    return value


def _resolve_mount_path(path: str) -> str:
    """Resolve a Storage path against the node mount (or a local file)."""
    if Path(path).exists():
        return path
    mounted = f"{MOUNT_PREFIX}/{path.lstrip('/')}"
    if Path(mounted).exists():
        return mounted
    raise ValueError(f"cannot resolve {path!r}: no local file, no mount at {mounted}")


def _load_join_table(source: str, on: str, column: str) -> dict[str, str]:
    """Build a {key: value} index from a manifest.jsonl or CSV join table."""
    text = Path(_resolve_mount_path(source)).read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    first = lines[0] if lines else ""
    if first.lstrip().startswith("{"):
        index: dict[str, str] = {}
        for line in lines:
            entry = json.loads(line)
            if entry.get(on) and entry.get(column):
                index[str(entry[on])] = str(entry[column])
        return index

    import csv
    from io import StringIO

    rows = list(csv.reader(StringIO(text)))
    header = rows[0] if rows else []
    idx = {name.lower(): i for i, name in enumerate(header)}
    if on not in idx or column not in idx:
        raise ValueError(
            f"join table {source!r} must be a JSONL with {on!r}/{column!r} "
            f"keys or a CSV with columns {on!r},{column!r}"
        )
    return {
        row[idx[on]]: row[idx[column]]
        for row in rows[1:]
        if len(row) > max(idx.values()) and row[idx[on]] and row[idx[column]]
    }


def build_join_indexes(template: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Load every ``{"join": ...}`` field spec into {field: {task_id: value}}."""
    indexes: dict[str, dict[str, str]] = {}
    for name, spec in template.get("fields", {}).items():
        if not (isinstance(spec, dict) and "join" in spec):
            continue
        join = spec["join"]
        if not isinstance(join, dict):
            raise ValueError(f"field {name!r}: join spec must be an object")
        if join.get("on_missing", "fail") != "fail":
            raise ValueError(
                f"field {name!r}: join on_missing={join['on_missing']!r} "
                "is not supported (only 'fail')"
            )
        indexes[name] = _load_join_table(
            str(join["source"]),
            str(join.get("on", "task_id")),
            str(join["column"]),
        )
    return indexes


def _resolve_spec(
    name: str,
    spec: Any,
    row: Any,
    row_index: int,
    task_id: str,
    record: dict[str, str],
    joins: dict[str, dict[str, str]],
) -> str:
    """Resolve one ``fields.*`` spec against the row, joins and Storage.

    ``record`` carries the already-rendered fields so that storage_file
    path templates can reference them (``{task_id}`` and friends).
    """
    if isinstance(spec, str):
        return str(_row_value(row, spec, row_index))
    if not isinstance(spec, dict):
        raise ValueError(f"invalid field spec for row {row_index}: {spec!r}")
    if "fixed" in spec:
        return str(spec["fixed"])
    if "field" in spec:
        return str(_row_value(row, str(spec["field"]), row_index))
    if "join" in spec:
        value = joins.get(name, {}).get(task_id)
        if value is None:
            raise JoinMiss(task_id, name)
        return str(value)
    kind = spec.get("kind")
    if kind == "message":
        source_field = str(spec.get("source_field", "messages"))
        messages = _row_value(row, source_field, row_index)
        if not isinstance(messages, list):
            raise ValueError(
                f"row {row_index}: field {name!r} message spec needs a list "
                f"column {source_field!r}, got {type(messages).__name__}"
            )
        role = str(spec.get("role", "user"))
        picked = [m for m in messages if isinstance(m, dict) and m.get("role") == role]
        if not picked:
            raise RenderSkip(task_id, name, f"no message with role {role!r}")
        message = picked[-1] if str(spec.get("index", "first")) == "last" else picked[0]
        content = message.get("content")
        if content is None:
            raise RenderSkip(
                task_id, name, f"message with role {role!r} has no content"
            )
        return str(content)
    if kind == "storage_file":
        path_template = str(spec["path_template"])
        try:
            path = path_template.format_map(record)
        except KeyError as exc:
            raise ValueError(
                f"row {row_index}: field {name!r} path_template references "
                f"unknown field {exc.args[0]!r}"
            ) from exc
        try:
            resolved = _resolve_mount_path(path)
        except ValueError as exc:
            raise RenderSkip(
                task_id,
                name,
                f"storage file missing: {path} "
                f"(path_template resolves against the storage root; "
                f"tried {MOUNT_PREFIX}/{path.lstrip('/')})",
            ) from exc
        try:
            return Path(resolved).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise RenderSkip(task_id, name, f"storage file unreadable: {path}") from exc
    raise ValueError(f"unsupported field spec for {name!r} (row {row_index}): {spec!r}")


def render_record(
    row: Any,
    template_fields: dict[str, Any],
    row_index: int,
    join_indexes: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    """Map one source row onto a runner manifest record."""
    joins = join_indexes or {}
    record: dict[str, str] = {
        "task_id": _resolve_spec(
            "task_id", template_fields["task_id"], row, row_index, "", {}, joins
        )
    }
    task_id = record["task_id"]

    def _resolve(name: str, spec: Any) -> str:
        return _resolve_spec(name, spec, row, row_index, task_id, record, joins)

    # storage_file specs are deferred so their path templates can reference
    # the plain fields rendered in the first pass ({task_id} and friends).
    deferred: list[tuple[str, Any]] = []
    for name, spec in template_fields.items():
        if name in ("task_id", "test_sh"):
            continue
        if isinstance(spec, dict) and spec.get("kind") == "storage_file":
            deferred.append((name, spec))
        else:
            record[name] = _resolve(name, spec)
    for name, spec in deferred:
        record[name] = _resolve(name, spec)
    if "workdir" not in record:
        record["workdir"] = "/home/user"

    test_sh_spec = template_fields.get("test_sh")
    if not test_sh_spec:
        raise ValueError(
            f"row {row_index}: template requires a test_sh field "
            "(pytest_wrapper or a plain spec)"
        )
    if isinstance(test_sh_spec, dict) and test_sh_spec.get("kind") == "pytest_wrapper":
        source = test_sh_spec.get("source", test_sh_spec.get("source_field"))
        if source is None:
            raise ValueError(
                f"row {row_index}: pytest_wrapper requires 'source' "
                "(any field spec) or legacy 'source_field'"
            )
        target_path = str(
            test_sh_spec.get("target_path", "/workspace/test_final_state.py")
        )
        record["test_sh"] = _pytest_wrapper(_resolve("test_sh", source), target_path)
    else:
        record["test_sh"] = _resolve("test_sh", test_sh_spec)
    return record


def _to_storage_path(node_path: str) -> str:
    """Strip the /target-workspace mount prefix -> Agent-facing Storage path."""
    if node_path.startswith(MOUNT_PREFIX):
        return node_path[len(MOUNT_PREFIX) :]
    return node_path


def _expand_parquet_files(data_source: str) -> list[str]:
    """Expand a glob pattern, a directory, or a single file into files."""
    if any(ch in data_source for ch in "*?["):
        # Try the pattern as-is first (absolute paths, cwd-relative), then
        # against the Storage mount — glob patterns never match Path.exists().
        files = sorted(globlib.glob(data_source))
        if not files and not data_source.startswith(MOUNT_PREFIX):
            mounted = f"{MOUNT_PREFIX}/{data_source.lstrip('/')}"
            files = sorted(globlib.glob(mounted))
    else:
        source = _resolve_mount_path(data_source)
        path = Path(source)
        if path.is_dir():
            files = sorted(str(f) for f in path.glob("*.parquet"))
        else:
            files = [source]
    if not files:
        raise ValueError(f"data source {data_source!r} matched no parquet files")
    return files


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _progress_payload(
    total: int, succeeded: int, failed: int, started_at: float
) -> dict[str, Any]:
    processed = succeeded + failed
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
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def render_product(
    parquet_path: str,
    template: dict[str, Any],
    *,
    shard_size: int = 500,
    limit: int | None = None,
    local_out: str | None = None,
    join_indexes: dict[str, dict[str, str]] | None = None,
) -> tuple[list[str], int]:
    """Segment + render shards.

    Returns (shard_paths, rendered). Join misses are skipped into
    render_failures.jsonl (and reported) instead of aborting the run.
    """
    import pyarrow.parquet as pq

    fields = template.get("fields")
    if (
        not isinstance(fields, dict)
        or "task_id" not in fields
        or "prompt" not in fields
    ):
        raise ValueError("template.fields requires at least task_id and prompt")

    out_root = Path(local_out) if local_out else Path("/out")
    out_root.mkdir(parents=True, exist_ok=True)
    progress_path = out_root / PROGRESS_FILENAME

    files = _expand_parquet_files(parquet_path)
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    if limit is not None:
        total = min(total, limit)

    shard_paths: list[str] = []
    rendered = 0
    failed = 0
    seen_rows = 0
    consecutive_skips = 0
    shard_index = 1
    shard_records: list[dict[str, str]] = []
    started_at = time.monotonic()

    def _flush() -> None:
        nonlocal shard_index, shard_records
        if not shard_records:
            return
        batch_dir = out_root / f"batch-{shard_index:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        with (batch_dir / "manifest.jsonl").open("w", encoding="utf-8") as out:
            for record in shard_records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        shard_paths.append(str(PurePosixPath(batch_dir) / "manifest.jsonl"))
        shard_index += 1
        shard_records = []
        _write_progress(
            progress_path, _progress_payload(total, rendered, failed, started_at)
        )

    with (out_root / FAILURES_FILENAME).open("a", encoding="utf-8") as failures:
        for file_path in files:
            file_ = pq.ParquetFile(file_path)
            for batch in file_.iter_batches(batch_size=shard_size):
                for row in batch.to_pylist():
                    if limit is not None and rendered + failed >= limit:
                        break
                    row_index = seen_rows
                    seen_rows += 1
                    try:
                        record = render_record(row, fields, row_index, join_indexes)
                    except RenderSkip as skip:
                        failed += 1
                        consecutive_skips += 1
                        if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                            raise RuntimeError(
                                f"aborting after {consecutive_skips} consecutive "
                                "row failures — the template is systematically "
                                "wrong (check path_template storage-root paths "
                                "and join sources), not scattered bad rows"
                            ) from skip
                        failures.write(
                            json.dumps(
                                {
                                    "task_id": skip.task_id,
                                    "field": skip.field,
                                    "reason": skip.reason,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        failures.flush()
                        print(
                            f"[{rendered + failed}/{total}] task_id={skip.task_id} "
                            f"FAILED {skip.reason}({skip.field}) "
                            f"-> {FAILURES_FILENAME}",
                            flush=True,
                        )
                        continue
                    shard_records.append(record)
                    rendered += 1
                    consecutive_skips = 0
                    print(
                        f"[{rendered + failed}/{total}] "
                        f"task_id={record['task_id']} -> batch-{shard_index:03d}",
                        flush=True,
                    )
                if len(shard_records) >= shard_size:
                    _flush()
                if limit is not None and rendered + failed >= limit:
                    break
            if limit is not None and rendered + failed >= limit:
                break
    _flush()
    _write_progress(
        progress_path, _progress_payload(total, rendered, failed, started_at)
    )
    report = {
        "total": total,
        "rendered": rendered,
        "failed": failed,
        "shards": [_to_storage_path(p) for p in shard_paths],
    }
    (out_root / REPORT_FILENAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return shard_paths, rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument(
        "--data-source",
        required=True,
        help="Storage parquet path, glob, or directory (mount or local)",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Storage root where batch-XXX/ + shards.json land",
    )
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--local-out", default=None, help="Optional local output dir (testing)"
    )
    args = parser.parse_args(argv)

    template_raw = Path(args.template).read_bytes()
    template: dict[str, Any] = json.loads(template_raw)

    local_out = args.local_out
    if local_out is None:
        # on the node, output-root is a Storage mount path (/target-workspace<path>)
        local_out = args.output_root

    try:
        join_indexes = build_join_indexes(template)
        shard_paths, rendered = render_product(
            args.data_source,
            template,
            shard_size=args.shard_size,
            limit=args.limit,
            local_out=local_out,
            join_indexes=join_indexes,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"render failed: {exc}", file=sys.stderr)
        return 1

    # Shards are recorded as Storage paths (no /target-workspace mount prefix)
    # so the agent can feed them straight to edp_submit's storage-manifest mode.
    storage_shards = [_to_storage_path(p) for p in shard_paths]
    index: dict[str, Any] = {"shards": storage_shards, "rendered": rendered}
    index_path = Path(local_out) / SHARDS_FILENAME
    index_path.write_text(json.dumps(index, ensure_ascii=False) + "\n")

    print(json.dumps(index, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
