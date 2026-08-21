"""DataFlow text pipeline template for canonical ``text`` output.

Batch-incremental mode with live progress tracking. Processes records in
batches of BATCH_SIZE, writing results and a progress.json snapshot after
each batch so ``df_check_progress`` can report progress and ETA while the
job is running. Supports automatic resume on restart.

Usage:
    python text_pipeline.py <input.jsonl> <processed.jsonl>
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path

import dataflow


# --- Import shim: bypass dataflow/serving/__init__.py ---
_pkg = types.ModuleType("dataflow.serving")
_pkg.__path__ = []
sys.modules["dataflow.serving"] = _pkg
_serving_file = (
    Path(dataflow.__file__).parent / "serving" / "api_llm_serving_request.py"
)
_spec = importlib.util.spec_from_file_location(
    "dataflow.serving.api_llm_serving_request", str(_serving_file)
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules["dataflow.serving.api_llm_serving_request"] = _module
_spec.loader.exec_module(_module)
APILLMServing_request = _module.APILLMServing_request
# --- End shim ---

from dataflow_config import (  # noqa: E402
    DEFAULT_DATAFLOW_API_URL,
    DEFAULT_DATAFLOW_MODEL_NAME,
)
from df_logging import LoggingLLMServing  # noqa: E402


SYSTEM_PROMPT = "You are a helpful assistant."
BATCH_SIZE = 20


def source_id(row: dict, index: int) -> str:
    return str(row.get("id") or f"text-{index}")


def build_user_prompt(row: dict) -> str:
    value = row.get("user_prompt") or row.get("prompt")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source record must provide a non-empty prompt")
    return value.strip()


def _read_input(path: Path) -> list[dict]:
    """Read all records from input JSONL."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _count_existing_lines(path: Path) -> int:
    """Count lines in existing output file for resume support."""
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _write_progress(
    progress_path: Path,
    *,
    total: int,
    processed: int,
    succeeded: int,
    failed: int,
    attempted_this_run: int,
    start_time: float,
) -> None:
    """Write an atomic progress.json snapshot consumed by df_check_progress."""
    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    elapsed_s = elapsed_ms / 1000.0
    rate = (
        attempted_this_run / elapsed_s if elapsed_s > 0 and attempted_this_run else 0.0
    )
    remaining = total - processed
    eta_ms = int(remaining / rate * 1000) if rate > 0 else None
    payload = {
        "total": total,
        "processed": processed,
        "succeeded": succeeded,
        "failed": failed,
        "elapsed_ms": elapsed_ms,
        "records_per_second": round(rate, 2),
        "eta_ms": eta_ms,
        "updated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    }
    # Atomic write: tmp + rename to avoid partial reads under FUSE
    tmp_path = Path(str(progress_path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        f.close()
    tmp_path.rename(progress_path)


def main(input_path: str, output_path: str) -> None:
    input_file = Path(input_path).resolve()
    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Read all input records
    records = _read_input(input_file)
    total = len(records)

    if total == 0:
        print("Input is empty. Nothing to do.")
        return

    # 2. Resume: skip already-processed records
    already_done = _count_existing_lines(output_file)
    if already_done > 0:
        print(f"Resuming: skipping {already_done}/{total} already-processed records")
    remaining = records[already_done:]

    if not remaining:
        print(f"All {total} records already processed. Nothing to do.")
        return

    # 3. Initialize LLM from environment variables
    #    Wrap with LoggingLLMServing for full call traceability.
    raw_llm = APILLMServing_request(
        api_url=os.environ.get("DF_API_URL", DEFAULT_DATAFLOW_API_URL),
        model_name=os.environ.get("DF_MODEL_NAME", DEFAULT_DATAFLOW_MODEL_NAME),
        key_name_of_api_key="DF_API_KEY",
        temperature=0.0,
        max_workers=8,
    )
    llm = LoggingLLMServing(raw_llm)

    # 4. Process in batches, write progress after each batch
    progress_path = output_file.parent / "progress.json"
    start_time = time.monotonic()
    succeeded = already_done
    failed = 0

    # Initial snapshot: job can be seen as started immediately
    _write_progress(
        progress_path,
        total=total,
        processed=already_done,
        succeeded=succeeded,
        failed=failed,
        attempted_this_run=0,
        start_time=start_time,
    )

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start : batch_start + BATCH_SIZE]
        prompts = [build_user_prompt(r) for r in batch]

        responses = None
        try:
            responses = llm.generate_from_input(
                user_inputs=prompts,
                system_prompt=SYSTEM_PROMPT,
            )
        except Exception as exc:
            api_key = os.environ.get("DF_API_KEY", "")
            err_msg = str(exc).replace(api_key, "<redacted>") if api_key else str(exc)
            print(
                f"[ERROR] batch at offset {batch_start}: {err_msg[:200]}",
                file=sys.stderr,
            )
            failed += len(batch)

        if responses is not None:
            # Open and close per batch so FUSE uploads chunks promptly
            with output_file.open("a", encoding="utf-8") as out:
                for record, response in zip(batch, responses):
                    if response is None:
                        failed += 1
                        continue
                    answer = response.strip()
                    if not answer:
                        failed += 1
                        continue
                    out.write(
                        json.dumps(
                            {
                                "id": source_id(record, already_done + succeeded),
                                "system_prompt": SYSTEM_PROMPT,
                                "user_prompt": build_user_prompt(record),
                                "gt": answer,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    succeeded += 1

        attempted = batch_start + len(batch)
        _write_progress(
            progress_path,
            total=total,
            processed=already_done + attempted,
            succeeded=succeeded,
            failed=failed,
            attempted_this_run=attempted,
            start_time=start_time,
        )
        print(
            f"Progress: {attempted}/{len(remaining)} processed, "
            f"{succeeded}/{total} succeeded, {failed} failed"
        )

    # Final snapshot
    _write_progress(
        progress_path,
        total=total,
        processed=total,
        succeeded=succeeded,
        failed=failed,
        attempted_this_run=len(remaining),
        start_time=start_time,
    )
    print(
        f"Done. total={total} succeeded={succeeded} failed={failed} "
        f"output={output_file}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: text_pipeline.py <input.jsonl> <processed.jsonl>")
    main(sys.argv[1], sys.argv[2])
