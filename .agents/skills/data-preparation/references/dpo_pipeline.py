"""DataFlow DPO pipeline template.

Usage:
    pipeline.py <input.jsonl> <processed.jsonl> [limit]

This template is for input-only or partially labeled records. It uses DataFlow
operators for normalization, generation, filtering, and deduplication, then writes
the canonical Pyromind DPO JSONL schema.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
import types
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path
from typing import Any

import dataflow
import pandas as pd


# open-dataflow 1.0.10 内部 storage.write 仍调用 pandas 1.x 的 applymap，
# pandas 3.0 已移除该方法；映射到语义相同的 DataFrame.map 保证兼容。
if not hasattr(pd.DataFrame, "applymap"):
    pd.DataFrame.applymap = pd.DataFrame.map  # type: ignore[attr-defined]

from dataflow.operators.core_text import PandasOperator
from dataflow.operators.general_text import ContentNullFilter
from dataflow.prompts.core_text import FormatStrPrompt
from dataflow.utils.storage import LazyFileStorage
from df_logging import LoggingLLMServing


_pkg = types.ModuleType("dataflow.serving")
_pkg.__path__ = []
sys.modules["dataflow.serving"] = _pkg
_serving_file = (
    Path(dataflow.__file__).parent / "serving" / "api_llm_serving_request.py"
)
_spec = importlib.util.spec_from_file_location(
    "dataflow.serving.api_llm_serving_request",
    str(_serving_file),
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["dataflow.serving.api_llm_serving_request"] = _mod
_spec.loader.exec_module(_mod)
APILLMServing_request = _mod.APILLMServing_request


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _first_string(row: pd.Series, names: list[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["id"] = [
        _first_string(row, ["id"], f"dpo-{index}") for index, row in result.iterrows()
    ]
    result["system_prompt"] = [
        _first_string(row, ["system_prompt", "system"], DEFAULT_SYSTEM_PROMPT)
        for _, row in result.iterrows()
    ]
    result["user_prompt"] = [
        _first_string(row, ["user_prompt", "prompt", "question", "input"])
        for _, row in result.iterrows()
    ]
    result["existing_gt"] = [
        _first_string(row, ["gt", "chosen", "preferred", "response"])
        for _, row in result.iterrows()
    ]
    result["existing_rejected"] = [
        _first_string(row, ["rejected_answer", "rejected", "bad"])
        for _, row in result.iterrows()
    ]
    return result


def parse_pair(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        payload = value
    elif isinstance(value, str):
        payload = json.loads(value)
    else:
        raise ValueError("dpo_pair must be a JSON object or JSON object string")
    chosen = payload.get("chosen")
    rejected = payload.get("rejected")
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        raise ValueError("dpo_pair must contain string chosen and rejected")
    return {"chosen": chosen.strip(), "rejected": rejected.strip()}


def apply_generated_pair(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    chosen_values: list[str] = []
    rejected_values: list[str] = []
    parse_ok: list[bool] = []
    for _, row in result.iterrows():
        existing_gt = row.get("existing_gt")
        existing_rejected = row.get("existing_rejected")
        try:
            parsed = parse_pair(row.get("dpo_pair"))
            chosen = (
                existing_gt
                if isinstance(existing_gt, str) and existing_gt
                else parsed["chosen"]
            )
            rejected = (
                existing_rejected
                if isinstance(existing_rejected, str) and existing_rejected
                else parsed["rejected"]
            )
            parse_ok.append(bool(chosen.strip() and rejected.strip()))
            chosen_values.append(chosen.strip())
            rejected_values.append(rejected.strip())
        except Exception:
            parse_ok.append(False)
            chosen_values.append("")
            rejected_values.append("")
    result["gt"] = chosen_values
    result["rejected_answer"] = rejected_values
    result["dpo_pair_parse_ok"] = parse_ok
    return result


def _write_progress_snapshot(
    progress_path: str | Path,
    *,
    total: int,
    processed: int,
    succeeded: int,
    failed: int,
    started_at: float,
    attempted_this_run: int,
) -> None:
    elapsed_s = time.monotonic() - started_at
    rate = (
        attempted_this_run / elapsed_s
        if elapsed_s > 0 and attempted_this_run > 0
        else 0.0
    )
    remaining = max(total - processed, 0)
    eta_ms = int(remaining / rate * 1000) if rate > 0 else None
    payload = {
        "total": total,
        "processed": processed,
        "succeeded": succeeded,
        "failed": failed,
        "elapsed_ms": int(elapsed_s * 1000),
        "records_per_second": round(rate, 2),
        "eta_ms": eta_ms,
        "updated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    }
    tmp_path = Path(str(progress_path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        f.close()
    tmp_path.rename(progress_path)


def _count_output_lines(path: str | Path) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except (FileNotFoundError, OSError):
        return 0


def write_canonical_output(
    input_path: str, output_path: str, limit: int | None
) -> None:
    # === Phase 1: Preprocessing (DataFlow operators) ===
    storage = LazyFileStorage(
        input_path,
        cache_path=str(Path(output_path).parent / ".dataflow_cache"),
        file_name_prefix="dpo",
        cache_type="jsonl",
    )
    storage = storage.step()

    if limit is not None:
        PandasOperator([lambda df: df.head(limit).copy()]).run(storage=storage)
        storage = storage.step()

    PandasOperator([normalize_columns]).run(storage=storage)
    storage = storage.step()

    ContentNullFilter().run(storage=storage, input_key="user_prompt")
    df = storage.read("dataframe")
    total = len(df)
    if total == 0:
        return

    # Progress tracking
    output_file = Path(output_path)
    progress_path = output_file.with_name("progress.json")
    started_at = time.monotonic()
    _write_progress_snapshot(
        progress_path,
        total=total,
        processed=0,
        succeeded=0,
        failed=0,
        started_at=started_at,
        attempted_this_run=0,
    )

    # === Phase 2: Batch LLM generation with progress ===
    BATCH_SIZE = int(os.environ.get("DF_BATCH_SIZE", "8"))

    raw_llm = APILLMServing_request(
        api_url=os.environ["DF_API_URL"],
        model_name=os.environ["DF_MODEL_NAME"],
        key_name_of_api_key="DF_API_KEY",
        max_workers=int(os.environ.get("DF_MAX_WORKERS", "8")),
    )
    llm = LoggingLLMServing(raw_llm)

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chosen": {"type": "string"},
            "rejected": {"type": "string"},
        },
        "required": ["chosen", "rejected"],
    }
    prompt = FormatStrPrompt(
        f_str_template=(
            "为下面用户问题生成 DPO 偏好回答对。"
            "如果给出了现有优质回答，请把它作为 chosen 的依据；否则生成"
            "一个完整、准确、有帮助的 chosen。"
            "rejected 必须看似相关但明显更差，例如过短、遗漏关键约束、"
            "推理错误或没有回答问题。"
            "不要让 rejected 包含有害内容、隐私泄露、歧视或违法指导。"
            '只返回 JSON：{"chosen":"...","rejected":"..."}\n\n'
            "用户问题：{question}\n"
            "现有优质回答（可为空）：{chosen_seed}\n"
            "现有负样本（可为空）：{rejected_seed}"
        )
    )

    # Build all prompt texts (same logic as FormatStrPromptedGenerator)
    prompts = []
    for _, row in df.iterrows():
        question = str(row.get("user_prompt", ""))
        chosen_seed = str(row.get("existing_gt", ""))
        rejected_seed = str(row.get("existing_rejected", ""))
        text = prompt.build_prompt(
            need_fields={"question", "chosen_seed", "rejected_seed"},
            question=question,
            chosen_seed=chosen_seed,
            rejected_seed=rejected_seed,
        )
        prompts.append(text)

    # Resume: skip already-processed records
    existing = _count_output_lines(output_file)
    start_index = existing

    attempted_this_run = 0
    succeeded = existing
    failed = 0

    batch_count = (total - start_index + BATCH_SIZE - 1) // BATCH_SIZE
    if batch_count > 0:
        print(
            f"Generating {total - start_index} DPO pairs in {batch_count} batches...",
            flush=True,
        )

    for batch_start in range(start_index, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_prompts = prompts[batch_start:batch_end]
        batch_rows = df.iloc[batch_start:batch_end].to_dict("records")

        # Generate batch via LLM
        outputs = llm.generate_from_input(
            user_inputs=batch_prompts,
            system_prompt="You generate preference-pair training data.",
            json_schema=schema,
        )
        attempted_this_run += len(outputs)

        # Parse, validate, append to output (open-append-write-close per batch)
        batch_succeeded = 0
        with open(output_file, "a", encoding="utf-8") as f:
            for row_obj, text in zip(batch_rows, outputs):
                try:
                    parsed = parse_pair(text)
                    chosen = parsed["chosen"].strip()
                    rejected = parsed["rejected"].strip()
                    if not chosen or not rejected or chosen == rejected:
                        continue
                except Exception:
                    continue
                f.write(
                    json.dumps(
                        {
                            "id": str(row_obj.get("id", "")),
                            "system_prompt": (
                                str(row_obj.get("system_prompt", "")).strip()
                            ),
                            "user_prompt": (
                                str(row_obj.get("user_prompt", "")).strip()
                            ),
                            "gt": chosen,
                            "rejected_answer": rejected,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                batch_succeeded += 1
            # close() on `with` exit triggers FUSE upload

        succeeded += batch_succeeded
        processed = batch_end
        failed = processed - succeeded

        _write_progress_snapshot(
            progress_path,
            total=total,
            processed=processed,
            succeeded=succeeded,
            failed=failed,
            started_at=started_at,
            attempted_this_run=attempted_this_run,
        )

        batch_num = (batch_start - start_index) // BATCH_SIZE + 1
        print(
            f"\rBatch {batch_num}/{batch_count}: "
            f"{processed}/{total} ({processed / total * 100:.0f}%), "
            f"succeeded {batch_succeeded},",
            end="",
            flush=True,
        )
    if batch_count > 0:
        print()

    # === Phase 3: Global dedup ===
    with open(output_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    seen = set()
    deduped = []
    for r in records:
        key = hashlib.md5(
            (
                r["system_prompt"] + r["user_prompt"] + r["gt"] + r["rejected_answer"]
            ).encode()
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    removed = len(records) - len(deduped)
    if removed:
        with open(output_file, "w", encoding="utf-8") as f:
            for r in deduped:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"Dedup: {len(records)} \u2192 {len(deduped)} rows (removed {removed})",
        flush=True,
    )

    # Final progress snapshot
    _write_progress_snapshot(
        progress_path,
        total=total,
        processed=total,
        succeeded=succeeded if attempted_this_run > 0 else total,
        failed=failed,
        started_at=started_at,
        attempted_this_run=attempted_this_run or total,
    )


if __name__ == "__main__":
    if len(sys.argv) not in {3, 4}:
        raise SystemExit("usage: pipeline.py <input.jsonl> <processed.jsonl> [limit]")
    write_canonical_output(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]) if len(sys.argv) == 4 else None,
    )
