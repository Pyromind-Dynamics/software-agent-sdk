"""LLM call logging wrapper for DataFlow pipelines.

Wraps APILLMServing_request to record every individual LLM API call
(request, response, latency, status) into a JSONL log file. The log is
written to the directory specified by the DF_LOG_DIR environment variable.

Usage in pipeline scripts:

    from df_logging import LoggingLLMServing

    raw_llm = APILLMServing_request(...)
    llm = LoggingLLMServing(raw_llm)
    # All operators use `llm` — interface is fully compatible.
"""

from __future__ import annotations  # noqa: I001

import json
import os
import threading
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from datetime import UTC as _UTC  # Python 3.11+
except ImportError:
    from datetime import timezone

    _UTC = timezone.utc  # noqa: UP017 — Python 3.10


class LoggingLLMServing:
    """Transparent proxy around APILLMServing_request with full call logging.

    Every individual LLM API call (per-item in the batch thread pool) is
    recorded as one JSON line in ``<log_dir>/llm_calls.jsonl``.

    The wrapper delegates all public methods to the inner serving object,
    so it is a drop-in replacement for any DataFlow operator.
    """

    def __init__(self, llm_serving: Any, log_dir: str | None = None):
        self._inner = llm_serving
        resolved_dir = log_dir or os.environ.get("DF_LOG_DIR", ".")
        Path(resolved_dir).mkdir(parents=True, exist_ok=True)
        self._log_path = Path(resolved_dir) / "llm_calls.jsonl"
        self._log_file = open(self._log_path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._seq = 0
        # Per-item wall-clock budget covering all internal retry attempts.
        # The serving gateway keeps hung connections alive, which defeats
        # socket read_timeout, so wall-clock is the only enforceable limit;
        # a cut item is logged with status="deadline" and its output slot
        # surfaces as None for the pipeline to ledger and skip.
        self._request_deadline = float(os.environ.get("DF_REQUEST_DEADLINE", "600"))
        self._slow_warn_seconds = float(os.environ.get("DF_SLOW_WARN_SECONDS", "120"))
        self._stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "total_latency_ms": 0.0,
        }

    # ------------------------------------------------------------------
    # Core interception: per-call logging at the retry level
    # ------------------------------------------------------------------

    def _api_chat_id_retry(
        self,
        id: int,
        payload: Any,
        model: str,
        is_embedding: bool = False,
        json_schema: dict | None = None,
    ):
        """Wrap the inner per-item call with logging."""
        seq = self._next_seq()
        record: dict[str, Any] = {
            "seq": seq,
            "batch_id": id,
            "timestamp": datetime.now(_UTC).isoformat(),
            "model": model,
            "is_embedding": is_embedding,
        }
        # Truncate payload for log sanity (keep first/last message)
        if isinstance(payload, list) and len(payload) > 0:
            record["request_messages"] = payload
        else:
            record["request_payload"] = str(payload)[:2000]

        start = time.time()
        try:
            result_id, response = self._inner._api_chat_id_retry(
                id, payload, model, is_embedding, json_schema
            )
            latency_ms = round((time.time() - start) * 1000)
            if response is not None:
                record["status"] = "success"
                record["response"] = (
                    response[:5000] if isinstance(response, str) else response
                )
                self._inc_stats(success=True, latency_ms=latency_ms)
            else:
                record["status"] = "empty_response"
                record["response"] = None
                self._inc_stats(success=False, latency_ms=latency_ms)
            record["latency_ms"] = latency_ms
            self._write_record(record)
            self._warn_if_slow(id, latency_ms)
            return result_id, response
        except Exception as e:
            latency_ms = round((time.time() - start) * 1000)
            record["status"] = "error"
            record["error"] = str(e)[:2000]
            record["latency_ms"] = latency_ms
            self._inc_stats(success=False, latency_ms=latency_ms)
            self._write_record(record)
            raise

    # ------------------------------------------------------------------
    # Threadpool override: route through wrapper's _api_chat_id_retry
    # ------------------------------------------------------------------

    def _run_threadpool(self, task_args_list: list[dict], desc: str = "") -> list:  # noqa: ARG002
        """Replicate inner threadpool but use wrapper's logging method.

        Each item gets a wall-clock budget (``DF_REQUEST_DEADLINE``) covering
        all of its internal retry attempts, so a hung gateway connection
        cannot stall the whole batch. ``shutdown(wait=False)`` matters:
        joining abandoned hung threads would reintroduce the stall this
        deadline exists to prevent.
        """
        responses: list[Any] = [None] * len(task_args_list)
        max_workers = getattr(self._inner, "max_workers", 8)
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = [
            (task_args["id"], executor.submit(self._api_chat_id_retry, **task_args))
            for task_args in task_args_list
        ]
        try:
            for task_id, future in futures:
                try:
                    result_id, response = future.result(timeout=self._request_deadline)
                    responses[result_id] = response
                except FuturesTimeoutError:
                    self._record_deadline(task_id)
                except Exception:
                    pass
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return responses

    # ------------------------------------------------------------------
    # Delegate all public methods to inner serving
    # ------------------------------------------------------------------

    def generate_from_input(
        self,
        user_inputs: list[str],
        system_prompt: str = "You are a helpful assistant",
        json_schema: dict | None = None,
    ) -> list[str]:
        """Batch generation with per-call logging via patched thread pool."""
        task_args_list = [
            dict(
                id=idx,
                payload=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                model=self._inner.model_name,
                json_schema=json_schema,
            )
            for idx, question in enumerate(user_inputs)
        ]
        return self._run_threadpool(
            task_args_list, desc="Generating responses from prompts......"
        )

    def generate_from_conversations(self, conversations: list[list[dict]]) -> list[str]:
        task_args_list = [
            dict(
                id=idx,
                payload=dialogue,
                model=self._inner.model_name,
            )
            for idx, dialogue in enumerate(conversations)
        ]
        return self._run_threadpool(
            task_args_list,
            desc="Generating responses from conversations......",
        )

    def generate_embedding_from_input(self, texts: list[str]) -> list[list[float]]:
        task_args_list = [
            dict(
                id=idx,
                payload=txt,
                model=self._inner.model_name,
                is_embedding=True,
            )
            for idx, txt in enumerate(texts)
        ]
        return self._run_threadpool(task_args_list, desc="Generating embedding......")

    def start_serving(self) -> None:
        self._inner.start_serving()

    def cleanup(self) -> None:
        self._inner.cleanup()
        self.close()

    def format_response(self, response: dict, is_embedding: bool = False):
        return self._inner.format_response(response, is_embedding)

    # ------------------------------------------------------------------
    # Accessors for stats (used by generate_report.py)
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._stats)

    @property
    def log_path(self) -> Path:
        return self._log_path

    def close(self) -> None:
        try:
            self._log_file.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Attribute passthrough (model_name, api_url, etc.)
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_seq(self) -> int:
        with self._lock:
            seq = self._seq
            self._seq += 1
            return seq

    def _inc_stats(self, *, success: bool, latency_ms: float) -> None:
        with self._lock:
            self._stats["total"] += 1
            if success:
                self._stats["success"] += 1
            else:
                self._stats["failed"] += 1
            self._stats["total_latency_ms"] += latency_ms

    def _record_deadline(self, task_id: int) -> None:
        record: dict[str, Any] = {
            "seq": self._next_seq(),
            "batch_id": task_id,
            "timestamp": datetime.now(_UTC).isoformat(),
            "model": getattr(self._inner, "model_name", None),
            "status": "deadline",
            "error": f"exceeded DF_REQUEST_DEADLINE={self._request_deadline:g}s",
            "latency_ms": round(self._request_deadline * 1000),
        }
        self._write_record(record)
        self._inc_stats(success=False, latency_ms=self._request_deadline * 1000)
        print(
            f"[df_logging] WARN deadline id={task_id} "
            f"after {self._request_deadline:g}s",
            flush=True,
        )

    def _warn_if_slow(self, task_id: int, latency_ms: float) -> None:
        if latency_ms >= self._slow_warn_seconds * 1000:
            print(
                f"[df_logging] WARN slow id={task_id} cost={latency_ms / 1000:.1f}s",
                flush=True,
            )

    def _write_record(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._log_file.write(line + "\n")
            self._log_file.flush()
