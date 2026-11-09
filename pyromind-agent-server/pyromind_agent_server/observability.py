from __future__ import annotations

import base64
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult
from opentelemetry.trace import Span, format_span_id, format_trace_id
from pyromind_runtime.domain.context import ExecutionTrace


logger = logging.getLogger(__name__)


class _LocalFailureExporter(OTLPSpanExporter):
    def __init__(self, base_url: str, public: str, secret: str, log_path: Path):
        auth = base64.b64encode(f"{public}:{secret}".encode()).decode()
        super().__init__(
            endpoint=f"{base_url.rstrip('/')}/api/public/otel/v1/traces",
            headers={
                "Authorization": f"Basic {auth}",
                "x-langfuse-ingestion-version": "4",
            },
            timeout=2,
        )
        self._log_path = log_path

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            result = super().export(spans)
        except Exception:
            result = SpanExportResult.FAILURE
        if result is SpanExportResult.FAILURE:
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                descriptor = os.open(
                    self._log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
                )
                with os.fdopen(descriptor, "a") as stream:
                    stream.write(json.dumps({"reason": "python_export_failed"}) + "\n")
            except OSError:
                logger.error("Unable to save Python observation export failure")
        return result


class LangfuseExecutionObserver:
    def __init__(self, provider: TracerProvider):
        self._provider = provider
        self._tracer = provider.get_tracer("pyromind-product-runtime")
        self._spans: dict[tuple[str, str], Span] = {}

    def start(self, conversation_id: str, run_id: str) -> ExecutionTrace:
        key = (conversation_id, run_id)
        span = self._spans.get(key)
        if span is None:
            span = self._tracer.start_span("Business user turn")
            span.set_attributes(
                {
                    "langfuse.observation.type": "agent",
                    "langfuse.trace.name": "Business user turn",
                    "langfuse.session.id": conversation_id,
                    "langfuse.observation.metadata.conversation_id": conversation_id,
                    "langfuse.observation.metadata.run_id": run_id,
                    "langfuse.observation.metadata.scope": (
                        "product dispatch through harness completion"
                    ),
                }
            )
            environment = os.getenv("LANGFUSE_TRACING_ENVIRONMENT")
            if environment:
                span.set_attribute("langfuse.environment", environment)
            if os.getenv("PYROMIND_OBSERVABILITY_RUN_KIND") == "scripted_test":
                span.set_attribute("langfuse.trace.tags", ["scripted-test"])
                span.set_attribute(
                    "langfuse.observation.metadata.run_kind", "scripted_test"
                )
            self._spans[key] = span
        context = span.get_span_context()
        assert context is not None
        return ExecutionTrace(
            trace_id=format_trace_id(context.trace_id),
            parent_span_id=format_span_id(context.span_id),
        )

    def finish(self, conversation_id: str, run_id: str, status: str) -> None:
        span = self._spans.pop((conversation_id, run_id), None)
        if span is not None:
            span.set_attribute(
                "langfuse.observation.output",
                json.dumps(
                    {
                        "status": status,
                        "execution_status": (
                            "completed" if status in {"idle", "completed"} else status
                        ),
                        "business_result": "not_evaluated_by_observer",
                    }
                ),
            )
            span.end()

    def close(self) -> None:
        for conversation_id, run_id in tuple(self._spans):
            self.finish(conversation_id, run_id, "server_closed_before_completion")
        self._provider.force_flush(timeout_millis=2500)
        self._provider.shutdown()


def build_execution_observer(root: Path) -> LangfuseExecutionObserver | None:
    if os.getenv("LANGFUSE_TRACING_ENABLED", "").lower() == "false":
        return None
    base = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    public = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret = os.getenv("LANGFUSE_SECRET_KEY")
    if not (base and public and secret):
        return None
    provider = TracerProvider(resource=Resource.create({"service.name": "pyromind"}))
    provider.add_span_processor(
        BatchSpanProcessor(
            _LocalFailureExporter(
                base, public, secret, root / "observability-failures.jsonl"
            ),
            max_queue_size=64,
            max_export_batch_size=16,
            schedule_delay_millis=500,
            export_timeout_millis=2500,
        )
    )
    return LangfuseExecutionObserver(provider)
