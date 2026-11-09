from __future__ import annotations

import asyncio
import json
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pyromind_agent_server.observability import LangfuseExecutionObserver
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.commands import UserMessageCommand
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.harness import SessionSpec

from .fake_adapter import FakeAdapter


async def test_product_traces_follow_commands_and_completion(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PYROMIND_OBSERVABILITY_RUN_KIND", "scripted_test")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    observer = LangfuseExecutionObserver(provider)
    adapter = FakeAdapter()
    runtime = ConversationRuntime(tmp_path, adapter, observer=observer)
    context = RequestContext(user_id="owner")
    try:
        for conversation_id in ("first", "second"):
            await runtime.create_conversation(
                SessionSpec(
                    conversation_id=conversation_id,
                    workspace_root=str(tmp_path / conversation_id),
                    user_id="owner",
                ),
                context,
            )
            for index in range(2):
                run_id = f"run-{index}"
                await runtime.submit_command(
                    conversation_id,
                    UserMessageCommand(
                        command_id=run_id, content=(TextContent(text="hello"),)
                    ),
                    context,
                )
                trace = adapter.sent[-1][2].execution_trace
                assert trace is not None
                adapter.emit(
                    conversation_id,
                    "run.finished",
                    (
                        {"outcome": {"status": "completed"}}
                        if index == 0
                        else {"status": "idle"}
                    ),
                    run_id=run_id,
                )
        for _ in range(20):
            if len(exporter.get_finished_spans()) == 4:
                break
            await asyncio.sleep(0.01)
        spans = exporter.get_finished_spans()
        assert len(spans) == 4
        assert len({span.context.trace_id for span in spans if span.context}) == 4
        assert sorted(
            json.loads(str(span.attributes["langfuse.observation.output"]))["status"]
            for span in spans
            if span.attributes
        ) == ["completed", "completed", "idle", "idle"]
        assert context.execution_trace is None
        for span in spans:
            assert span.attributes
            assert (
                span.attributes["langfuse.observation.metadata.run_kind"]
                == "scripted_test"
            )
            assert span.attributes["langfuse.trace.tags"] == ("scripted-test",)
            output = json.loads(str(span.attributes["langfuse.observation.output"]))
            assert output["execution_status"] == "completed"
            assert output["business_result"] == "not_evaluated_by_observer"
    finally:
        await runtime.close()
