from __future__ import annotations

import pytest
from pydantic import ValidationError
from pyromind_runtime.contracts import (
    HarnessCapabilities,
    HarnessEvent,
    TextContentBlock,
    ToolResult,
)


def test_harness_event_does_not_serialize_provider_metadata() -> None:
    event = HarnessEvent(
        session_id="session-1",
        type="message.delta",
        payload={"text": "hello"},
        provider_metadata={"piMessageType": "assistant_message_delta"},
    )

    serialized = event.model_dump(mode="json")

    assert serialized["payload"] == {"text": "hello"}
    assert "provider_metadata" not in serialized


def test_capabilities_report_only_missing_requirements() -> None:
    capabilities = HarnessCapabilities(cancel=True, partial_message=True)

    missing = capabilities.missing(
        frozenset({"cancel", "partial_message", "custom_tools"})
    )

    assert missing == frozenset({"custom_tools"})


def test_tool_result_requires_error_code_for_errors() -> None:
    with pytest.raises(ValidationError):
        ToolResult(
            content=(TextContentBlock(text="failed"),),
            is_error=True,
        )
