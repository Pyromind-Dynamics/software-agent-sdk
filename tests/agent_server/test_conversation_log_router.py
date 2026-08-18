import logging

import pytest
from starlette.requests import Request

from openhands.agent_server.event_router import _bind_conversation_log_context


def _request_with_conversation(conversation_id: str | None) -> Request:
    scope = {
        "type": "http",
        "path": "/api/conversations/conv-router/events/search",
        "method": "GET",
        "scheme": "http",
        "server": ("test", 80),
        "client": ("1.2.3.4", 123),
        "path_params": {"conversation_id": conversation_id},
    }
    return Request(scope)


async def test_router_dependency_binds_conversation_id(
    caplog: pytest.LogCaptureFixture,
):
    generator = _bind_conversation_log_context(
        _request_with_conversation("conv-router")
    )
    await generator.__anext__()
    try:
        with caplog.at_level(logging.INFO):
            logging.getLogger("openhands.agent_server").info("request log")
        assert "[cid=conv-router] request log" in caplog.text
    finally:
        with pytest.raises(StopAsyncIteration):
            await generator.__anext__()


async def test_router_dependency_without_conversation_id(
    caplog: pytest.LogCaptureFixture,
):
    generator = _bind_conversation_log_context(_request_with_conversation(None))
    await generator.__anext__()
    try:
        with caplog.at_level(logging.INFO):
            logging.getLogger("openhands.agent_server").info("no-conversation log")
        assert "[cid=" not in caplog.text
    finally:
        with pytest.raises(StopAsyncIteration):
            await generator.__anext__()
