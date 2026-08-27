"""Tests for WebSocketCallbackClient."""

import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets
import websockets.frames

from openhands.sdk.conversation.impl.remote_conversation import (
    WebSocketCallbackClient,
    WebSocketConnectionState,
)
from openhands.sdk.event.conversation_state import (
    FULL_STATE_KEY,
    ConversationStateUpdateEvent,
)
from openhands.sdk.event.llm_convertible import MessageEvent
from openhands.sdk.llm import Message, TextContent


@pytest.fixture
def mock_event():
    """Create a test event."""
    return MessageEvent(
        id="test-event-id",
        timestamp=datetime.now().isoformat(),
        source="agent",
        llm_message=Message(
            role="assistant", content=[TextContent(text="Test message")]
        ),
    )


def _state_update_message() -> str:
    return ConversationStateUpdateEvent(key=FULL_STATE_KEY, value={}).model_dump_json()


class _FakeWebSocket:
    """Async context manager yielding canned messages, then ending cleanly."""

    def __init__(self, messages: list[str]):
        self._messages = iter(messages)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._messages)
        except StopIteration:
            raise StopAsyncIteration


def test_websocket_client_lifecycle():
    """Test WebSocket client start/stop lifecycle with idempotency."""
    callback_events = []

    def test_callback(event):
        callback_events.append(event)

    client = WebSocketCallbackClient(
        host="http://localhost:8000",
        conversation_id="test-conv-id",
        callback=test_callback,
    )

    assert isinstance(client, WebSocketCallbackClient)

    with patch.object(client, "_run"):
        # Start the client
        client.start()
        assert client._thread is not None
        assert client._thread.daemon is True

        # Starting again should be idempotent
        original_thread = client._thread
        client.start()
        assert client._thread is original_thread

        # Stop the client
        client.stop()
        assert client._stop.is_set()
        assert client._thread is None


def test_websocket_client_error_resilience(mock_event):
    """Test that callback exceptions are logged but don't crash the client."""

    def failing_callback(event):
        raise ValueError("Test error")

    client = WebSocketCallbackClient(
        host="http://localhost:8000",
        conversation_id="test-conv-id",
        callback=failing_callback,
    )

    with patch(
        "openhands.sdk.conversation.impl.remote_conversation.logger"
    ) as mock_logger:
        try:
            client.callback(mock_event)
        except Exception:
            mock_logger.exception("ws_event_processing_error", stack_info=True)

        mock_logger.exception.assert_called_with(
            "ws_event_processing_error", stack_info=True
        )


def test_websocket_client_stop_timeout():
    """Test WebSocket client handles thread join timeout gracefully."""

    def noop_callback(event):
        pass

    client = WebSocketCallbackClient(
        host="http://localhost:8000",
        conversation_id="test-conv-id",
        callback=noop_callback,
    )

    # Mock thread that simulates delay
    mock_thread = MagicMock()
    mock_thread.join.side_effect = lambda timeout: time.sleep(0.1)
    client._thread = mock_thread

    start_time = time.time()
    client.stop()
    end_time = time.time()

    mock_thread.join.assert_called_with(timeout=5)
    assert end_time - start_time < 1.0
    assert client._thread is None


def test_websocket_client_callback_invocation(mock_event):
    """Test callback is invoked with events."""
    callback_events = []

    def test_callback(event):
        callback_events.append(event)

    client = WebSocketCallbackClient(
        host="http://localhost:8000",
        conversation_id="test-conv-id",
        callback=test_callback,
    )

    client.callback(mock_event)

    assert len(callback_events) == 1
    assert callback_events[0].id == mock_event.id


def test_websocket_client_url_encodes_api_key():
    """Test that API key special characters are URL-encoded in the WebSocket URL."""
    captured_urls = []

    class _MockAsyncContextManager:
        def __init__(self, url):
            self.url = url

        async def __aenter__(self):
            captured_urls.append(self.url)
            raise websockets.exceptions.ConnectionClosed(
                rcvd=websockets.frames.Close(1000, "test"),
                sent=websockets.frames.Close(1000, "test"),
                rcvd_then_sent=False,
            )

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _MockConnect:
        def __call__(self, url, *args, **kwargs):
            return _MockAsyncContextManager(url)

    client = WebSocketCallbackClient(
        host="http://localhost:8000",
        conversation_id="test-conv-id",
        callback=lambda event: None,
        api_key="1+FYh/SRE=ds 8Q",
    )

    with patch(
        "openhands.sdk.conversation.impl.remote_conversation.websockets.connect",
        _MockConnect(),
    ):
        asyncio.run(client._client_loop())

    assert len(captured_urls) == 1
    assert "session_api_key=1%2BFYh%2FSRE%3Dds%208Q" in captured_urls[0]


def test_websocket_client_notifies_state_transitions_around_server_restart():
    """Client surfaces connect/reconnect/stop transitions and keeps retrying
    after abnormal and service-restart closes instead of dying silently."""
    states: list[tuple[WebSocketConnectionState, str | None]] = []

    def on_state_change(state, reason):
        states.append((state, reason))

    client = WebSocketCallbackClient(
        host="http://localhost:8000",
        conversation_id="test-conv-id",
        callback=lambda event: None,
        on_state_change=on_state_change,
    )

    abnormal = websockets.exceptions.ConnectionClosed(
        rcvd=None, sent=None, rcvd_then_sent=None
    )
    restart_close = websockets.frames.Close(1012, "Server is restarting, please wait")
    restart = websockets.exceptions.ConnectionClosed(
        rcvd=restart_close,
        sent=restart_close,
        rcvd_then_sent=False,
    )
    refused_close = websockets.frames.Close(4004, "Conversation not found")
    refused = websockets.exceptions.ConnectionClosed(
        rcvd=refused_close,
        sent=refused_close,
        rcvd_then_sent=False,
    )

    with (
        patch(
            "openhands.sdk.conversation.impl.remote_conversation.websockets.connect",
            side_effect=[
                abnormal,
                restart,
                _FakeWebSocket([_state_update_message()]),
                refused,
            ],
        ),
        patch(
            "openhands.sdk.conversation.impl.remote_conversation.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        asyncio.run(client._client_loop())

    assert [(state.value, reason) for state, reason in states] == [
        ("connecting", None),
        ("reconnecting", "code 1006"),
        ("reconnecting", "Server is restarting, please wait"),
        ("connected", None),
        ("reconnecting", None),
        ("stopped", "Conversation not found"),
    ]


def test_websocket_client_reports_http_503_as_temporarily_unavailable():
    """HTTP 503 during the WS handshake maps to a reconnecting state."""
    states: list[tuple[WebSocketConnectionState, str | None]] = []

    client = WebSocketCallbackClient(
        host="http://localhost:8000",
        conversation_id="test-conv-id",
        callback=lambda event: None,
        on_state_change=lambda state, reason: states.append((state, reason)),
    )

    refused_close = websockets.frames.Close(4004, "Conversation not found")
    refused = websockets.exceptions.ConnectionClosed(
        rcvd=refused_close,
        sent=refused_close,
        rcvd_then_sent=False,
    )

    with (
        patch(
            "openhands.sdk.conversation.impl.remote_conversation.websockets.connect",
            side_effect=[
                websockets.exceptions.InvalidStatus(MagicMock(status_code=503)),
                _FakeWebSocket([_state_update_message()]),
                refused,
            ],
        ),
        patch(
            "openhands.sdk.conversation.impl.remote_conversation.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        asyncio.run(client._client_loop())

    assert [(state.value, reason) for state, reason in states] == [
        ("connecting", None),
        ("reconnecting", "Service temporarily unavailable"),
        ("connected", None),
        ("reconnecting", None),
        ("stopped", "Conversation not found"),
    ]


def test_websocket_client_stops_on_deliberate_close_without_retrying():
    """A 4xxx application close ends the stream after a single attempt."""
    states: list[tuple[WebSocketConnectionState, str | None]] = []

    client = WebSocketCallbackClient(
        host="http://localhost:8000",
        conversation_id="test-conv-id",
        callback=lambda event: None,
        on_state_change=lambda state, reason: states.append((state, reason)),
    )

    refused_close = websockets.frames.Close(4001, "Authentication failed")
    refused = websockets.exceptions.ConnectionClosed(
        rcvd=refused_close,
        sent=refused_close,
        rcvd_then_sent=False,
    )

    connect = MagicMock(side_effect=[refused])
    with (
        patch(
            "openhands.sdk.conversation.impl.remote_conversation.websockets.connect",
            connect,
        ),
        patch(
            "openhands.sdk.conversation.impl.remote_conversation.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        asyncio.run(client._client_loop())

    connect.assert_called_once()
    assert [(state.value, reason) for state, reason in states] == [
        ("connecting", None),
        ("stopped", "Authentication failed"),
    ]


def test_websocket_client_reports_generic_connect_failure_as_reconnecting():
    """Generic connection failures surface as RECONNECTING with a reason."""
    states: list[tuple[WebSocketConnectionState, str | None]] = []

    client = WebSocketCallbackClient(
        host="http://localhost:8000",
        conversation_id="test-conv-id",
        callback=lambda event: None,
        on_state_change=lambda state, reason: states.append((state, reason)),
    )

    refused_close = websockets.frames.Close(4004, "Conversation not found")
    refused = websockets.exceptions.ConnectionClosed(
        rcvd=refused_close,
        sent=refused_close,
        rcvd_then_sent=False,
    )

    with (
        patch(
            "openhands.sdk.conversation.impl.remote_conversation.websockets.connect",
            side_effect=[ConnectionError("server down"), refused],
        ),
        patch(
            "openhands.sdk.conversation.impl.remote_conversation.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        asyncio.run(client._client_loop())

    assert [(state.value, reason) for state, reason in states] == [
        ("connecting", None),
        ("reconnecting", "Connection failed"),
        ("stopped", "Conversation not found"),
    ]
