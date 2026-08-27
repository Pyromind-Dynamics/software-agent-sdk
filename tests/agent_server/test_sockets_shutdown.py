"""Tests for gracefully closing client WebSocket connections on shutdown."""

from unittest.mock import AsyncMock, MagicMock

import openhands.agent_server.sockets as sockets_mod
from openhands.agent_server.drain import RESTART_REASON
from openhands.agent_server.sockets import (
    _active_client_sockets,
    close_client_sockets,
)


def _make_ws() -> MagicMock:
    ws = MagicMock()
    ws.close = AsyncMock()
    return ws


async def test_close_client_sockets_closes_registered_sockets_with_restart_reason():
    ws1, ws2 = _make_ws(), _make_ws()
    sockets_mod._register_client_socket(ws1)
    sockets_mod._register_client_socket(ws2)
    try:
        await close_client_sockets()
        ws1.close.assert_awaited_once_with(code=1012, reason=RESTART_REASON)
        ws2.close.assert_awaited_once_with(code=1012, reason=RESTART_REASON)
        assert _active_client_sockets == set()
    finally:
        sockets_mod._unregister_client_socket(ws1)
        sockets_mod._unregister_client_socket(ws2)


async def test_close_client_sockets_ignores_failed_closes():
    ws1, ws2 = _make_ws(), _make_ws()
    ws1.close.side_effect = RuntimeError("already closed")
    sockets_mod._register_client_socket(ws1)
    sockets_mod._register_client_socket(ws2)
    try:
        await close_client_sockets()
        ws2.close.assert_awaited_once_with(code=1012, reason=RESTART_REASON)
        assert _active_client_sockets == set()
    finally:
        sockets_mod._unregister_client_socket(ws1)
        sockets_mod._unregister_client_socket(ws2)
