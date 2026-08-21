"""Tests for terminal backend I/O error normalization."""

import errno
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openhands.tools.terminal.definition import TerminalAction
from openhands.tools.terminal.impl import TerminalExecutor


def _executor_with_failing_session(tmp_path: Path, error: OSError) -> TerminalExecutor:
    """Build an executor whose session.execute raises *error*."""
    with patch("openhands.tools.terminal.impl.create_terminal_session") as factory:
        executor = TerminalExecutor(
            working_dir=str(tmp_path), terminal_type="subprocess"
        )
        session = MagicMock()
        session._closed = False
        session.work_dir = str(tmp_path)
        session.username = None
        session.no_change_timeout_seconds = None
        session.execute.side_effect = error
        factory.return_value = session
        executor._session = session
    return executor


def test_eio_error_returns_structured_hint(tmp_path: Path) -> None:
    """EIO (errno 5) from the backend must return a structured observation
    explaining the sandbox is unavailable instead of a raw OSError."""
    executor = _executor_with_failing_session(
        tmp_path, OSError(errno.EIO, "Input/output error")
    )
    observation = executor(
        TerminalAction(command="mkdir -p public_data/data-preparation")
    )

    assert observation.is_error
    assert observation.exit_code == -1
    assert "[Terminal-unavailable]" in observation.text
    assert "errno 5: Input/output error" in observation.text
    assert "file_editor/apply_patch" in observation.text
    assert observation.command == "mkdir -p public_data/data-preparation"


def test_eagain_error_returns_structured_hint(tmp_path: Path) -> None:
    """EAGAIN (resource exhaustion) is normalized the same way."""
    executor = _executor_with_failing_session(
        tmp_path, OSError(errno.EAGAIN, "Resource temporarily unavailable")
    )
    observation = executor(TerminalAction(command="pwd"))

    assert observation.is_error
    assert "[Terminal-unavailable]" in observation.text
    assert f"errno {errno.EAGAIN}: Resource temporarily unavailable" in observation.text


def test_unrelated_oserror_propagates(tmp_path: Path) -> None:
    """OSErrors that are not backend resource failures still propagate."""
    executor = _executor_with_failing_session(
        tmp_path, OSError(errno.EPERM, "Operation not permitted")
    )
    with pytest.raises(OSError):
        executor(TerminalAction(command="pwd"))
