"""Helpers for sanitising the environment seen by tmux panes.

Shared between :mod:`tmux_terminal` and :mod:`tmux_pane_pool`.  Kept in its
own module so neither has to import the other (they already have a
``TmuxPanePool -> TmuxTerminal`` dependency, and reversing it would create a
circular import at module load time).
"""

from __future__ import annotations

import os
import subprocess

import libtmux

from openhands.sdk.logger import get_logger
from openhands.sdk.utils import sanitized_env
from openhands.sdk.utils.command import (
    _SENSITIVE_ENV_PREFIXES,
    _SENSITIVE_ENV_VARS,
)


logger = get_logger(__name__)


def kill_tmux_server(socket_name: str) -> None:
    """Kill any existing tmux server on the given socket.

    libtmux's ``Server(environment=...)`` parameter only takes effect when a
    **new** tmux server is spawned.  If a server is already running on the
    socket (e.g. left over from a previous agent-server process, or started
    under a build whose ``sanitized_env()`` did not strip sensitive vars),
    libtmux simply connects to it and the ``environment`` argument is
    ignored — the stale server's process environ (with leaked credentials)
    is then inherited by every new pane.

    Killing the server before creating a new session guarantees the new
    server starts with the current sanitized environment.
    """
    try:
        subprocess.run(
            ["tmux", "-L", socket_name, "kill-server"],
            capture_output=True,
            timeout=2,
            env=sanitized_env(),
        )
    except Exception as e:
        logger.debug(f"kill-server on socket {socket_name} failed: {e}")


def strip_sensitive_tmux_env(session: libtmux.Session) -> None:
    """Unset sensitive env vars from the tmux server's global environment.

    tmux builds each new pane's environ by **copying the server process
    environ** and then overlaying the server-global / session environments.
    Even when ``set-environment`` populates the session env with sanitized
    values, vars that the server inherited (e.g. from a stale process) still
    leak into panes unless explicitly *unset* in the tmux environment.

    ``set-environment -u -g <KEY>`` removes ``KEY`` from tmux's global
    environment so new panes won't inherit it, regardless of the server
    process environ.
    """
    keys_to_unset: set[str] = set(_SENSITIVE_ENV_VARS)
    for prefix in _SENSITIVE_ENV_PREFIXES:
        for key in os.environ:
            if key.startswith(prefix):
                keys_to_unset.add(key)
    for key in keys_to_unset:
        try:
            session.cmd("set-environment", "-u", "-g", key)
        except Exception:
            pass
