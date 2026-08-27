"""Draining state used during graceful server shutdown.

The state lives on each app instance (``app.state.draining``) so separate
apps in the same process never affect each other. Once draining starts, new
HTTP requests fail fast with 503 (see ``DrainingMiddleware``) instead of
hanging until the process exits, and active WebSocket streams are closed
with a user-readable restart reason.
"""

from fastapi import FastAPI


RESTART_REASON = "Server is restarting, please wait"


def mark_draining(app: FastAPI) -> None:
    """Enter the draining state; new HTTP requests are rejected with 503."""
    app.state.draining = True


def reset_draining(app: FastAPI) -> None:
    """Clear the draining state (called at server startup)."""
    app.state.draining = False


def is_draining(app: FastAPI) -> bool:
    return getattr(app.state, "draining", False)
