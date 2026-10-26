from __future__ import annotations

from typing import Literal


type ProductErrorCode = Literal[
    "conversation_not_found",
    "checkpoint_not_found",
    "conversation_busy",
    "capability_not_supported",
    "command_conflict",
    "fork_target_conflict",
    "capacity_exceeded",
    "harness_operation_failed",
    "input_not_pending",
    "input_pending",
]


class ProductRuntimeError(RuntimeError):
    """Stable application error that does not expose harness internals."""

    def __init__(self, code: ProductErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class UserInputValidationError(ValueError):
    """Invalid answer structure or semantics (HTTP 422, the request stays pending)."""


class EvaluationRunNotFoundError(RuntimeError):
    """The evaluation run or its conversation is confirmed absent or foreign.

    Raised only for outcomes the caller can fix by naming a different
    target: unknown conversation, foreign conversation, malformed run id,
    unknown or foreign run. Messages are safe to surface (HTTP 404); every
    other failure mode must be reported as an availability error instead of
    masquerading as "not found".
    """


class EvaluationRunUnavailableError(RuntimeError):
    """The run's durable local state cannot be read (HTTP 5xx, sanitized).

    A corrupt or unreadable Product Store means the state exists but the
    server cannot serve it; the cause stays in logs, never in the response.
    """
