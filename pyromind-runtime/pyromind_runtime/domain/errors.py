from __future__ import annotations

from typing import Literal


type ProductErrorCode = Literal[
    "conversation_not_found",
    "checkpoint_not_found",
    "conversation_busy",
    "capability_not_supported",
    "command_conflict",
    "fork_target_conflict",
    "harness_operation_failed",
]


class ProductRuntimeError(RuntimeError):
    """Stable application error that does not expose harness internals."""

    def __init__(self, code: ProductErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

