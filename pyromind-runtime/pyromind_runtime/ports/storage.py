"""Storage port for evaluation annotation artifacts.

The save protocol only ever creates new ``*.r{N}`` object names and reads
them back; it never relies on atomic overwrite semantics.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class EvaluationStorageMissingError(LookupError):
    """A Storage object is confirmed absent (the backend answered "not found").

    Implementations raise this only when absence is explicit. Transport,
    permission, payload, and size-limit failures must raise
    :class:`EvaluationStorageUnavailableError` instead — never this one.
    """


class EvaluationStorageUnavailableError(RuntimeError):
    """A Storage operation failed for a reason other than confirmed absence.

    Covers transport errors, rejected requests, business failures riding
    HTTP 200, and unexpected payloads. The operation may be retryable, and
    the failure must never be classified as "the object does not exist".
    """


class EvaluationStorage(Protocol):
    """Byte-level access to one run's Storage output directory.

    Paths are Storage-relative (``<output_dir>/<name>``). Implementations
    must raise on failure: ``list`` returning an empty set means an empty
    directory, never an unreachable one.
    """

    def read(self, path: str) -> bytes: ...

    def write(self, path: str, payload: bytes) -> None: ...

    def list(self, prefix: str) -> set[str]:
        """Names directly inside ``prefix``; raises when unreachable."""
        ...


@runtime_checkable
class DeletableEvaluationStorage(Protocol):
    """Optional garbage-collection capability of a storage implementation."""

    def delete(self, path: str) -> None: ...
