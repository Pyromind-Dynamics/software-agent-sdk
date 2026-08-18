from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from threading import RLock


class ToolRequestContextNotAvailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class ToolRequestContext:
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "headers",
            {name.lower(): value for name, value in self.headers.items()},
        )


class SessionToolContextStore:
    """Keeps request credentials in memory and outside persisted session state."""

    def __init__(self, *, allowed_header_names: Iterable[str] = ()) -> None:
        self._allowed_header_names = frozenset(
            name.lower() for name in allowed_header_names
        )
        self._contexts: dict[str, ToolRequestContext] = {}
        self._lock = RLock()

    def bind(self, session_id: str, context: ToolRequestContext) -> None:
        headers = dict(context.headers)
        denied = set(headers) - self._allowed_header_names
        if denied:
            names = ", ".join(sorted(denied))
            raise ValueError(f"Tool request headers are not allowed: {names}")
        with self._lock:
            self._contexts[session_id] = ToolRequestContext(headers=headers)

    def get(self, session_id: str) -> ToolRequestContext:
        with self._lock:
            context = self._contexts.get(session_id)
            if context is None:
                raise ToolRequestContextNotAvailableError(
                    f"Tool request context is unavailable: {session_id}"
                )
            return ToolRequestContext(headers=dict(context.headers))

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._contexts.pop(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._contexts.clear()
