from __future__ import annotations

from typing import Protocol


class WorkQueue(Protocol):
    def claim(
        self, lease_seconds: int = 90, kinds: tuple[str, ...] | None = None
    ) -> dict | None: ...

    def heartbeat(self, identity: str, generation: int, seconds: int = 90) -> None: ...

    def finish(
        self, identity: str, generation: int, checkpoint: dict, state: str = "completed"
    ) -> None: ...

    def expired(self) -> list[dict]: ...

    def reconcile_expired(self, identity: str, safe_to_retry: bool) -> None: ...
