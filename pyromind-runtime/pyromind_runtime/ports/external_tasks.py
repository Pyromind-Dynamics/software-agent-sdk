from __future__ import annotations

from typing import Protocol

from pyromind_runtime.domain.content import JsonObject


class ExternalTaskRegistry(Protocol):
    """Durable business-task lookup behind the Product application layer."""

    def owner(self, task_id: str) -> str | None: ...

    def resolve(self, conversation_id: str, task_id: str) -> JsonObject | None: ...

    def update_status(
        self,
        conversation_id: str,
        task_id: str,
        status: str,
    ) -> None: ...
