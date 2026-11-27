from collections.abc import Callable
from typing import Protocol

from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext


class GraphRepairSubmitter(Protocol):
    def inspect(
        self,
        *,
        conversation_id: str,
        project_id: str,
        run_id: str,
        handoff_id: str,
        context: RequestContext,
    ) -> JsonObject: ...

    def submit(
        self,
        *,
        conversation_id: str,
        project_id: str,
        run_id: str,
        handoff_id: str,
        proposal: JsonObject,
        context: RequestContext,
        read_asset: Callable[[str], bytes] | None = None,
    ) -> JsonObject: ...
