from __future__ import annotations

from pyromind_runtime.domain.base import ContractModel


class HarnessCapabilities(ContractModel):
    resume: bool = False
    cancel: bool = False
    permission_reply: bool = False
    partial_message: bool = False
    fork: bool = False
    workflow_rollback: bool = False
    native_workspace_tools: frozenset[str] = frozenset()
