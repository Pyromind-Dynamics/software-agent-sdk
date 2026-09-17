from __future__ import annotations

from pydantic import Field

from pyromind_runtime.domain.base import ContractModel


class ResourceLimits(ContractModel):
    memory_limit_bytes: int = Field(default=500 * 1024 * 1024, gt=0)
    nproc_limit: int = Field(default=2, ge=2)


class HarnessCapabilities(ContractModel):
    resume: bool = False
    cancel: bool = False
    permission_reply: bool = False
    partial_message: bool = False
    fork: bool = False
    workflow_rollback: bool = False
    external_task_resume: bool = False
    native_workspace_tools: frozenset[str] = frozenset()
    enforced_limits: frozenset[str] = frozenset()
