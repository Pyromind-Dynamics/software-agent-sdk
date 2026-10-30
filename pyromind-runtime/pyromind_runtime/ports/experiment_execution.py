from __future__ import annotations

from typing import Protocol

from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.ports.work_queue import WorkQueue


class ExperimentExecutionStore(Protocol):
    def request(
        self,
        owner: PlanOwner,
        project: str,
        revision: int,
        task: str,
        role: str,
        request_id: str,
    ) -> dict: ...

    def list(self, owner: PlanOwner) -> list[dict]: ...

    def propose(self, owner: PlanOwner, run_id: str, proposal: dict) -> dict: ...

    def decide(
        self,
        owner: PlanOwner,
        run_id: str,
        command_id: str,
        revision: int,
        fingerprint: str,
        decision: str,
    ) -> dict: ...

    def pending_deliveries(self) -> list[dict]: ...

    def acknowledge_delivery(self, identity: str) -> None: ...


class ExperimentPlatform(Protocol):
    async def prepare(self, configuration: dict, context: object) -> dict: ...
    async def submit(self, frozen: dict, context: object) -> str: ...
    async def query(self, task_id: str, context: object) -> dict: ...
    async def stop(self, task_id: str, context: object) -> None: ...


class DelegatedExecutionStore(ExperimentExecutionStore, Protocol):
    @property
    def queue(self) -> WorkQueue: ...

    def role_scope(
        self, owner: PlanOwner, run_id: str, role: str, scope: dict
    ) -> dict: ...

    def propose_from_role(
        self, owner: PlanOwner, run_id: str, role: str, scope: dict, proposal: dict
    ) -> dict: ...

    def save_work_checkpoint(
        self, identity: str, generation: int, value: dict
    ) -> None: ...

    def resume_authorized(self, owner: PlanOwner) -> None: ...

    def preparation_needs_input(
        self, owner: PlanOwner, run_id: str, scope: dict, reason: str
    ) -> None: ...
