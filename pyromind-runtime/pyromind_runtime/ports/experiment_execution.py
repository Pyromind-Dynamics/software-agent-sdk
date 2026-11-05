from __future__ import annotations

from typing import Protocol

from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.ports.work_queue import WorkQueue


class ExperimentExecutionStore(Protocol):
    def work_items(self, owner: PlanOwner) -> tuple[dict, ...]: ...

    def request(
        self,
        owner: PlanOwner,
        project: str,
        revision: int | None,
        task: str,
        role: str,
        request_id: str,
        *,
        task_spec: dict | None = None,
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
    def mark_submission_uncertain(
        self, owner: PlanOwner, run_id: str, submission_id: str
    ) -> None: ...

    def retire_legacy_work(self, owner: PlanOwner) -> bool: ...

    def recover_submissions(self) -> int: ...

    def recover_managed_receipt(
        self, owner: PlanOwner, run_id: str, submission_id: str, receipt: dict
    ) -> None: ...

    def note_wait(
        self,
        owner: PlanOwner,
        run_id: str,
        reason: str,
        *,
        artifact_failure: bool = False,
    ) -> None: ...

    def conflicting_tasks(self) -> set[str]: ...

    def abandon(self, owner: PlanOwner, run_id: str, command_id: str) -> dict: ...

    def resolve(
        self,
        owner: PlanOwner,
        run_id: str,
        task_id: str,
        command_id: str,
        verified_evidence: dict,
    ) -> dict: ...

    def cancel(self, owner: PlanOwner, run_id: str) -> dict: ...

    def enqueue_manager(
        self, owner: PlanOwner, identity: str, payload: dict, priority: int = 100
    ) -> str: ...

    def continue_task(
        self, owner: PlanOwner, run_id: str, command_id: str, message: str
    ) -> str: ...

    def guidance_target(self, owner: PlanOwner, run_id: str, work_id: str) -> dict: ...

    def enqueue_continuation(
        self, owner: PlanOwner, run_id: str, guidance_id: str, message: str
    ) -> str: ...

    def platform_run(self, task_id: str) -> dict | None: ...

    def cancellation_result(
        self,
        owner: PlanOwner,
        run_id: str,
        scope: dict,
        native_stopped: bool,
        platform_stopped: bool,
    ) -> dict: ...

    def ingest(
        self, identity: str, task_id: str, source: str, payload: dict
    ) -> None: ...

    def apply_observations(self) -> int: ...

    def register_artifacts(
        self, owner: PlanOwner, run_id: str, verified: list[dict], metrics: dict
    ) -> None: ...

    def enqueue_report(
        self, owner: PlanOwner, run_id: str, message: str = "", command_id: str = ""
    ) -> str: ...

    def complete_agent_turn(
        self,
        owner: PlanOwner,
        run_id: str,
        scope: dict,
        text: str,
        report: bool = False,
    ) -> dict: ...

    def native_reference(self, owner: PlanOwner, run_id: str) -> str | None: ...

    def managed_submission(
        self, owner: PlanOwner, run_id: str, scope: dict, arguments: dict
    ) -> dict: ...

    def managed_receipt(
        self, owner: PlanOwner, run_id: str, scope: dict, operation: str, receipt: dict
    ) -> None: ...

    def record_result(
        self,
        owner: PlanOwner,
        run_id: str,
        scope: dict,
        result: dict,
        report_artifact: dict | None,
        verified: list[dict] | None = None,
    ) -> None: ...

    def record_result_failure(
        self, owner: PlanOwner, run_id: str, scope: dict, reason: str
    ) -> None: ...

    def record_checkpoint(
        self, owner: PlanOwner, run_id: str, scope: dict, checkpoint: dict
    ) -> None: ...

    def resolve_checkpoint(
        self,
        owner: PlanOwner,
        run_id: str,
        checkpoint_id: str,
        command_id: str,
        approved: bool,
    ) -> None: ...

    def queue_artifact_report(self, owner: PlanOwner, run_id: str) -> None: ...

    def stage_workflow(
        self, owner: PlanOwner, run_id: str, scope: dict, request_id: str
    ) -> None: ...

    def begin_workflow(
        self,
        owner: PlanOwner,
        run_id: str,
        request_id: str,
        definition_fingerprint: str,
        allow: bool,
    ) -> str | None: ...

    def pending_runs(self) -> list[dict]: ...

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
