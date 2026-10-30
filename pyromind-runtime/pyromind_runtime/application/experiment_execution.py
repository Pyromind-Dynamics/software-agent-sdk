"""Harness-neutral admission and confirmation of experiment executions."""

from __future__ import annotations

from pyromind_runtime.domain.experiment_execution import (
    ExperimentExecutionView,
    RequestExperimentTask,
)
from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.ports.experiment_execution import ExperimentExecutionStore
from pyromind_runtime.ports.experiment_plans import ExperimentPlanStore


class ExperimentExecutions:
    def __init__(self, plans: ExperimentPlanStore, store: ExperimentExecutionStore):
        self.plans = plans
        self.store = store

    def request(
        self,
        owner: PlanOwner,
        request: RequestExperimentTask,
        allowed_roles: tuple[str, ...],
    ) -> dict:
        if request.role_id not in allowed_roles:
            raise PermissionError("execution role is not enabled")
        view = self.plans.view(owner)
        if not view.current or view.current_revision != request.expected_revision:
            raise ValueError("execution requires the current confirmed plan")
        task = next(
            (t for t in view.current.plan.tasks if t.id == request.task_id), None
        )
        if task is None:
            raise ValueError("task is absent from confirmed plan")
        stage = next(s for s in view.current.plan.stages if s.id == task.stage_id)
        if not stage.enabled or stage.placeholder:
            raise ValueError("task stage is disabled")
        assert view.project_id is not None
        return self.store.request(
            owner,
            view.project_id,
            request.expected_revision,
            request.task_id,
            request.role_id,
            request.request_id,
        )

    def decide(
        self,
        owner: PlanOwner,
        run_id: str,
        command_id: str,
        revision: int,
        fingerprint: str,
        decision: str,
    ) -> dict:
        view = self.plans.view(owner)
        if decision == "allow" and view.current_revision != revision:
            raise ValueError("plan changed; prepare a new execution proposal")
        return self.store.decide(
            owner, run_id, command_id, revision, fingerprint, decision
        )

    def list(self, owner: PlanOwner) -> tuple[ExperimentExecutionView, ...]:
        versions = {}
        result = []
        for row in self.store.list(owner):
            revision = row["revision"]
            if revision not in versions:
                versions[revision] = self.plans.revision(owner, revision)
            plan = versions[revision].plan
            task = next(t for t in plan.tasks if t.id == row["task_id"])
            submission = row.get("submission") or {}
            result.append(
                ExperimentExecutionView(
                    run_id=row["id"],
                    project_id=row["project_id"],
                    plan_revision=revision,
                    task_id=task.id,
                    stage_id=task.stage_id,
                    task_name=task.name,
                    role_id=row["role_id"],
                    state=row["state"],
                    report_state=row["report_state"],
                    fingerprint=row.get("fingerprint"),
                    proposal=row.get("proposal"),
                    metrics=row.get("metrics"),
                    platform_task_id=submission.get("task_id"),
                    artifacts=tuple(
                        {"name": a["name"], "uri": a["uri"], "sha256": a["sha256"]}
                        for a in row.get("artifacts", [])
                    ),
                )
            )
        return tuple(result)
