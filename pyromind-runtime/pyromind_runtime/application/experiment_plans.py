from __future__ import annotations

from uuid import uuid4

from pyromind_runtime.domain.experiment_plans import (
    CreateExperimentPlan,
    PlanOwner,
    PlanProposal,
    ReviseExperimentPlan,
    WorkflowPlan,
)
from pyromind_runtime.ports.experiment_plans import ExperimentPlanStore


def plan_changes(before: WorkflowPlan | None, after: WorkflowPlan) -> tuple[str, ...]:
    if before is None:
        return ("创建实验计划",)
    changes: list[str] = []
    for key, label in (
        ("name", "项目名称"),
        ("goal", "实验目标"),
        ("resources", "已知资源"),
        ("unresolved", "待补充事项"),
        ("edges", "阶段连线"),
    ):
        if getattr(before, key) != getattr(after, key):
            changes.append(f"修改{label}")
    for key, label in (("stages", "阶段"), ("tasks", "任务")):
        old = {item.id: item for item in getattr(before, key)}
        new = {item.id: item for item in getattr(after, key)}
        for item_id, item in new.items():
            if item_id not in old:
                changes.append(f"新增{label}：{item.name}")
            elif old[item_id] != item:
                changes.append(f"修改{label}：{item.name}")
        for item_id, item in old.items():
            if item_id not in new:
                changes.append(f"删除{label}：{item.name}")
    return tuple(changes)


class ExperimentPlans:
    def __init__(self, store: ExperimentPlanStore) -> None:
        self.store = store

    def propose(
        self,
        owner: PlanOwner,
        request: CreateExperimentPlan | ReviseExperimentPlan,
    ) -> PlanProposal:
        view = self.store.view(owner)
        initial = not isinstance(request, ReviseExperimentPlan)
        proposal = PlanProposal(
            proposal_id=uuid4().hex,
            base_revision=(request.expected_revision if not initial else 0),
            plan=request.plan,
            summary=request.summary,
            changes=plan_changes(
                view.current.plan if view.current else None, request.plan
            ),
        )
        return self.store.propose(
            owner,
            proposal,
            source_proposal_id=(None if initial else request.source_proposal_id),
            initial=initial,
        )
