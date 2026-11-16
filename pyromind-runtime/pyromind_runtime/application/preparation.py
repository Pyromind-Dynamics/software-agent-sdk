from __future__ import annotations

from pyromind_runtime.domain.experiment_execution import (
    DelegatedTask,
    DelegateExperimentTask,
)
from pyromind_runtime.domain.experiment_plans import (
    ExperimentContext,
    PlanConflictError,
    PlanOwner,
)
from pyromind_runtime.domain.preparation import (
    PreparationPolicy,
    PreparationState,
    SetPreparationPlan,
)
from pyromind_runtime.ports.preparation import PreparationStore


def validate_workflow_catalog(
    policy: PreparationPolicy, plan: ExperimentContext
) -> None:
    nodes = {node.id: node for node in policy.workflow_nodes}
    stages = {stage.id: stage for stage in plan.stages}
    for node in policy.workflow_nodes:
        if (
            node.max_instances is not None
            and sum(stage.kind == node.id for stage in plan.stages) > node.max_instances
        ):
            raise ValueError("workflow node exceeds its catalog instance limit")
    for index, stage in enumerate(plan.stages):
        node = nodes.get(stage.kind or "")
        if node is None:
            raise ValueError(
                f"plan.stages[{index}].kind: choose one of {', '.join(nodes)}"
            )
        if stage.enabled and not node.roles:
            raise ValueError("unimplemented catalog nodes must remain disabled")
    for index, task in enumerate(plan.tasks):
        node = nodes[stages[task.stage_id].kind or ""]
        allowed = tuple(item.id for item in node.tasks)
        if task.kind not in allowed:
            raise ValueError(
                f"plan.tasks[{index}].kind for stage {task.stage_id!r}: "
                f"allowed task kinds: {', '.join(allowed) or '(none)'}"
            )


class Preparations:
    def __init__(self, store: PreparationStore):
        self.store = store

    def set_plan(
        self,
        owner: PlanOwner,
        request: SetPreparationPlan,
        policy: PreparationPolicy,
        expected_revision: int,
    ) -> PreparationState:
        current = self.store.get(owner)
        if current.project_id:
            raise ValueError("preparation is already activated")
        if current.revision != expected_revision:
            raise PlanConflictError(
                "Review context changed; consider the latest conversation "
                "before submitting a new plan."
            )
        if not request.plan.goal.strip() or not request.plan.stages:
            raise ValueError("a review plan requires a goal and stages")
        validate_workflow_catalog(policy, request.plan)
        task = next(
            (task for task in request.plan.tasks if task.id == request.entry_task_id),
            None,
        )
        if task is None:
            raise ValueError("entry_task_id must reference plan.tasks")
        stage = next(
            stage for stage in request.plan.stages if stage.id == task.stage_id
        )
        if not stage.enabled or stage.placeholder:
            raise ValueError("entry task requires an enabled plan stage")
        node = next(node for node in policy.workflow_nodes if node.id == stage.kind)
        if len(node.roles) != 1:
            raise ValueError("entry stage must select exactly one execution role")
        resources = {resource.id: resource for resource in request.plan.resources}
        missing_inputs = (
            f"Entry task {task.id!r} requires concrete inputs: "
            "set inputs[].resource_id to a plan.resources[].id "
            "with a nonempty storage_path."
        )
        if not task.inputs:
            raise ValueError(missing_inputs)
        inputs = {}
        for item in task.inputs:
            resource = resources.get(item.resource_id or "")
            if resource is None or not resource.storage_path:
                raise ValueError(missing_inputs)
            inputs[item.name] = resource.storage_path
        entry = DelegateExperimentTask(
            role_id=node.roles[0],
            request_id=task.id,
            stage_id=task.stage_id,
            task=DelegatedTask(
                name=task.name,
                instruction=task.description or request.plan.goal,
                inputs=inputs,
                constraints=request.plan.constraints,
                expected_outputs=tuple(output.name for output in task.outputs),
            ),
        )
        updated = current.model_copy(
            update={
                "plan": request.plan,
                "entry_task": entry,
                "phase": "ready",
                "review_pending": False,
            }
        )
        return self.store.save(owner, updated, expected_revision)
