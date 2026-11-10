from __future__ import annotations

import json

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
    PreparationFact,
    PreparationPolicy,
    PreparationView,
    SetPreparationPlan,
    UpdatePreparation,
)
from pyromind_runtime.ports.preparation import PreparationStore


def validate_workflow_catalog(
    policy: PreparationPolicy, plan: ExperimentContext
) -> None:
    if not policy.workflow_nodes:
        return
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
            choices = ", ".join(allowed) or "(none)"
            raise ValueError(
                f"plan.tasks[{index}].kind for stage {task.stage_id!r}: "
                f"allowed task kinds: {choices}"
            )


def validate_workflow_role(
    policy: PreparationPolicy, plan: ExperimentContext, stage_id: str | None, role: str
) -> None:
    if not policy.workflow_nodes:
        return
    nodes = {node.id: node for node in policy.workflow_nodes}
    targets = [
        {
            "stage_id": stage.id,
            "stage_kind": stage.kind,
            "stage_name": stage.name,
            "roles": list(nodes[stage.kind].roles),
        }
        for stage in plan.stages
        if stage.enabled
        and not stage.placeholder
        and stage.kind in nodes
        and nodes[stage.kind].roles
    ]
    target = next((item for item in targets if item["stage_id"] == stage_id), None)
    if target is None or role not in target["roles"]:
        raise ValueError(
            json.dumps(
                {
                    "error": "invalid_execution_target",
                    "reason": (
                        "stage_id is required and must identify an enabled "
                        "non-placeholder stage whose catalog allows role_id"
                    ),
                    "requested": {"stage_id": stage_id, "role_id": role},
                    "valid_targets": targets,
                    "action": (
                        "Choose a stage_id and role_id from valid_targets, then "
                        "submit the complete task with a new request_id."
                    ),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )


class Preparations:
    def __init__(self, store: PreparationStore):
        self.store = store

    @staticmethod
    def _check_revision(current: PreparationView, expected: int) -> None:
        if current.project_id:
            raise ValueError("preparation is already activated")
        if current.revision != expected:
            raise ValueError(
                json.dumps(
                    {
                        "error": "preparation_revision_conflict",
                        "expected_revision": expected,
                        "current_revision": current.revision,
                        "current_preparation": current.model_dump(
                            mode="json", include={"facts"}
                        ),
                        "action": "Merge intended changes with current facts and "
                        "retry with current_revision after checking user intent.",
                    },
                    ensure_ascii=False,
                )
            )

    def update(
        self, owner: PlanOwner, request: UpdatePreparation, evidence: set[str]
    ) -> PreparationView:
        current = self.store.get(owner)
        self._check_revision(current, request.expected_revision)
        fields = {field.key: field for field in current.policy.fields}
        if unknown := request.facts.keys() - fields.keys():
            raise ValueError(
                f"unknown checklist field(s): {', '.join(sorted(unknown))}; "
                "use only keys listed in get_preparation.policy.fields"
            )
        facts = {**current.facts, **request.facts}
        expected_types = {
            "string": str,
            "object": dict,
            "array": list,
            "integer": int,
            "boolean": bool,
        }
        for key, fact in request.facts.items():
            if fact.status != "resolved":
                continue
            value = fact.value
            if (
                type(value) is not expected_types[fields[key].value_type]
                or not str(value).strip()
                or value in ("", {}, [])
            ):
                raise ValueError(f"invalid value for checklist field {key}")
            if isinstance(value, dict) and any(
                value.get(name) in (None, "", {}, [])
                for name in fields[key].required_keys
            ):
                raise ValueError(f"missing required properties for {key}")
            if not fact.evidence_refs or not set(fact.evidence_refs) <= evidence:
                raise ValueError(f"missing or unknown evidence for {key}")
        changed = {
            key
            for key in fields
            if (
                current.facts.get(key, PreparationFact()).value,
                current.facts.get(key, PreparationFact()).status,
            )
            != (
                facts.get(key, PreparationFact()).value,
                facts.get(key, PreparationFact()).status,
            )
        }
        while True:
            invalid = {
                key
                for key, field in fields.items()
                if key not in changed
                and changed.intersection(field.depends_on)
                and facts.get(key) == current.facts.get(key)
            }
            if not invalid:
                break
            for key in invalid:
                facts[key] = PreparationFact()
            changed.update(invalid)
        missing = tuple(
            key
            for key, field in fields.items()
            if field.required and facts.get(key, PreparationFact()).status != "resolved"
        )
        updated = current.model_copy(
            update={
                "facts": facts,
                "missing": missing,
                "plan": None,
                "entry_task": None,
                "phase": "collecting" if missing else "planning",
                "review_pending": False,
            }
        )
        return self._save(owner, updated, request.expected_revision)

    def _save(
        self, owner: PlanOwner, updated: PreparationView, expected_revision: int
    ) -> PreparationView:
        try:
            return self.store.save(owner, updated, expected_revision)
        except PlanConflictError:
            self._check_revision(self.store.get(owner), expected_revision)
            raise

    def set_plan(
        self, owner: PlanOwner, request: SetPreparationPlan
    ) -> PreparationView:
        current = self.store.get(owner)
        self._check_revision(current, request.expected_revision)
        if current.missing:
            raise ValueError("complete the checklist before generating a workflow plan")
        validate_workflow_catalog(current.policy, request.plan)
        task = next(
            (task for task in request.plan.tasks if task.id == request.entry_task_id),
            None,
        )
        if task is None:
            raise ValueError("entry_task_id must reference plan.tasks")
        validate_workflow_role(
            current.policy, request.plan, task.stage_id, request.role_id
        )
        stage = next(
            stage for stage in request.plan.stages if stage.id == task.stage_id
        )
        if not stage.enabled or stage.placeholder:
            raise ValueError("entry task requires an enabled plan stage")
        if request.role_id not in current.policy.allowed_roles:
            raise ValueError("entry task requires an enabled execution role")
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
        if any(item.storage_path is None for item in request.plan.resources):
            raise ValueError("resolve plan resource locations before creation")
        entry = DelegateExperimentTask(
            role_id=request.role_id,
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
        updated = PreparationView.model_validate(
            {
                **current.model_dump(mode="json"),
                "plan": request.plan,
                "entry_task": entry,
                "phase": "ready",
                "review_pending": False,
            }
        )
        return self._save(owner, updated, request.expected_revision)
