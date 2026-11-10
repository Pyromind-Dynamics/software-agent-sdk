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
    for stage in plan.stages:
        node = nodes.get(stage.kind or "")
        if node is None:
            raise ValueError("select stage.kind from preparation.policy.workflow_nodes")
        if stage.enabled and not node.roles:
            raise ValueError("unimplemented catalog nodes must remain disabled")
    for task in plan.tasks:
        node = nodes[stages[task.stage_id].kind or ""]
        if task.kind not in {item.id for item in node.tasks}:
            raise ValueError("select task.kind from the selected catalog node tasks")


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

    def update(
        self, owner: PlanOwner, request: UpdatePreparation, evidence: set[str]
    ) -> PreparationView:
        current = self.store.get(owner)
        if current.project_id:
            raise ValueError("preparation is already activated")
        fields = {field.key: field for field in current.policy.fields}
        if set(request.facts) - fields.keys():
            raise ValueError("unknown checklist field")
        facts = dict(request.facts)
        expected_types = {
            "string": str,
            "object": dict,
            "array": list,
            "integer": int,
            "boolean": bool,
        }
        for key, fact in facts.items():
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
        if missing and (request.plan is not None or request.entry_task is not None):
            raise ValueError("complete the checklist before generating a workflow plan")
        if (current.missing or changed) and request.plan is not None:
            raise ValueError("save the complete checklist before generating a plan")
        plan, entry = request.plan, request.entry_task
        if (plan is None) != (entry is None):
            raise ValueError("a plan requires its first executable task")
        if plan is not None and entry is not None:
            validate_workflow_catalog(current.policy, plan)
            validate_workflow_role(current.policy, plan, entry.stage_id, entry.role_id)
            stage = next(
                (stage for stage in plan.stages if stage.id == entry.stage_id), None
            )
            if stage is None or not stage.enabled or stage.placeholder:
                raise ValueError("entry task requires an enabled plan stage")
            if (
                entry.role_id not in current.policy.allowed_roles
                or entry.intent != "execute"
            ):
                raise ValueError("entry task requires an enabled execution role")
            declared = next(
                (task for task in plan.tasks if task.id == entry.request_id), None
            )
            if declared is None or declared.stage_id != entry.stage_id:
                raise ValueError(
                    "entry task must reference a task in the selected stage"
                )
            resources = {resource.id: resource for resource in plan.resources}
            if not entry.task.inputs or not declared.inputs:
                raise ValueError("entry task requires concrete inputs")
            for item in declared.inputs:
                if (
                    item.resource_id is None
                    or entry.task.inputs.get(item.name)
                    != resources[item.resource_id].storage_path
                ):
                    raise ValueError(
                        "entry task inputs must resolve declared resources"
                    )
            if any(item.storage_path is None for item in plan.resources):
                raise ValueError("resolve plan resource locations before creation")
        updated = PreparationView.model_validate(
            {
                **current.model_dump(mode="json"),
                "facts": facts,
                "missing": missing,
                "plan": plan,
                "entry_task": entry,
                "phase": "collecting" if missing else "ready" if plan else "planning",
                "review_pending": False,
            }
        )
        try:
            return self.store.save(owner, updated, request.expected_revision)
        except PlanConflictError as exc:
            latest = self.store.get(owner)
            raise ValueError(
                json.dumps(
                    {
                        "error": "preparation_revision_conflict",
                        "reason": str(exc),
                        "expected_revision": request.expected_revision,
                        "current_revision": latest.revision,
                        "current_preparation": latest.model_dump(mode="json"),
                        "action": (
                            "Merge the requested fact changes into "
                            "current_preparation, preserve unrelated current "
                            "facts, and retry with expected_revision equal to "
                            "current_revision."
                        ),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ) from exc

    def set_plan(
        self, owner: PlanOwner, request: SetPreparationPlan
    ) -> PreparationView:
        current = self.store.get(owner)
        task = next(
            (task for task in request.plan.tasks if task.id == request.entry_task_id),
            None,
        )
        if task is None:
            raise ValueError("entry_task_id must reference plan.tasks")
        resources = {resource.id: resource for resource in request.plan.resources}
        inputs = {}
        for item in task.inputs:
            resource = resources.get(item.resource_id or "")
            if resource is None or resource.storage_path is None:
                raise ValueError("the first task requires concrete resource inputs")
            inputs[item.name] = resource.storage_path
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
        return self.update(
            owner,
            UpdatePreparation(
                expected_revision=request.expected_revision,
                facts=current.facts,
                plan=request.plan,
                entry_task=entry,
            ),
            {
                reference
                for fact in current.facts.values()
                for reference in fact.evidence_refs
            },
        )
