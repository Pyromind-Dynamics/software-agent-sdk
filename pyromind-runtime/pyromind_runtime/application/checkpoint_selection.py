"""Select only saved, comparable checkpoints using an explicit business policy."""

import math
from typing import Any

from pyromind_runtime.domain.training_evaluation import (
    CheckpointSelection,
    CheckpointSelectionPolicy,
    StepEvaluation,
)


def metric_value(metrics: dict[str, Any], path: str) -> float | None:
    value: Any = metrics
    # Exact names may contain dots (for example rubric.verdict).
    if path in metrics:
        value = metrics[path]
    else:
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def select_checkpoint(
    evaluations: list[StepEvaluation], policy: CheckpointSelectionPolicy
) -> CheckpointSelection:
    excluded: dict[str, list[str]] = {}
    candidates: list[StepEvaluation] = []
    objectives = [policy.primary, *policy.secondary]
    groups = {
        row.comparison_group
        for row in evaluations
        if row.eligible
        and row.comparison_group
        and row.role != "baseline"
        and row.dataset_role == "validation"
    }
    group = policy.comparison_group
    if group is None and len(groups) == 1:
        group = next(iter(groups))

    def rank(row: StepEvaluation) -> tuple[float, ...]:
        values = [metric_value(row.metrics, item.metric) for item in objectives]
        return tuple(
            (-value if item.direction == "max" else value)
            for item, value in zip(objectives, values, strict=True)
            if value is not None
        ) + (float(row.step),)

    comparable: list[StepEvaluation] = []
    for row in evaluations:
        if row.dataset_role != "validation":
            excluded[str(row.step)] = ["only_validation_can_select_checkpoint"]
            continue
        if row.role == "baseline":
            excluded[str(row.step)] = ["baseline_reference_only"]
            continue
        issues = list(row.issues)
        if not row.eligible:
            issues.append("incomplete_or_unverified_evaluation")
        if group is None or row.comparison_group != group:
            issues.append("comparison_group_not_selected_or_mismatched")
        for objective in objectives:
            if metric_value(row.metrics, objective.metric) is None:
                issues.append(f"missing_metric:{objective.metric}")
        if not issues:
            comparable.append(row)
        if not row.checkpoint or not row.checkpoint.complete:
            issues.append("no_complete_checkpoint_at_same_step")
        elif row.checkpoint.step != row.step:
            issues.append("checkpoint_step_mismatch")
        if issues:
            excluded[str(row.step)] = sorted(set(issues))
        else:
            candidates.append(row)

    def unmet(row: StepEvaluation) -> list[str]:
        failed = []
        for constraint in policy.constraints:
            value = metric_value(row.metrics, constraint.metric)
            if value is None or (
                value < constraint.value
                if constraint.operator == "ge"
                else value > constraint.value
            ):
                failed.append(
                    f"{constraint.metric} {constraint.operator} {constraint.value}"
                )
        return failed

    candidates.sort(key=lambda row: (bool(unmet(row)), rank(row)))
    comparable.sort(key=rank)
    best = candidates[0] if candidates else None
    return CheckpointSelection(
        selected_step=best.step if best else None,
        selected_run_id=best.run_id if best else None,
        model=best.checkpoint if best else None,
        meets_target=(not unmet(best)) if best and policy.constraints else None,
        unmet_constraints=unmet(best) if best else [],
        ranking=[row.step for row in candidates],
        excluded=excluded,
        best_evaluated_step=comparable[0].step if comparable else None,
        reason=(
            "Best saved checkpoint under the frozen business policy; "
            "telemetry is diagnostic evidence, not an alternative ranking."
            if best
            else "No complete checkpoint with a comparable, valid evaluation."
        ),
    )
