"""Business selection is deterministic and independent of harness and telemetry."""

from pyromind_runtime.application.checkpoint_selection import select_checkpoint
from pyromind_runtime.domain.training_evaluation import (
    CheckpointArtifact,
    CheckpointSelectionPolicy,
    MetricConstraint,
    MetricObjective,
    StepEvaluation,
)


def evaluated(step, rate, *, saved=True, group="g", complete=True):
    return StepEvaluation(
        step=step,
        run_id=f"run-{step}",
        output_dir=f"/eval/{step}",
        eligible=True,
        comparison_group=group,
        metrics={"pass_rate": rate, "recall": 1 - rate / 2},
        checkpoint=CheckpointArtifact(
            step=step, path=f"/checkpoint-{step}", kind="full", complete=complete
        )
        if saved
        else None,
    )


def test_unsaved_best_eval_does_not_select_nearest_checkpoint():
    result = select_checkpoint(
        [evaluated(300, 0.99, saved=False), evaluated(400, 0.7), evaluated(200, 0.8)],
        CheckpointSelectionPolicy(),
    )
    assert result.selected_step == 200
    assert result.best_evaluated_step == 300
    assert result.ranking == [200, 400]
    assert result.meets_target is None


def test_test_dataset_never_selects_checkpoint_even_with_better_metrics():
    validation = evaluated(100, 0.7)
    test = evaluated(200, 1.0, group="test").model_copy(update={"dataset_role": "test"})
    result = select_checkpoint([validation, test], CheckpointSelectionPolicy())
    assert result.selected_step == result.best_evaluated_step == 100
    assert result.ranking == [100]
    assert result.excluded["200"] == ["only_validation_can_select_checkpoint"]


def test_constraints_precede_primary_and_unmet_targets_are_explicit():
    rows = [evaluated(100, 0.9), evaluated(200, 0.7)]
    result = select_checkpoint(
        rows,
        CheckpointSelectionPolicy(
            constraints=[MetricConstraint(metric="recall", operator="ge", value=0.6)]
        ),
    )
    assert result.selected_step == 200 and result.meets_target
    result = select_checkpoint(
        rows,
        CheckpointSelectionPolicy(
            constraints=[MetricConstraint(metric="recall", operator="ge", value=0.99)]
        ),
    )
    assert result.selected_step == 100 and result.meets_target is False
    assert result.unmet_constraints


def test_ties_use_only_declared_secondary_then_earlier_step():
    rows = [evaluated(200, 0.8), evaluated(100, 0.8)]
    assert select_checkpoint(rows, CheckpointSelectionPolicy()).selected_step == 100
    rows[0] = rows[0].model_copy(update={"metrics": {"pass_rate": 0.8, "recall": 0.95}})
    result = select_checkpoint(
        rows, CheckpointSelectionPolicy(secondary=[MetricObjective(metric="recall")])
    )
    assert result.selected_step == 200


def test_unverified_or_incomparable_groups_and_incomplete_models_do_not_rank():
    rows = [evaluated(100, 0.9, group="a"), evaluated(200, 0.8, group="b")]
    assert select_checkpoint(rows, CheckpointSelectionPolicy()).selected_step is None
    assert (
        select_checkpoint(
            rows, CheckpointSelectionPolicy(comparison_group="a")
        ).selected_step
        == 100
    )
    rows = [
        evaluated(100, 0.9, complete=False),
        evaluated(200, 0.8).model_copy(
            update={"eligible": False, "issues": ["duplicate sample ID"]}
        ),
    ]
    result = select_checkpoint(rows, CheckpointSelectionPolicy())
    assert result.selected_step is None and result.model is None
    assert "incomplete_or_unverified_evaluation" in result.excluded["200"]
