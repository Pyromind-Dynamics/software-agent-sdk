"""Persist batch evaluations and checkpoint choices through product ports."""

from __future__ import annotations

import hashlib
import html
import json
from collections import Counter
from typing import Any, cast
from uuid import UUID, uuid5

from pyromind_runtime.application.checkpoint_selection import select_checkpoint
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.training_evaluation import (
    CheckpointSelectionPolicy,
    StepEvaluation,
)
from pyromind_runtime.ports.product_store import ProductStore
from pyromind_runtime.ports.storage import EvaluationStorage


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def write_json(storage: EvaluationStorage, path: str, value: Any) -> None:
    storage.write(path, json.dumps(value, ensure_ascii=False, allow_nan=False).encode())


class TrainingEvaluations:
    def __init__(self, store: ProductStore, storage: EvaluationStorage):
        self.store = store
        self.storage = storage

    def finalize(self, run_id: str) -> dict[str, Any]:
        runs = self.store.load_pipeline_runs()
        parent = runs[run_id]
        definition = parent.definition
        if definition.get("kind") != "prediction_evaluation":
            raise ValueError("not a prediction evaluation batch")
        manifest = cast(dict[str, Any], definition["source_manifest"])
        spec = cast(dict[str, Any], definition["scoring_spec"])
        assert isinstance(manifest, dict) and isinstance(spec, dict)
        batch = json.loads(self.storage.read(parent.output_dir + "/batch_result.json"))
        if (
            batch.get("run_id") != run_id
            or batch.get("source_manifest_sha256") != digest(manifest)
            or batch.get("scoring_spec_sha256") != digest(spec)
            or batch.get("script_sha256")
            != hashlib.sha256(str(definition["scoring_script"]).encode()).hexdigest()
        ):
            raise ValueError("batch provenance does not match the frozen run")
        steps = [StepEvaluation.model_validate(value) for value in batch["steps"]]
        expected_steps = [entry["step"] for entry in manifest["evaluations"]]
        if sorted(row.step for row in steps) != sorted(expected_steps):
            raise ValueError(
                "batch must account for every frozen evaluation step exactly once"
            )
        checkpoints = {item["step"]: item for item in manifest.get("checkpoints", [])}
        for row in steps:
            child_id = str(uuid5(UUID(run_id), f"step:{row.step}"))
            output_dir = parent.output_dir + f"/steps/{row.step}"
            if row.run_id != child_id or row.output_dir != output_dir:
                raise ValueError("step report identity does not match its batch")
            if row.checkpoint:
                source = checkpoints.get(row.step)
                if not source or row.checkpoint.path != source["path"]:
                    raise ValueError(
                        "checkpoint was not declared in the frozen source manifest"
                    )
            child = PipelineRun(
                run_id=UUID(child_id),
                conversation_id=parent.conversation_id,
                definition={
                    "kind": "step_evaluation",
                    "parent_run_id": run_id,
                    "step": row.step,
                },
                fingerprint=parent.fingerprint,
                output_dir=output_dir,
                submission_status="submitted",
            )
            if child_id not in runs:
                self.store.save_pipeline_run(child, expected_revision=None)
        policy = CheckpointSelectionPolicy.model_validate(spec.get("selection", {}))
        selection = select_checkpoint(steps, policy)
        telemetry = manifest.get("telemetry", {})
        if isinstance(telemetry, dict):
            selection = selection.model_copy(update={"telemetry": telemetry})
        result = selection.model_dump(mode="json")
        write_json(
            self.storage, parent.output_dir + "/checkpoint_selection.json", result
        )
        previous_context = self.store.load_snapshot().external_tasks
        context = {
            "schema_version": 1,
            "run_id": run_id,
            "business_rule": spec["business_rule"],
            "scoring_spec": spec,
            "script_sha256": batch["script_sha256"],
            "source": {
                key: manifest.get(key)
                for key in ("task_id", "training_node_id", "output_path")
            },
            "selection_policy": policy.model_dump(mode="json"),
            "selection": result,
            "datasets": manifest.get("datasets", {}),
            "training": manifest.get("training", {}),
            "dataset_distribution": batch.get("dataset_distribution", {}),
            "steps": [row.model_dump(mode="json") for row in steps],
            "telemetry": telemetry,
            "warnings": batch.get("warnings", []),
            "prior_actions": manifest.get("prior_actions", []),
            "budget": manifest.get("budget"),
            "previous_runs": [
                {
                    "run_id": str(run.run_id),
                    "selection": run.result,
                    "iteration_context_path": run.output_dir
                    + "/iteration_context.json",
                    "same_scoring_spec": digest(run.definition.get("scoring_spec"))
                    == digest(spec),
                    "same_scoring_script": run.definition.get("scoring_script")
                    == definition["scoring_script"],
                }
                for run in runs.values()
                if run.definition.get("kind") == "prediction_evaluation"
                and str(run.run_id) != run_id
            ],
            "task_history": [
                {"task_id": task.task_id, "status": task.status}
                for task in previous_context
            ],
            "human_review": {"status": "not_loaded", "coverage": None},
        }
        # Preserve live human feedback when duplicate completion callbacks arrive.
        commits = getattr(self.store, "load_evaluation_commit", None)
        reviews = {}
        review_sources = [(row.run_id, row.output_dir, row.run_id) for row in steps]
        step_ids = {row.run_id for row in steps}
        for run in runs.values():
            if run.definition.get("kind") != "badcase_review":
                continue
            review_manifest = run.definition.get("products_manifest", {})
            assert isinstance(review_manifest, dict)
            source = review_manifest.get("source", {})
            assert isinstance(source, dict)
            scoring_id = source.get("scoring_run_id")
            if isinstance(scoring_id, str) and scoring_id in step_ids:
                review_sources.append((str(run.run_id), run.output_dir, scoring_id))
        if commits:
            for review_id, review_output, scoring_id in review_sources:
                commit = commits(review_id)
                if commit:
                    reviews[review_id] = {
                        "scoring_run_id": scoring_id,
                        "revision": commit.revision,
                        "annotations_object": review_output
                        + "/"
                        + commit.annotations_object,
                        "summary": commit.history[-1].result.model_dump(mode="json")
                        if commit.history
                        else {},
                    }
                    records = [
                        json.loads(line)
                        for line in self.storage.read(
                            review_output + "/" + commit.annotations_object
                        ).splitlines()
                        if line.strip()
                    ]
                    reviewed = [
                        record
                        for record in records
                        if record.get("review_status") == "reviewed"
                    ]
                    reviews[review_id].update(
                        {
                            "scope": "machine_failed_cases_only",
                            "total_badcases": len(records),
                            "reviewed": len(reviewed),
                            "coverage": len(reviewed) / len(records)
                            if records
                            else 1.0,
                            "conclusions": dict(
                                Counter(
                                    record.get("human_annotation", {}).get(
                                        "resolution_code"
                                    )
                                    or record.get("human_annotation", {}).get(
                                        "decision", "unknown"
                                    )
                                    for record in reviewed
                                )
                            ),
                            "representative_case_ids": [
                                r["case_id"] for r in reviewed[:10]
                            ],
                        }
                    )
        context["human_review"] = {
            "status": "available" if reviews else "unreviewed",
            "by_run": {k: v for k, v in reviews.items() if k in step_ids},
            "by_task": {k: v for k, v in reviews.items() if k not in step_ids},
            "pending_runs": {
                row.run_id: {
                    "scope": "machine_failed_cases_only",
                    "reviewed": 0,
                    "total_badcases": row.metrics.get("failed"),
                    "coverage": 0.0 if row.metrics.get("failed") else None,
                }
                for row in steps
                if row.run_id not in {v["scoring_run_id"] for v in reviews.values()}
            },
        }
        datasets = manifest.get("datasets", {})
        context["final_test"] = {
            "status": "required_after_validation_selection"
            if datasets.get("test")
            else "not_applicable_reuse_validation_predictions",
            "selection_run_id": run_id,
            "selected_model": result.get("model"),
            "dataset": datasets.get("test"),
            "participates_in_selection": False,
            "runs": [
                {"run_id": str(run.run_id), "output_dir": run.output_dir}
                for run in runs.values()
                if run.definition.get("dataset_role") == "test"
                and cast(dict, run.definition.get("inference", {})).get(
                    "selection_run_id"
                )
                == run_id
            ],
        }
        write_json(self.storage, parent.output_dir + "/iteration_context.json", context)
        page = batch_report(parent.conversation_id, steps, context)
        self.storage.write(parent.output_dir + "/evaluation_report.html", page.encode())
        write_json(
            self.storage,
            parent.output_dir + "/report.json",
            {
                "status": "succeeded",
                "evaluation_status": "validated"
                if selection.model
                else "no_eligible_checkpoint",
                "selection": result,
                "steps": len(steps),
                "request_count": 0,
                "iteration_context_path": parent.output_dir + "/iteration_context.json",
                "artifacts": {
                    "evaluation_report.html": parent.output_dir
                    + "/evaluation_report.html"
                },
            },
        )
        if parent.result != result:
            self.store.save_pipeline_run(
                parent.model_copy(
                    update={"result": result, "revision": parent.revision + 1}
                ),
                expected_revision=parent.revision,
            )
        return result


def batch_report(
    conversation_id: str, steps: list[StepEvaluation], context: dict
) -> str:
    """Small static batch overview; individual cases use the shared report UI."""
    selection = context["selection"]

    def escape(value: Any) -> str:
        return html.escape(str(value), quote=True)

    def url(run_id: str) -> str:
        return (
            f"/api/v2/pyromind/conversations/{conversation_id}"
            f"/evaluation-runs/{run_id}/report"
        )

    rows = []
    for row in sorted(steps, key=lambda value: value.step):
        marker = " ← Selected" if row.step == selection["selected_step"] else ""
        cells = [
            f'<a href="{escape(url(row.run_id))}">{row.step}</a>{marker}',
            escape(row.metrics.get("total", row.metrics.get("total_samples"))),
            escape(row.metrics.get("pass_rate")),
            escape(row.metrics.get("macro_f1")),
            escape(row.checkpoint.path if row.checkpoint else "Not saved"),
            escape(row.comparison_group),
            escape("; ".join(selection["excluded"].get(str(row.step), []))),
        ]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    selected = selection["selected_run_id"]
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>Checkpoint evaluation</title><style>"
        "body{font:15px system-ui;margin:32px;color:#172b4d}"
        "table{border-collapse:collapse;width:100%}td,th{padding:10px;"
        "border:1px solid #dce3ed;text-align:left;overflow-wrap:anywhere}"
        "pre{white-space:pre-wrap}details{margin:20px 0}"
        "</style><h1>Checkpoint evaluation</h1><p>"
        + escape(selection["reason"])
        + "</p>"
        + (
            f'<p><a href="{escape(url(selected))}">'
            "Review selected checkpoint badcases</a></p>"
            if selected
            else "<p>No eligible model selected.</p>"
        )
        + "<p>Business rule: "
        + escape(context["business_rule"])
        + "</p>"
        + "<p>Meets acceptance target: "
        + escape(selection["meets_target"])
        + "; best evaluated step: "
        + escape(selection["best_evaluated_step"])
        + "</p>"
        + "<p>Unmet constraints: "
        + escape(selection["unmet_constraints"])
        + "</p>"
        + "<table><tr>"
        + "".join(
            f"<th>{name}</th>"
            for name in (
                "Step",
                "Samples",
                "Pass rate",
                "Macro F1",
                "Model",
                "Comparison group",
                "Excluded",
            )
        )
        + "</tr>"
        + "".join(rows)
        + "</table>"
    )
    for title, data in (
        ("Frozen selection policy", context["selection_policy"]),
        ("Dataset distributions", context["dataset_distribution"]),
        ("W&B diagnostics", context["telemetry"]),
        ("Human review coverage and conclusions", context["human_review"]),
        ("Evidence gaps", context["warnings"]),
    ):
        page += (
            "<details><summary>"
            + escape(title)
            + "</summary><pre>"
            + escape(json.dumps(data, ensure_ascii=False, indent=2))
            + "</pre></details>"
        )
    return page + "</html>"
