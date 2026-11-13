"""Reconcile the existing inference output files with the durable run record."""

import json
import re

from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.domain.training_evaluation import (
    CheckpointArtifact,
    StepEvaluation,
)
from pyromind_runtime.domain.versions import check_persistable, fingerprint
from pyromind_runtime.ports.product_store import ProductStore
from pyromind_runtime.ports.storage import (
    EvaluationStorage,
    EvaluationStorageMissingError,
)


def finalize_inference(
    store: ProductStore,
    storage: EvaluationStorage,
    run: PipelineRun,
    checkpoint: CheckpointArtifact,
    *,
    succeeded: bool,
) -> None:
    if run.result.get("inference_delivery"):
        return
    root = run.output_dir.rstrip("/")
    report_path = root + "/report.json"
    try:
        report = json.loads(storage.read(report_path))
    except EvaluationStorageMissingError:
        if not succeeded:
            return
        raise
    if not isinstance(report, dict):
        raise ValueError("inference report must be an object")
    if not succeeded:
        if report.get("status") != "failed":
            return
        delivery = {
            "report_path": report_path,
            "evaluation_status": report.get("evaluation_status"),
        }
        check_persistable(delivery)
        store.save_pipeline_run(
            run.model_copy(
                update={
                    "result": {**run.result, "inference_delivery": delivery},
                    "revision": run.revision + 1,
                }
            ),
            expected_revision=run.revision,
        )
        return

    manifest = json.loads(storage.read(root + "/run_manifest.json"))
    metrics = json.loads(storage.read(root + "/metrics.json"))
    if (
        not all(isinstance(value, dict) for value in (manifest, metrics))
        or report.get("status") not in {"succeeded", "failed"}
        or manifest.get("run_id") != str(run.run_id)
        or to_storage_api_path(manifest.get("model_reference") or "")
        != to_storage_api_path(checkpoint.path)
        or metrics != report.get("metrics")
        or not isinstance(manifest.get("dataset_sha256"), str)
        or not re.fullmatch(r"[a-f0-9]{64}", manifest["dataset_sha256"])
        or not isinstance(manifest.get("generation"), dict)
        or not isinstance(manifest.get("evaluation_config_sha256"), str)
    ):
        raise ValueError("inference report does not match the run manifest and metrics")
    references = report.get("artifacts", {})
    if not isinstance(references, dict) or any(
        references.get(name) != f"{root}/{name}"
        for name in ("metrics.json", "predictions.jsonl")
    ):
        raise ValueError("inference outputs do not belong to the run directory")
    dataset_role = run.definition.get("dataset_role", "validation")
    delivery = {
        "report_path": root + "/report.json",
        "metrics_path": root + "/metrics.json",
        "predictions_path": root + "/predictions.jsonl",
        "evaluation_status": report.get("evaluation_status"),
        "report_status": report.get("report_status"),
    }
    check_persistable(delivery)
    check_persistable(metrics)
    # JSON accepts nonfinite constants by default. The ledger uses strict JSON.
    json.dumps(metrics, allow_nan=False)
    step = StepEvaluation(
        step=checkpoint.step,
        run_id=f"{run.run_id}:{run.execution_revision}",
        output_dir=root,
        dataset_role=dataset_role,
        checkpoint=checkpoint,
        metrics=metrics,
        eligible=report.get("evaluation_status") == "validated",
        comparison_group=fingerprint(
            [
                "inference",
                dataset_role,
                manifest["dataset_sha256"],
                manifest.get("media_sha256"),
                manifest.get("dataset_config"),
                manifest["evaluation_config_sha256"],
                manifest["generation"],
                manifest.get("runtime_sha256"),
            ]
        ),
    )
    store.save_pipeline_run(
        run.model_copy(
            update={
                "result": {**run.result, "inference_delivery": delivery},
                "evaluations": (step,),
                "revision": run.revision + 1,
            }
        ),
        expected_revision=run.revision,
    )
