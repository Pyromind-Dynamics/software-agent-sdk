"""Prepare frozen evaluation inputs in a terminal's execution environment."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID
from zipfile import ZIP_DEFLATED, ZipFile

import httpx

from pyromind_common.platform.evaluation_core import (
    _PREFLIGHT,
    INFERENCE_RUNTIME_FILENAMES,
    DatasetMapping,
    EvaluationOptions,
    EvaluationProductsOptions,
    InferenceOptions,
    PredictionEvaluationOptions,
    ValidationSelection,
    _validate_report_ui_bundle,
    build_inference_dsl,
    build_prediction_evaluation_dsl,
    build_products_dsl,
    model_reference,
    prepare_definition,
    prepare_products,
    read_evaluation_config,
    storage_path,
)
from pyromind_common.platform.preparation import LocalFiles, StagingDirectory
from pyromind_common.platform.scene_core import validate_pipeline_topology
from pyromind_common.platform.workflows import prepare_workflow


def _scoring_preflight(definition: dict[str, Any], runtime_dir: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="evaluation-preflight-") as directory:
        archive = Path(directory) / "preflight.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as files:
            for name in ("evaluate_predictions.py", "evaluate_inference.py"):
                files.write(runtime_dir / name, name)
            files.writestr("__main__.py", _PREFLIGHT)
            files.writestr("score_eval.py", definition["scoring_script"])
            files.writestr("spec.json", json.dumps(definition["scoring_spec"]))
        result = subprocess.run(
            [sys.executable, str(archive)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError("Scoring validation examples failed: " + result.stderr)


@dataclass(frozen=True)
class StagedEvaluation:
    name: str
    dsl: str
    workflow_path: Path
    frozen_files: tuple[Path, ...]
    requires_inference: bool


def stage_evaluation(
    options: EvaluationOptions,
    *,
    files: LocalFiles,
    staging: StagingDirectory,
    runtime_dir: Path,
    run_id: UUID,
    frozen_reward: dict[str, Any] | None = None,
    selection: ValidationSelection | None = None,
) -> StagedEvaluation:
    """Freeze evaluation inputs and write DSL without HTTP validation or submission.

    Run this in the terminal execution environment: scoring validation executes
    the supplied examples in a subprocess of that environment. Project callers
    must supply the already-authorized frozen Reward; this function does not
    create Project runs or authorize held-out checkpoint selection. A held-out
    run uses the caller's already-authorized ValidationSelection evidence.
    Omitted resources reuse the initial frozen definition. To inherit a later
    allocation, the caller supplies its Project's latest CPU and memory values.
    """
    request = options.request()
    prior = None
    if options.operation == "resume":
        if options.resume_run_id != run_id:
            raise ValueError("resume_run_id must match the original evaluation")
        prior = json.loads((staging.local / "evaluation-definition.json").read_text())
        if prior.get("run_id") != str(run_id):
            raise ValueError("Evaluation staging directory belongs to another run")
        if prior["input_path"] != storage_path(request.input_path):
            raise ValueError("Evaluation input changed; use a new full run")
        if prior.get("frozen_reward") != frozen_reward:
            raise ValueError("Evaluation Reward changed; use a new full run")
        if prior.get("test_source"):
            frozen_selection = ValidationSelection.model_validate_json(
                (staging.local / "validation-selection.json").read_text()
            )
            if selection is not None and selection != frozen_selection:
                raise ValueError("Validation selection changed; use a new full run")
            selection = frozen_selection
    kind = prior.get("kind") if prior else None
    prediction = isinstance(request.options, PredictionEvaluationOptions) or (
        kind == "prediction_evaluation"
    )
    products = isinstance(request.options, EvaluationProductsOptions) or kind in {
        "badcase_review",
        "evaluation_products",
    }
    final_test = bool(
        isinstance(request.options, InferenceOptions)
        and request.options.selection_run_id
        or prior
        and prior.get("test_source")
    )
    if final_test and selection is None:
        raise ValueError("Held-out evaluation requires its frozen ValidationSelection")
    if selection is not None and not final_test:
        raise ValueError("ValidationSelection only applies to held-out inference")
    assets = {name: runtime_dir / name for name in INFERENCE_RUNTIME_FILENAMES}
    if prediction or products or final_test or frozen_reward is not None:
        assets["evaluate_predictions.py"] = runtime_dir / "evaluate_predictions.py"
        assets["training_artifacts.py"] = (
            runtime_dir.parent.parent
            / "training-analysis/scripts/training_artifacts.py"
        )
    if products:
        assets["evaluation_products.py"] = runtime_dir / "evaluation_products.py"
    if frozen_reward is not None:
        assets["frozen_reward.py"] = runtime_dir / "frozen_reward.py"
    _validate_report_ui_bundle(runtime_dir)
    runtime_hash = hashlib.sha256(
        "".join(
            name + hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(assets.items())
        ).encode()
    ).hexdigest()
    if prior is not None:
        definition = prior
        if definition["runtime_sha256"] != runtime_hash:
            raise ValueError("Evaluation runtime changed; use a new full run")
    elif isinstance(request.options, EvaluationProductsOptions):
        definition = prepare_products(
            request.options,
            files,
            runtime_hash,
            input_path=request.input_path,
            cpu=request.cpu,
            memory=request.memory,
        )
    elif isinstance(request.options, PredictionEvaluationOptions):
        definition = prepare_definition(
            request.options,
            files,
            runtime_hash,
            input_path=request.input_path,
            cpu=request.cpu,
            memory=request.memory,
            evaluation_reward=frozen_reward,
        )
        if frozen_reward is None:
            _scoring_preflight(definition, runtime_dir)
    elif isinstance(request.options, InferenceOptions):
        inference = request.options
        dataset = DatasetMapping.model_validate_json(
            files.read_text(inference.dataset_config_path)
        ).model_dump(exclude_none=True)
        if frozen_reward is None and not dataset.get("reference_field"):
            raise ValueError("dataset config requires reference_field")
        evaluation = read_evaluation_config(
            files,
            inference.evaluation_config_path,
            frozen_reward=frozen_reward is not None,
        )
        inference_config = inference.model_dump(
            exclude={
                "dataset_config_path",
                "evaluation_config_path",
                "report_spec_path",
            },
            exclude_none=True,
        )
        inference_config["model_path"] = model_reference(
            inference.model_path, evaluation.model_source
        )
        report_spec = (
            files.read_text(inference.report_spec_path)
            if inference.report_spec_path
            else None
        )
        if report_spec:
            json.loads(report_spec)
        definition = {
            "inference": inference_config,
            "dataset_config": dataset,
            "evaluation_config": evaluation.model_dump(exclude_none=True),
            "report_spec": report_spec,
            "input_path": storage_path(request.input_path),
            "cpu": request.cpu,
            "memory": request.memory,
            "runtime_sha256": runtime_hash,
            **({"frozen_reward": frozen_reward} if frozen_reward is not None else {}),
        }
    else:
        raise ValueError("Evaluation options are required for a new run")
    if selection is not None:
        definition = selection.apply(definition)
    definition["run_id"] = str(run_id)
    execution_definition = dict(definition)
    if "cpu" in request.resource_overrides:
        execution_definition["cpu"] = request.cpu
    if "memory" in request.resource_overrides:
        execution_definition["memory"] = request.memory
    output = staging.storage_path()
    if products:
        dsl = build_products_dsl(execution_definition, output, str(run_id))
    elif prediction:
        dsl = build_prediction_evaluation_dsl(execution_definition, output, str(run_id))
    else:
        dsl = build_inference_dsl(execution_definition, output, str(run_id), files)
    frozen_files = []
    artifacts = {
        "evaluation-definition.json": definition,
        "dataset_config.json": definition["dataset_config"],
        "evaluation_config.json": definition["evaluation_config"],
    }
    if frozen_reward is not None:
        artifacts["frozen_reward.json"] = {
            **frozen_reward,
            **({"test_source": definition["test_source"]} if final_test else {}),
        }
    if selection is not None:
        artifacts["validation-selection.json"] = selection.model_dump()
    if definition.get("business_scoring"):
        artifacts["business_scoring.json"] = {
            **definition["business_scoring"],
            "test_source": definition["test_source"],
        }
    if prediction:
        artifacts["prediction_evaluation.json"] = {
            "source_manifest": definition["source_manifest"],
            **(
                {"frozen_reward": frozen_reward}
                if frozen_reward is not None
                else {"scoring_spec": definition["scoring_spec"]}
            ),
        }
        if frozen_reward is None:
            frozen_files.append(
                staging.write("score_eval.py", definition["scoring_script"].encode())
            )
    if products:
        artifacts["products_definition.json"] = definition
        if definition.get("extraction_script"):
            frozen_files.append(
                staging.write(
                    "extract_dataset.py", definition["extraction_script"].encode()
                )
            )
    if definition.get("report_spec"):
        frozen_files.append(
            staging.write("report_spec.json", definition["report_spec"].encode())
        )
    for name, document in artifacts.items():
        frozen_files.append(
            staging.write(
                name, json.dumps(document, ensure_ascii=False, sort_keys=True).encode()
            )
        )
    for name, source in assets.items():
        frozen_files.append(staging.freeze(source, name))
    frozen_files.append(
        staging.write(
            f"workflows/{hashlib.sha256(dsl.encode()).hexdigest()}.py", dsl.encode()
        )
    )
    workflow_path = staging.local / "workflow.py"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=staging.local, delete=False
    ) as generated:
        generated.write(dsl)
        temporary = Path(generated.name)
    try:
        temporary.replace(workflow_path)
    finally:
        temporary.unlink(missing_ok=True)
    return StagedEvaluation(
        name=f"agent-evaluation-{str(run_id)[:8]}",
        dsl=dsl,
        workflow_path=workflow_path,
        frozen_files=tuple(frozen_files),
        requires_inference=not (prediction or products),
    )


def prepare_evaluation(
    options: EvaluationOptions,
    *,
    files: LocalFiles,
    staging: StagingDirectory,
    runtime_dir: Path,
    run_id: UUID,
    validator: httpx.Client,
    frozen_reward: dict[str, Any] | None = None,
    selection: ValidationSelection | None = None,
) -> dict[str, Any]:
    """Stage inputs, then validate with the caller's separately authenticated client.

    SDK-only callers use stage_evaluation, the retained validation Tool, then
    submit its actual validated graph with the SDK.
    """
    prepared = stage_evaluation(
        options,
        files=files,
        staging=staging,
        runtime_dir=runtime_dir,
        run_id=run_id,
        frozen_reward=frozen_reward,
        selection=selection,
    )
    workflow = prepare_workflow(validator, dsl=prepared.dsl, name=prepared.name)
    validate_pipeline_topology(workflow, require_inference=prepared.requires_inference)
    return workflow
