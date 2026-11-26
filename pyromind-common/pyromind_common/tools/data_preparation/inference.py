from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from pyromind_common.platform.evaluation_core import (
    INFERENCE_RUNTIME_FILENAMES as INFERENCE_RUNTIME_FILENAMES,
    CheckpointModelSource as CheckpointModelSource,
    CloneModelSource as CloneModelSource,
    DatasetMapping as DatasetMapping,
    DownloadModelSource as DownloadModelSource,
    EvaluationConfig as EvaluationConfig,
    EvaluationConfigurationError as EvaluationConfigurationError,
    GenerationOptions as GenerationOptions,
    InferenceOptions as InferenceOptions,
    ModelSource as ModelSource,
    Rubric as Rubric,
    RubricEvaluator as RubricEvaluator,
    StorageModelSource as StorageModelSource,
    _validate_report_ui_bundle as _validate_report_ui_bundle,
    build_inference_dsl as build_inference_dsl,
    lora_merge_model_type as lora_merge_model_type,
    model_reference as model_reference,
    read_evaluation_config as read_evaluation_config,
    storage_path as storage_path,
)
from pyromind_common.sdk.logger import get_logger
from pyromind_common.tools.data_preparation.runner import runtime_bundle_fingerprint
from pyromind_common.tools.utils.workspace_staging import publish_workspace_path
from pyromind_common.tools.workflow.constants import WORKFLOW_RELATIVE_PATH
from pyromind_common.tools.workflow.dsl_to_xyflow import convert_dsl_to_xyflow
from pyromind_common.tools.workflow.task_submission import (
    WorkflowResourceValidationError,
    submit_workflow_task,
)
from pyromind_common.tools.workflow.validate_workflow_dsl import (
    ValidateWorkflowDslAction,
    ValidateWorkflowDslExecutor,
    ValidateWorkflowDslObservation,
)


if TYPE_CHECKING:
    from pyromind_common.sdk.conversation.base import ToolConversation
    from pyromind_common.tools.data_preparation.evaluation_submit import (
        EvaluationRequest,
    )


from pyromind_common.tools.data_preparation.evaluation_products import (
    EvaluationProductsOptions,
)
from pyromind_common.tools.data_preparation.prediction_evaluation import (
    PredictionEvaluationOptions,
)
from pyromind_common.tools.data_preparation.submission import (
    SubmissionConfig,
    SubmissionFiles,
    register_submitted_task,
    stage_submission_file,
    submission_client,
    validate_pipeline_topology as validate_pipeline_topology,
)
from pyromind_common.tools.utils.storage_paths import (
    to_agent_storage_path,
)


logger = get_logger(__name__)


class PipelineRunBackend(Protocol):
    def resolve(self, run_id: str) -> dict[str, Any] | None: ...

    def reserve(
        self, definition: dict[str, Any], resume_run_id: str | None
    ) -> dict[str, Any]: ...

    def finish_submission(
        self,
        run_id: str,
        revision: int,
        task_id: str | None,
        status: Literal["submitted", "failed", "uncertain"],
    ) -> None: ...

    def prepare_execution(
        self, run_id: str, workflow: dict[str, Any], dsl: str, script: str
    ) -> tuple[dict[str, Any], str]: ...


class _WorkflowValidationError(ValueError):
    def __init__(self, validation: ValidateWorkflowDslObservation) -> None:
        super().__init__(f"platform rejected evaluation workflow: {validation.text}")
        self.validation = validation


class InferencePipelineHandler:
    def __init__(
        self,
        runtime_dir: Path,
        runs: PipelineRunBackend,
        validator: ValidateWorkflowDslExecutor,
        submission_guard: Callable[[], None] | None = None,
        *,
        evaluation_reward: dict | None = None,
    ) -> None:
        self.runtime_dir = runtime_dir
        self.runs = runs
        self.validator = validator
        self._submission_guard = submission_guard
        self._evaluation_reward = evaluation_reward

    def _validate(
        self,
        dsl: str,
        conversation: ToolConversation,
        *,
        env: str | None,
        cluster: str | None,
    ) -> ValidateWorkflowDslObservation:
        validation = self.validator(ValidateWorkflowDslAction(dsl=dsl), conversation)
        logger.info(
            "evaluation.dsl_validation env=%s cluster=%s dsl_sha256=%s "
            "valid=%s errors=%d is_error=%s failure_stage=%s",
            env,
            cluster,
            hashlib.sha256(dsl.encode()).hexdigest(),
            validation.valid,
            len(validation.errors),
            validation.is_error,
            validation.failure_stage,
        )
        if validation.is_error or validation.valid is not True or validation.errors:
            raise _WorkflowValidationError(validation)
        return validation

    @staticmethod
    def _publish_canvas(conversation: ToolConversation, dsl: str) -> None:
        with tempfile.TemporaryDirectory(prefix="inference-canvas-") as directory:
            canvas = Path(directory) / "workflow.py"
            canvas.write_text(dsl)
            publish_workspace_path(
                cast(Any, conversation).workspace, canvas, WORKFLOW_RELATIVE_PATH
            )

    def submit(
        self,
        config: SubmissionConfig,
        request: EvaluationRequest,
        conversation: ToolConversation,
        files: SubmissionFiles,
    ) -> dict[str, Any]:
        inference_options = (
            request.options if isinstance(request.options, InferenceOptions) else None
        )
        evaluation_options = (
            request.options
            if isinstance(request.options, PredictionEvaluationOptions)
            else None
        )
        products_options = (
            request.options
            if isinstance(request.options, EvaluationProductsOptions)
            else None
        )
        prior = (
            self.runs.resolve(str(request.resume_run_id))
            if request.resume_run_id
            else None
        )
        saved_predictions = evaluation_options is not None or bool(
            prior and prior["definition"].get("kind") == "prediction_evaluation"
        )
        products = products_options is not None or bool(
            prior
            and prior["definition"].get("kind")
            in {"badcase_review", "evaluation_products"}
        )
        cpu_only = saved_predictions or products
        reward = None if products else self._evaluation_reward
        if (
            prior
            and not products
            and prior["definition"].get("frozen_reward") != reward
        ):
            raise ValueError("evaluation Reward changed; use a new full run")
        final_test = bool(
            inference_options
            and inference_options.selection_run_id
            or prior
            and prior["definition"].get("business_scoring")
        )
        if cpu_only and inference_options is not None:
            raise ValueError("cannot change evaluation kind on resume")
        assets = {name: self.runtime_dir / name for name in INFERENCE_RUNTIME_FILENAMES}
        if cpu_only or final_test or reward is not None:
            assets["evaluate_predictions.py"] = (
                self.runtime_dir / "evaluate_predictions.py"
            )
            assets["training_artifacts.py"] = (
                self.runtime_dir.parent.parent
                / "training-analysis/scripts/training_artifacts.py"
            )
        if products:
            assets["evaluation_products.py"] = (
                self.runtime_dir / "evaluation_products.py"
            )
        if reward is not None:
            assets["frozen_reward.py"] = self.runtime_dir / "frozen_reward.py"
        _validate_report_ui_bundle(self.runtime_dir)
        runtime_hash = runtime_bundle_fingerprint(
            self.runtime_dir, INFERENCE_RUNTIME_FILENAMES
        )
        if cpu_only or final_test or reward is not None:
            runtime_hash = hashlib.sha256(
                "".join(
                    name + hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in sorted(assets.items())
                ).encode()
            ).hexdigest()
        if products_options:
            from pyromind_common.tools.data_preparation.evaluation_products import (
                prepare_products,
            )

            definition = prepare_products(
                products_options,
                files,
                runtime_hash,
                input_path=request.input_path,
                cpu=request.cpu,
                memory=request.memory,
            )
        elif evaluation_options:
            from pyromind_common.tools.data_preparation.prediction_evaluation import (
                preflight_scoring,
                prepare_definition,
            )

            definition = prepare_definition(
                evaluation_options,
                files,
                runtime_hash,
                input_path=request.input_path,
                cpu=request.cpu,
                memory=request.memory,
                evaluation_reward=reward,
            )
            if reward is None:
                preflight_scoring(
                    cast(Any, conversation).workspace, self.runtime_dir, definition
                )
        elif inference_options:
            options = inference_options
            dataset = DatasetMapping.model_validate_json(
                files.read_text(options.dataset_config_path)
            ).model_dump(exclude_none=True)
            if reward is None and not dataset.get("reference_field"):
                raise ValueError("dataset config requires reference_field")
            evaluation_options = read_evaluation_config(
                files, options.evaluation_config_path, frozen_reward=reward is not None
            )
            evaluation = evaluation_options.model_dump(exclude_none=True)
            if "model_source" not in evaluation_options.model_fields_set:
                evaluation.pop("model_source")
            inference = options.model_dump(
                exclude={
                    "dataset_config_path",
                    "evaluation_config_path",
                    "report_spec_path",
                }
            )
            if inference["source_checkpoint_version_id"] is None:
                inference.pop("source_checkpoint_version_id")
            inference["model_path"] = model_reference(
                options.model_path, evaluation_options.model_source
            )
            report_spec = None
            if options.report_spec_path:
                report_spec = files.read_text(options.report_spec_path)
                try:
                    json.loads(report_spec)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "invalid report_spec JSON in "
                        f"{options.report_spec_path}: {error}"
                    ) from error
            definition = {
                "inference": inference,
                "dataset_config": dataset,
                "evaluation_config": evaluation,
                "report_spec": report_spec,
                "input_path": storage_path(request.input_path, "input_path"),
                "cpu": request.cpu,
                "memory": request.memory,
                "runtime_sha256": runtime_hash,
                **({"frozen_reward": reward} if reward is not None else {}),
            }
        elif prior:
            definition = dict(prior["definition"])
            if definition["input_path"] != storage_path(
                request.input_path, "input_path"
            ):
                raise ValueError("input_path changed; use a new full run")
            for resource in ("cpu", "memory"):
                if resource in request.resource_overrides:
                    definition[resource] = getattr(request, resource)
            if definition["runtime_sha256"] != runtime_hash:
                raise ValueError("evaluation runtime changed; use a new full run")
        else:
            raise ValueError("inference options are required for a new evaluation")
        run = self.runs.reserve(
            definition, str(request.resume_run_id) if request.resume_run_id else None
        )
        definition = run["definition"]
        if definition.get("business_scoring"):
            assets["evaluate_predictions.py"] = (
                self.runtime_dir / "evaluate_predictions.py"
            )
            assets["training_artifacts.py"] = (
                self.runtime_dir.parent.parent
                / "training-analysis/scripts/training_artifacts.py"
            )
        run_id, output_dir = run["run_id"], run["output_dir"]
        agent_output_dir = to_agent_storage_path(output_dir)
        revision = run["revision"]
        submitting = False
        task_id = None
        response_status = "Pending"
        warning = ""
        validation: ValidateWorkflowDslObservation | None = None
        try:
            if products:
                from pyromind_common.tools.data_preparation.evaluation_products import (
                    build_products_dsl,
                )

                dsl = build_products_dsl(definition, output_dir, run_id)
            elif saved_predictions:
                from pyromind_common.tools.data_preparation import prediction_evaluation

                dsl = prediction_evaluation.build_prediction_evaluation_dsl(
                    definition, output_dir, run_id
                )
            else:
                dsl = build_inference_dsl(definition, output_dir, run_id, files)
            workflow = convert_dsl_to_xyflow(dsl, name=f"agent-evaluation-{run_id[:8]}")
            validate_pipeline_topology(workflow, require_inference=not cpu_only)
            self._publish_canvas(conversation, dsl)
            validation = self._validate(
                dsl, conversation, env=config.env, cluster=config.cluster
            )
            workflow, effective_dsl = self.runs.prepare_execution(
                run_id,
                workflow,
                dsl,
                (self.runtime_dir / "evaluate_inference.py").read_text(),
            )
            if effective_dsl != dsl:
                dsl = effective_dsl
                if convert_dsl_to_xyflow(dsl, name=workflow["name"]) != workflow:
                    raise ValueError("frozen evaluation DSL and workflow differ")
                validate_pipeline_topology(workflow, require_inference=not cpu_only)
                self._publish_canvas(conversation, dsl)
                validation = self._validate(
                    dsl, conversation, env=config.env, cluster=config.cluster
                )
            client = submission_client(config, conversation)
            with tempfile.TemporaryDirectory(prefix="inference-config-") as directory:
                for name, frozen_config in (
                    ("dataset_config.json", definition["dataset_config"]),
                    ("evaluation_config.json", definition["evaluation_config"]),
                    *(
                        (
                            (
                                "frozen_reward.json",
                                {
                                    **definition["frozen_reward"],
                                    **(
                                        {"test_source": definition["test_source"]}
                                        if definition.get("test_source")
                                        else {}
                                    ),
                                },
                            ),
                        )
                        if definition.get("frozen_reward")
                        else ()
                    ),
                    *(
                        (
                            (
                                "business_scoring.json",
                                {
                                    **definition["business_scoring"],
                                    "test_source": definition["test_source"],
                                },
                            ),
                        )
                        if definition.get("business_scoring")
                        else ()
                    ),
                ):
                    path = Path(directory) / name
                    path.write_text(json.dumps(frozen_config, ensure_ascii=False))
                    stage_submission_file(
                        config,
                        str(path),
                        output_dir,
                        conversation,
                        frozen_script_name=name,
                    )
                if saved_predictions:
                    scoring = (
                        {"frozen_reward": definition["frozen_reward"]}
                        if definition.get("frozen_reward")
                        else {"scoring_spec": definition["scoring_spec"]}
                    )
                    artifacts = {
                        "prediction_evaluation.json": json.dumps(
                            {
                                "source_manifest": definition["source_manifest"],
                                **scoring,
                            },
                            ensure_ascii=False,
                        )
                    }
                    if not definition.get("frozen_reward"):
                        artifacts["score_eval.py"] = definition["scoring_script"]
                    for name, content in artifacts.items():
                        path = Path(directory) / name
                        path.write_text(content)
                        stage_submission_file(
                            config,
                            str(path),
                            output_dir,
                            conversation,
                            frozen_script_name=name,
                        )
                if products:
                    contents = {
                        "products_definition.json": json.dumps(
                            definition, ensure_ascii=False
                        )
                    }
                    if definition.get("extraction_script"):
                        contents["extract_dataset.py"] = definition["extraction_script"]
                    for name, content in contents.items():
                        path = Path(directory) / name
                        path.write_text(content)
                        stage_submission_file(
                            config,
                            str(path),
                            output_dir,
                            conversation,
                            frozen_script_name=name,
                        )
                report_spec = definition.get("report_spec")
                if report_spec:
                    path = Path(directory) / "report_spec.json"
                    path.write_text(report_spec)
                    stage_submission_file(
                        config,
                        str(path),
                        output_dir,
                        conversation,
                        frozen_script_name="report_spec.json",
                    )
            for filename, asset_path in assets.items():
                stage_submission_file(
                    config,
                    str(asset_path),
                    output_dir,
                    conversation,
                    frozen_script_name=filename,
                )
            if self._submission_guard:
                self._submission_guard()
            submitting = True
            response = submit_workflow_task(
                client=client,
                workflow=workflow,
                name=str(workflow["name"]),
                conversation_id=str(conversation.id),
                task_kind="data_preparation",
                retry_create=False,
            )
            task_id = response.task_id
            response_status = response.status
            self.runs.finish_submission(run_id, revision, task_id, "submitted")
        except Exception as exc:
            if task_id is None:
                if isinstance(exc, WorkflowResourceValidationError):
                    submitting = False
                self.runs.finish_submission(
                    run_id, revision, None, "uncertain" if submitting else "failed"
                )
                return {
                    "text": (
                        f"Evaluation submission failed: {exc}. run_id={run_id}. "
                        + (
                            "Platform acceptance is uncertain; do not resubmit blindly."
                            if submitting
                            else "No platform task was submitted."
                        )
                    ),
                    "status": "Failed",
                    "is_error": True,
                    "run_id": run_id,
                    "output_dir": agent_output_dir,
                    "resumed": request.resume_run_id is not None,
                    "execution_revision": run["execution_revision"],
                    **(
                        {
                            "failure_stage": exc.validation.failure_stage,
                            "retryable": exc.validation.retryable,
                            "validation_errors": exc.validation.errors,
                            "validation_warnings": exc.validation.warnings,
                        }
                        if isinstance(exc, _WorkflowValidationError)
                        else {}
                    ),
                    **(
                        {"failure_stage": "resources"}
                        if isinstance(exc, WorkflowResourceValidationError)
                        else {}
                    ),
                }
            warning = (
                " Platform accepted the task, but saving its association failed; "
                "do not submit it again."
            )
        register_submitted_task(conversation, task_id, response_status)
        return {
            "text": (
                f"Evaluation submitted: {task_id}. Check "
                f"{agent_output_dir}/report.json "
                "after the callback; it contains the HTML artifact path." + warning
            ),
            "task_id": task_id,
            "run_id": run_id,
            "output_dir": agent_output_dir,
            "status": response_status,
            "resumed": request.resume_run_id is not None,
            "execution_revision": run["execution_revision"],
            "validation_warnings": validation.warnings if validation else [],
        }
