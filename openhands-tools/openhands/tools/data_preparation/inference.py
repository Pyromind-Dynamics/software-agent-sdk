from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from openhands.sdk.logger import get_logger
from openhands.tools.data_preparation.runner import runtime_bundle_fingerprint
from openhands.tools.utils.workspace_staging import publish_workspace_path
from openhands.tools.workflow.definition import WORKFLOW_RELATIVE_PATH
from openhands.tools.workflow.dsl_to_xyflow import convert_dsl_to_xyflow
from openhands.tools.workflow.task_submission import (
    submit_workflow_task,
)
from openhands.tools.workflow.validate_workflow_dsl import (
    ValidateWorkflowDslAction,
    ValidateWorkflowDslExecutor,
    ValidateWorkflowDslObservation,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.tools.data_preparation.evaluation_submit import EvaluationRequest


from openhands.tools.data_preparation.evaluation_products import (
    EvaluationProductsOptions,
)
from openhands.tools.data_preparation.prediction_evaluation import (
    PredictionEvaluationOptions,
)
from openhands.tools.data_preparation.submission import (
    SubmissionConfig,
    SubmissionFiles,
    register_submitted_task,
    stage_submission_file,
    submission_client,
    validate_pipeline_topology as validate_pipeline_topology,
)


logger = get_logger(__name__)


INFERENCE_RUNTIME_FILENAMES = (
    "evaluate_inference.py",
    "evaluation_report.css",
    "evaluation_report.js",
)


def _validate_report_ui_bundle(runtime_dir: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(runtime_dir / "build_report_ui.py"), "--check"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            "evaluation report UI bundle is stale or its build check failed; run "
            ".agents/skills/inference-evaluation/scripts/build_report_ui.py"
        )


class InferenceOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model_path: str = Field(
        min_length=1,
        description=(
            "Model reference interpreted by model_source in the evaluation config "
            "JSON: an absolute Storage path for storage (default), a central "
            "catalog model identifier for clone (exact enum value, including its "
            "namespace), or a repository name for download."
        ),
    )
    selection_run_id: str | None = Field(
        default=None,
        description="Validation selection batch authorizing this independent test run.",
    )
    dataset_config_path: str = Field(min_length=1)
    evaluation_config_path: str = Field(min_length=1)
    report_spec_path: str | None = Field(default=None, min_length=1)
    served_model_name: str = Field(default="default", min_length=1)
    gpu_product: str = "NVIDIA-L40S"
    gpu_count: int = Field(default=1, ge=1)
    port: int = Field(default=3000, ge=1, le=65535)
    max_model_len: int | None = Field(default=None, ge=1)


class DatasetMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_field: str = Field(min_length=1)
    user_prompt_field: str | None = Field(default=None, min_length=1)
    messages_field: str | None = Field(default=None, min_length=1)
    system_prompt_field: str | None = Field(default=None, min_length=1)
    id_field: str = Field(default="id", min_length=1)
    media_field: str | None = Field(default=None, min_length=1)
    media_base_dir: str | None = Field(default=None, min_length=1)
    image_order: list[str] | None = None

    @model_validator(mode="after")
    def has_input(self) -> DatasetMapping:
        if not (self.user_prompt_field or self.messages_field):
            raise ValueError("user_prompt_field or messages_field is required")
        return self


class RubricEvaluator(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    type: Literal[
        "exact_match",
        "json_valid",
        "required_fields",
        "field_equals",
        "non_empty",
        "number_range",
        "list_count_match",
        "bbox_iou",
    ]
    prediction_path: str = ""
    reference_path: str = ""
    prediction_paths: list[str] = Field(default_factory=list)
    case_sensitive: bool = False
    min: float | None = None
    max: float | None = None
    iou_threshold: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def valid_bounds(self) -> RubricEvaluator:
        if self.type == "required_fields" and not all(self.prediction_paths):
            raise ValueError("required_fields paths must be nonempty")
        if self.type == "required_fields" and not self.prediction_paths:
            raise ValueError("required_fields requires prediction_paths")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("number_range min exceeds max")
        return self


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(min_length=1)
    criterion: str = Field(min_length=1)
    weight: float = Field(gt=0)
    required: bool = False
    evaluator: RubricEvaluator


class GenerationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    workers: int = Field(default=4, ge=1, le=64)
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0, ge=0, le=2)
    timeout_seconds: int = Field(default=240, ge=1)
    max_retries: int = Field(default=3, ge=0, le=10)
    limit: int = Field(default=0, ge=0)


class StorageModelSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["storage"] = "storage"


class CloneModelSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["clone"]


class DownloadModelSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["download"]
    download_source: Literal["huggingface", "modelscope"] = "huggingface"


class CheckpointModelSource(BaseModel):
    """Explicit saved checkpoint; its reference must never be rebound to latest."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["checkpoint"]
    kind: Literal["full", "lora"]
    base_model: str | None = None

    @model_validator(mode="after")
    def lora_base(self):
        if self.kind == "lora":
            if not self.base_model:
                raise ValueError("LoRA requires its resolved base_model Storage path")
            storage_path(self.base_model, "model_source.base_model")
        return self


ModelSource = Annotated[
    StorageModelSource | CloneModelSource | DownloadModelSource | CheckpointModelSource,
    Field(discriminator="type"),
]


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mode: Literal["agent_rubric"]
    rubrics: list[Rubric] = Field(min_length=1, max_length=20)
    pass_threshold: float = Field(default=0.7, ge=0, le=1)
    rubric_pass_threshold: float = Field(default=0.7, ge=0, le=1)
    generation: GenerationOptions = Field(default_factory=GenerationOptions)
    model_source: ModelSource = Field(default_factory=StorageModelSource)

    @model_validator(mode="after")
    def unique_names(self) -> EvaluationConfig:
        names = [rubric.name.strip() for rubric in self.rubrics]
        if not all(names) or len(set(names)) != len(names):
            raise ValueError("rubric names must be nonempty and unique")
        return self


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


def storage_path(value: str, field: str = "Storage path") -> str:
    if value == "/workspace" or value.startswith("/workspace/"):
        value = value[len("/workspace") :] or "/"
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) == "/":
        raise ValueError(
            f"{field}: expected an absolute Storage path without traversal; "
            f"received {value!r}"
        )
    return str(path)


def model_reference(value: str, source: ModelSource) -> str:
    if source.type == "checkpoint":
        reference = storage_path(value, "inference.model_path")
        if not re.fullmatch(r"checkpoint-\d+", PurePosixPath(reference).name):
            raise ValueError("selected model must reference an exact checkpoint-n")
        return reference
    if source.type == "storage":
        return storage_path(value, "inference.model_path (model_source.type=storage)")
    path = PurePosixPath(value)
    if (
        value != value.strip()
        or not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or any(ord(character) < 32 for character in value)
        or str(path) in {".", "/"}
    ):
        raise ValueError(
            f"inference.model_path (model_source.type={source.type}): "
            f"expected a model identifier, received {value!r}"
        )
    return value


def build_inference_dsl(
    definition: dict[str, Any], output_dir: str, run_id: str
) -> str:
    options = definition["inference"]
    source = EvaluationConfig.model_validate(
        definition["evaluation_config"]
    ).model_source
    reference = model_reference(options["model_path"], source)
    output = "/workspace" + storage_path(output_dir)
    command = " ".join(
        [
            "python3",
            shlex.quote(output + "/evaluate_inference.py"),
            "--run-id",
            shlex.quote(run_id),
            '--endpoint "$param"',
            "--model",
            shlex.quote(options["served_model_name"]),
            "--model-reference",
            shlex.quote(options["model_path"]),
            "--dataset-path",
            shlex.quote("/workspace" + definition["input_path"]),
            "--dataset-config",
            shlex.quote(output + "/dataset_config.json"),
            "--evaluation-config",
            shlex.quote(output + "/evaluation_config.json"),
            *(
                ["--report-spec", shlex.quote(output + "/report_spec.json")]
                if definition.get("report_spec")
                else []
            ),
            "--output-dir",
            shlex.quote(output),
            "--report-assets-dir",
            shlex.quote(output),
            *(
                [
                    "--dataset-provenance",
                    shlex.quote(output + "/dataset_provenance.json"),
                ]
                if definition.get("dataset_provenance")
                else []
            ),
            *(
                ["--business-scoring", shlex.quote(output + "/business_scoring.json")]
                if definition.get("business_scoring")
                else []
            ),
        ]
    )
    node_config = {
        key: options[key]
        for key in ("port", "gpu_count", "gpu_product", "max_model_len")
        if options.get(key) is not None
    }
    preparation = ""
    if source.type == "checkpoint" and source.kind == "lora":
        base = "/workspace" + storage_path(source.base_model or "")
        merged = output + "/merged-model"
        preparation = (
            f"model = ModelMergeLoraNode(id=3, lora_path={'/workspace' + reference!r}, "
            f"model_path={base!r}, output_path={merged!r}, "
            f"gpu_count={options['gpu_count']!r}, "
            f"gpu_product={options['gpu_product']!r})\n\n"
        )
        model_input = "model.merged_model_path"
    elif source.type in {"storage", "checkpoint"}:
        model_input = repr("/workspace" + reference)
    elif source.type == "clone":
        preparation = (
            f"model = CloneAndCacheModel(id=3, model={reference!r}, "
            "target_path='/workspace/models/')\n\n"
        )
        model_input = "model.model_path"
    else:
        assert isinstance(source, DownloadModelSource)
        cache_dir = "/workspace/models/" + reference
        preparation = (
            f"model = DownloadAndCacheModel(id=3, modelname={reference!r}, "
            f"cache_dir={cache_dir!r}, "
            f"download_source={source.download_source!r})\n\n"
        )
        model_input = "model.model_path"
    args = ", ".join(f"{key}={value!r}" for key, value in node_config.items())
    return (
        preparation
        + f"inference = VLLMInference(id=1, model_path={model_input}, {args})\n\n"
        f"evaluation = CustomCommandCPUNode(id=2, command={command!r}, "
        f"cpu={definition['cpu']}, memory={definition['memory']}, "
        "param=inference.endpoint)\n"
    )


class EvaluationConfigurationError(ValueError):
    pass


def read_evaluation_config(files, path: str) -> EvaluationConfig:
    try:
        raw = json.loads(files.read_text(path))
    except json.JSONDecodeError as exc:
        raise EvaluationConfigurationError(
            f"{path}: JSON line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    allowed = {
        "exact_match": {"type", "prediction_path", "reference_path", "case_sensitive"},
        "json_valid": {"type", "prediction_path"},
        "required_fields": {"type", "prediction_paths"},
        "field_equals": {"type", "prediction_path", "reference_path", "case_sensitive"},
        "non_empty": {"type", "prediction_path"},
        "number_range": {"type", "prediction_path", "min", "max"},
        "list_count_match": {"type", "prediction_path", "reference_path"},
        "bbox_iou": {"type", "prediction_path", "reference_path", "iou_threshold"},
    }
    for index, rubric in enumerate(
        raw.get("rubrics", []) if isinstance(raw, dict) else []
    ):
        evaluator = rubric.get("evaluator", {}) if isinstance(rubric, dict) else {}
        kind = evaluator.get("type") if isinstance(evaluator, dict) else None
        if kind in allowed and set(evaluator) - allowed[kind]:
            extra = sorted(set(evaluator) - allowed[kind])
            raise EvaluationConfigurationError(
                f"{path}: rubrics.{index}.evaluator rubric={rubric.get('name')!r}: "
                f"unsupported fields {extra}; {kind} accepts: "
                f"{', '.join(sorted(allowed[kind]))}"
            )
    try:
        return EvaluationConfig.model_validate(raw)
    except ValidationError as exc:
        raise EvaluationConfigurationError(
            f"{path}: {exc.errors(include_url=False)}"
        ) from exc


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
    ) -> None:
        self.runtime_dir = runtime_dir
        self.runs = runs
        self.validator = validator

    def _validate(
        self,
        dsl: str,
        conversation: BaseConversation,
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
    def _publish_canvas(conversation: BaseConversation, dsl: str) -> None:
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
        conversation: BaseConversation,
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
        final_test = bool(
            inference_options
            and inference_options.selection_run_id
            or prior
            and prior["definition"].get("business_scoring")
        )
        if cpu_only and inference_options is not None:
            raise ValueError("cannot change evaluation kind on resume")
        assets = {name: self.runtime_dir / name for name in INFERENCE_RUNTIME_FILENAMES}
        if cpu_only or final_test:
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
        _validate_report_ui_bundle(self.runtime_dir)
        runtime_hash = runtime_bundle_fingerprint(
            self.runtime_dir, INFERENCE_RUNTIME_FILENAMES
        )
        if cpu_only or final_test:
            runtime_hash = hashlib.sha256(
                "".join(
                    name + hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in sorted(assets.items())
                ).encode()
            ).hexdigest()
        if products_options:
            from openhands.tools.data_preparation.evaluation_products import (
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
            from openhands.tools.data_preparation.prediction_evaluation import (
                prepare_definition,
            )

            definition = prepare_definition(
                evaluation_options,
                files,
                runtime_hash,
                input_path=request.input_path,
                cpu=request.cpu,
                memory=request.memory,
            )
        elif inference_options:
            options = inference_options
            dataset = DatasetMapping.model_validate_json(
                files.read_text(options.dataset_config_path)
            ).model_dump(exclude_none=True)
            evaluation_options = read_evaluation_config(
                files, options.evaluation_config_path
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
        revision = run["revision"]
        submitting = False
        task_id = None
        response_status = "Pending"
        warning = ""
        validation: ValidateWorkflowDslObservation | None = None
        try:
            if products:
                from openhands.tools.data_preparation.evaluation_products import (
                    build_products_dsl,
                )

                dsl = build_products_dsl(definition, output_dir, run_id)
            elif saved_predictions:
                from openhands.tools.data_preparation.prediction_evaluation import (
                    build_prediction_evaluation_dsl,
                )

                dsl = build_prediction_evaluation_dsl(definition, output_dir, run_id)
            else:
                dsl = build_inference_dsl(definition, output_dir, run_id)
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
                        (("dataset_provenance.json", definition["dataset_provenance"]),)
                        if definition.get("dataset_provenance")
                        else ()
                    ),
                    *(
                        (
                            (
                                "business_scoring.json",
                                {
                                    **definition["business_scoring"],
                                    **(
                                        {"test_source": definition["test_source"]}
                                        if definition.get("test_source")
                                        else {}
                                    ),
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
                    for name, content in (
                        (
                            "prediction_evaluation.json",
                            json.dumps(
                                {
                                    "source_manifest": definition["source_manifest"],
                                    "scoring_spec": definition["scoring_spec"],
                                },
                                ensure_ascii=False,
                            ),
                        ),
                        ("score_eval.py", definition["scoring_script"]),
                    ):
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
            submitting = True
            response = submit_workflow_task(
                client=client,
                workflow=workflow,
                name=str(workflow["name"]),
                conversation_id=str(conversation.id),
                retry_create=False,
            )
            task_id = response.task_id
            response_status = response.status
            self.runs.finish_submission(run_id, revision, task_id, "submitted")
        except Exception as exc:
            if task_id is None:
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
                    "output_dir": output_dir,
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
                }
            warning = (
                " Platform accepted the task, but saving its association failed; "
                "do not submit it again."
            )
        register_submitted_task(conversation, task_id, response_status)
        return {
            "text": (
                f"Evaluation submitted: {task_id}. Check {output_dir}/report.json "
                "after the callback; it contains the HTML artifact path." + warning
            ),
            "task_id": task_id,
            "run_id": run_id,
            "output_dir": output_dir,
            "status": response_status,
            "resumed": request.resume_run_id is not None,
            "execution_revision": run["execution_revision"],
            "validation_warnings": validation.warnings if validation else [],
        }
