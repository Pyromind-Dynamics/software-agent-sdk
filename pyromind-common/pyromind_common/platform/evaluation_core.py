"""Evaluation contracts and pure DSL/definition builders, shared with old Tools."""

from __future__ import annotations

import ast
import json
import re
import shlex
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from pyromind_common.platform.paths import to_agent_storage_path, to_storage_api_path


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
            "JSON: a user Storage path such as '/workspace/models/model' for "
            "storage (default), a central "
            "catalog model identifier for clone (exact enum value, including its "
            "namespace), or a repository name for download."
        ),
    )
    source_checkpoint_version_id: str | None = Field(
        default=None,
        min_length=1,
        description="Version ID of the checkpoint used as model_path.",
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
    reference_field: str | None = Field(default=None, min_length=1)
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
    rubrics: list[Rubric] = Field(default_factory=list, max_length=20)
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


class FrozenDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def canonical_path(cls, value: str) -> str:
        return to_storage_api_path(value)


class ValidationCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    kind: Literal["full", "lora"]
    base_model: str | None = None

    @field_validator("path", "base_model")
    @classmethod
    def canonical_path(cls, value: str | None) -> str | None:
        return to_storage_api_path(value) if value is not None else None

    @model_validator(mode="after")
    def checkpoint_reference(self) -> ValidationCheckpoint:
        if not re.fullmatch(r"checkpoint-\d+", PurePosixPath(self.path).name):
            raise ValueError("Validation selection requires an exact checkpoint-n")
        if self.kind == "lora" and self.base_model is None:
            raise ValueError("Validation-selected LoRA requires its frozen base model")
        return self


class ValidationSelection(BaseModel):
    """Already-authorized Project evidence supplied by the execution caller."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    selection_run_id: str = Field(min_length=1)
    train: FrozenDataset
    validation: FrozenDataset
    test: FrozenDataset
    model: ValidationCheckpoint
    frozen_reward: dict[str, Any] | None = None
    scoring_script: str | None = None
    scoring_spec: dict[str, Any] | None = None

    @model_validator(mode="after")
    def independent_test(self) -> ValidationSelection:
        for source in (self.train, self.validation):
            if self.test.path == source.path or self.test.sha256 == source.sha256:
                raise ValueError("test must be independent of training and validation")
        if self.frozen_reward is None and (
            not self.scoring_script or not self.scoring_spec
        ):
            raise ValueError(
                "Validation selection requires its frozen scoring contract"
            )
        return self

    def apply(self, definition: dict[str, Any]) -> dict[str, Any]:
        inference = definition["inference"]
        config = EvaluationConfig.model_validate(definition["evaluation_config"])
        source = config.model_source
        if inference.get("selection_run_id") != self.selection_run_id:
            raise ValueError("final test requires the matching validation selection")
        if definition["input_path"] != self.test.path:
            raise ValueError("final test input must match the frozen test dataset")
        if (
            inference["model_path"] != self.model.path
            or not isinstance(source, CheckpointModelSource)
            or source.kind != self.model.kind
            or (
                self.model.kind == "lora"
                and to_storage_api_path(source.base_model or "")
                != self.model.base_model
            )
        ):
            raise ValueError("final test must use the validation-selected checkpoint")
        if config.generation.limit:
            raise ValueError("final test requires the complete test dataset")
        if definition.get("frozen_reward") != self.frozen_reward:
            raise ValueError("final test must use the validation run's frozen Reward")
        scoring = (
            {}
            if self.frozen_reward is not None
            else {
                "business_scoring": {
                    "script": self.scoring_script,
                    "spec": self.scoring_spec,
                    "selection_run_id": self.selection_run_id,
                }
            }
        )
        return {
            **definition,
            "dataset_role": "test",
            "test_source": self.test.model_dump(),
            **scoring,
        }


def storage_path(value: str, field: str = "Storage path") -> str:
    return to_storage_api_path(value, field=field)


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


class _InferenceFiles(Protocol):
    def read_text(self, path: str) -> str: ...


def build_inference_dsl(
    definition: dict[str, Any], output_dir: str, run_id: str, files: _InferenceFiles
) -> str:
    options = definition["inference"]
    source = EvaluationConfig.model_validate(
        definition["evaluation_config"]
    ).model_source
    reference = model_reference(options["model_path"], source)
    output = to_agent_storage_path(output_dir, field="output_dir")
    dataset_path = to_agent_storage_path(definition["input_path"], field="input_path")
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
            shlex.quote(dataset_path),
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
                ["--frozen-reward", shlex.quote(output + "/frozen_reward.json")]
                if definition.get("frozen_reward")
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
        base = to_agent_storage_path(
            source.base_model or "", field="model_source.base_model"
        )
        lora = to_agent_storage_path(reference, field="inference.model_path")
        merged = output + "/merged-model"
        model_type = lora_merge_model_type(files, base)
        preparation = (
            f"model = ModelMergeLoraNode(id=3, lora_path={lora!r}, "
            f"model_path={base!r}, output_path={merged!r}, "
            f"gpu_count=1, gpu_product={options['gpu_product']!r}, "
            f"model_type={model_type!r})\n\n"
        )
        model_input = "model.merged_model_path"
    elif source.type in {"storage", "checkpoint"}:
        model_input = repr(
            to_agent_storage_path(reference, field="inference.model_path")
        )
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


def lora_merge_model_type(files: _InferenceFiles, base_model: str) -> str:
    """Pick the platform loader from the base model's config, including aliases."""
    try:
        config = json.loads(files.read_text(f"{base_model}/config.json"))
    except (OSError, ValueError):
        return "auto"
    model_type = config.get("model_type") if isinstance(config, dict) else None
    if model_type == "qwen3_vl":
        return "qwen3vl"
    if model_type == "qwen3_5":
        return "qwen3.5"
    return "auto"


class EvaluationConfigurationError(ValueError):
    pass


def read_evaluation_config(
    files, path: str, *, frozen_reward: bool = False
) -> EvaluationConfig:
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
        config = EvaluationConfig.model_validate(raw)
        if frozen_reward and config.rubrics:
            raise EvaluationConfigurationError(
                "omit rubrics for the Project's frozen Reward"
            )
        if not frozen_reward and not config.rubrics:
            raise EvaluationConfigurationError(
                "agent_rubric requires a nonempty rubrics list"
            )
        return config
    except ValidationError as exc:
        raise EvaluationConfigurationError(
            f"{path}: {exc.errors(include_url=False)}"
        ) from exc


_PREFLIGHT = """\
import datetime
import json
import sys
import types
from zipfile import ZipFile

# The sandbox can run Python 3.10. The CPU worker runs a newer Python.
if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc

from evaluate_predictions import validate_script

with ZipFile(sys.argv[0]) as files:
    module = types.ModuleType("frozen_business_scoring")
    exec(compile(files.read("score_eval.py"), "score_eval.py", "exec"), module.__dict__)
    validate_script(module, json.loads(files.read("spec.json")))
"""


class PredictionEvaluationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_manifest_path: str = Field(min_length=1)
    scoring_spec_path: str | None = Field(default=None, min_length=1)
    scoring_script_path: str | None = Field(default=None, min_length=1)


def prepare_definition(
    options: PredictionEvaluationOptions,
    files: Any,
    runtime_hash: str,
    *,
    input_path: str,
    cpu: int,
    memory: int,
    evaluation_reward: dict | None = None,
) -> dict[str, Any]:

    manifest = json.loads(files.read_text(options.source_manifest_path))
    if not isinstance(manifest, dict):
        raise ValueError("source manifest must be a JSON object")
    scoring = {}
    if evaluation_reward is not None:
        if options.scoring_spec_path or options.scoring_script_path:
            raise ValueError("this Project uses its frozen Reward for scoring")
        scoring = {"frozen_reward": evaluation_reward}
    else:
        if not options.scoring_spec_path or not options.scoring_script_path:
            raise ValueError("score requires scoring_spec_path and scoring_script_path")
        spec = json.loads(files.read_text(options.scoring_spec_path))
        script = files.read_text(options.scoring_script_path)
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("business_rule"), str)
            or not spec["business_rule"].strip()
        ):
            raise ValueError("scoring_spec.business_rule must state the confirmed rule")
        if (
            not isinstance(spec.get("validation_examples"), list)
            or not spec["validation_examples"]
        ):
            raise ValueError("scoring_spec.validation_examples is required")
        functions = {
            node.name
            for node in ast.parse(script).body
            if isinstance(node, ast.FunctionDef)
        }
        if not {"parse_record", "score_case"} <= functions:
            raise ValueError("script must define parse_record and score_case")
        scoring = {"scoring_spec": spec, "scoring_script": script}
    entries = manifest.get("evaluations", [])
    if not entries or not isinstance(entries, list):
        raise ValueError("source manifest requires evaluations")
    steps = set()
    for entry in entries:
        step = entry.get("step")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or step in steps
        ):
            raise ValueError("evaluation steps must be unique nonnegative integers")
        steps.add(step)
        entry["path"] = storage_path(entry["path"])
        if (
            manifest.get("schema_version", 1) == 1
            and Path(entry["path"]).name != f"eval_step_{step}.jsonl"
        ):
            # Explicit baseline entries are independently identifiable sources.
            if entry.get("role") != "baseline":
                raise ValueError("evaluation filename must match its step")
        if evaluation_reward is not None and entry.get("dataset_role") == "test":
            raise ValueError(
                "held-out evaluation requires the validation-selected checkpoint"
            )
        if entry.get("dataset_role", "validation") not in {
            "validation",
            "test",
            "train",
        }:
            raise ValueError("unknown evaluation dataset role")
        if re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")) is None:
            raise ValueError("each source requires its SHA-256 fingerprint")
    for checkpoint in manifest.get("checkpoints", []):
        checkpoint["path"] = storage_path(checkpoint["path"])
        if Path(checkpoint["path"]).name != f"checkpoint-{checkpoint['step']}":
            raise ValueError("checkpoint directory must match its step")
    for entry in manifest.get("datasets", {}).values():
        entry["path"] = storage_path(entry["path"])
        if re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")) is None:
            raise ValueError("datasets require SHA-256 fingerprints")
    return {
        "kind": "prediction_evaluation",
        "source_manifest": manifest,
        **scoring,
        "input_path": storage_path(input_path),
        "dataset_config": {},
        "evaluation_config": {},
        "cpu": cpu,
        "memory": memory,
        "runtime_sha256": runtime_hash,
    }


def build_prediction_evaluation_dsl(
    definition: dict, output_dir: str, run_id: str
) -> str:

    output = to_agent_storage_path(output_dir, field="output_dir")
    command = " ".join(
        shlex.quote(value)
        for value in (
            "python3",
            output + "/evaluate_predictions.py",
            "--run-id",
            run_id,
            "--output-dir",
            output,
        )
    )
    return (
        f"evaluation = CustomCommandCPUNode(id=1, command={command!r}, "
        f"cpu={definition['cpu']}, memory={definition['memory']})\n"
    )


class EvaluationProductsOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["profile", "review", "report"]
    manifest_path: str = Field(min_length=1)
    extraction_script_path: str | None = None


def prepare_products(
    options: EvaluationProductsOptions,
    files,
    runtime_hash: str,
    *,
    input_path: str,
    cpu: int,
    memory: int,
) -> dict:

    manifest = json.loads(files.read_text(options.manifest_path))
    script = None
    if options.operation == "profile":
        if not options.extraction_script_path:
            raise ValueError("profile requires extraction_script_path")
        script = files.read_text(options.extraction_script_path)
        if "parse_record" not in {
            node.name
            for node in ast.parse(script).body
            if isinstance(node, ast.FunctionDef)
        }:
            raise ValueError("extraction script must define parse_record")
    elif options.extraction_script_path:
        raise ValueError("review/report do not execute extraction or scoring scripts")
    if options.operation in {"profile", "review"}:
        if not {"train", "validation"} <= manifest.get("datasets", {}).keys():
            raise ValueError("train and validation datasets are required")
    if options.operation in {"review", "report"}:
        source = manifest["source"]
        if source["kind"] not in {"scoring", "import"}:
            raise ValueError("source.kind must be scoring or import")
        if source["kind"] == "scoring":
            for name in ("scoring_run_id", "scoring_spec_sha256"):
                if not source.get(name):
                    raise ValueError(f"scoring source requires {name}")
            if "scoring_manifest" not in manifest:
                raise ValueError("scoring source requires frozen scoring_manifest")
        if options.operation == "review" and "profile" not in manifest:
            raise ValueError("review requires a frozen, completed profile")
    return {
        "kind": "badcase_review"
        if options.operation == "review"
        else "evaluation_products",
        "products_operation": options.operation,
        "products_manifest": manifest,
        "extraction_script": script,
        "input_path": storage_path(input_path),
        "dataset_config": {},
        "evaluation_config": {},
        "cpu": cpu,
        "memory": memory,
        "runtime_sha256": runtime_hash,
    }


def build_products_dsl(definition: dict, output_dir: str, run_id: str) -> str:

    output = to_agent_storage_path(output_dir, field="output_dir")
    command = " ".join(
        map(
            shlex.quote,
            (
                "python3",
                output + "/evaluation_products.py",
                "--run-id",
                run_id,
                "--output-dir",
                output,
            ),
        )
    )
    return (
        f"products = CustomCommandCPUNode(id=1, command={command!r}, "
        f"cpu={definition['cpu']}, memory={definition['memory']})\n"
    )


EvaluationOperation = Literal[
    "inference", "score", "profile", "review", "report", "resume"
]


_INFERENCE_FIELDS = frozenset(InferenceOptions.model_fields)


_SCORE_FIELDS = frozenset(PredictionEvaluationOptions.model_fields)


_PRODUCT_FIELDS = frozenset({"manifest_path", "extraction_script_path"})


_COMMON_FIELDS = frozenset({"operation", "input_path", "cpu", "memory"})


_REQUIRED_FIELDS = {
    "inference": ("model_path", "dataset_config_path", "evaluation_config_path"),
    "score": ("source_manifest_path",),
    "profile": ("manifest_path", "extraction_script_path"),
    "review": ("manifest_path",),
    "report": ("manifest_path",),
    "resume": ("resume_run_id",),
}


class EvaluationOptions(BaseModel):
    """Submit an evaluation operation with flat, operation-specific fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: EvaluationOperation = Field(
        description="Task: inference, score, profile, review, report, or resume."
    )
    input_path: str = Field(
        min_length=1,
        description=(
            "Source data path in user Storage, e.g. "
            "'/workspace/datasets/test.jsonl'; unchanged on resume. "
            "For inference, a directory uses its val.jsonl."
        ),
    )
    model_path: str | None = Field(
        default=None,
        min_length=1,
        description="inference: model reference; source is in evaluation config JSON.",
    )
    dataset_config_path: str | None = Field(
        default=None,
        min_length=1,
        description="inference: workspace JSON mapping input and reference fields.",
    )
    evaluation_config_path: str | None = Field(
        default=None,
        min_length=1,
        description="inference: workspace rubric, generation and model_source JSON.",
    )
    source_checkpoint_version_id: str | None = Field(
        default=None,
        min_length=1,
        description="Version ID of the checkpoint used as model_path.",
    )
    selection_run_id: str | None = Field(
        default=None,
        min_length=1,
        description="inference: validation selection run authorizing independent test.",
    )
    report_spec_path: str | None = Field(
        default=None,
        min_length=1,
        description="inference: optional workspace report specification JSON.",
    )
    served_model_name: str | None = Field(default=None, min_length=1)
    gpu_product: str | None = Field(default=None, min_length=1)
    gpu_count: int | None = Field(default=None, ge=1)
    port: int | None = Field(default=None, ge=1, le=65535)
    max_model_len: int | None = Field(default=None, ge=1)
    source_manifest_path: str | None = Field(
        default=None,
        min_length=1,
        description="score: workspace prediction sources and checkpoints manifest.",
    )
    scoring_spec_path: str | None = Field(
        default=None,
        min_length=1,
        description="score: workspace business rules and validation examples JSON.",
    )
    scoring_script_path: str | None = Field(
        default=None,
        min_length=1,
        description="score: workspace Python with parse_record and score_case.",
    )
    manifest_path: str | None = Field(
        default=None,
        min_length=1,
        description="profile/review/report: workspace artifact manifest JSON.",
    )
    extraction_script_path: str | None = Field(
        default=None,
        min_length=1,
        description="profile only: workspace Python script defining parse_record.",
    )
    resume_run_id: uuid.UUID | None = Field(
        default=None, description="resume only: existing evaluation run UUID."
    )
    cpu: int | None = Field(
        default=None,
        ge=1,
        le=64,
        description="CPU cores; new runs default to 4, resume inherits when omitted.",
    )
    memory: int | None = Field(
        default=None,
        ge=1,
        le=256,
        description="Memory GiB; new runs default to 32, resume inherits when omitted.",
    )

    @model_validator(mode="after")
    def operation_fields(self) -> EvaluationOptions:
        missing = [
            key
            for key in _REQUIRED_FIELDS[self.operation]
            if getattr(self, key) is None
        ]
        if self.operation == "score" and bool(self.scoring_spec_path) != bool(
            self.scoring_script_path
        ):
            raise ValueError(
                "Supply scoring_spec_path and scoring_script_path together. "
                "Omit both when using the frozen Reward."
            )
        if missing:
            raise ValueError(
                f"operation={self.operation} requires top-level fields: "
                + ", ".join(missing)
            )
        allowed = (
            _COMMON_FIELDS
            | {
                "inference": _INFERENCE_FIELDS,
                "score": _SCORE_FIELDS,
                "profile": _PRODUCT_FIELDS,
                "review": frozenset({"manifest_path"}),
                "report": frozenset({"manifest_path"}),
                "resume": frozenset({"resume_run_id"}),
            }[self.operation]
        )
        supplied = {
            key for key in self.model_fields_set if getattr(self, key, None) is not None
        }
        invalid = supplied - allowed - {"kind"}
        if invalid:
            raise ValueError(
                f"operation={self.operation} does not accept: "
                + ", ".join(sorted(invalid))
            )
        return self

    def request(self) -> EvaluationRequest:
        """Build one typed request independent of the script tool."""
        values = self.model_dump(exclude_none=True)
        options: (
            InferenceOptions
            | PredictionEvaluationOptions
            | EvaluationProductsOptions
            | None
        )
        if self.operation == "inference":
            options = InferenceOptions.model_validate(
                {key: values[key] for key in _INFERENCE_FIELDS if key in values}
            )
        elif self.operation == "score":
            options = PredictionEvaluationOptions.model_validate(
                {key: values[key] for key in _SCORE_FIELDS if key in values}
            )
        elif self.operation in {"profile", "review", "report"}:
            options = EvaluationProductsOptions.model_validate(
                {
                    "operation": self.operation,
                    **{key: values[key] for key in _PRODUCT_FIELDS if key in values},
                }
            )
        else:
            options = None
        return EvaluationRequest(
            input_path=self.input_path,
            options=options,
            resume_run_id=self.resume_run_id,
            cpu=self.cpu if self.cpu is not None else 4,
            memory=self.memory if self.memory is not None else 32,
            resource_overrides=frozenset(
                key for key in ("cpu", "memory") if values.get(key) is not None
            ),
        )


@dataclass(frozen=True, kw_only=True)
class EvaluationRequest:
    input_path: str
    options: (
        InferenceOptions
        | PredictionEvaluationOptions
        | EvaluationProductsOptions
        | None
    )
    resume_run_id: uuid.UUID | None = None
    cpu: int = 4
    memory: int = 32
    resource_overrides: frozenset[str] = frozenset()
