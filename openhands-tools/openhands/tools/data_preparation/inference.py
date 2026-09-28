from __future__ import annotations

import hashlib
import json
import shlex
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openhands.sdk.conversation.state import ActiveLongTask
from openhands.tools.data_preparation.workspace_paths import resolve_workspace_file
from openhands.tools.workflow.dsl_to_xyflow import convert_dsl_to_xyflow
from openhands.tools.workflow.task_submission import (
    PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET,
    create_workflow_api_client,
    submit_workflow_task,
)
from openhands.tools.workflow.validate_workflow_dsl import (
    ValidateWorkflowDslAction,
    ValidateWorkflowDslExecutor,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState
    from openhands.tools.data_preparation.platform_submit import (
        DfSubmitPipelineAction,
        DfSubmitPipelineExecutor,
    )


class InferenceOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model_path: str = Field(min_length=1)
    dataset_config_path: str = Field(min_length=1)
    evaluation_config_path: str = Field(min_length=1)
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


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mode: Literal["agent_rubric"]
    rubrics: list[Rubric] = Field(min_length=1, max_length=20)
    pass_threshold: float = Field(default=0.7, ge=0, le=1)
    rubric_pass_threshold: float = Field(default=0.7, ge=0, le=1)
    generation: GenerationOptions = Field(default_factory=GenerationOptions)

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


ALLOWED_NODE_TYPES = frozenset({"CustomCommandCPUNode", "VLLMInference"})


def validate_pipeline_topology(workflow: dict[str, Any]) -> None:
    nodes, edges = workflow.get("nodes", []), workflow.get("edges", [])
    kinds = [node["data"]["nodeType"] for node in nodes]
    if not set(kinds) <= ALLOWED_NODE_TYPES:
        raise ValueError("pipeline contains a node outside the server allowlist")
    if kinds == ["CustomCommandCPUNode"] and not edges:
        return
    if sorted(kinds) != ["CustomCommandCPUNode", "VLLMInference"] or len(edges) != 1:
        raise ValueError("only one CPU node or inference -> CPU is allowed")
    source = next(node for node in nodes if node["data"]["nodeType"] == "VLLMInference")
    target = next(
        node for node in nodes if node["data"]["nodeType"] == "CustomCommandCPUNode"
    )
    edge = edges[0]
    if (
        str(source["id"]) == str(target["id"])
        or str(edge.get("source")) != str(source["id"])
        or str(edge.get("target")) != str(target["id"])
        or edge.get("sourceHandle") != "endpoint"
        or edge.get("targetHandle") != "param"
    ):
        raise ValueError("pipeline must connect inference.endpoint to CPU.param")


def storage_path(value: str) -> str:
    if value == "/workspace" or value.startswith("/workspace/"):
        value = value[len("/workspace") :] or "/"
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) == "/":
        raise ValueError("expected an absolute Storage path without traversal")
    return str(path)


def build_inference_dsl(definition: dict[str, Any], output_dir: str) -> str:
    options = definition["inference"]
    output = "/workspace" + storage_path(output_dir)
    command = " ".join(
        [
            "python3",
            shlex.quote(output + "/evaluate_inference.py"),
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
            "--output-dir",
            shlex.quote(output),
        ]
    )
    node_config = {
        key: options[key]
        for key in ("port", "gpu_count", "gpu_product", "max_model_len")
        if options.get(key) is not None
    }
    node_config["model_path"] = "/workspace" + options["model_path"]
    args = ", ".join(f"{key}={value!r}" for key, value in node_config.items())
    return (
        f"inference = VLLMInference(id=1, {args})\n\n"
        f"evaluation = CustomCommandCPUNode(id=2, command={command!r}, "
        f"cpu={definition['cpu']}, memory={definition['memory']}, "
        "param=inference.endpoint)\n"
    )


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

    def submit(
        self,
        owner: DfSubmitPipelineExecutor,
        action: DfSubmitPipelineAction,
        conversation: BaseConversation,
    ) -> dict[str, Any]:
        if any(
            (
                action.script_path,
                action.support_file_path,
                action.output_schema,
                action.model_profile,
                action.reuse_assessment,
                action.prompt_fingerprint,
            )
        ):
            raise ValueError(
                "inference uses frozen built-in scripts and no cleaning options"
            )
        if action.convert_format != "none":
            raise ValueError("inference does not support format conversion")
        if action.mode == "full" and action.resume_run_id:
            raise ValueError("resume_run_id requires mode=resume")
        if action.mode == "resume" and not action.resume_run_id:
            raise ValueError("resume requires resume_run_id")
        prior = (
            self.runs.resolve(str(action.resume_run_id))
            if action.resume_run_id
            else None
        )
        script = self.runtime_dir / "evaluate_inference.py"
        runtime_hash = hashlib.sha256(script.read_bytes()).hexdigest()
        if action.inference:
            options = action.inference
            dataset = DatasetMapping.model_validate_json(
                resolve_workspace_file(
                    conversation, options.dataset_config_path
                ).read_text()
            ).model_dump(exclude_none=True)
            evaluation = EvaluationConfig.model_validate_json(
                resolve_workspace_file(
                    conversation, options.evaluation_config_path
                ).read_text()
            ).model_dump(exclude_none=True)
            inference = options.model_dump(
                exclude={"dataset_config_path", "evaluation_config_path"}
            )
            inference["model_path"] = storage_path(options.model_path)
            definition = {
                "inference": inference,
                "dataset_config": dataset,
                "evaluation_config": evaluation,
                "input_path": storage_path(action.input_path),
                "cpu": action.cpu,
                "memory": action.memory,
                "runtime_sha256": runtime_hash,
            }
        elif prior:
            definition = dict(prior["definition"])
            if definition["input_path"] != storage_path(action.input_path):
                raise ValueError("input_path changed; use a new full run")
            for resource in ("cpu", "memory"):
                if resource in action.model_fields_set:
                    definition[resource] = getattr(action, resource)
            if definition["runtime_sha256"] != runtime_hash:
                raise ValueError("evaluation runtime changed; use a new full run")
        else:
            raise ValueError("inference options are required for a new evaluation")
        run = self.runs.reserve(
            definition, str(action.resume_run_id) if action.resume_run_id else None
        )
        run_id, output_dir = run["run_id"], run["output_dir"]
        revision = run["revision"]
        submitting = False
        task_id = None
        response_status = "Pending"
        warning = ""
        try:
            dsl = build_inference_dsl(definition, output_dir)
            workflow = convert_dsl_to_xyflow(dsl, name=f"agent-evaluation-{run_id[:8]}")
            validate_pipeline_topology(workflow)
            validation = self.validator(
                ValidateWorkflowDslAction(dsl=dsl), conversation
            )
            if validation.is_error or validation.valid is not True:
                raise ValueError(
                    f"platform rejected evaluation workflow: {validation.text}"
                )
            state = cast("ConversationState", conversation.state)
            client = create_workflow_api_client(
                env=owner._env,
                cluster=owner._cluster,
                headers=owner._headers,
                timeout=owner._timeout,
                auth_token=state.secret_registry.get_secret_value(
                    PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET
                ),
            )
            with tempfile.TemporaryDirectory(prefix="inference-config-") as directory:
                for name, config in (
                    ("dataset_config.json", definition["dataset_config"]),
                    ("evaluation_config.json", definition["evaluation_config"]),
                ):
                    path = Path(directory) / name
                    path.write_text(json.dumps(config, ensure_ascii=False))
                    owner._stage_script(
                        str(path), output_dir, conversation, frozen_script_name=name
                    )
            owner._stage_script(
                str(script),
                output_dir,
                conversation,
                frozen_script_name="evaluate_inference.py",
            )
            workspace = Path(cast(Any, conversation).workspace.working_dir)
            canvas = workspace / "public_data/workflow_canvas/workflow.py"
            canvas.parent.mkdir(parents=True, exist_ok=True)
            canvas.write_text(dsl)
            submitting = True
            response = submit_workflow_task(
                client=client,
                workflow=workflow,
                name=str(workflow["name"]),
                conversation_id=str(conversation.id),
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
                    "resumed": action.mode == "resume",
                    "execution_revision": run["execution_revision"],
                }
            warning = (
                " Platform accepted the task, but saving its association failed; "
                "do not submit it again."
            )
        conversation.register_active_long_task(
            ActiveLongTask(
                task_id=task_id,
                kind="data_preparation",
                status=response_status,
            )
        )
        return {
            "text": (
                f"Evaluation submitted: {task_id}. Check {output_dir}/report.json "
                "after the callback; it contains the HTML artifact path." + warning
            ),
            "task_id": task_id,
            "run_id": run_id,
            "output_dir": output_dir,
            "status": response_status,
            "resumed": action.mode == "resume",
            "execution_revision": run["execution_revision"],
        }
