"""Platform I/O for the shared checkpoint coordinator; no Harness session needed."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx
import requests
from pydantic import TypeAdapter
from pyromind_runtime.application.checkpoint_evaluations import fingerprint, job_state
from pyromind_runtime.domain.checkpoint_evaluation import (
    CheckpointCandidate,
    CheckpointEvaluationRequest,
    EvaluationExecution,
    TaskStatus,
    TrainingObservation,
)
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.training_evaluation import StepEvaluation
from pyromind_runtime.ports.checkpoint_evaluation import EvaluationCredentialsRequired
from pyromind_runtime.ports.storage import (
    EvaluationStorage,
    EvaluationStorageMissingError,
)

from harness_adapter.workflow_submission import PlatformWorkflowSubmission
from openhands.agent_server.pyromind_auth import parse_auth_token_from_cookie_header
from openhands.tools.data_preparation.inference import (
    INFERENCE_RUNTIME_FILENAMES,
    DatasetMapping,
    EvaluationConfig,
    InferenceOptions,
    build_inference_dsl,
    storage_path,
    validate_pipeline_topology,
)
from openhands.tools.workflow.dsl_to_xyflow import convert_dsl_to_xyflow
from openhands.tools.workflow.task_submission import create_workflow_api_client
from openhands.tools.workflow.validate_workflow_dsl import (
    ValidateWorkflowDslAction,
    ValidateWorkflowDslExecutor,
)


def checkpoint_skills_root() -> Path:
    repository = (
        Path(getattr(sys, "_MEIPASS", ""))
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[2]
    )
    return Path(os.getenv("PYROMIND_SKILLS_PATH") or repository / ".agents/skills")


def task_status(value: str) -> TaskStatus:
    normalized = value.lower().replace("_", "").replace(" ", "")
    if normalized in {"success", "succeeded", "completed", "complete", "finished"}:
        return "succeeded"
    if normalized in {"failed", "error", "failure"}:
        return "failed"
    if normalized in {"stopped", "terminated", "cancelled", "canceled", "deleted"}:
        return "stopped"
    if normalized in {"running", "executing"}:
        return "running"
    if normalized in {
        "pending",
        "created",
        "queued",
        "waiting",
        "initializing",
        "scheduling",
    }:
        return "pending"
    return "unknown"


class PlatformCheckpointEvaluation:
    def __init__(
        self,
        skills_root: Path,
        mount_root: Path,
        storage_factory: Callable[[RequestContext], EvaluationStorage],
    ):
        self.skills_root, self.mount_root = skills_root, mount_root
        self.storage_factory = storage_factory
        self.submission = PlatformWorkflowSubmission()

    @staticmethod
    async def _io(operation, *args):
        try:
            return await asyncio.to_thread(operation, *args)
        except Exception as error:
            cause = error
            while cause is not None:
                if isinstance(cause, (httpx.HTTPStatusError, requests.HTTPError)):
                    response = cause.response
                    if response is not None and response.status_code in {401, 403}:
                        raise EvaluationCredentialsRequired() from None
                if "requires login" in str(cause):
                    raise EvaluationCredentialsRequired() from None
                cause = cause.__cause__
            raise

    @staticmethod
    def headers(context: RequestContext) -> dict[str, str]:
        return {
            name: value
            for name, value in (
                ("cookie", context.cookie),
                ("authorization", context.authorization),
                ("x-cluster", context.x_cluster),
            )
            if value
        }

    @staticmethod
    def client(context: RequestContext):
        if not context.cookie:
            raise EvaluationCredentialsRequired()
        route = context.x_cluster or ""
        cluster, separator, routed_env = route.partition("#")
        env = (
            routed_env
            if separator
            else ("prod" if route else os.getenv("PYROMIND_ENV", "pre"))
        )
        auth_token = parse_auth_token_from_cookie_header(context.cookie)
        if not auth_token:
            raise EvaluationCredentialsRequired()
        return create_workflow_api_client(
            env=env,
            cluster=cluster or "default",
            auth_token=auth_token,
            headers={"x-cluster": route} if route else {},
        )

    def _observer(
        self, operation: str, definition: dict, context: RequestContext
    ) -> dict:
        payload = {
            "operation": operation,
            "task_id": definition["training_task_id"],
            "training_node_id": definition["training_node_id"],
            "headers": self.headers(context),
            "mount_root": str(self.mount_root),
            "output_path": definition.get("output_path"),
            "base_model": definition.get("base_model"),
        }
        script = self.skills_root / "training-analysis/scripts/checkpoint_observer.py"
        interpreter = os.getenv("PYROMIND_CHECKPOINT_OBSERVER_PYTHON") or (
            shutil.which("python3") if getattr(sys, "frozen", False) else sys.executable
        )
        if not interpreter:
            raise ValueError(
                "A Python interpreter is required for checkpoint observation"
            )
        process = subprocess.run(
            [interpreter, str(script)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=150,
        )
        if process.returncode:
            raise ValueError("Training artifacts could not be inspected")
        return json.loads(process.stdout)

    async def freeze(
        self,
        request: CheckpointEvaluationRequest,
        output_dir: str,
        context: RequestContext,
    ) -> JsonObject:
        return await self._io(self._freeze, request, output_dir, context)

    def _freeze(
        self,
        request: CheckpointEvaluationRequest,
        output_dir: str,
        context: RequestContext,
    ) -> JsonObject:
        self.client(context).studio.get_task(request.training_task_id)
        try:
            source = self._observer("resolve", request.model_dump(mode="json"), context)
        except ValueError:
            source = {"output_path": None, "inputs": {}}
        template: dict[str, Any] = dict(request.template)
        allowed = {
            "input_path",
            "dataset_config",
            "evaluation_config",
            "inference",
            "cpu",
            "memory",
            "scoring_script",
            "scoring_spec",
            "datasets",
        }
        if template.keys() - allowed:
            raise ValueError("unknown checkpoint evaluation template fields")
        dataset = DatasetMapping.model_validate(template["dataset_config"]).model_dump(
            exclude_none=True
        )
        evaluation = EvaluationConfig.model_validate(
            template["evaluation_config"]
        ).model_dump(exclude_none=True)
        if evaluation.get("generation", {}).get("limit", 0):
            raise ValueError(
                "checkpoint selection requires the complete validation dataset"
            )
        raw_options = dict(template.get("inference", {}))
        if raw_options.keys() - {
            "served_model_name",
            "gpu_product",
            "gpu_count",
            "port",
            "max_model_len",
        }:
            raise ValueError(
                "checkpoint inference options only accept serving and GPU settings"
            )
        options = InferenceOptions.model_validate(
            {
                **raw_options,
                "model_path": "/placeholder/checkpoint-0",
                "dataset_config_path": "frozen",
                "evaluation_config_path": "frozen",
            }
        ).model_dump(
            exclude={
                "model_path",
                "dataset_config_path",
                "evaluation_config_path",
                "report_spec_path",
            }
        )
        storage = self.storage_factory(context)
        input_path = storage_path(str(template["input_path"]))
        raw = storage.read(input_path)
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if not rows or any(not isinstance(row, dict) for row in rows):
            raise ValueError("validation dataset must contain JSON objects")
        dataset.setdefault(
            "media_base_dir", "/workspace" + str(Path(input_path).parent)
        )
        datasets = dict(template.get("datasets", {}))
        datasets["validation"] = {
            "path": input_path,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for role in ("train", "test"):
            if role in datasets:
                path = storage_path(str(datasets[role]["path"]))
                datasets[role] = {
                    "path": path,
                    "sha256": hashlib.sha256(storage.read(path)).hexdigest(),
                }
        media = self._media_fingerprints(rows, dataset, input_path)
        runtime = self.skills_root / "inference-evaluation/scripts"
        asset_paths = {name: runtime / name for name in INFERENCE_RUNTIME_FILENAMES}
        asset_paths["evaluate_predictions.py"] = runtime / "evaluate_predictions.py"
        asset_paths["training_artifacts.py"] = (
            self.skills_root / "training-analysis/scripts/training_artifacts.py"
        )
        assets = {name: path.read_bytes() for name, path in asset_paths.items()}
        hashes = {
            name: hashlib.sha256(data).hexdigest() for name, data in assets.items()
        }
        for name, data in assets.items():
            storage.write(output_dir + "/frozen/" + name, data)
        frozen_input = output_dir + "/frozen/validation.jsonl"
        storage.write(frozen_input, raw)
        template = {
            "input_path": frozen_input,
            "dataset_config": dataset,
            "evaluation_config": evaluation,
            "inference": options,
            "cpu": int(template.get("cpu", 4)),
            "memory": int(template.get("memory", 32)),
            "runtime_sha256": fingerprint(hashes),
            "dataset_provenance": {
                "sha256": datasets["validation"]["sha256"],
                "dataset_role": "validation",
            },
        }
        if not 1 <= template["cpu"] <= 64 or not 1 <= template["memory"] <= 256:
            raise ValueError("invalid evaluation CPU or memory resources")
        script = request.template.get("scoring_script")
        spec = request.template.get("scoring_spec")
        if script is not None or spec is not None:
            if not isinstance(script, str) or not isinstance(spec, dict):
                raise ValueError(
                    "business scoring requires scoring_script and scoring_spec"
                )
            functions = {
                node.name
                for node in ast.parse(script).body
                if isinstance(node, ast.FunctionDef)
            }
            if not {"parse_record", "score_case"} <= functions or not spec.get(
                "validation_examples"
            ):
                raise ValueError(
                    "business scoring requires parse_record, score_case "
                    "and validation_examples"
                )
            template["business_scoring"] = {
                "script": script,
                "spec": spec,
                "dataset_role": "validation",
                "dataset_source": datasets["validation"],
            }
        worker_digest = hashlib.sha256()
        for name in sorted(INFERENCE_RUNTIME_FILENAMES):
            worker_digest.update(name.encode() + b"\0" + assets[name] + b"\0")
        comparison = fingerprint(
            {
                "template": template,
                "dataset": datasets["validation"],
                "media": media,
                "assets": hashes,
            }
        )
        inputs = source.get("inputs", {})
        return TypeAdapter(JsonObject).validate_python(
            {
                "training_task_id": request.training_task_id,
                "training_node_id": request.training_node_id,
                "output_path": source["output_path"],
                "base_model": inputs.get("base_model") or inputs.get("model_path"),
                "template": template,
                "datasets": datasets,
                "media_sha256": media,
                "worker_runtime_sha256": worker_digest.hexdigest(),
                "sample_count": len(rows),
                "assets": hashes,
                "comparison_group": comparison,
                "retention_limit": inputs.get("training_config", {}).get(
                    "save_total_limit"
                ),
            }
        )

    def _media_fingerprints(self, rows: list, config: dict, input_path: str) -> dict:
        helpers = runpy.run_path(
            str(self.skills_root / "inference-evaluation/scripts/evaluate_inference.py")
        )
        media = {}
        for row in rows:
            for value in helpers["_input_media"](row, config):
                if value.startswith(("http://", "https://")):
                    raise ValueError(
                        "checkpoint comparison requires versioned Storage media; "
                        "remote URLs are not supported"
                    )
                if value.startswith("data:"):
                    continue
                path = helpers["_media_path"](
                    value, Path("/workspace" + input_path), config
                )
                reference = storage_path(str(path))
                local = (self.mount_root / reference.lstrip("/")).resolve()
                if not local.is_relative_to(self.mount_root.resolve()):
                    raise ValueError("media path escapes Storage mount")
                with local.open("rb") as stream:
                    media[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
        return media

    async def observe(
        self, definition: JsonObject, context: RequestContext
    ) -> TrainingObservation:
        def read():
            client = self.client(context)
            status = task_status(
                client.studio.get_task(str(definition["training_task_id"])).status
            )
            observed = self._observer("observe", definition, context)
            telemetry = observed.get("telemetry", {})
            warnings = [str(telemetry["warning"])] if telemetry.get("warning") else []
            warnings.append(
                "Checkpoint readiness is heuristic; "
                "deleted checkpoints cannot be recovered."
            )
            return TrainingObservation(
                status=status,
                checkpoints=observed["checkpoints"],
                training_step=telemetry.get("training_step"),
                telemetry=telemetry,
                warnings=warnings,
            )

        return await self._io(read)

    async def prepare(
        self, parent: PipelineRun, child: PipelineRun, context: RequestContext
    ) -> JsonObject:
        return await self._io(self._prepare, parent, child, context)

    def _prepare(
        self, parent: PipelineRun, child: PipelineRun, context: RequestContext
    ) -> JsonObject:
        storage = self.storage_factory(context)
        definition = child.definition
        datasets = cast(dict, parent.definition["datasets"])
        raw = storage.read(str(definition["input_path"]))
        if hashlib.sha256(raw).hexdigest() != datasets["validation"]["sha256"]:
            raise ValueError("frozen validation dataset changed")
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        media = self._media_fingerprints(
            rows,
            cast(dict, definition["dataset_config"]),
            str(definition["input_path"]),
        )
        if media != parent.definition["media_sha256"]:
            raise ValueError("validation media changed since registration")
        dsl = build_inference_dsl(definition, child.output_dir, str(child.run_id))
        validation = ValidateWorkflowDslExecutor(headers=self.headers(context))(
            ValidateWorkflowDslAction(dsl=dsl)
        )
        if validation.is_error or validation.valid is not True:
            raise ValueError("checkpoint evaluation workflow validation failed")
        workflow = convert_dsl_to_xyflow(
            dsl, name=f"checkpoint-evaluation-{child.run_id}"
        )
        validate_pipeline_topology(workflow, require_inference=True)
        configs = {
            "dataset_config.json": definition["dataset_config"],
            "evaluation_config.json": definition["evaluation_config"],
        }
        configs["dataset_provenance.json"] = definition["dataset_provenance"]
        if definition.get("business_scoring"):
            configs["business_scoring.json"] = definition["business_scoring"]
        for name, value in configs.items():
            storage.write(child.output_dir + "/" + name, json.dumps(value).encode())
        assets = cast(dict, parent.definition["assets"])
        for name, expected in assets.items():
            payload = storage.read(parent.output_dir + "/frozen/" + name)
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("frozen evaluation runtime changed")
            storage.write(child.output_dir + "/" + name, payload)
        storage.write(child.output_dir + "/workflow.py", dsl.encode())
        return workflow

    async def submit(
        self, child: PipelineRun, workflow: JsonObject, context: RequestContext
    ) -> str:
        return await self.submission.submit(workflow, child.conversation_id, context)

    async def inspect(
        self,
        parent: PipelineRun,
        candidate: CheckpointCandidate,
        context: RequestContext,
    ) -> EvaluationExecution:
        return await self._io(self._inspect, parent, candidate, context)

    def _inspect(
        self,
        parent: PipelineRun,
        candidate: CheckpointCandidate,
        context: RequestContext,
    ) -> EvaluationExecution:
        assert candidate.task_id and candidate.run_id
        task = self.client(context).studio.get_task(candidate.task_id)
        status = task_status(task.status)
        if status == "failed":
            message = task.error_message or ""
            not_ready = any(
                marker in message.lower()
                for marker in (
                    "safetensorerror",
                    "header too small",
                    "incomplete metadata",
                    "unable to open file",
                    "no file named",
                    "unexpected eof",
                )
            ) and any(
                marker in message.lower()
                for marker in (
                    "safetensor",
                    "pytorch_model",
                    "adapter_model",
                    "checkpoint-",
                )
            )
            return EvaluationExecution(
                status=status,
                weights_not_ready=not_ready,
                error="Weights not ready"
                if not_ready
                else "Platform evaluation failed",
            )
        if status != "succeeded":
            return EvaluationExecution(
                status=status,
                error="Platform evaluation failed" if status == "failed" else None,
            )
        output = (
            f"/.pyromind-agent/{parent.conversation_id}/data_preparation/"
            f"{candidate.run_id}"
        )
        storage = self.storage_factory(context)
        try:
            manifest = json.loads(storage.read(output + "/run_manifest.json"))
            metrics = json.loads(storage.read(output + "/metrics.json"))
        except (
            EvaluationStorageMissingError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            return EvaluationExecution(
                status="succeeded",
                error="Completed evaluation is missing result artifacts",
            )
        if not isinstance(manifest, dict) or not isinstance(metrics, dict):
            return EvaluationExecution(
                status="succeeded", error="Invalid result artifacts"
            )
        template = cast(dict, parent.definition["template"])
        dataset = cast(dict, parent.definition["datasets"])["validation"]
        issues = []
        expected_evaluation = dict(template["evaluation_config"])
        expected_evaluation["model_source"] = {
            "type": "checkpoint",
            "kind": candidate.artifact.kind,
            "base_model": candidate.artifact.base_model,
        }
        if (
            manifest.get("evaluation_config") != expected_evaluation
            or manifest.get("runtime_sha256")
            != parent.definition["worker_runtime_sha256"]
            or manifest.get("business_scoring") != template.get("business_scoring")
        ):
            issues.append("evaluation_rule_provenance_mismatch")
        if (
            manifest.get("run_id") != str(candidate.run_id)
            or manifest.get("model_reference") != candidate.artifact.path
            or manifest.get("dataset_sha256") != dataset["sha256"]
            or manifest.get("dataset_config") != template["dataset_config"]
            or manifest.get("media_sha256") != parent.definition["media_sha256"]
        ):
            issues.append("evaluation_input_provenance_mismatch")
        for key in ("total", "successful_predictions", "evaluated_cases"):
            if metrics.get(key) != parent.definition["sample_count"]:
                issues.append("incomplete_" + key)
        return EvaluationExecution(
            status=status,
            evaluation=StepEvaluation(
                step=candidate.artifact.step,
                run_id=str(candidate.run_id),
                output_dir=output,
                checkpoint=candidate.artifact,
                metrics=metrics,
                comparison_group=str(parent.definition["comparison_group"]),
                eligible=not issues,
                issues=issues,
            ),
        )

    async def verify_submission(
        self, child: PipelineRun, task_id: str, context: RequestContext
    ) -> None:
        def verify():
            task = self.client(context).studio.get_task(task_id)
            if task.name != f"checkpoint-evaluation-{child.run_id}":
                raise ValueError("task name does not identify this evaluation run")
            self._observer(
                "verify",
                {
                    "training_task_id": task_id,
                    "training_node_id": "evaluation",
                    "output_path": child.output_dir,
                },
                context,
            )

        await self._io(verify)

    async def stop(self, task_id: str, context: RequestContext) -> None:
        await self._io(lambda: self.client(context).studio.stop(task_id))

    async def publish(self, parent: PipelineRun, context: RequestContext) -> None:
        def write():
            storage = self.storage_factory(context)
            state = job_state(parent)
            storage.write(
                parent.output_dir + "/checkpoint_selection.json",
                json.dumps(parent.result, ensure_ascii=False).encode(),
            )
            storage.write(
                parent.output_dir + "/iteration_context.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": str(parent.run_id),
                        "selection": parent.result,
                        "datasets": parent.definition["datasets"],
                        "steps": [
                            c.evaluation.model_dump(mode="json")
                            for c in state.candidates.values()
                            if c.evaluation
                        ],
                        "coverage": {
                            key: {"status": c.status, "error": c.error}
                            for key, c in state.candidates.items()
                        },
                        "warnings": state.warnings,
                    },
                    ensure_ascii=False,
                ).encode(),
            )

        await self._io(write)
