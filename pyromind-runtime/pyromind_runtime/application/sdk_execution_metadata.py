"""Bind owned SDK task facts to the existing version and Recipe records."""

from __future__ import annotations

import re
from http.cookies import SimpleCookie
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from pyromind_runtime.application.model_evaluation_plan import sdk_inference_paths
from pyromind_runtime.application.versions import Versions
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.domain.storage_paths import (
    from_storage_api_path,
    to_storage_api_path,
)
from pyromind_runtime.domain.versions import RunSource, fingerprint
from pyromind_runtime.domain.workflow_execution import (
    DatasetRecipeReceipt,
    WorkflowExecutionRequest,
)
from pyromind_runtime.infrastructure.file_product_store import FileProductStore


class _ExecutionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    task_id: str = Field(min_length=1)
    workflow: JsonObject
    code: str | None = None
    output_dir: str | None = None
    dataset_recipe: DatasetRecipeReceipt | None = None


_CREDENTIAL_ASSIGNMENT = re.compile(
    r"\b(?:DF_API_KEY|PYROMIND_API_KEY|PYROMIND_AUTH_TOKEN|LLM_AUTH_TOKEN)"
    r"\s*=\s*(['\"]?)([^\s;'\"]+)"
)


def _validate_metadata(request: _ExecutionMetadata, context: RequestContext) -> None:
    workflow = request.workflow
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list) or not isinstance(workflow.get("edges"), list):
        raise ValueError("Supply the actual validated SDK workflow")
    commands = [request.code] if request.code else []
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Invalid workflow node")
        data = node.get("data")
        if not isinstance(data, dict):
            raise ValueError("Invalid workflow node data")
        config = data.get("config", {})
        validate_persisted_payload(config)
        if isinstance(config, dict):
            command = config.get("command")
            if isinstance(command, str):
                commands.append(command)
    validate_persisted_payload(workflow.get("execution_argos", []))
    if request.dataset_recipe is not None:
        validate_persisted_payload(request.dataset_recipe.model_dump())
        commands.append(request.dataset_recipe.command_template)
    document = request.model_dump_json()
    for command in commands:
        for assignment in _CREDENTIAL_ASSIGNMENT.finditer(command):
            if not assignment[2].startswith("$"):
                raise ValueError(
                    "Worker credentials cannot be saved in execution metadata"
                )
    secrets = [context.authorization or ""]
    if context.authorization and context.authorization.startswith("Bearer "):
        secrets.append(context.authorization.removeprefix("Bearer "))
    cookies = SimpleCookie(context.cookie or "")
    if "auth_token" in cookies:
        secrets.append(cookies["auth_token"].value)
    if any(secret and secret in document for secret in secrets):
        raise ValueError("Request credentials cannot be saved in execution metadata")


def _validate_recipe(
    recipe: DatasetRecipeReceipt, output_dir: str, store: FileProductStore
) -> None:
    if to_storage_api_path(recipe.output_directory) != to_storage_api_path(output_dir):
        raise ValueError("Recipe output does not match the SDK task output")
    root = PurePosixPath(to_storage_api_path(output_dir), "recipe-assets")
    roles = [asset.role for asset in recipe.assets]
    if len(roles) != len(set(roles)) or not {"pipeline", "export", "verifier"} <= set(
        roles
    ):
        raise ValueError(
            "Recipe needs unique frozen pipeline, export and verifier assets"
        )
    to_storage_api_path(recipe.source_path)
    for asset in recipe.assets:
        path = PurePosixPath(to_storage_api_path(asset.path))
        if not path.is_relative_to(root) or path.parent != root / asset.sha256:
            raise ValueError("Recipe assets must belong to this frozen SDK run")
    if recipe.replay_of_task_id is not None:
        parent = recipe.replay_of_task_id
        snapshot = store.load_snapshot()
        if not any(
            task.task_id == parent
            and task.kind == "data_preparation"
            and task.status == "succeeded"
            and task.graph_run_id is None
            for task in snapshot.external_tasks
        ) or not any(
            receipt.task_id == parent
            and receipt.status == "submitted"
            and receipt.dataset_recipe is not None
            for receipt in store.load_workflow_execution_requests().values()
        ):
            raise ValueError(
                "Replay requires a successful Dataset Recipe in this conversation"
            )


def bind_sdk_execution_metadata(
    store: FileProductStore,
    versions: Versions,
    arguments: JsonObject,
    context: RequestContext,
) -> tuple[WorkflowExecutionRequest, str | None]:
    """Record metadata only; a trusted callback must already establish ownership."""
    store.authorize(context.user_id, context.profile_id)
    request = _ExecutionMetadata.model_validate(arguments)
    _validate_metadata(request, context)
    snapshot = store.load_snapshot()
    task = next(
        (task for task in snapshot.external_tasks if task.task_id == request.task_id),
        None,
    )
    correlation = task.correlation if task is not None else None
    if (
        task is None
        or correlation is None
        or correlation.signature is None
        or correlation.type != "main"
        or not store.verify_task_correlation(correlation)
        or (context.task_origin is not None and context.task_origin.type != "main")
    ):
        raise PermissionError(
            "Wait for a trusted SDK callback confirming this conversation's task; "
            "retry metadata binding without resubmitting"
        )
    reference = correlation.execution
    if reference is None and store.preparation_policy() is not None:
        raise PermissionError(
            "Project metadata needs its signed Project task association"
        )
    if reference is not None and not any(
        run.project_id == reference.project_id
        and run.task_id == reference.task_id
        and run.plan_sequence == reference.plan_sequence
        and run.plan_attempt == reference.attempt
        and task.task_id in (run.platform_task_ids or (run.platform_task_id,))
        for run in snapshot.experiment_executions or ()
    ):
        raise PermissionError("SDK task does not match its Project/task/attempt")
    output_dir = request.output_dir
    if request.dataset_recipe is not None:
        if correlation.task_kind != "data_preparation" or task.graph_run_id is not None:
            raise ValueError(
                "Dataset Recipe requires a development data-preparation task"
            )
        output_dir = output_dir or request.dataset_recipe.output_directory
        _validate_recipe(request.dataset_recipe, output_dir, store)
    if output_dir is not None:
        output_dir = from_storage_api_path(output_dir)
        if task.output_dir is not None and to_storage_api_path(
            task.output_dir
        ) != to_storage_api_path(output_dir):
            raise ValueError("SDK task output directory cannot change")
    identity = "sdk:" + task.task_id
    digest = fingerprint(request.model_dump(mode="json"))
    previous = store.load_workflow_execution_requests().get(identity)
    if previous is not None:
        if previous.fingerprint != digest or previous.correlation != correlation:
            raise ValueError(
                "SDK execution metadata is already frozen with different facts"
            )
        return previous, output_dir
    run = versions.prepare_run(
        snapshot.conversation_id,
        context,
        identity,
        RunSource(kind="workflow", source_id=identity, task_id=task.task_id),
        request.workflow,
        code=request.code,
        resolved_inputs={},
    )
    receipt = WorkflowExecutionRequest(
        request_id=identity,
        conversation_id=snapshot.conversation_id,
        workflow_path="version://" + run.id,
        fingerprint=digest,
        workflow=request.workflow,
        dataset_recipe=request.dataset_recipe,
        summary={"name": str(request.workflow.get("name", "workflow"))},
        correlation=correlation,
        task_id=task.task_id,
        status="submitted",
    )
    saved, _ = store.save_workflow_execution_request(
        receipt, expected_revision=None, reuse_duplicate=True
    )
    return saved, output_dir


def verified_sdk_inference_paths(
    store: FileProductStore,
) -> dict[str, tuple[str, str]]:
    """Read frozen inference graphs owned by their signed Project task/attempt."""
    snapshot = store.load_snapshot()
    tasks = {task.task_id: task for task in snapshot.external_tasks}
    paths = {}
    for receipt in store.load_workflow_execution_requests().values():
        correlation = receipt.correlation
        if correlation is None:
            continue
        reference = correlation.execution
        task = tasks.get(receipt.task_id) if receipt.task_id else None
        if (
            receipt.status != "submitted"
            or receipt.conversation_id != snapshot.conversation_id
            or correlation.conversation_id != snapshot.conversation_id
            or correlation.type != "main"
            or correlation.task_kind != "evaluation"
            or reference is None
            or task is None
            or task.correlation != correlation
            or not store.verify_task_correlation(correlation)
        ):
            continue
        if not any(
            run.project_id == reference.project_id
            and run.task_id == reference.task_id
            and run.plan_sequence == reference.plan_sequence
            and run.plan_attempt == reference.attempt
            and task.task_id in (run.platform_task_ids or (run.platform_task_id,))
            for run in snapshot.experiment_executions or ()
        ):
            continue
        if inference := sdk_inference_paths(receipt.workflow):
            paths[task.task_id] = inference
    return paths
