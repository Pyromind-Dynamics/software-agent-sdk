"""Bind owned SDK task facts to the existing version and Recipe records."""

from __future__ import annotations

import re
import shlex
from copy import deepcopy
from http.cookies import SimpleCookie
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from pyromind_common.platform.recipe import (
    RecipeAsset,
    compose_recipe,
    read_prepared_dataset_recipe,
)
from pyromind_runtime.application.model_evaluation_plan import (
    sdk_inference_paths,
    validate_evaluation_inputs,
)
from pyromind_runtime.application.plan_execution import recover_plan_delivery_rows
from pyromind_runtime.application.versions import Versions
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.domain.preparation import ProjectView
from pyromind_runtime.domain.sft_reader import archived_sft_input
from pyromind_runtime.domain.storage_paths import (
    from_storage_api_path,
    to_storage_api_path,
)
from pyromind_runtime.domain.task_correlation import TaskCorrelation
from pyromind_runtime.domain.versions import (
    RunSource,
    fingerprint,
)
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
    r"\b(?:WANDB_API_KEY|DF_API_KEY|PYROMIND_API_KEY|PYROMIND_AUTH_TOKEN|LLM_AUTH_TOKEN)"
    r"\s*=\s*(['\"]?)([^\s;'\"]+)"
)


def portable_sdk_workflow(workflow: JsonObject) -> JsonObject:
    """Remove Studio's task annotations and resolved values of connected inputs."""
    result = deepcopy(workflow)
    nodes, edges = result.get("nodes"), result.get("edges")
    if not isinstance(nodes, list) or not nodes or not isinstance(edges, list):
        raise ValueError("Studio returned no executable graph")
    configs = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(data := node.get("data"), dict):
            raise ValueError("Invalid Studio node")
        node.pop("inputs_values", None)
        node.pop("outputs_values", None)
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id or node_id in configs:
            raise ValueError("Invalid Studio node identity")
        if not isinstance(data.get("nodeType"), str):
            raise ValueError("Missing Studio node type")
        props = node.get("properties")
        if isinstance(props, dict):
            props.pop("dystatus", None)
            if not props:
                node.pop("properties")
        config = data.get("config", {})
        if not isinstance(config, dict):
            raise ValueError("Invalid Studio node configuration")
        configs[node_id] = config
        if data["nodeType"] in {
            "WandbConfigBuilderNode",
            "PyroMindWandbConfigBuilderNode",
        }:
            key = config.get("wandb_api_key")
            if key is not None and not isinstance(key, str):
                raise ValueError("Invalid W&B credential reference")
            if key:
                config["wandb_api_key"] = "WANDB_API_KEY"
    for edge in edges:
        if (
            not isinstance(edge, dict)
            or edge.get("source") not in configs
            or edge.get("target") not in configs
        ):
            raise ValueError("Invalid Studio edge")
        handle = edge.get("targetHandle")
        if isinstance(handle, str) and handle:
            configs[edge["target"]].pop(handle, None)
    extra = result.get("extra")
    if isinstance(extra, dict):
        extra.pop("customType", None)
        extra.pop("taskId", None)
        if not extra:
            result.pop("extra")
    return result


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
                if request.dataset_recipe and read_prepared_dataset_recipe(command):
                    command = command.partition("\n")[2]
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


def submitted_sdk_dataset_recipe(workflow: JsonObject) -> DatasetRecipeReceipt | None:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 1 or workflow.get("edges") != []:
        return None
    node = nodes[0]
    data = node.get("data") if isinstance(node, dict) else None
    if not isinstance(data, dict) or data.get("nodeType") != "CustomCommandCPUNode":
        return None
    config = data.get("config")
    if not isinstance(config, dict) or not isinstance(
        command := config.get("command"), str
    ):
        return None
    prepared = read_prepared_dataset_recipe(command)
    if prepared is None:
        return None
    if prepared.uses_labeling_gateway:
        if not isinstance(config.get("param"), str) or not config["param"]:
            raise ValueError("Missing Dataset labeling credential binding")
        config["param"] = "DF_API_KEY"
    elif config.get("param"):
        raise ValueError("Unexpected Dataset command parameter")
    return DatasetRecipeReceipt.model_validate(prepared.model_dump())


class IndependentReplayNeedsStudio(ValueError):
    def __init__(self, task_id: str, cluster: str):
        self.task_id = task_id
        self.cluster = cluster
        super().__init__("独立换批小样需要项目所有者的登录态核对平台实际执行的工作流。")


def _verify_replay_command(workflow: JsonObject, recipe: DatasetRecipeReceipt) -> None:
    nodes = workflow.get("nodes")
    node = nodes[0] if isinstance(nodes, list) and len(nodes) == 1 else None
    data = node.get("data") if isinstance(node, dict) else None
    config = data.get("config") if isinstance(data, dict) else None
    command = config.get("command") if isinstance(config, dict) else None
    invalid = ValueError(
        "平台任务未执行这份完整配方的导出、校验和小样条数检查。"
        "请用完整 Dataset 配方提交换批小样。"
    )
    if not (
        isinstance(command, str)
        and isinstance(data, dict)
        and data.get("nodeType") == "CustomCommandCPUNode"
        and workflow.get("edges") == []
    ):
        raise invalid
    before, begin, rest = recipe.command_template.partition("\n(\n")
    processing, end, _ = rest.rpartition("\n)\n")
    if not begin or not end:
        raise invalid
    env = {}
    try:
        for line in before.splitlines():
            if not line.startswith("export "):
                continue
            for item in shlex.split(line)[1:]:
                key, _, value = item.partition("=")
                if key in ("DF_API_URL", "DF_API_BASE_URL", "DF_MODEL_NAME"):
                    env[key] = value
        embedded = submitted_sdk_dataset_recipe(workflow)
        if embedded is not None and embedded != recipe:
            raise invalid
        if recipe.uses_labeling_gateway:
            env["DF_API_KEY"] = "$param"
        expected, expected_command = compose_recipe(
            input_path=recipe.source_path,
            output_dir=recipe.output_directory,
            assets=tuple(
                RecipeAsset.model_validate(asset.model_dump())
                for asset in recipe.assets
            ),
            dataflow_command=processing.format(),
            llm_env=env,
            fixed_validation_path=recipe.fixed_validation_path,
            replay_of_task_id=recipe.replay_of_task_id,
            batch_label_source=recipe.batch_label_source,
            prepared_gateway=recipe.uses_labeling_gateway
            if embedded is not None
            else None,
            development_environment='export DF_API_KEY="$param"',
        )
    except (ValueError, KeyError, IndexError):
        raise invalid from None
    if (
        DatasetRecipeReceipt.model_validate(expected.model_dump()) != recipe
        or expected_command != command
    ):
        raise invalid


def _validate_recipe(
    recipe: DatasetRecipeReceipt,
    output_dir: str,
    store: FileProductStore,
    correlation: TaskCorrelation,
    task_id: str,
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
        source = next(
            (item for item in snapshot.external_tasks if item.task_id == parent), None
        )
        parent_receipt = next(
            (
                item
                for item in store.load_workflow_execution_requests().values()
                if item.task_id == parent
                and item.status == "submitted"
                and source is not None
                and item.correlation == source.correlation
            ),
            None,
        )
        if (
            parent == task_id
            or source is None
            or source.kind != "data_preparation"
            or source.status != "succeeded"
            or source.graph_run_id is not None
            or any(run.task_id == parent for run in store.load_pipeline_runs().values())
            or parent_receipt is None
            or not parent_receipt.workflow.get("nodes")
            or (
                parent_receipt.dataset_recipe
                and parent_receipt.dataset_recipe.replay_of_task_id
            )
        ):
            raise ValueError(
                "Replay requires a successful Dataset task in this conversation"
            )
        if store.preparation_policy() is not None:
            original = parent_receipt.correlation
            ref = original.execution if original else None
            if not (
                original
                and original.type == "main"
                and ref
                and original.task_kind == "data_preparation"
                and store.verify_task_correlation(original)
                and original.cluster
                and correlation.cluster == original.cluster
                and (correlation.execution is None or correlation.execution == ref)
                and any(
                    run.role_id == "main"
                    and run.project_id == ref.project_id
                    and run.task_id == ref.task_id
                    and run.plan_sequence == ref.plan_sequence
                    and run.plan_attempt == ref.attempt
                    and parent in (run.platform_task_id, *run.platform_task_ids)
                    for run in snapshot.experiment_executions or ()
                )
            ):
                raise PermissionError(
                    "Replay requires its signed Project Dataset receipt"
                )


def bind_sdk_execution_metadata(
    store: FileProductStore,
    versions: Versions,
    arguments: JsonObject,
    context: RequestContext,
    *,
    platform_archive: bool = False,
    accepted_workflow: JsonObject | None = None,
) -> tuple[WorkflowExecutionRequest, str | None]:
    """Record metadata only; a trusted callback must already establish ownership."""
    store.authorize(context.user_id, context.profile_id)
    request = _ExecutionMetadata.model_validate(arguments)
    with store.lock_sdk_submission(request.task_id):
        return _bind_sdk_metadata(
            store, versions, request, context, platform_archive, accepted_workflow
        )


def _bind_sdk_metadata(
    store: FileProductStore,
    versions: Versions,
    request: _ExecutionMetadata,
    context: RequestContext,
    platform_archive: bool,
    accepted_workflow: JsonObject | None,
) -> tuple[WorkflowExecutionRequest, str | None]:
    request.workflow = deepcopy(request.workflow)
    embedded = submitted_sdk_dataset_recipe(request.workflow)
    if embedded is not None:
        if request.dataset_recipe is not None and request.dataset_recipe != embedded:
            raise ValueError("Dataset recipe differs from the submitted command")
        request.dataset_recipe = embedded
    if platform_archive:
        request.workflow = portable_sdk_workflow(request.workflow)
        validate_persisted_payload(request.workflow)
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
    if (
        reference is None
        and store.preparation_policy() is not None
        and not (request.dataset_recipe and request.dataset_recipe.replay_of_task_id)
    ):
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
        _validate_recipe(
            request.dataset_recipe, output_dir, store, correlation, task.task_id
        )
    if output_dir is not None:
        output_dir = from_storage_api_path(output_dir)
        if task.output_dir is not None and to_storage_api_path(
            task.output_dir
        ) != to_storage_api_path(output_dir):
            raise ValueError("SDK task output directory cannot change")
    identity = "sdk:" + task.task_id
    previous = store.load_workflow_execution_requests().get(identity)
    independently_verified = False
    if reference is None and store.preparation_policy() is not None:
        assert (
            request.dataset_recipe is not None
            and request.dataset_recipe.replay_of_task_id
        )
        verified = request.workflow if platform_archive else accepted_workflow
        if (
            verified is None
            and previous
            and previous.summary.get("metadata_source") == "studio_task"
        ):
            verified = previous.workflow
        if verified is None:
            if not correlation.cluster:
                raise ValueError("独立换批小样没有已验签的集群归属。")
            raise IndependentReplayNeedsStudio(task.task_id, correlation.cluster)
        verified = deepcopy(verified)
        _verify_replay_command(verified, request.dataset_recipe)
        if portable_sdk_workflow(verified) != portable_sdk_workflow(request.workflow):
            raise ValueError("提交的配方工作流与平台实际执行的工作流不一致。")
        independently_verified = True
    digest = fingerprint(request.model_dump(mode="json"))
    if previous is not None:
        if previous.correlation != correlation:
            raise ValueError("SDK task association cannot change")
        if platform_archive or previous.fingerprint == digest:
            return previous, output_dir
        if (
            (
                previous.dataset_recipe is not None
                and request.dataset_recipe is not None
                and previous.dataset_recipe != request.dataset_recipe
            )
            or previous.summary.get("metadata_source") != "studio_task"
            or portable_sdk_workflow(request.workflow) != previous.workflow
        ):
            raise ValueError(
                "SDK execution metadata is already frozen with different facts"
            )
        if not (request.code or output_dir or request.dataset_recipe):
            return previous, output_dir
        request.workflow = previous.workflow
        request.dataset_recipe = request.dataset_recipe or previous.dataset_recipe
    if platform_archive:
        project = versions.project(snapshot.conversation_id, context)
        assert project is not None
        version = versions.save_run_config(
            project, identity, "workflow", request.workflow, context
        )
        workflow_path = "version://" + version.id
    else:
        run = versions.prepare_run(
            snapshot.conversation_id,
            context,
            identity,
            RunSource(kind="workflow", source_id=identity, task_id=task.task_id),
            request.workflow,
            code=request.code,
            resolved_inputs={},
        )
        workflow_path = "version://" + run.id
    train_input = archived_sft_input(request.workflow)
    receipt = WorkflowExecutionRequest(
        request_id=identity,
        conversation_id=snapshot.conversation_id,
        workflow_path=workflow_path,
        fingerprint=digest,
        workflow=request.workflow,
        dataset_recipe=request.dataset_recipe,
        summary={
            "name": str(request.workflow.get("name", "workflow")),
            **({"train_input": train_input} if train_input else {}),
            **(
                {"metadata_source": "studio_task"}
                if platform_archive or independently_verified
                else {}
            ),
        },
        correlation=correlation,
        task_id=task.task_id,
        status="submitted",
        revision=previous.revision + 1 if previous else 1,
    )
    saved, _ = store.save_workflow_execution_request(
        receipt,
        expected_revision=previous.revision if previous else None,
        reuse_duplicate=True,
    )
    return saved, output_dir


def verified_sdk_inference_paths(
    store: FileProductStore,
    project: ProjectView | None = None,
    model_evals: set[str] | None = None,
) -> dict[str, tuple[str, str]]:
    """Read frozen inference graphs owned by their signed Project task/attempt."""
    snapshot = store.load_snapshot()
    tasks = {task.task_id: task for task in snapshot.external_tasks}
    paths = {}
    approved = (
        {
            item.id: item
            for item in project.plan.tasks
            if model_evals and item.id in model_evals
        }
        if project
        else {}
    )
    rows = None
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
            or correlation.task_kind not in {"evaluation", "workflow_run"}
            or reference is None
            or task is None
            or task.correlation != correlation
            or not store.verify_task_correlation(correlation)
        ):
            continue
        owners = [
            run
            for run in snapshot.experiment_executions or ()
            if task.task_id in (run.platform_task_ids or (run.platform_task_id,))
        ]
        if len(owners) != 1:
            continue
        owner = owners[0]
        if (
            owner.role_id != "main"
            or owner.project_id != reference.project_id
            or owner.task_id != reference.task_id
            or owner.plan_sequence != reference.plan_sequence
            or owner.plan_attempt != reference.attempt
        ):
            continue
        inference = sdk_inference_paths(receipt.workflow)
        if not inference:
            continue
        if correlation.task_kind == "workflow_run":
            planned = approved.get(reference.task_id)
            if (
                project is None
                or planned is None
                or reference.project_id != project.project_id
                or reference.plan_sequence != project.plan_sequence
                or receipt.request_id != "sdk:" + task.task_id
                or task.status != "succeeded"
            ):
                continue
            if rows is None:
                rows = recover_plan_delivery_rows(snapshot, project)
            try:
                validate_evaluation_inputs(
                    project, planned, rows, *inference, require_bindings=True
                )
            except ValueError:
                continue
        paths[task.task_id] = inference
    return paths
