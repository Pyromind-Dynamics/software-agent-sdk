from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import cache
from hashlib import sha256
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from filelock import Timeout as FileLockTimeout
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    model_validator,
)

from pyromind_agent_server.api.router import (
    _product_error,
    _require_profile,
    configure_session,
    context_for_profile,
    get_conversation_service,
    get_product_runtime,
)
from pyromind_agent_server.config import Config
from pyromind_agent_server.dependencies import check_session_api_key
from pyromind_agent_server.echomind_acquisition import (
    AcquisitionSimulation,
    AcquisitionSimulationStore,
    AcquisitionSource,
    AcquisitionSplit,
)
from pyromind_agent_server.echomind_acquisition_runtime import (
    AcquisitionSimulationRunner,
    JsonlAcquisitionSampleSource,
    TracerBatchMaterializer,
)
from pyromind_agent_server.echomind_acquisition_tasks import (
    EchoMindAcquisitionTaskManager,
)
from pyromind_agent_server.echomind_consolidation import (
    ConsolidationBlocker,
    ConsolidationContext,
    ConsolidationProposal,
    ConsolidationRecipe,
    ConsolidationReview,
    apply_consolidation,
    consolidation_context,
)
from pyromind_agent_server.echomind_consolidation_references import (
    consolidation_evaluation_reference,
    consolidation_references,
)
from pyromind_agent_server.echomind_consolidation_replay import recheck_dataset_replay
from pyromind_agent_server.echomind_failure_evidence import (
    read_graph_failure_diagnostics,
    redact_graph_failure_text,
)
from pyromind_agent_server.echomind_graph_runs import (
    ActiveGraphRunExistsError,
    GraphRun,
    GraphRunStore,
    TaskRun,
)
from pyromind_agent_server.echomind_platform_tracer import (
    is_platform_tracer_script,
    prepare_platform_tracer_setup,
)
from pyromind_agent_server.echomind_projects import (
    ProjectClusterMismatchError,
    ProjectInfo,
    ProjectNode,
    ProjectNotFoundError,
    ProjectStore,
    ProjectSummary,
)
from pyromind_agent_server.echomind_publication import (
    DebugExecutionAdoptionPreview,
    DebugExecutionCandidate,
    ProjectLifecycleError,
    ProjectPublicationService,
    PublicationImmutableError,
    PublicationMode,
    PublicationPreflight,
    PublicationStore,
    PublicationValidationError,
    PublishedGraph,
    PublishedGraphDraft,
    RuntimeContractApplication,
    RuntimeContractUpgradeDraft,
    RuntimeGraphRepairProposal,
    RuntimeImplementationAdoption,
    RuntimeImplementationAdoptionError,
    adopt_runtime_implementation,
    apply_runtime_contract_draft,
    build_publication_preflight,
    build_runtime_contract_draft,
    compile_working_graph,
    list_runtime_validation_candidates,
    load_runtime_validation_project,
)
from pyromind_agent_server.echomind_repair_proposals import GraphRepairProposals
from pyromind_agent_server.echomind_resource_proposals import (
    RuntimeResourceProposal,
    read_runtime_resource_proposal,
)
from pyromind_agent_server.echomind_runtime_access import (
    build_runtime_credential_store,
    refresh_runtime_access,
)
from pyromind_agent_server.echomind_runtime_contracts import (
    RuntimeCredentialUnavailableError,
)
from pyromind_agent_server.echomind_runtime_factory import (
    build_echomind_runtime_scheduler,
)
from pyromind_agent_server.echomind_runtime_scheduler import (
    RuntimeGraphScheduler,
    RuntimeSchedulerError,
)
from pyromind_agent_server.echomind_runtime_slots import (
    EchoMindTracerProvider,
    RuntimeSlot,
    RuntimeSlotEnsureError,
    RuntimeSlotService,
    RuntimeSlotSetupRequest,
    RuntimeSlotStore,
)
from pyromind_agent_server.echomind_sandbox_storage import (
    build_acquisition_storage,
    build_bound_evaluation_storage,
)
from pyromind_agent_server.echomind_script_consolidation import (
    build_script_compilation,
    load_script_compilation,
    save_script_compilation,
)
from pyromind_agent_server.echomind_stage_executors import (
    get_project_node_executor,
)
from pyromind_agent_server.echomind_stage_runs import (
    ProjectArtifactDocument,
    ProjectRunStore,
    StageRun,
    run_project_node,
)
from pyromind_agent_server.echomind_storage import (
    ProjectLeaseTimeout,
)
from pyromind_agent_server.echomind_tracer_admin import (
    build_tracer_admin_cipher,
    load_tracer_admin_key,
    save_tracer_admin_key,
)
from pyromind_agent_server.evaluation_storage import build_evaluation_storage
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_common.platform.auth import (
    parse_auth_token_from_cookie_header,
)
from pyromind_common.sdk.utils.cipher import Cipher
from pyromind_common.tools.utils.platform_env import (
    platform_cluster_route,
    platform_cluster_target,
)
from pyromind_common.tools.utils.pyromind_api_client import (
    get_api_key,
    get_pyromind_api_client,
)
from pyromind_runtime.application.consolidation_repairs import (
    ConsolidationRepair,
    conversation_has_work,
    plan_repair,
    repair_recipe_path,
)
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.commands import CommandReceipt, UserMessageCommand
from pyromind_runtime.domain.content import JsonObject, JsonValue, TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.errors import ProductRuntimeError
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.domain.workflow_execution import DatasetRecipeReceipt
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.storage import (
    DatasetReplayReader,
    EvaluationStorage,
    EvaluationStorageMissingError,
    EvaluationStorageUnavailableError,
)


router = APIRouter(
    prefix="/api/echomind",
    tags=["EchoMind Projects"],
    dependencies=[Depends(check_session_api_key)],
)
logger = logging.getLogger(__name__)


class StartEchoMindProjectNodeResponse(BaseModel):
    schema_version: Literal[1] = 1
    project_id: str
    graph_id: str
    node_id: str
    conversation_id: str
    cluster: str
    event_id: str
    status: Literal["started"] = "started"


class RuntimeContractApplyRequest(BaseModel):
    expected_draft_sha256: str
    validation_project_id: str | None = None


class ConsolidationTaskSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str | None = Field(default=None, pattern=r"^[0-9]{1,64}$")
    train: str | None = Field(default=None, pattern=r"^[0-9]{1,64}$")
    eval: str | None = Field(default=None, pattern=r"^[0-9]{1,64}$")


class RequestConsolidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_ids: ConsolidationTaskSelection = Field(
        default_factory=ConsolidationTaskSelection
    )


class ReviseConsolidation(RequestConsolidation):
    review_note: str = Field(default="", max_length=2000)


class CompileConsolidation(RequestConsolidation):
    recipe_path: str | None = Field(
        default=None,
        max_length=240,
        pattern=r"^public_data/execution-recipes/[A-Za-z0-9][A-Za-z0-9._-]*\.json$",
    )


class ApplyConsolidation(RequestConsolidation):
    proposal_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ConsolidationRequestReceipt(BaseModel):
    command_id: str
    proposal_path: str


class _ConsolidationProposalUnavailable(RuntimeError):
    pass


class RuntimeImplementationAdoptionRequest(BaseModel):
    validation_project_id: str | None = None


class TracerAdminRequest(BaseModel):
    admin_key: SecretStr


class ConfirmRuntimeResourceProposal(BaseModel):
    expected_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class DebugExecutionAdoptionRequest(BaseModel):
    source_project_id: str
    run_id: str
    expected_preview_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class PublishEchoMindProjectRequest(BaseModel):
    publication_mode: PublicationMode = "debug"
    expected_draft_sha256: str | None = None
    expected_contract_sha256: str | None = None
    validation_project_id: str | None = None


class StartEchoMindGraphRunRequest(BaseModel):
    request_id: str
    inputs: dict[str, object] = Field(default_factory=dict)


class StartEchoMindDebugGraphRunRequest(StartEchoMindGraphRunRequest):
    expected_contract_sha256: str
    validation_project_id: str | None = None


class StartGraphRepairDebugRequest(BaseModel):
    expected_proposal_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class RequestGraphRepairRequest(BaseModel):
    revise_invalid_proposal: bool = False


class GraphRepairProposalView(BaseModel):
    node_id: str
    reason: str
    content_sha256: str
    changed_fields: list[str]


class EchoMindGraphRunCallbackRequest(BaseModel):
    provider_run_id: str
    status: str


class RetryEchoMindGraphRunRequest(BaseModel):
    inputs: dict[str, object] | None = None


class EchoMindGraphRunDetail(BaseModel):
    run: GraphRun
    stage_runs: list[StageRun]
    task_runs: list[TaskRun]


class CreateEchoMindCollectionSimulationRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=256)
    source: AcquisitionSource
    limit: int = Field(gt=0)
    rounds: int = Field(gt=0)
    splits: list[AcquisitionSplit] = Field(
        default_factory=lambda: ["train", "val"],
        min_length=1,
    )
    auto_start_graph_run: bool = True
    model: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def validate_splits(self) -> CreateEchoMindCollectionSimulationRequest:
        if len(set(self.splits)) != len(self.splits):
            raise ValueError("collection simulation splits must be unique")
        return self


def _build_dataset_start_context(project: ProjectInfo, node_id: str) -> str:
    node = next(item for item in project.graph.nodes if item.id == node_id)
    context = {
        "project": {
            "project_id": project.project.project_id,
            "graph_id": project.graph.graph_id,
            "name": project.project.name,
            "description": project.project.description,
            "skill": project.project.skill,
            "lifecycle": project.graph.lifecycle,
            "requirements": {
                requirement.key: requirement.value
                for requirement in project.project.requirements
            },
        },
        "stage": {
            "node_id": node.id,
            "node_type": node.type,
            "title": node.title,
            "launch_mode": node.launch_mode,
            "config": node.config,
        },
    }
    return (
        "A user explicitly started this Project stage from the EchoMind UI.\n\n"
        "Server-generated Project and Stage context:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        "Continue in this existing EchoMind conversation and develop only the "
        "Dataset stage. Use the `data-processing` Skill as the source of truth.\n"
        "Required development sequence:\n"
        "1. Inspect the referenced dataset in Storage and determine its actual "
        "structure. Treat `input_hint` as a hint, not verified metadata.\n"
        "2. Select the matching data-processing paradigm, create the Pipeline, "
        "and run a small sample in the conversation sandbox.\n"
        "3. Show the sample result and wait for explicit user confirmation before "
        "submitting a full validation task.\n"
        "4. If the user confirms, submit the validation task through the "
        "data-processing platform flow and report its task/run/output identifiers.\n\n"
        "Do not create another Project, do not write `graph.json`, do not invent "
        "server identifiers, and do not start Inference, Evaluator, or Train. "
        "Those stages are outside this Dataset development action."
    )


def _require_sandbox_credentials(request: Request) -> None:
    if os.getenv("PYROMIND_HARNESS_BACKEND", "pi").strip().lower() != "pi":
        return
    if os.getenv("PYROMIND_PI_TERMINAL_BACKEND", "").strip().lower() != "sandbox":
        return

    auth_token = parse_auth_token_from_cookie_header(request.headers.get("cookie"))
    authorization = request.headers.get("authorization", "")
    scheme, _, value = authorization.partition(" ")
    if not auth_token and not (scheme.lower() == "bearer" and value.strip()):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Dataset development runs in the conversation sandbox. "
                "Enter a Cookie containing auth_token=... or a Bearer "
                "Authorization, and set x-cluster, then click Apply."
            ),
        )


def _load_project(
    request: Request,
    project_id: str,
    *,
    require_cluster: bool = False,
) -> tuple[RequestContext, ProjectInfo, ProjectStore]:
    context = context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
    service = get_conversation_service(request)
    store = ProjectStore.for_conversations_dir(service.conversations_dir)
    try:
        project = store.load(project_id, owner_user_id=context.user_id)
        if require_cluster:
            store.require_cluster(project, context.x_cluster)
    except ProjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_id}",
        ) from exc
    except ProjectClusterMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return context, project, store


def _runtime_scheduler(
    request: Request,
    *,
    projects_root: Path,
    context: RequestContext | None,
) -> RuntimeGraphScheduler:
    return build_echomind_runtime_scheduler(
        projects_root=projects_root,
        conversations_dir=get_conversation_service(request).conversations_dir,
        request_context=context,
        recorded_storage=lambda cid, ctx: get_product_runtime(
            request
        ).recorded_workflow_storage(cid, ctx),
        credential_store=build_runtime_credential_store(
            projects_root, request.app.state.config
        ),
    )


async def _refresh_runtime_access_for_request(
    request: Request,
    projects_root: Path,
    project: ProjectInfo,
    context: RequestContext,
) -> None:
    credential_store = build_runtime_credential_store(
        projects_root, request.app.state.config
    )
    if credential_store is not None:
        await asyncio.to_thread(
            refresh_runtime_access, credential_store, project, context, force=True
        )


def _load_graph_run_release(
    projects_root: Path, project_id: str, run_id: str
) -> PublishedGraph:
    try:
        run = GraphRunStore(projects_root).load(project_id=project_id, run_id=run_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"GraphRun not found: {run_id}",
        ) from exc
    try:
        return PublicationStore(projects_root).load(
            project_id=project_id, release_id=run.release_id
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="GraphRun's immutable release is unavailable.",
        ) from exc


@router.get(
    "/projects",
    response_model=list[ProjectSummary],
)
async def list_echomind_projects(request: Request) -> list[ProjectSummary]:
    context = context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
    service = get_conversation_service(request)
    return ProjectStore.for_conversations_dir(service.conversations_dir).list(
        owner_user_id=context.user_id
    )


@router.get(
    "/projects/{project_id}",
    response_model=ProjectInfo,
    responses={404: {"description": "Project not found"}},
)
async def get_echomind_project(project_id: str, request: Request) -> ProjectInfo:
    _, project, _ = _load_project(request, project_id)
    return project


@router.get(
    "/projects/{project_id}/runtime-contract/draft",
    response_model=RuntimeContractUpgradeDraft,
    responses={404: {"description": "Project not found"}},
)
async def draft_echomind_runtime_contract(
    project_id: str,
    request: Request,
    validation_project_id: str | None = None,
) -> RuntimeContractUpgradeDraft:
    _, project, store = _load_project(request, project_id)
    try:
        service = ProjectPublicationService(store.root)
        draft = service.runtime_contract_draft(project, validation_project_id)
    except RuntimeImplementationAdoptionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return draft.model_copy(
        update={
            "validation_candidates": list_runtime_validation_candidates(
                store=store, project=project
            ),
            "resource_release_id": _resource_release_id(service, project, draft),
        }
    )


async def _load_consolidation_context(
    request: Request, project: ProjectInfo, task_ids: dict[str, str]
) -> ConsolidationContext:
    try:
        conversation = project.project.conversation_id
        root = get_conversation_service(request).conversations_dir
        receipts = await asyncio.to_thread(
            FileProductStore(root / conversation).load_workflow_execution_requests
        )
        approved = None
        if any(
            item.request_id.startswith("sdk:")
            and item.correlation
            and item.correlation.execution
            for item in receipts.values()
        ):
            approved = await get_product_runtime(request).get_project(
                conversation, context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
            )
        return await asyncio.to_thread(
            consolidation_context,
            project,
            root,
            task_ids=task_ids,
            approved_project=approved,
        )
    except (FileNotFoundError, PermissionError) as exc:
        raise HTTPException(
            404, detail="Project execution evidence not found."
        ) from exc
    except ValidationError as exc:
        raise HTTPException(
            409, detail="Project execution evidence is invalid."
        ) from exc
    except ValueError as exc:
        raise HTTPException(409, detail=str(exc)) from exc


async def _read_consolidation_proposal(
    request: Request, consolidation: ConsolidationContext, context: RequestContext
) -> ConsolidationProposal | None:
    try:
        async with asyncio.timeout(8):
            source = await get_product_runtime(request).read_persisted_workspace_text(
                consolidation.evidence.conversation_id,
                consolidation.proposal_path,
                context,
                max_bytes=256 * 1024,
            )
    except TimeoutError as exc:
        raise _ConsolidationProposalUnavailable(
            "读取执行方案超时，可以继续请求 Agent 准备方案。"
        ) from exc
    except (OSError, ProductRuntimeError) as exc:
        raise _ConsolidationProposalUnavailable(
            "暂时无法读取工作区中的执行方案，可以继续请求 Agent 准备方案。"
        ) from exc
    return (
        ConsolidationProposal.model_validate_json(source)
        if source is not None
        else None
    )


def _consolidation_issue(error: ValueError, context: RequestContext) -> str:
    if isinstance(error, ValidationError):
        fields = ", ".join(
            ".".join(map(str, item["loc"])) or "JSON"
            for item in error.errors(include_input=False)[:4]
        )
        fields = redact_graph_failure_text(fields, context)[:300]
        return f"Invalid execution proposal fields: {fields}."
    return redact_graph_failure_text(str(error), context)[:600]


@router.get(
    "/projects/{project_id}/runtime-contract/consolidation",
    response_model=ConsolidationReview,
)
async def get_echomind_consolidation(
    project_id: str,
    request: Request,
    dataset_task_id: str | None = None,
    train_task_id: str | None = None,
    eval_task_id: str | None = None,
    script_only: bool = False,
) -> ConsolidationReview:
    context, project, store = _load_project(request, project_id)
    task_ids = {
        stage: task_id
        for stage, task_id in (
            ("dataset", dataset_task_id),
            ("train", train_task_id),
            ("eval", eval_task_id),
        )
        if task_id is not None
    }
    consolidation = await _load_consolidation_context(request, project, task_ids)
    review = ConsolidationReview(
        context=consolidation, proposal_path=consolidation.proposal_path
    )
    try:
        compilation = await asyncio.to_thread(
            load_script_compilation, store, consolidation
        )
        if compilation is not None:
            return await asyncio.to_thread(compilation.review, project, consolidation)
        if script_only:
            return review
        proposal = await _read_consolidation_proposal(request, consolidation, context)
        if proposal is None:
            return review

    except _ConsolidationProposalUnavailable as exc:
        return review.model_copy(update={"issue": str(exc)})
    except ValueError as exc:
        return review.model_copy(update={"issue": _consolidation_issue(exc, context)})
    return review.model_copy(
        update={"issue": "已有静态提案，请生成并验证执行方案以校验换批小样。"}
    )


@router.post(
    "/projects/{project_id}/runtime-contract/consolidation/compile",
    response_model=ConsolidationReview,
)
async def compile_echomind_script_plan(
    project_id: str, body: CompileConsolidation, request: Request
) -> ConsolidationReview:
    context, project, store = _load_project(request, project_id, require_cluster=True)
    consolidation = await _load_consolidation_context(
        request, project, body.task_ids.model_dump(exclude_none=True)
    )
    runtime = get_product_runtime(request)
    conversations_root = get_conversation_service(request).conversations_dir
    try:
        recipe = None
        if body.recipe_path is not None:
            async with asyncio.timeout(8):
                source = await runtime.read_persisted_workspace_text(
                    consolidation.evidence.conversation_id,
                    body.recipe_path,
                    context,
                    max_bytes=256 * 1024,
                )
            if source is None:
                raise ValueError("工作区中找不到配方，请确认已保存并具有读取权限。")
            recipe = ConsolidationRecipe.model_validate_json(source)
        read, check = await _replay_checks(
            request,
            store.root,
            project_id,
            context,
            consolidation.evidence.conversation_id,
            consolidation.source_batch,
        )
        compiled = await asyncio.to_thread(
            build_script_compilation,
            project,
            consolidation,
            conversations_root,
            context,
            versions=runtime.versions,
            recipe=recipe,
            read_replay_file=read,
            check_replay=check,
        )
        return await asyncio.to_thread(
            save_script_compilation,
            store,
            project,
            consolidation,
            compiled,
            conversations_root,
        )
    except ProductRuntimeError as exc:
        raise _product_error(exc) from exc
    except ProjectLeaseTimeout as exc:
        raise HTTPException(409, detail="Project 正在更新，请稍后重新编译。") from exc
    except (TimeoutError, OSError) as exc:
        raise HTTPException(503, detail="暂时无法读取或保存执行方案，请重试。") from exc
    except ValueError as exc:
        raise HTTPException(409, detail=_consolidation_issue(exc, context)) from exc


class GenerateConsolidation(RequestConsolidation):
    retry_command_id: str | None = Field(default=None, max_length=180)


class GeneratedConsolidation(BaseModel):
    review: ConsolidationReview
    repair: ConsolidationRepair | None = None


async def _replay_checks(
    request: Request,
    root: Path,
    project_id: str,
    ctx: RequestContext,
    conversation_id: str,
    approved_source: str,
):
    runtime = get_product_runtime(request)
    development_proxy = await runtime.development_storage_proxy(conversation_id, ctx)
    replay_reader = await runtime.dataset_replay_reader(conversation_id, ctx)

    @cache
    def storage() -> EvaluationStorage | DatasetReplayReader:
        return (
            build_bound_evaluation_storage(
                projects_root=root,
                project_id=project_id,
                context=ctx,
                credential_store=build_runtime_credential_store(
                    root, request.app.state.config
                ),
                development_proxy=development_proxy,
            )
            or replay_reader
            or build_evaluation_storage(ctx)
        )

    def read(path: str) -> bytes:
        return storage().read(to_storage_api_path(path))

    def check(receipt: DatasetRecipeReceipt) -> bytes:
        return recheck_dataset_replay(
            storage(), receipt, approved_source=approved_source
        )

    return read, check


async def _read_repair_recipes(
    runtime: ConversationRuntime,
    context: ConsolidationContext,
    ctx: RequestContext,
    receipts: tuple[CommandReceipt, ...],
) -> tuple[ConsolidationRecipe, list[ConsolidationBlocker], dict[str, str]]:
    recipe_fields: JsonObject = {"evidence_sha256": context.evidence_sha256}
    invalid: list[ConsolidationBlocker] = []
    revisions: dict[str, str] = {}
    conversation = context.evidence.conversation_id
    for stage in ("train", "eval"):
        prefix = f"consolidation-{context.evidence_sha256[7:]}-{stage}-"
        if not any(r.command_id.startswith(prefix) for r in receipts):
            continue
        path = repair_recipe_path(context.evidence_sha256, stage)
        async with asyncio.timeout(12):
            source = await runtime.read_persisted_workspace_text(
                conversation, path, ctx, max_bytes=256 * 1024
            )
        if source is None:
            continue
        revisions[stage] = sha256(source.encode()).hexdigest()
        field = "evaluator_runtime" if stage == "eval" else f"{stage}_runtime"
        try:
            patch = ConsolidationRecipe.model_validate_json(source)
            if (
                patch.evidence_sha256 != context.evidence_sha256
                or patch.model_fields_set != {"evidence_sha256", field}
            ):
                raise ValueError("补丁必须只包含当前证据摘要与本节点配置")
            value = patch.model_dump()[field]
            if not isinstance(value, dict):
                raise ValueError("节点配置为空")
            recipe_fields[field] = value
        except ValueError as exc:
            invalid.append(
                ConsolidationBlocker(
                    stage=stage,
                    code="repair_invalid",
                    message=_consolidation_issue(exc, ctx),
                )
            )
    return ConsolidationRecipe.model_validate(recipe_fields), invalid, revisions


@router.post(
    "/projects/{project_id}/runtime-contract/consolidation/generate",
    response_model=GeneratedConsolidation,
)
async def generate_echomind_consolidation(
    project_id: str, body: GenerateConsolidation, request: Request
) -> GeneratedConsolidation:
    ctx, project, store = _load_project(request, project_id, require_cluster=True)
    context = await _load_consolidation_context(
        request, project, body.task_ids.model_dump(exclude_none=True)
    )
    conversation = context.evidence.conversation_id
    runtime = get_product_runtime(request)
    root = get_conversation_service(request).conversations_dir
    receipts, snapshot = await runtime.consolidation_command_state(
        conversation, context.evidence_sha256, ctx
    )
    prior = await asyncio.to_thread(load_script_compilation, store, context)
    awaiting = (
        snapshot.pending_inputs
        or conversation_has_work(snapshot)
        or snapshot.status
        in {
            "waiting_for_confirmation",
            "waiting_for_input",
        }
    )
    try:
        revisions: dict[str, str] = {}
        if receipts and prior and prior.blockers and awaiting:
            review = prior.review(project, context)
        else:
            recipe, invalid, revisions = await _read_repair_recipes(
                runtime, context, ctx, receipts
            )
            read, check = await _replay_checks(
                request,
                store.root,
                project_id,
                ctx,
                conversation,
                context.source_batch,
            )
            compiled = await asyncio.to_thread(
                build_script_compilation,
                project,
                context,
                root,
                ctx,
                versions=runtime.versions,
                recipe=recipe,
                read_replay_file=read,
                check_replay=check,
            )
            if invalid:
                stages = {error.stage for error in invalid}
                compiled = compiled.model_copy(
                    update={
                        "proposal": None,
                        "blockers": (
                            *invalid,
                            *(b for b in compiled.blockers if b.stage not in stages),
                        ),
                        "stages": tuple(
                            row.model_copy(update={"status": "blocked"})
                            if row.stage in stages
                            else row
                            for row in compiled.stages
                        ),
                    }
                )
            review = await asyncio.to_thread(
                save_script_compilation, store, project, context, compiled, root
            )

        if not review.blockers:
            return GeneratedConsolidation(review=review)
        storage_issue = next(
            (
                b
                for b in review.blockers
                if b.code == "dataset_replay_storage_unavailable"
            ),
            None,
        )
        if storage_issue:
            return GeneratedConsolidation(
                review=review,
                repair=ConsolidationRepair(
                    stage=storage_issue.stage,
                    command_id="",
                    status="blocked",
                    message=storage_issue.message,
                ),
            )
        stage = review.blockers[0].stage
        evidence = context.evidence.model_dump(mode="json")[stage]
        facts: JsonObject = {
            "stage": stage,
            "evidence_sha256": context.evidence_sha256,
            "development_task": evidence,
            "plan": [
                redact_graph_failure_text(
                    json.dumps(node.config.get("tasks", []), ensure_ascii=False), ctx
                )[:6000]
                for node in project.graph.nodes
                if node.type == ("evaluator" if stage == "eval" else stage)
            ],
            "inputs": {
                "source_batch": context.source_batch,
                "base_model": context.base_model,
                "fixed_eval": context.validation_dataset,
                "training_validation": context.training_validation,
            },
            "assets": review.assets.get(stage),
            "train_input_fields": review.assets.get("train_input_fields"),
            "train_development_input": review.assets.get("train_development_input"),
        }
        repair, command = plan_repair(
            stage=stage,
            evidence=context.evidence_sha256,
            facts=facts,
            issues=[
                f"{b.code} · {b.message}" for b in review.blockers if b.stage == stage
            ],
            revision=revisions.get(stage, ""),
            receipts=receipts,
            snapshot=snapshot,
            retry_command_id=body.retry_command_id,
        )
        if command:
            await asyncio.to_thread(
                _require_profile, request, conversation, ECHOMIND_PRODUCT_PROFILE
            )
            await configure_session(request, conversation, ECHOMIND_PRODUCT_PROFILE)
            result = await runtime.submit_command(conversation, command, ctx)
            if result.status == "failed":
                repair = repair.model_copy(
                    update={
                        "status": "blocked",
                        "message": "修订请求未送达，稍后可重试。",
                    }
                )
        return GeneratedConsolidation(review=review, repair=repair)
    except ProductRuntimeError as exc:
        raise _product_error(exc) from exc
    except ProjectLeaseTimeout as exc:
        raise HTTPException(409, detail="Project 正在更新，请稍后重试。") from exc
    except (TimeoutError, OSError, EvaluationStorageUnavailableError) as exc:
        raise HTTPException(
            503, detail="读取或保存固化证据暂时失败，可以继续本次验证。"
        ) from exc
    except ValueError as exc:
        raise HTTPException(409, detail=_consolidation_issue(exc, ctx)) from exc


@router.post(
    "/projects/{project_id}/runtime-contract/consolidation",
    response_model=ConsolidationRequestReceipt,
    status_code=status.HTTP_202_ACCEPTED,
)
async def request_echomind_consolidation(
    project_id: str, body: ReviseConsolidation, request: Request
) -> ConsolidationRequestReceipt:
    context, project, _ = _load_project(request, project_id, require_cluster=True)
    conversation_id = project.project.conversation_id
    await asyncio.to_thread(
        _require_profile, request, conversation_id, ECHOMIND_PRODUCT_PROFILE
    )
    consolidation = await _load_consolidation_context(
        request, project, body.task_ids.model_dump(exclude_none=True)
    )
    runtime = get_product_runtime(request)
    try:
        references = await asyncio.to_thread(
            consolidation_references,
            FileProductStore(
                get_conversation_service(request).conversations_dir / conversation_id
            ),
            consolidation.evidence,
            approved_project=consolidation.approved_project,
        )
        train_reference = references.get("train")
        if isinstance(train_reference, dict) and "runtime_candidate" in train_reference:
            train_reference.pop("workflow", None)
        if runtime.versions is not None:
            eval_reference = references.get("eval")
            if isinstance(eval_reference, dict):
                eval_reference.update(
                    await asyncio.to_thread(
                        consolidation_evaluation_reference,
                        runtime.versions,
                        consolidation.evidence,
                        context,
                    )
                )
        command = UserMessageCommand(
            command_id=uuid4().hex,
            content=(
                TextContent(
                    text=_consolidation_prompt(
                        consolidation,
                        references,
                        review_note=redact_graph_failure_text(
                            body.review_note.strip(), context
                        ),
                    )
                ),
            ),
        )
        await configure_session(request, conversation_id, ECHOMIND_PRODUCT_PROFILE)
        receipt = await runtime.submit_command(conversation_id, command, context)
    except ProductRuntimeError as exc:
        if exc.code == "conversation_busy":
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail={
                    "code": exc.code,
                    "message": "Agent 正在处理当前回合，结束后再生成执行方案。",
                },
            ) from exc
        raise _product_error(exc) from exc
    except OSError as exc:
        raise HTTPException(
            503, detail="Could not request an execution proposal."
        ) from exc
    except ValueError as exc:
        raise HTTPException(409, detail=_consolidation_issue(exc, context)) from exc
    if receipt.status == "failed":
        raise HTTPException(409, detail="Could not deliver the request to Manager.")
    return ConsolidationRequestReceipt(
        command_id=receipt.command_id, proposal_path=consolidation.proposal_path
    )


@router.post(
    "/projects/{project_id}/runtime-contract/consolidation/apply",
    response_model=ProjectInfo,
)
async def apply_echomind_consolidation(
    project_id: str, body: ApplyConsolidation, request: Request
) -> ProjectInfo:
    context, project, store = _load_project(request, project_id, require_cluster=True)
    task_ids = body.task_ids.model_dump(exclude_none=True)
    consolidation = await _load_consolidation_context(request, project, task_ids)
    try:
        compilation = await asyncio.to_thread(
            load_script_compilation, store, consolidation
        )
        proposal = (
            compilation.proposal
            if compilation is not None
            else await _read_consolidation_proposal(request, consolidation, context)
        )
        if proposal is None:
            raise ValueError("The execution proposal is missing. Review it again.")

        def load_current_proposal(
            fresh: ConsolidationContext,
        ) -> ConsolidationProposal | None:
            current = load_script_compilation(store, fresh)
            return current.proposal if current is not None else None

        return await asyncio.to_thread(
            apply_consolidation,
            store,
            project,
            conversations_root=get_conversation_service(request).conversations_dir,
            task_ids=task_ids,
            proposal=proposal,
            expected_proposal_sha256=body.proposal_sha256,
            approved_project=consolidation.approved_project,
            load_current_proposal=load_current_proposal
            if compilation is not None
            else None,
        )
    except _ConsolidationProposalUnavailable as exc:
        raise HTTPException(503, detail=str(exc)) from exc
    except (ValueError, ProjectLeaseTimeout) as exc:
        issue = (
            _consolidation_issue(exc, context)
            if isinstance(exc, ValueError)
            else "The Project is being updated. Try again."
        )
        raise HTTPException(409, detail=issue) from exc


def _consolidation_prompt(
    consolidation: ConsolidationContext, references: JsonObject, *, review_note: str
) -> str:
    evidence = (
        json.dumps(
            {
                **consolidation.model_dump(mode="json"),
                "proposal_path": consolidation.proposal_path,
                "references": references,
                "review_note": review_note,
            },
            ensure_ascii=False,
        )
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    return (
        "你是当前 EchoMind Project 的 Manager。根据已成功的 Dataset、Train 和 Eval "
        "研发任务，准备该 Project 的执行方案供用户审阅。核对真实工具回执、已确认的"
        "输入和可从平台任务容器读取的脚本与配置。只准备提案及其版本化资产。"
        "不要提交平台任务、修改现有产物或 Project 状态。"
        "若已有提案，补齐已核验缺口。review_note 是用户在审阅时指定的修改意见。"
        "涉及已确认输入或标签政策变更时，向用户确认。\n\n"
        "逐节点准备配置，每次工具调用只写一份小配置。已有的长脚本只传版本路径，"
        "不要在回复、提案或工具参数里转抄脚本。三个节点都齐备后，用 JSON 解析"
        "合并配置并写到 proposal_path。顶层字段是 evidence_sha256（原样填写）、"
        "dataset_runtime、train_runtime、evaluator_runtime。每个 runtime 中的 workflow "
        "是冻结的 Studio XYFlow JSON，包含 nodes 和 edges；不要把执行 DSL 字符串写入"
        " workflow。workflow_bindings 是参数路径到 Graph 输入端口的映射。使用独立的"
        " /workspace 输出目录，路径含 {graph_run_id} 和 {stage_run_id}。命令用"
        " command_template 和 command_binding_path 写入节点参数，静态文本中的花括号"
        "要转义。只引用已核验的平台可读资产，不把凭据写进配置或命令。\n\n"
        "Dataset 从 source_batch 处理真实源数据，输出 train_messages.jsonl 和"
        " val_messages.jsonl。绑定 {source_batch} 和 {output_uri}。从本 Project 的"
        "已确认数据格式、标签策略、切分规则和成功回执提取实现。确有预处理时，将"
        "整理源数据、生成中间清单和下游处理编入执行器支持的工作流。当前执行器直接"
        "提交平台工作流。若需要在提交前额外执行一次沙箱命令，注明该执行能力缺口。"
        "每轮使用独立中间目录，按实际挂载映射 Storage 路径。已封批的采集数据可按"
        "契约直接转训练数据，保持 Project 的标签策略和训练格式。\n"
        "Manager 只核对已知回执和明确的资产路径。仅在必须时，请 Dataset 子 Agent"
        "用它的技能做只读核验，返回可用的 Storage 文件、校验摘要和输入输出约定。"
        "此阶段严禁提交任务、安装环境、申请凭据或遍历无关目录。版本运行标识"
        "仅用于追溯。原图标注缺少可执行步骤、已确认标签或受支持的授权引用时，"
        "明确报告缺口和待补证据，结束本轮提案准备。\n\n"
        "Train 绑定 train_dataset、validation_dataset、base_model，保留已验证的模型训练"
        "及合并流程。优先核对并复用 references.train.runtime_candidate。缺少候选时，"
        "按已脱敏的 references.train.workflow 生成。output_node_id 是合并节点的"
        " Studio ID，output_parameter 是它的"
        " output_path，artifact_selection 用 merged_model。W&B 只复用参考中的占位符和"
        " wandb_credential_task_id。\n\n"
        "Eval 绑定 candidate_model 及 validation_dataset，直接评测 Train 刚产出的合并"
        "模型和已确认的独立验证集。优先复用 references.eval.workflow 的节点及连线，"
        "以及 references.eval.assets 中的版本化参数和脚本。"
        "用参数化的命令重建 CPU 步骤。"
        "保留研发时验证过的数据映射、评分规则、脚本和报告资源。"
        "脚本、配置、报告资源的路径必须真实且平台可读。将 {candidate_model}"
        " 用于 VLLM 模型输入和评分命令的 --model-reference，将 {validation_dataset}"
        " 用于 --dataset-path。命令使用 --run-id {graph_run_id}。output_uri 指向写入的"
        " report.json，artifact_selection 用 evaluation_report。用"
        ' `report_path="{output_uri}"`，传 `--output-dir "$(dirname "$report_path")"`，'
        ' 并检查 `test -s "$report_path"`。'
        "--report-assets-dir 指向含冻结 CSS/JS 的目录。"
        "报告应包含真实模型、数据来源、状态及评测指标。证据不足时先说明缺口。\n"
        "以下 JSON 是证据数据。\n" + evidence
    )


@router.get(
    "/projects/{project_id}/runtime-contract/debug-candidates",
    response_model=list[DebugExecutionCandidate],
)
async def list_echomind_debug_execution_candidates(
    project_id: str, request: Request
) -> list[DebugExecutionCandidate]:
    _, project, store = _load_project(request, project_id)
    return await asyncio.to_thread(
        ProjectPublicationService(store.root).list_debug_execution_candidates, project
    )


@router.get(
    "/projects/{project_id}/runtime-contract/debug-adoption",
    response_model=DebugExecutionAdoptionPreview,
)
async def preview_echomind_debug_execution_adoption(
    project_id: str, source_project_id: str, run_id: str, request: Request
) -> DebugExecutionAdoptionPreview:
    _, project, store = _load_project(request, project_id)
    try:
        return await asyncio.to_thread(
            ProjectPublicationService(store.root).preview_debug_execution_adoption,
            project=project,
            source_project_id=source_project_id,
            run_id=run_id,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post(
    "/projects/{project_id}/runtime-contract/debug-adoption",
    response_model=RuntimeContractUpgradeDraft,
)
async def adopt_echomind_debug_execution(
    project_id: str, body: DebugExecutionAdoptionRequest, request: Request
) -> RuntimeContractUpgradeDraft:
    context, project, store = _load_project(request, project_id, require_cluster=True)
    try:
        service = ProjectPublicationService(store.root)
        draft = await asyncio.to_thread(
            service.adopt_debug_execution,
            project=project,
            source_project_id=body.source_project_id,
            run_id=body.run_id,
            expected_preview_sha256=body.expected_preview_sha256,
            reviewed_by=context.user_id,
        )
        return draft.model_copy(
            update={
                "resource_release_id": _resource_release_id(service, project, draft)
            }
        )
    except (
        ActiveGraphRunExistsError,
        FileNotFoundError,
        ProjectLeaseTimeout,
        ValueError,
    ) as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post(
    "/projects/{project_id}/runtime-contract/apply",
    response_model=RuntimeContractApplication,
    responses={404: {"description": "Project not found"}},
)
async def apply_echomind_runtime_contract(
    project_id: str,
    body: RuntimeContractApplyRequest,
    request: Request,
) -> RuntimeContractApplication:
    _, project, store = _load_project(request, project_id, require_cluster=True)
    try:
        validation = (
            load_runtime_validation_project(
                store=store,
                project=project,
                validation_project_id=body.validation_project_id,
            )
            if body.validation_project_id is not None
            else None
        )
        draft = build_runtime_contract_draft(project, validation_project=validation)
        if body.expected_draft_sha256 != draft.content_sha256:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Runtime contract draft is stale; rebuild it before applying.",
            )
        return await asyncio.to_thread(
            apply_runtime_contract_draft,
            store=store,
            project=project,
            draft=draft,
        )
    except (ProjectLeaseTimeout, RuntimeImplementationAdoptionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except PublicationValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Runtime contract can not be applied.",
                "issues": [issue.model_dump(mode="json") for issue in exc.issues],
            },
        ) from exc


@router.post(
    "/projects/{project_id}/runtime-implementation/adopt",
    response_model=RuntimeImplementationAdoption,
    responses={404: {"description": "Project not found"}},
)
async def adopt_echomind_runtime_implementation(
    project_id: str,
    body: RuntimeImplementationAdoptionRequest,
    request: Request,
) -> RuntimeImplementationAdoption:
    _, project, store = _load_project(request, project_id, require_cluster=True)
    try:
        return adopt_runtime_implementation(
            store=store,
            project=project,
            validation_project_id=body.validation_project_id,
        )
    except RuntimeImplementationAdoptionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.post(
    "/projects/{project_id}/publish",
    response_model=PublishedGraph,
    responses={404: {"description": "Project not found"}},
)
async def publish_echomind_project(
    project_id: str,
    request: Request,
    body: PublishEchoMindProjectRequest | None = None,
) -> PublishedGraph:
    context, project, store = _load_project(
        request,
        project_id,
        require_cluster=True,
    )
    try:
        return await asyncio.to_thread(
            ProjectPublicationService(store.root, project_store=store).publish,
            project=project,
            published_by=context.user_id,
            publication_mode=(body.publication_mode if body is not None else "debug"),
            expected_draft_sha256=body.expected_draft_sha256 if body else None,
            expected_contract_sha256=body.expected_contract_sha256 if body else None,
            validation_project_id=body.validation_project_id if body else None,
            require_ready_resources=True,
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(404, detail=f"Project not found: {project_id}") from exc
    except PublicationValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Project can not be published.",
                "issues": [issue.model_dump(mode="json") for issue in exc.issues],
            },
        ) from exc
    except (
        ActiveGraphRunExistsError,
        ProjectLeaseTimeout,
        ProjectLifecycleError,
        PublicationImmutableError,
        RuntimeImplementationAdoptionError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.post(
    "/projects/{project_id}/reopen",
    response_model=ProjectInfo,
    responses={404: {"description": "Project not found"}},
)
async def reopen_echomind_project(
    project_id: str,
    request: Request,
) -> ProjectInfo:
    _, project, store = _load_project(request, project_id, require_cluster=True)
    try:
        return await asyncio.to_thread(
            ProjectPublicationService(store.root, project_store=store).reopen,
            project=project,
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(404, detail=f"Project not found: {project_id}") from exc
    except (
        ActiveGraphRunExistsError,
        ProjectLeaseTimeout,
        ProjectLifecycleError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.get(
    "/projects/{project_id}/publication",
    response_model=PublishedGraph,
    responses={404: {"description": "Active publication not found"}},
)
async def get_echomind_project_publication(
    project_id: str,
    request: Request,
) -> PublishedGraph:
    _, _, store = _load_project(request, project_id)
    try:
        return PublicationStore(store.root).load_active(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Active publication not found: {project_id}",
        ) from exc


@router.get(
    "/projects/{project_id}/publication/preflight",
    response_model=PublicationPreflight,
    responses={404: {"description": "Project not found"}},
)
async def preflight_echomind_project_publication(
    project_id: str,
    request: Request,
    publication_mode: PublicationMode = "debug",
) -> PublicationPreflight:
    _, project, _ = _load_project(request, project_id)
    return build_publication_preflight(
        project,
        publication_mode=publication_mode,
    )


@router.post(
    "/projects/{project_id}/publication/compile",
    response_model=PublishedGraphDraft,
    responses={404: {"description": "Project not found"}},
)
async def compile_echomind_project_publication(
    project_id: str,
    request: Request,
    publication_mode: PublicationMode = "debug",
) -> PublishedGraphDraft:
    _, project, _ = _load_project(request, project_id)
    try:
        return compile_working_graph(
            project,
            publication_mode=publication_mode,
        )
    except PublicationValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Project cannot be compiled for publication.",
                "issues": [issue.model_dump(mode="json") for issue in exc.issues],
            },
        ) from exc


@router.get(
    "/projects/{project_id}/runtime-slot",
    response_model=RuntimeSlot,
    responses={404: {"description": "RuntimeSlot not found"}},
)
async def get_echomind_runtime_slot(
    project_id: str,
    request: Request,
) -> RuntimeSlot:
    _, _, store = _load_project(request, project_id)
    try:
        return RuntimeSlotStore(store.root).load(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"RuntimeSlot not found: {project_id}",
        ) from exc


@router.post(
    "/projects/{project_id}/runtime-slot/ensure",
    response_model=RuntimeSlot,
)
async def ensure_echomind_runtime_slot(
    project_id: str,
    body: RuntimeSlotSetupRequest,
    request: Request,
) -> RuntimeSlot:
    context, project, store = _load_project(
        request,
        project_id,
        require_cluster=True,
    )
    published = _runtime_setup_release(store, project)
    try:
        service = await _runtime_slot_service(
            context,
            project_cluster=project.project.cluster,
            projects_root=store.root,
        )
        await _check_inference_model(body, context=context)
        setup = await _prepare_tracer_script(
            body, project_id=project_id, context=context
        )
        return await _setup_runtime_slot(
            service, project, published, setup, request.app.state.config
        )
    except RuntimeSlotEnsureError as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, detail="RuntimeSlot creation failed.") from exc


def _runtime_setup_release(store: ProjectStore, project: ProjectInfo) -> PublishedGraph:
    try:
        return ProjectPublicationService(store.root).resource_setup_release(project)
    except (
        FileNotFoundError,
        ProjectLifecycleError,
        PublicationValidationError,
        RuntimeImplementationAdoptionError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="先采用验证通过的执行方案，再搭建常驻资源。",
        ) from exc


def _resource_release_id(
    service: ProjectPublicationService,
    project: ProjectInfo,
    draft: RuntimeContractUpgradeDraft,
) -> str | None:
    if draft.issues or draft.validation_project_id is not None:
        return None
    try:
        return service.resource_setup_release(project).release_id
    except (
        FileNotFoundError,
        ProjectLifecycleError,
        PublicationValidationError,
        RuntimeImplementationAdoptionError,
    ):
        return None


@router.get("/projects/{project_id}/runtime-slot/proposal")
async def get_echomind_runtime_resource_proposal(
    project_id: str, request: Request
) -> RuntimeResourceProposal | None:
    _, project, store = _load_project(request, project_id)
    try:
        published = _runtime_setup_release(store, project)
    except HTTPException as exc:
        if exc.status_code == 409:
            return None
        raise
    return read_runtime_resource_proposal(store.root, project_id, published.release_id)


@router.post("/projects/{project_id}/runtime-slot/proposal/apply", status_code=202)
async def confirm_echomind_runtime_resource_proposal(
    project_id: str,
    body: ConfirmRuntimeResourceProposal,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    context, project, store = _load_project(request, project_id, require_cluster=True)
    published = _runtime_setup_release(store, project)
    proposal = read_runtime_resource_proposal(
        store.root, project_id, published.release_id
    )
    if proposal is None or body.expected_digest != proposal.digest:
        raise HTTPException(
            409, detail="The resource proposal changed. Review it again."
        )
    service = await _runtime_slot_service(
        context, project_cluster=project.project.cluster, projects_root=store.root
    )
    await _check_inference_model(proposal.setup, context=context)
    setup = await _prepare_tracer_script(
        proposal.setup, project_id=project_id, context=context
    )
    background_tasks.add_task(
        _run_confirmed_resource_proposal,
        service,
        project,
        published,
        setup,
        request.app.state.config,
    )
    return {"status": "accepted"}


async def _prepare_tracer_script(
    setup: RuntimeSlotSetupRequest, *, project_id: str, context: RequestContext
) -> RuntimeSlotSetupRequest:
    try:
        return await asyncio.to_thread(
            prepare_platform_tracer_setup,
            setup,
            project_id=project_id,
            storage=build_evaluation_storage(context),
        )
    except (EvaluationStorageMissingError, EvaluationStorageUnavailableError) as exc:
        raise HTTPException(
            424, detail="无法在 Storage 中准备 EchoMind 触发脚本。"
        ) from exc


async def _check_inference_model(
    setup: RuntimeSlotSetupRequest, *, context: RequestContext
) -> None:
    inference = setup.serving_inference_create
    if inference is None:
        return

    def list_model_files() -> set[str]:
        storage = build_evaluation_storage(context)
        return storage.list(to_storage_api_path(inference.model_path))

    try:
        files = await asyncio.to_thread(list_model_files)
    except EvaluationStorageMissingError:
        files = set()
    except (EvaluationStorageUnavailableError, RuntimeCredentialUnavailableError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "暂时无法读取 Storage 模型目录，请检查平台登录和 Storage 权限后重试。"
            ),
        ) from None

    if "config.json" in files:
        return
    detail = (
        "这个目录只有 LoRA adapter。请选择评测产出的合并模型目录，"
        "或先合并 LoRA 再搭建 Inference。"
        if "adapter_config.json" in files
        else "Storage 模型目录缺少 config.json。请选择包含完整模型的目录后重试。"
    )
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


async def _run_confirmed_resource_proposal(
    service: RuntimeSlotService,
    project: ProjectInfo,
    published: PublishedGraph,
    setup: RuntimeSlotSetupRequest,
    config: Config,
) -> None:
    try:
        store = ProjectStore(service.store.projects_root)
        current = store.load(
            project.project.project_id, owner_user_id=project.project.owner_user_id
        )
        current_release = _runtime_setup_release(store, current)
        if current_release.release_id != published.release_id:
            return
        await _setup_runtime_slot(service, current, published, setup, config)
    except Exception as exc:
        logger.warning(
            "Runtime resource setup ended for %s (%s)",
            project.project.project_id,
            type(exc).__name__,
        )


async def _setup_runtime_slot(
    service: RuntimeSlotService,
    project: ProjectInfo,
    published: PublishedGraph,
    setup: RuntimeSlotSetupRequest,
    config: Config,
) -> RuntimeSlot:
    project_id = project.project.project_id
    slot = await service.setup(
        project_id=project_id, release_id=published.release_id, request=setup
    )
    return await _discover_tracer_batch_access(service, slot, config)


async def _discover_tracer_batch_access(
    service: RuntimeSlotService, slot: RuntimeSlot, config: Config
) -> RuntimeSlot:
    project_id = slot.project_id
    cipher = build_tracer_admin_cipher(config)
    if (
        slot.tracer is None
        or not slot.tracer.resource_id
        or slot.batch_execution_available
    ):
        return slot
    resource_id = slot.tracer.resource_id
    try:
        if is_platform_tracer_script(
            project_id, slot.tracer.custom_runtime_script_path
        ):
            path = await asyncio.to_thread(
                service.provider.tracer_script_path, resource_id
            )
            if path == slot.tracer.custom_runtime_script_path:
                async with service.store.lease(project_id):
                    current = service.store.load(project_id)
                    if current.tracer == slot.tracer:
                        return service.store.save(
                            current.model_copy(
                                update={"batch_execution_available": True}
                            )
                        )
                    return current
            return slot
        if cipher is None:
            return slot
        key = await asyncio.to_thread(service.provider.batch_admin_key, resource_id)
        if key and await asyncio.to_thread(
            service.provider.verify_batch_admin, resource_id=resource_id, admin_key=key
        ):
            return await _record_tracer_admin(
                service.store, project_id, resource_id, cipher, key
            )
    except Exception as exc:
        logger.info(
            "Tracer batch permissions are pending for %s (%s)",
            project_id,
            type(exc).__name__,
        )
    return slot


@router.post(
    "/projects/{project_id}/runtime-slot/health",
    response_model=RuntimeSlot,
)
async def health_check_echomind_runtime_slot(
    project_id: str,
    request: Request,
) -> RuntimeSlot:
    context, project, store = _load_project(
        request,
        project_id,
        require_cluster=True,
    )
    try:
        service = await _runtime_slot_service(
            context,
            project_cluster=project.project.cluster,
            projects_root=store.root,
        )
        slot = await service.refresh_health(
            project_id=project_id,
        )
        return await _discover_tracer_batch_access(
            service, slot, request.app.state.config
        )
    except RuntimeSlotEnsureError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"RuntimeSlot not found: {project_id}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="RuntimeSlot health check failed.",
        ) from exc


@router.post(
    "/projects/{project_id}/runtime-slot/tracer-admin", response_model=RuntimeSlot
)
async def configure_echomind_tracer_admin(
    project_id: str, body: TracerAdminRequest, request: Request
) -> RuntimeSlot:
    context, project, store = _load_project(request, project_id, require_cluster=True)
    slot_store = RuntimeSlotStore(store.root)
    slot = slot_store.load_optional(project_id)
    if slot is None or slot.tracer is None or not slot.tracer.resource_id:
        raise HTTPException(
            409, detail="Bind a Tracer before configuring batch access."
        )
    cipher = build_tracer_admin_cipher(request.app.state.config)
    if cipher is None:
        raise HTTPException(503, detail="Tracer credential encryption is unavailable.")
    key = body.admin_key.get_secret_value().strip()
    if not key or len(key) > 2048 or any(char.isspace() for char in key):
        raise HTTPException(422, detail="Enter a valid Tracer admin key.")
    service = await _runtime_slot_service(
        context, project_cluster=project.project.cluster, projects_root=store.root
    )
    resource_id = slot.tracer.resource_id
    if not await asyncio.to_thread(
        service.provider.verify_batch_admin, resource_id=resource_id, admin_key=key
    ):
        raise HTTPException(
            409, detail="Tracer batch API rejected the admin key or is unavailable."
        )
    return await _record_tracer_admin(slot_store, project_id, resource_id, cipher, key)


async def _record_tracer_admin(
    slot_store: RuntimeSlotStore,
    project_id: str,
    resource_id: str,
    cipher: Cipher,
    key: str,
) -> RuntimeSlot:
    try:
        async with slot_store.lease(project_id):
            current = slot_store.load(project_id)
            if current.tracer is None or current.tracer.resource_id != resource_id:
                raise HTTPException(
                    409, detail="The Tracer changed. Refresh and retry."
                )
            save_tracer_admin_key(
                projects_root=slot_store.projects_root,
                project_id=project_id,
                resource_id=resource_id,
                cipher=cipher,
                key=key,
            )
            return slot_store.save(
                current.model_copy(update={"batch_execution_available": True})
            )
    except FileLockTimeout as exc:
        raise HTTPException(409, detail="Project is busy. Try again shortly.") from exc


@router.post(
    "/projects/{project_id}/collection-simulations",
    response_model=AcquisitionSimulation,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_echomind_collection_simulation(
    project_id: str,
    body: CreateEchoMindCollectionSimulationRequest,
    request: Request,
) -> AcquisitionSimulation:
    context, project, store = _load_project(
        request,
        project_id,
        require_cluster=True,
    )
    simulation = AcquisitionSimulationStore(store.root).create(
        project_id=project_id,
        request_id=body.request_id,
        source=body.source,
        requested_samples=body.limit,
        requested_rounds=body.rounds,
        splits=body.splits,
        auto_start_graph_run=body.auto_start_graph_run,
        model=body.model,
    )
    await _launch_collection_simulation(
        context=context,
        project=project,
        projects_root=store.root,
        simulation=simulation,
        request=request,
    )
    return simulation


@router.get(
    "/projects/{project_id}/collection-simulations",
    response_model=list[AcquisitionSimulation],
)
async def list_echomind_collection_simulations(
    project_id: str,
    request: Request,
) -> list[AcquisitionSimulation]:
    _, _, store = _load_project(request, project_id)
    return AcquisitionSimulationStore(store.root).list(project_id)


@router.get(
    "/projects/{project_id}/collection-simulations/{simulation_id}",
    response_model=AcquisitionSimulation,
)
async def get_echomind_collection_simulation(
    project_id: str,
    simulation_id: str,
    request: Request,
) -> AcquisitionSimulation:
    _, _, store = _load_project(request, project_id)
    try:
        return AcquisitionSimulationStore(store.root).load(
            project_id=project_id,
            simulation_id=simulation_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection simulation not found: {simulation_id}",
        ) from exc


@router.post(
    "/projects/{project_id}/collection-simulations/{simulation_id}/resume",
    response_model=AcquisitionSimulation,
    status_code=status.HTTP_202_ACCEPTED,
)
async def resume_echomind_collection_simulation(
    project_id: str,
    simulation_id: str,
    request: Request,
) -> AcquisitionSimulation:
    context, project, store = _load_project(
        request,
        project_id,
        require_cluster=True,
    )
    simulation_store = AcquisitionSimulationStore(store.root)
    try:
        simulation = simulation_store.load(
            project_id=project_id,
            simulation_id=simulation_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection simulation not found: {simulation_id}",
        ) from exc
    if simulation.status in {"succeeded", "failed", "canceled"}:
        return simulation
    await _launch_collection_simulation(
        context=context,
        project=project,
        projects_root=store.root,
        simulation=simulation,
        request=request,
    )
    return simulation


@router.post(
    "/projects/{project_id}/graph-runs",
    response_model=GraphRun,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_echomind_graph_run(
    project_id: str,
    body: StartEchoMindGraphRunRequest,
    request: Request,
) -> GraphRun:
    context, project, store = _load_project(
        request,
        project_id,
        require_cluster=True,
    )
    try:
        published = PublicationStore(store.root).load_active(project_id)
        await _refresh_runtime_access_for_request(request, store.root, project, context)
        return await asyncio.to_thread(
            _runtime_scheduler(
                request,
                projects_root=store.root,
                context=context,
            ).start,
            project=project,
            published=published,
            request_id=body.request_id,
            inputs=body.inputs,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project has no active publication.",
        ) from exc
    except (ActiveGraphRunExistsError, RuntimeSchedulerError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.post(
    "/projects/{project_id}/graph-runs/debug",
    response_model=GraphRun,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_echomind_debug_graph_run(
    project_id: str,
    body: StartEchoMindDebugGraphRunRequest,
    request: Request,
) -> GraphRun:
    context, project, store = _load_project(request, project_id, require_cluster=True)
    try:
        if project.project.lifecycle != "development":
            raise ProjectLifecycleError(
                "Debug GraphRuns require a development Project."
            )
        snapshot = await asyncio.to_thread(
            ProjectPublicationService(store.root, project_store=store).prepare_debug,
            project=project,
            reviewed_by=context.user_id,
            expected_contract_sha256=body.expected_contract_sha256,
            validation_project_id=body.validation_project_id,
        )
        await _refresh_runtime_access_for_request(request, store.root, project, context)
        return await asyncio.to_thread(
            _runtime_scheduler(
                request,
                projects_root=store.root,
                context=context,
            ).start,
            project=project,
            published=snapshot,
            request_id=body.request_id,
            inputs=body.inputs,
            execution_mode="debug",
        )
    except PublicationValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Project cannot start a debug GraphRun.",
                "issues": [issue.model_dump(mode="json") for issue in exc.issues],
            },
        ) from exc
    except (
        ActiveGraphRunExistsError,
        FileNotFoundError,
        ProjectLeaseTimeout,
        ProjectLifecycleError,
        PublicationImmutableError,
        RuntimeImplementationAdoptionError,
        RuntimeSchedulerError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.get(
    "/projects/{project_id}/graph-runs/{run_id}/repair/proposal",
    response_model=GraphRepairProposalView | None,
)
async def get_graph_repair_proposal(
    project_id: str, run_id: str, request: Request
) -> GraphRepairProposalView | None:
    context, project, store = _load_project(request, project_id, require_cluster=True)
    run = _repair_source_run(store, project_id, run_id)
    proposal = await _read_graph_repair_proposal(
        request, project, run, context, store.root
    )
    if proposal is None:
        return None
    return GraphRepairProposalView(
        node_id=proposal.node_id,
        reason=proposal.reason,
        content_sha256=proposal.content_sha256,
        changed_fields=sorted(
            key
            for key, value in proposal.runtime_config.model_dump().items()
            if value is not None and (key != "workflow_parameter_updates" or value)
        ),
    )


@router.post(
    "/projects/{project_id}/graph-runs/{run_id}/repair/debug",
    response_model=GraphRun,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_graph_repair_debug(
    project_id: str, run_id: str, body: StartGraphRepairDebugRequest, request: Request
) -> GraphRun:
    context, project, store = _load_project(request, project_id, require_cluster=True)
    run = _repair_source_run(store, project_id, run_id)
    request_id = f"repair:{run_id}:{body.expected_proposal_sha256}"
    graph_runs = GraphRunStore(store.root)
    existing = graph_runs.find_by_request_id(project_id, request_id)
    if existing is not None and existing.repair_of_run_id == run_id:
        return existing
    proposal = await _read_graph_repair_proposal(
        request, project, run, context, store.root
    )
    if proposal is None or proposal.content_sha256 != body.expected_proposal_sha256:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Repair proposal is missing or changed. Review the latest proposal.",
        )
    try:
        snapshot = await asyncio.to_thread(
            ProjectPublicationService(
                store.root, project_store=store
            ).prepare_repair_debug,
            project=project,
            release_id=run.release_id,
            proposal=proposal,
            reviewed_by=context.user_id,
        )
        await _refresh_runtime_access_for_request(request, store.root, project, context)
        return await asyncio.to_thread(
            _runtime_scheduler(
                request, projects_root=store.root, context=context
            ).start_repair,
            project=project,
            published=snapshot,
            repair_of_run_id=run_id,
            request_id=request_id,
            inputs=run.inputs,
        )
    except PublicationValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Repair proposal failed graph validation.",
                "issues": [issue.model_dump(mode="json") for issue in exc.issues],
            },
        ) from exc
    except (
        ProjectLifecycleError,
        PublicationImmutableError,
        RuntimeSchedulerError,
        ActiveGraphRunExistsError,
        ProjectLeaseTimeout,
        FileNotFoundError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


def _repair_source_run(store: ProjectStore, project_id: str, run_id: str) -> GraphRun:
    try:
        return GraphRunStore(store.root).load(project_id=project_id, run_id=run_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="GraphRun not found"
        ) from exc


async def _read_graph_repair_proposal(
    request: Request,
    project: ProjectInfo,
    run: GraphRun,
    context: RequestContext,
    projects_root: Path,
) -> RuntimeGraphRepairProposal | None:
    handoff = run.current_repair_handoff()
    if handoff is None and run.repair_restart is not None and run.repair_handoffs:
        handoff = run.repair_handoffs[-1]
    if handoff is None or handoff.delivered_at is None:
        return None
    source = await _read_repair_proposal_text(
        request, project, context, handoff.handoff_id, run.run_id, projects_root
    )
    if source is None:
        return None
    published = _load_graph_run_release(projects_root, run.project_id, run.run_id)
    if published.content_sha256 != run.release_digest:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The debug snapshot no longer matches the run.",
        )
    try:
        proposal = RuntimeGraphRepairProposal.model_validate_json(source)
        proposal.repaired_node(published)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=_repair_proposal_error(exc, context),
        ) from exc
    except ProjectLifecycleError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=_repair_proposal_error(exc, context),
        ) from exc
    return proposal


async def _read_repair_proposal_text(
    request: Request,
    project: ProjectInfo,
    context: RequestContext,
    handoff_id: str,
    run_id: str,
    projects_root: Path,
) -> str | None:
    submitted = await asyncio.to_thread(
        GraphRepairProposals(projects_root).load,
        project.project.project_id,
        run_id,
        handoff_id,
    )
    if submitted is not None:
        return submitted.model_dump_json()
    path = f"public_data/graph-repairs/{handoff_id}.json"
    try:
        async with asyncio.timeout(8):
            return await get_product_runtime(request).read_persisted_workspace_text(
                project.project.conversation_id, path, context, max_bytes=128 * 1024
            )
    except (OSError, ProductRuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not read the repair proposal from the workspace.",
        ) from exc


def _repair_proposal_error(
    exc: ValidationError | ProjectLifecycleError, context: RequestContext
) -> str:
    fields = str(exc)
    if isinstance(exc, ValidationError):
        fields = ", ".join(
            ".".join(map(str, item["loc"])) or "JSON"
            for item in exc.errors(include_input=False)[:4]
        )
    fields = redact_graph_failure_text(fields, context)[:300]
    return f"Agent 修复提案的字段格式不正确（{fields}）。"


@router.get(
    "/projects/{project_id}/graph-runs",
    response_model=list[GraphRun],
)
async def list_echomind_graph_runs(
    project_id: str,
    request: Request,
) -> list[GraphRun]:
    _, _, store = _load_project(request, project_id)
    return GraphRunStore(store.root).list(project_id)


@router.get(
    "/projects/{project_id}/graph-runs/{run_id}",
    response_model=EchoMindGraphRunDetail,
)
async def get_echomind_graph_run(
    project_id: str,
    run_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> EchoMindGraphRunDetail:
    context, project, store = _load_project(request, project_id)
    graph_run_store = GraphRunStore(store.root)
    try:
        run = graph_run_store.load(project_id=project_id, run_id=run_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"GraphRun not found: {run_id}",
        ) from exc
    if run.status in {"pending", "running", "waiting_external"}:
        credential_store = build_runtime_credential_store(
            store.root, request.app.state.config
        )
        if credential_store is not None:
            background_tasks.add_task(
                refresh_runtime_access, credential_store, project, context
            )
    return EchoMindGraphRunDetail(
        run=run,
        stage_runs=ProjectRunStore(store.root).list_for_graph_run(
            project_id=project_id,
            graph_run_id=run_id,
        ),
        task_runs=graph_run_store.list_task_runs(
            project_id=project_id,
            graph_run_id=run_id,
        ),
    )


def _debug_repair_node_evidence(
    node: ProjectNode, context: RequestContext
) -> JsonObject:
    evidence: JsonObject = {
        "node_id": node.id,
        "type": node.type,
        "executor": node.executor,
        "inputs": list(node.inputs),
        "outputs": list(node.outputs),
    }
    config = node.config.get(node.executor)
    if isinstance(config, dict):
        template = config.get("command_template")
        if (
            isinstance(template, str)
            and len(template) <= 4096
            and redact_graph_failure_text(template, context) == template
        ):
            evidence["command_template"] = template
    return evidence


@router.post(
    "/projects/{project_id}/graph-runs/{run_id}/repair",
    response_model=ProductEvent,
    status_code=status.HTTP_202_ACCEPTED,
)
async def request_echomind_debug_repair(
    project_id: str,
    run_id: str,
    request: Request,
    body: RequestGraphRepairRequest | None = None,
) -> ProductEvent:
    context, project, store = _load_project(request, project_id, require_cluster=True)
    conversation_id = project.project.conversation_id
    await asyncio.to_thread(
        _require_profile, request, conversation_id, ECHOMIND_PRODUCT_PROFILE
    )
    try:
        handoff = await asyncio.to_thread(
            _runtime_scheduler(
                request, projects_root=store.root, context=context
            ).get_repair_handoff,
            project_id=project_id,
            run_id=run_id,
        )
        if handoff is None:
            raise RuntimeSchedulerError("No failed Graph Debug attempt to repair.")
        published = _load_graph_run_release(store.root, project_id, run_id)
        if published.content_sha256 != handoff.release_digest:
            raise RuntimeSchedulerError("Graph Debug snapshot no longer matches.")
        stage = ProjectRunStore(store.root).load(
            project_id=project_id, run_id=handoff.stage_run_id
        )
        input_paths: list[JsonValue] = [
            path
            for path in stage.input_refs
            if path.startswith("/workspace/") and "?" not in path
        ]
        evidence: JsonObject = {
            "handoff_id": handoff.handoff_id,
            "proposal_path": f"public_data/graph-repairs/{handoff.handoff_id}.json",
            "project_id": handoff.project_id,
            "run_id": handoff.run_id,
            "release_id": handoff.release_id,
            "release_digest": handoff.release_digest,
            "failed_node_id": handoff.node_id,
            "stage_run_id": handoff.stage_run_id,
            "attempt": handoff.attempt,
            "provider_task_id": handoff.provider_task_id,
            "failure_category": handoff.failure.category,
            "failure_message": redact_graph_failure_text(
                handoff.failure.message, context
            )[:2000],
            "input_paths": input_paths[:16],
            "nodes": [
                _debug_repair_node_evidence(node, context) for node in published.nodes
            ],
        }
        if handoff.provider_task_id is not None and handoff.delivered_at is None:
            diagnostics = await asyncio.to_thread(
                read_graph_failure_diagnostics,
                project,
                handoff.provider_task_id,
                context,
                credential_store=build_runtime_credential_store(
                    store.root, request.app.state.config
                ),
            )
            if diagnostics:
                evidence["stage_log_excerpt"] = diagnostics
        notification_id = handoff.handoff_id
        if body is not None and body.revise_invalid_proposal:
            previous = await _read_repair_proposal_text(
                request, project, context, handoff.handoff_id, run_id, store.root
            )
            if previous is None or handoff.delivered_at is None:
                raise RuntimeSchedulerError("No Agent repair proposal to revise.")
            try:
                proposal = RuntimeGraphRepairProposal.model_validate_json(previous)
                proposal.repaired_node(published)
            except (ValidationError, ProjectLifecycleError) as exc:
                evidence["proposal_validation_error"] = _repair_proposal_error(
                    exc, context
                )
            else:
                raise RuntimeSchedulerError("The repair proposal is already valid.")
            digest = sha256(previous.encode()).hexdigest()
            notification_id = f"{handoff.handoff_id}:{digest}"
        await configure_session(request, conversation_id, ECHOMIND_PRODUCT_PROFILE)
        event = await get_product_runtime(request).request_echomind_debug_repair(
            conversation_id,
            handoff_id=notification_id,
            evidence=evidence,
            context=context,
        )
        await asyncio.to_thread(
            GraphRunStore(store.root).mark_repair_handoff_delivered,
            project_id=project_id,
            run_id=run_id,
            handoff_id=handoff.handoff_id,
        )
    except (FileNotFoundError, PermissionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Graph Debug repair context not found.",
        ) from exc
    except (RuntimeSchedulerError, ProjectLeaseTimeout) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return event


@router.post(
    "/projects/{project_id}/graph-runs/{run_id}/advance",
    response_model=GraphRun,
)
async def advance_echomind_graph_run(
    project_id: str,
    run_id: str,
    request: Request,
) -> GraphRun:
    context, project, store = _load_project(
        request,
        project_id,
        require_cluster=True,
    )
    published = _load_graph_run_release(store.root, project_id, run_id)
    await _refresh_runtime_access_for_request(request, store.root, project, context)
    return await asyncio.to_thread(
        _runtime_scheduler(
            request,
            projects_root=store.root,
            context=context,
        ).advance,
        project=project,
        published=published,
        run_id=run_id,
    )


@router.post(
    "/projects/{project_id}/graph-runs/{run_id}/callback",
    response_model=GraphRun,
)
async def callback_echomind_graph_run(
    project_id: str,
    run_id: str,
    body: EchoMindGraphRunCallbackRequest,
    request: Request,
) -> GraphRun:
    context, project, store = _load_project(
        request,
        project_id,
        require_cluster=True,
    )
    published = _load_graph_run_release(store.root, project_id, run_id)
    try:
        return await asyncio.to_thread(
            _runtime_scheduler(
                request,
                projects_root=store.root,
                context=context,
            ).callback,
            project=project,
            published=published,
            run_id=run_id,
            provider_run_id=body.provider_run_id,
            status=body.status,
        )
    except RuntimeSchedulerError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post(
    "/projects/{project_id}/graph-runs/{run_id}/pause",
    response_model=GraphRun,
)
async def pause_echomind_graph_run(
    project_id: str,
    run_id: str,
    request: Request,
) -> GraphRun:
    context, project, store = _load_project(
        request,
        project_id,
        require_cluster=True,
    )
    return await asyncio.to_thread(
        _runtime_scheduler(
            request,
            projects_root=store.root,
            context=context,
        ).pause,
        project=project,
        run_id=run_id,
    )


@router.post(
    "/projects/{project_id}/graph-runs/{run_id}/resume",
    response_model=GraphRun,
)
async def resume_echomind_graph_run(
    project_id: str,
    run_id: str,
    request: Request,
) -> GraphRun:
    context, project, store = _load_project(
        request,
        project_id,
        require_cluster=True,
    )
    published = _load_graph_run_release(store.root, project_id, run_id)
    await _refresh_runtime_access_for_request(request, store.root, project, context)
    return await asyncio.to_thread(
        _runtime_scheduler(
            request,
            projects_root=store.root,
            context=context,
        ).resume,
        project=project,
        published=published,
        run_id=run_id,
    )


@router.post(
    "/projects/{project_id}/graph-runs/{run_id}/retry",
    response_model=GraphRun,
)
async def retry_echomind_graph_run(
    project_id: str,
    run_id: str,
    body: RetryEchoMindGraphRunRequest,
    request: Request,
) -> GraphRun:
    context, project, store = _load_project(
        request,
        project_id,
        require_cluster=True,
    )
    published = _load_graph_run_release(store.root, project_id, run_id)
    await _refresh_runtime_access_for_request(request, store.root, project, context)
    try:
        return await asyncio.to_thread(
            _runtime_scheduler(
                request,
                projects_root=store.root,
                context=context,
            ).retry,
            project=project,
            published=published,
            run_id=run_id,
            inputs=body.inputs,
        )
    except RuntimeSchedulerError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.post(
    "/projects/{project_id}/graph-runs/{run_id}/cancel",
    response_model=GraphRun,
)
async def cancel_echomind_graph_run(
    project_id: str,
    run_id: str,
    request: Request,
) -> GraphRun:
    context, project, store = _load_project(
        request,
        project_id,
        require_cluster=True,
    )
    return await asyncio.to_thread(
        _runtime_scheduler(
            request,
            projects_root=store.root,
            context=context,
        ).cancel,
        project=project,
        run_id=run_id,
    )


@router.post(
    "/projects/{project_id}/nodes/{node_id}/start",
    response_model=StartEchoMindProjectNodeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={404: {"description": "Project or node not found"}},
)
async def start_echomind_project_node(
    project_id: str,
    node_id: str,
    request: Request,
) -> StartEchoMindProjectNodeResponse:
    _require_sandbox_credentials(request)
    context, project, _ = _load_project(request, project_id, require_cluster=True)

    node = next((item for item in project.graph.nodes if item.id == node_id), None)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project node not found: {node_id}",
        )
    if project.graph.lifecycle != "development":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Frozen Projects cannot start Agent-led stage development.",
        )
    if node.type != "dataset" or node.launch_mode != "agent":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Project node is not Agent-led Dataset development: {node_id}",
        )
    if node.config.get("capability") != "data-processing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Dataset node is not bound to the data-processing capability.",
        )

    conversation_id = project.project.conversation_id
    await asyncio.to_thread(
        _require_profile, request, conversation_id, ECHOMIND_PRODUCT_PROFILE
    )
    command = UserMessageCommand(
        command_id=uuid4().hex,
        content=(TextContent(text=_build_dataset_start_context(project, node_id)),),
    )
    await configure_session(request, conversation_id, ECHOMIND_PRODUCT_PROFILE)
    receipt = await get_product_runtime(request).submit_command(
        conversation_id,
        command,
        context,
    )
    if receipt.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project stage context could not be delivered.",
        )
    return StartEchoMindProjectNodeResponse(
        project_id=project.project.project_id,
        graph_id=project.graph.graph_id,
        node_id=node.id,
        conversation_id=conversation_id,
        cluster=project.project.cluster,
        event_id=receipt.command_id,
    )


@router.post(
    "/projects/{project_id}/nodes/{node_id}/run",
    response_model=StageRun,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"description": "Project or node not found"}},
)
async def run_echomind_project_node(
    project_id: str,
    node_id: str,
    request: Request,
) -> StageRun:
    _, project, store = _load_project(request, project_id, require_cluster=True)
    node = next((item for item in project.graph.nodes if item.id == node_id), None)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project node not found: {node_id}",
        )
    if node.launch_mode == "agent":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Agent-led Project nodes must use the start endpoint.",
        )
    try:
        get_project_node_executor(node.executor)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    try:
        return await asyncio.to_thread(
            run_project_node,
            project=project,
            node=node,
            run_store=ProjectRunStore(store.root),
            project_store=store,
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(404, detail=f"Project not found: {project_id}") from exc
    except ProjectLeaseTimeout as exc:
        raise HTTPException(409, detail=str(exc)) from exc


@router.get(
    "/projects/{project_id}/runs",
    response_model=list[StageRun],
    responses={404: {"description": "Project not found"}},
)
async def list_echomind_project_runs(
    project_id: str,
    request: Request,
) -> list[StageRun]:
    _, _, store = _load_project(request, project_id)
    return ProjectRunStore(store.root).list(project_id)


@router.get(
    "/projects/{project_id}/artifacts/{artifact_id}",
    response_model=ProjectArtifactDocument,
    responses={404: {"description": "Project or artifact not found"}},
)
async def get_echomind_project_artifact(
    project_id: str,
    artifact_id: str,
    request: Request,
) -> ProjectArtifactDocument:
    _, _, store = _load_project(request, project_id)
    try:
        return ProjectRunStore(store.root).load_artifact(
            project_id=project_id,
            artifact_id=artifact_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project artifact not found: {artifact_id}",
        ) from exc


class _EvaluationStorageReader:
    def __init__(self, storage: EvaluationStorage) -> None:
        self.storage = storage

    def read(self, path: str) -> bytes:
        return self.storage.read("/" + path.lstrip("/"))


async def _launch_collection_simulation(
    *,
    context: RequestContext,
    project: ProjectInfo,
    projects_root: Path,
    simulation: AcquisitionSimulation,
    request: Request,
) -> None:
    if simulation.status in {"succeeded", "failed", "canceled"}:
        return
    if project.project.lifecycle != "frozen":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project must be published before collection simulation.",
        )
    try:
        published = PublicationStore(projects_root).load_active(
            project.project.project_id
        )
        slot = RuntimeSlotStore(projects_root).load(project.project.project_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project runtime resources are not ready.",
        ) from exc
    if (
        published.publication_mode != "production"
        or slot.status != "ready"
        or slot.active_release_id != published.release_id
        or slot.tracer is None
        or not slot.tracer.resource_id
        or not slot.tracer.endpoint
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="RuntimeSlot Tracer is not ready for the active release.",
        )
    try:
        slot_service = await _runtime_slot_service(
            context,
            project_cluster=project.project.cluster,
            projects_root=projects_root,
        )
        tracer_secret = await asyncio.to_thread(
            slot_service.provider.secret_key,
            slot.tracer.resource_id,
        )
    except RuntimeSlotEnsureError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    if not tracer_secret:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Tracer access key is unavailable.",
        )
    tracer_admin_secret = await asyncio.to_thread(
        load_tracer_admin_key,
        projects_root=projects_root,
        project_id=project.project.project_id,
        resource_id=slot.tracer.resource_id,
        cipher=build_tracer_admin_cipher(request.app.state.config),
    )
    platform_batch_trigger = is_platform_tracer_script(
        project.project.project_id, slot.tracer.custom_runtime_script_path
    )
    if simulation.auto_start_graph_run and (
        not slot.batch_execution_available
        or (not platform_batch_trigger and not tracer_admin_secret)
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Configure Tracer batch admin access before starting a GraphRun.",
        )
    if simulation.auto_start_graph_run:
        await _refresh_runtime_access_for_request(
            request, projects_root, project, context
        )
    try:
        storage = await asyncio.to_thread(
            build_acquisition_storage,
            projects_root=projects_root,
            project_id=project.project.project_id,
            simulation_id=simulation.simulation_id,
            context=context,
            credential_store=build_runtime_credential_store(
                projects_root, request.app.state.config
            ),
        )
    except EvaluationStorageUnavailableError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    runner = AcquisitionSimulationRunner(
        projects_root=str(projects_root),
        project=project,
        published=published,
        slot=slot,
        tracer_secret=tracer_secret,
        tracer_admin_secret=tracer_admin_secret,
        platform_batch_trigger=platform_batch_trigger,
        sample_source=JsonlAcquisitionSampleSource(_EvaluationStorageReader(storage)),
        batch_materializer=TracerBatchMaterializer(storage),
        scheduler=_runtime_scheduler(
            request,
            projects_root=projects_root,
            context=context,
        ),
        poll_seconds=_positive_float_env(
            "PYROMIND_ECHOMIND_ACQUISITION_POLL_SECONDS",
            3,
        ),
        backpressure_timeout_seconds=_positive_float_env(
            "PYROMIND_ECHOMIND_ACQUISITION_BACKPRESSURE_TIMEOUT_SECONDS",
            300,
        ),
        dispatch_timeout_seconds=_positive_float_env(
            "PYROMIND_ECHOMIND_ACQUISITION_DISPATCH_TIMEOUT_SECONDS",
            3600,
        ),
    )
    manager = getattr(request.app.state, "echomind_acquisition_tasks", None)
    if not isinstance(manager, EchoMindAcquisitionTaskManager):
        manager = EchoMindAcquisitionTaskManager()
        request.app.state.echomind_acquisition_tasks = manager
    manager.start(simulation=simulation, runner=runner)


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


async def _runtime_slot_service(
    context: RequestContext,
    *,
    project_cluster: str,
    projects_root: Path,
) -> RuntimeSlotService:
    auth_token = parse_auth_token_from_cookie_header(context.cookie)
    if not auth_token and context.authorization:
        scheme, _, value = context.authorization.partition(" ")
        if scheme.lower() == "bearer":
            auth_token = value.strip() or None
    if not auth_token:
        raise RuntimeSlotEnsureError(
            "RuntimeSlot creation requires current Pyromind credentials"
        )

    route = platform_cluster_route(context.x_cluster or project_cluster)
    env, _ = platform_cluster_target(route)
    headers = {
        name: value
        for name, value in (
            ("cookie", context.cookie),
            ("authorization", context.authorization),
            ("x-cluster", route),
        )
        if value
    }
    access_key = await asyncio.to_thread(
        get_api_key,
        env,
        auth_token,
        headers,
    )
    client = get_pyromind_api_client(
        env=env,
        cluster=route,
        api_key=access_key,
        max_retries=0,
    )
    provider = EchoMindTracerProvider(client=client, cluster=route)
    return RuntimeSlotService(
        projects_root=projects_root,
        provider=provider,
    )
