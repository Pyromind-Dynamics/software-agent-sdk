"""Manual collection demo for Projects with an explicitly prepared sample roster."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from filelock import AsyncFileLock, Timeout
from pydantic import BaseModel, Field, field_validator

from pyromind_agent_server.api.echomind_projects import (
    _launch_collection_simulation,
    _load_project,
    _runtime_slot_service,
)
from pyromind_agent_server.dependencies import check_session_api_key
from pyromind_agent_server.echomind_acquisition import (
    AcquisitionSimulation,
    AcquisitionSimulationStore,
)
from pyromind_agent_server.echomind_acquisition_runtime import (
    HttpTracerAcquisitionTransport,
    dataset_labels_batch,
    fixed_batch_validation,
)
from pyromind_agent_server.echomind_demo_collection import (
    demo_expected_source,
    demo_source_directory,
    make_inference_labeler,
    prepare_demo_source,
)
from pyromind_agent_server.echomind_failure_evidence import redact_graph_failure_text
from pyromind_agent_server.echomind_graph_runs import GraphRunStore
from pyromind_agent_server.echomind_projects import ProjectInfo
from pyromind_agent_server.echomind_publication import PublicationStore
from pyromind_agent_server.echomind_runtime_access import build_runtime_credential_store
from pyromind_agent_server.echomind_runtime_slots import RuntimeSlotStore
from pyromind_agent_server.echomind_sandbox_storage import (
    SandboxEvaluationStorage,
    build_bound_evaluation_storage,
)
from pyromind_agent_server.echomind_storage import ProjectLease, project_directory
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.storage_paths import normalize_workspace_path


router = APIRouter(
    prefix="/api/echomind/projects/{project_id}",
    tags=["EchoMind Collection Demo"],
    dependencies=[Depends(check_session_api_key)],
)


class DemoSelection(BaseModel):
    request_id: UUID
    image_directory: str = Field(min_length=1, max_length=4000)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)

    @field_validator("image_directory")
    @classmethod
    def normalize_directory(cls, value: str) -> str:
        return normalize_workspace_path(value.strip().rstrip("/"))


def _reserve(
    root: Path, project: ProjectInfo, selection: DemoSelection
) -> tuple[AcquisitionSimulation, AcquisitionSimulation]:
    pid = project.project.project_id
    simulations = AcquisitionSimulationStore(root)
    prefix = f"demo:{selection.request_id}:"
    request_id = f"{prefix}{selection.offset}"
    with ProjectLease(root, pid).acquire():
        history = simulations.list(pid)
        seed = next((item for item in history if item.manual_start), None)
        if seed is None:
            raise HTTPException(404, "当前 Project 未准备演示采集。")
        source = demo_expected_source(project, request_id, selection.image_directory)
        previous = next((s for s in history if s.request_id.startswith(prefix)), None)
        if previous is not None:
            if (
                previous.request_id != request_id
                or previous.source != source
                or previous.requested_samples != selection.limit
            ):
                raise HTTPException(409, "这个请求已经用于另一组采集参数。")
            return previous, seed

        slot = RuntimeSlotStore(root).load(pid)
        if (
            project.project.lifecycle != "frozen"
            or not project.project.active_release_id
            or slot.active_release_id != project.project.active_release_id
            or slot.status != "ready"
            or not slot.batch_execution_available
            or slot.tracer is None
        ):
            raise HTTPException(409, "请先发布 Project 并确认 Tracer 已就绪。")
        if selection.limit != slot.tracer.buffer_config.batch_size:
            raise HTTPException(422, "单轮采集条数必须等于当前 Tracer 的封批阈值。")
        if GraphRunStore(root).active(pid) or any(
            (not item.manual_start or item.status != "pending")
            and item.status not in {"succeeded", "failed", "canceled"}
            for item in history
        ):
            raise HTTPException(409, "这个 Project 已有采集或训练在进行。")
        published = PublicationStore(root).load_active(pid)
        return simulations.create(
            project_id=pid,
            request_id=request_id,
            source=source,
            requested_samples=selection.limit,
            requested_rounds=1,
            splits=["train"] if fixed_batch_validation(published) else ["train", "val"],
            auto_start_graph_run=True,
        ), seed


async def _prepare_and_start(
    request: Request,
    context: RequestContext,
    project: ProjectInfo,
    root: Path,
    simulation: AcquisitionSimulation,
    seed: AcquisitionSimulation,
    selection: DemoSelection,
) -> None:
    pid = project.project.project_id
    store = AcquisitionSimulationStore(root)
    lock_path = project_directory(root, pid) / f".demo-{simulation.simulation_id}.lock"
    try:
        async with AsyncFileLock(lock_path, timeout=0):
            simulation = store.load(
                project_id=pid, simulation_id=simulation.simulation_id
            )
            if simulation.status != "pending":
                return
            try:
                storage = await asyncio.to_thread(
                    build_bound_evaluation_storage,
                    projects_root=root,
                    project_id=pid,
                    context=context,
                    credential_store=build_runtime_credential_store(
                        root, request.app.state.config
                    ),
                    write_prefixes=(
                        demo_source_directory(project, simulation.request_id),
                    ),
                )
                if not isinstance(storage, SandboxEvaluationStorage):
                    raise ValueError("此演示需要绑定当前 Project 的 Storage 沙箱。")
                published = PublicationStore(root).load_active(pid)
                if published.release_id != project.project.active_release_id:
                    raise ValueError("采集准备期间已发布方案发生变化，请重新选择轮次。")
                model = None
                infer = None
                if not dataset_labels_batch(published):
                    slot = RuntimeSlotStore(root).load(pid)
                    inference = slot.serving_inference
                    if inference is None or not inference.resource_id:
                        raise ValueError("当前 Project 的 Inference 尚未就绪。")
                    service = await _runtime_slot_service(
                        context,
                        project_cluster=project.project.cluster,
                        projects_root=root,
                    )
                    binding, key = await asyncio.to_thread(
                        service.provider.serving_binding, inference.resource_id
                    )
                    endpoint = binding.endpoint.rstrip("/").removesuffix("/v1")
                    models = await asyncio.to_thread(
                        HttpTracerAcquisitionTransport().models,
                        endpoint=endpoint,
                        access_key=key,
                    )
                    if not models:
                        raise ValueError("Inference 未返回可用模型。")
                    model = models[0]
                    infer = make_inference_labeler(endpoint, key, model)
                source = await asyncio.to_thread(
                    prepare_demo_source,
                    project=project,
                    storage=storage,
                    seed_source=seed.source,
                    raw_directory=selection.image_directory,
                    offset=selection.offset,
                    limit=selection.limit,
                    request_id=simulation.request_id,
                    model=model,
                    infer=infer,
                    fixed_validation=fixed_batch_validation(published),
                )
                if source != simulation.source:
                    raise ValueError("采集样本清单与所选范围不一致。")
                await _launch_collection_simulation(
                    context=context,
                    project=project,
                    projects_root=root,
                    simulation=simulation,
                    request=request,
                )
            except Exception as exc:
                message = exc.detail if isinstance(exc, HTTPException) else str(exc)
                safe = redact_graph_failure_text(str(message), context)[:600]
                now = datetime.now(UTC)
                store.save(
                    simulation.model_copy(
                        update={
                            "status": "failed",
                            "error": safe or "准备演示采集失败",
                            "finished_at": now,
                            "updated_at": now,
                        }
                    )
                )
    except Timeout:
        return


@router.post("/demo-collection", response_model=AcquisitionSimulation, status_code=202)
async def start_demo_collection(
    project_id: str,
    selection: DemoSelection,
    request: Request,
    background: BackgroundTasks,
) -> AcquisitionSimulation:
    context, project, store = _load_project(request, project_id, require_cluster=True)
    simulation, seed = await asyncio.to_thread(_reserve, store.root, project, selection)
    if simulation.status == "pending":
        background.add_task(
            _prepare_and_start,
            request,
            context,
            project,
            store.root,
            simulation,
            seed,
            selection,
        )
    return simulation
