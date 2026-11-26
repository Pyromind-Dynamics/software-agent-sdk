from __future__ import annotations

from collections.abc import Awaitable
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel
from pyromind_sdk.client.base import PyroMindAPIError

from pyromind_agent_server.api.router import context_for_profile, get_product_runtime
from pyromind_agent_server.dependencies import check_session_api_key
from pyromind_agent_server.echomind_projects import ProjectClusterMismatchError
from pyromind_agent_server.echomind_storage import ProjectLeaseTimeout
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.application.evaluation_cases import (
    EvaluationCaseError,
    EvaluationCaseRequestError,
    EvaluationMediaNotFoundError,
)
from pyromind_runtime.domain.batch_schedule import (
    BatchScheduleConfig,
    BatchScheduleState,
)
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.echomind_deployments import (
    DeploymentConflictError,
    DeploymentRecord,
    DeploymentRequest,
)
from pyromind_runtime.domain.echomind_model_reports import (
    ModelEvaluationRecord,
    ModelReportCatalog,
)
from pyromind_runtime.domain.project_storage import (
    ProjectStorageBinding,
    StorageProxyConfig,
)
from pyromind_runtime.ports.storage import EvaluationStorageUnavailableError


router = APIRouter(
    prefix="/api/echomind/projects/{project_id}",
    tags=["Project Flywheel"],
    dependencies=[Depends(check_session_api_key)],
)


class ReportDocument(BaseModel):
    html: str


async def _execute[T](operation: Awaitable[T]) -> T:
    try:
        return await operation
    except (
        ProjectClusterMismatchError,
        ProjectLeaseTimeout,
        BlockingIOError,
        DeploymentConflictError,
    ):
        raise HTTPException(
            409, "Project is busy or belongs to a different cluster"
        ) from None
    except (
        PermissionError,
        LookupError,
        FileNotFoundError,
        EvaluationMediaNotFoundError,
    ):
        raise HTTPException(404, "Project resource not found") from None
    except (ValueError, EvaluationCaseRequestError):
        raise HTTPException(
            422, "The selected report or configuration is invalid"
        ) from None
    except (EvaluationStorageUnavailableError, httpx.HTTPError, PyroMindAPIError):
        raise HTTPException(
            503,
            "Storage or platform access unavailable; check the Project Storage "
            "binding and platform authorization",
        ) from None
    except (EvaluationCaseError, RuntimeError):
        raise HTTPException(
            409, "Project operation is not ready; check its current status"
        ) from None


@router.get("/model-reports", response_model=ModelReportCatalog)
async def reports(project_id: str, request: Request):
    return await _execute(
        get_product_runtime(request).project_flywheel_reports(
            project_id, context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
        )
    )


@router.get("/storage-binding", response_model=ProjectStorageBinding | None)
async def storage_binding(project_id: str, request: Request):
    return await _execute(
        get_product_runtime(request).project_flywheel_storage_binding(
            project_id, context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
        )
    )


@router.put("/storage-binding", response_model=ProjectStorageBinding)
async def bind_storage(project_id: str, body: StorageProxyConfig, request: Request):
    return await _execute(
        get_product_runtime(request).project_flywheel_bind_storage(
            project_id, body, context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
        )
    )


@router.post("/storage-binding/refresh", response_model=ProjectStorageBinding)
async def refresh_storage_binding(project_id: str, request: Request):
    return await _execute(
        get_product_runtime(request).project_flywheel_refresh_storage_binding(
            project_id, context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
        )
    )


@router.post(
    "/model-reports/{candidate_id}/refresh", response_model=ModelEvaluationRecord
)
async def refresh_report(project_id: str, candidate_id: str, request: Request):
    return await _execute(
        get_product_runtime(request).project_flywheel_refresh_report(
            project_id,
            candidate_id,
            context_for_profile(request, ECHOMIND_PRODUCT_PROFILE),
        )
    )


@router.get("/model-reports/{candidate_id}/document", response_model=ReportDocument)
async def report_document(project_id: str, candidate_id: str, request: Request):
    html = await _execute(
        get_product_runtime(request).project_flywheel_report_document(
            project_id,
            candidate_id,
            context_for_profile(request, ECHOMIND_PRODUCT_PROFILE),
        )
    )
    return ReportDocument(html=html)


@router.get("/model-reports/{candidate_id}/cases")
async def report_cases(
    project_id: str,
    candidate_id: str,
    request: Request,
    cursor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    query: Annotated[str, Query(max_length=256)] = "",
) -> JsonObject:
    return await _execute(
        get_product_runtime(request).project_flywheel_report_cases(
            project_id,
            candidate_id,
            context_for_profile(request, ECHOMIND_PRODUCT_PROFILE),
            cursor,
            limit,
            query,
        )
    )


@router.get("/model-reports/{candidate_id}/media/{media_id}")
async def report_media(
    project_id: str, candidate_id: str, media_id: str, request: Request
):
    payload, mime = await _execute(
        get_product_runtime(request).project_flywheel_report_media(
            project_id,
            candidate_id,
            media_id,
            context_for_profile(request, ECHOMIND_PRODUCT_PROFILE),
        )
    )
    return Response(
        content=payload,
        media_type=mime,
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "sandbox; default-src 'none'",
        },
    )


@router.get("/batch-schedule", response_model=BatchScheduleState)
async def schedule(project_id: str, request: Request):
    return await _execute(
        get_product_runtime(request).project_flywheel_schedule(
            project_id, context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
        )
    )


@router.put("/batch-schedule", response_model=BatchScheduleState)
async def configure_schedule(
    project_id: str, body: BatchScheduleConfig, request: Request
):
    return await _execute(
        get_product_runtime(request).project_flywheel_configure_schedule(
            project_id, body, context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
        )
    )


@router.get("/deployments", response_model=list[DeploymentRecord])
async def deployments(project_id: str, request: Request):
    return await _execute(
        get_product_runtime(request).project_flywheel_deployments(
            project_id, context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
        )
    )


@router.post("/deployments", response_model=DeploymentRecord, status_code=202)
async def deploy(project_id: str, body: DeploymentRequest, request: Request):
    return await _execute(
        get_product_runtime(request).project_flywheel_deploy(
            project_id, body, context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
        )
    )


@router.post("/deployments/{deployment_id}/advance", response_model=DeploymentRecord)
async def advance_deployment(project_id: str, deployment_id: str, request: Request):
    return await _execute(
        get_product_runtime(request).project_flywheel_advance_deployment(
            project_id,
            deployment_id,
            context_for_profile(request, ECHOMIND_PRODUCT_PROFILE),
        )
    )
