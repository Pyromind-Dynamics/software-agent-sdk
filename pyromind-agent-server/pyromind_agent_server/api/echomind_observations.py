from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from pyromind_agent_server.api.echomind_projects import _load_project
from pyromind_agent_server.api.router import get_product_runtime
from pyromind_agent_server.dependencies import check_session_api_key
from pyromind_runtime.domain.observations import (
    ObservationOverview,
    ObservationReportDetails,
    SeriesPage,
)


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


router = APIRouter(
    prefix="/api/echomind/projects/{project_id}/observations",
    tags=["EchoMind observations"],
    dependencies=[Depends(check_session_api_key), Depends(_no_store)],
)


@router.get("/overview")
async def overview(project_id: str, request: Request) -> ObservationOverview:
    context, info, _ = _load_project(request, project_id, require_cluster=True)
    try:
        return await get_product_runtime(request).get_project_observations(
            info.project.conversation_id, info.project.project_id, context
        )
    except PermissionError as exc:
        raise HTTPException(403, "Project conversation authorization required") from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, "Project conversation is unavailable") from exc
    except ValueError as exc:
        raise HTTPException(409, "Project plan is unavailable") from exc


class ArtifactDeliveryRetry(BaseModel):
    project_node_id: str = Field(min_length=1, max_length=128)


@router.post("/retry-artifact-delivery", status_code=202)
async def retry_artifact_delivery(
    project_id: str, request: Request, payload: ArtifactDeliveryRetry
) -> dict[str, str]:
    context, info, _ = _load_project(request, project_id, require_cluster=True)
    try:
        await get_product_runtime(request).retry_project_artifact_delivery(
            info.project.conversation_id,
            info.project.project_id,
            payload.project_node_id,
            context,
        )
    except PermissionError as exc:
        raise HTTPException(403, "Storage authorization required") from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, "Project conversation is unavailable") from exc
    except ValueError as exc:
        raise HTTPException(409, "Artifact delivery cannot be retried") from exc
    return {"status": "accepted"}


@router.get("/report-details")
async def report_details(
    project_id: str,
    request: Request,
    project_node_id: Annotated[str, Query(min_length=1, max_length=128)],
    analysis_job_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    report_version_id: Annotated[
        str | None, Query(min_length=1, max_length=128)
    ] = None,
) -> ObservationReportDetails:
    context, info, _ = _load_project(request, project_id, require_cluster=True)
    try:
        return await get_product_runtime(request).get_project_observation_report(
            info.project.conversation_id,
            info.project.project_id,
            project_node_id,
            analysis_job_id,
            context,
            report_version_id=report_version_id,
        )
    except PermissionError as exc:
        raise HTTPException(403, "Observation report authorization required") from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, "Observation report is unavailable") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(409, "Observation report could not be read") from exc


@router.get("/series")
async def series(
    project_id: str,
    request: Request,
    project_node_id: Annotated[str, Query(min_length=1, max_length=128)],
    run_id: Annotated[str, Query(min_length=1, max_length=512)],
    keys: Annotated[list[str] | None, Query(max_length=20)] = None,
    cursor: Annotated[str | None, Query(max_length=2048)] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 1000,
) -> SeriesPage:
    context, info, _ = _load_project(request, project_id, require_cluster=True)
    try:
        return await get_product_runtime(request).get_project_training_series(
            info.project.conversation_id,
            info.project.project_id,
            project_node_id,
            run_id,
            tuple(keys or ()),
            cursor,
            limit,
            context,
        )
    except PermissionError as exc:
        raise HTTPException(403, "Training source authorization required") from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, "Project conversation is unavailable") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
