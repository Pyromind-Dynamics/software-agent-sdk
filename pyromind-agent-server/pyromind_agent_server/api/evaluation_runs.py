"""Same-origin evaluation report and annotation API for one run.

Route family (mounted under the product conversations prefix):

    GET  /api/v2/pyromind/conversations/{cid}/evaluation-runs/{rid}/report
    GET  /api/v2/pyromind/conversations/{cid}/evaluation-runs/{rid}/annotations
    PUT  /api/v2/pyromind/conversations/{cid}/evaluation-runs/{rid}/annotations
    GET  /api/v2/pyromind/conversations/{cid}/evaluation-runs/{rid}/exports/{name}

Storage authorizes access using the current request's credentials. This
route family does not authenticate a local user or enforce user ownership.
The run must still resolve inside its conversation. The server holds no
credentials of its own; an optional user label only attributes annotations.

The ``report`` endpoint is the only sanctioned way to open a new report: it
serves the Storage HTML same-origin and injects the online marker that
switches the embedded report library into autosave mode. Storage-signed
URLs serve the raw file without the marker and the page stays read-only.

The runtime owns the shared annotation lock registry. Cross-instance mutual
exclusion requires conversation affinity or an external single-writer guarantee.

Status codes (the runtime translates store/Storage failures into typed
domain exceptions; this router maps them):

- 403: cross-origin write without a trusted origin.
- 404: confirmed absence only — unknown conversation,
  malformed run id, unknown or foreign run, never-initialized annotations,
  a Storage object the backend reports as absent, an unknown export name.
  Resolution failures share one uniform body ("evaluation run not found"):
  the specific resolution failure is logged server-side.
- 409: ``base_revision`` behind the committed revision.
- 422: annotation record validation failure or replay-content mismatch.
- 500/503: sanitized availability failures (Storage transport errors,
  corrupt Product Store or commit state, unexpected payload shapes). The
  cause is logged server-side and never reflected in the response body.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from functools import partial
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from pyromind_runtime.application.evaluation_annotations import (
    AnnotationConflictError,
    AnnotationNotInitializedError,
    AnnotationReplayMismatchError,
    AnnotationStateFault,
    AnnotationValidationError,
    EvaluationAnnotationError,
)
from pyromind_runtime.application.evaluation_cases import (
    EvaluationCaseError,
    EvaluationCaseRequestError,
    EvaluationMediaNotFoundError,
)
from pyromind_runtime.domain.errors import (
    EvaluationRunNotFoundError,
    EvaluationRunUnavailableError,
)
from pyromind_runtime.ports.storage import (
    EvaluationStorage,
    EvaluationStorageMissingError,
    EvaluationStorageUnavailableError,
)
from starlette.concurrency import run_in_threadpool

from pyromind_agent_server.api.router import get_product_runtime
from pyromind_agent_server.auth_context import evaluation_request_context
from pyromind_agent_server.evaluation_storage import build_evaluation_storage


logger = logging.getLogger(__name__)

EXPORT_NAMES = frozenset(
    {
        "evaluation_annotations.json",
        "evaluation_annotations.jsonl",
        "labeled_evaluation_results.jsonl",
    }
)
# The single-file report needs inline script/style; CSP is a
# blast-radius limiter after a hypothetical injection, not a permission
# boundary (Storage enforces access to the underlying files).
REPORT_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src 'self' data: http: https:; connect-src 'self'; form-action 'none'; "
    "base-uri 'none'; object-src 'none'"
)


class EvaluationRunHttpError(HTTPException):
    pass


class SaveAnnotationsBody(BaseModel):
    save_id: str = Field(min_length=1)
    base_revision: int = Field(ge=0)
    records: list[dict[str, Any]]


class SaveAnnotationChangesBody(BaseModel):
    save_id: str = Field(min_length=1)
    base_revision: int = Field(ge=0)
    changes: list[dict[str, Any]] = Field(min_length=1)


async def get_evaluation_storage(
    conversation_id: str, request: Request
) -> EvaluationStorage:
    context = await _execute(
        partial(
            get_product_runtime(request).evaluation_storage_context,
            conversation_id,
            evaluation_request_context(request),
        )
    )
    return build_evaluation_storage(context)


async def _execute[T](operation: Callable[[], T]) -> T:
    """Run one runtime operation and map only its typed failure vocabulary.

    404 is reserved for confirmed absence (resolution failures and
    explicitly-missing Storage objects) and its body is uniform: the
    specific reason is logged, never returned. Availability failures become
    sanitized 5xx responses with the cause logged server-side; annotation
    protocol errors stay typed so each endpoint can apply its own mapping.
    """
    try:
        return await run_in_threadpool(operation)
    except (EvaluationRunNotFoundError, EvaluationStorageMissingError) as exc:
        logger.info("evaluation run request rejected as not found: %s", exc)
        raise EvaluationRunHttpError(
            status.HTTP_404_NOT_FOUND,
            detail="evaluation run not found",
        ) from exc
    except EvaluationStorageUnavailableError as exc:
        logger.error("evaluation storage unavailable: %s", exc)
        raise EvaluationRunHttpError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="evaluation storage is temporarily unavailable",
        ) from exc
    except EvaluationRunUnavailableError as exc:
        logger.error("evaluation run state unavailable: %s", exc)
        raise EvaluationRunHttpError(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="evaluation run state is currently unavailable",
        ) from exc


def _save_http_error(exc: Exception) -> EvaluationRunHttpError:
    if isinstance(exc, AnnotationConflictError):
        return EvaluationRunHttpError(status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, AnnotationValidationError):
        return EvaluationRunHttpError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(exc), "errors": exc.errors},
        )
    if isinstance(exc, AnnotationReplayMismatchError):
        return EvaluationRunHttpError(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    if isinstance(exc, AnnotationNotInitializedError):
        return EvaluationRunHttpError(status.HTTP_404_NOT_FOUND, detail=str(exc))
    logger.error("evaluation annotation save failed: %s", exc)
    return EvaluationRunHttpError(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="saving annotations failed; nothing was committed",
    )


def _reject_cross_origin_write(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is None:
        return
    # Trust boundary, most to least explicit:
    # 1. PYROMIND_REPORT_ALLOWED_ORIGINS: deployment-declared public origins
    #    (the reverse-proxy-safe path — client-supplied forwarded headers are
    #    never trusted).
    # 2. PYROMIND_TRUST_PROXY_HEADERS=1: the entry proxy overwrites/clears
    #    X-Forwarded-*, so its values define the request origin.
    # 3. Default: compare against the request's own scheme/host.
    allowed = os.getenv("PYROMIND_REPORT_ALLOWED_ORIGINS", "")
    if allowed.strip():
        allowed_origins = {
            item.strip().lower() for item in allowed.split(",") if item.strip()
        }
        if origin not in allowed_origins:
            raise EvaluationRunHttpError(
                status.HTTP_403_FORBIDDEN,
                detail=f"cross-origin write rejected: origin {origin!r}",
            )
        return
    if os.getenv("PYROMIND_TRUST_PROXY_HEADERS", "") == "1":
        scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
        host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    else:
        scheme = request.url.scheme
        host = request.headers.get("host")
    scheme = (scheme or "").split(",")[0].strip().lower()
    expected = f"{scheme}://{(host or '').strip().lower()}"
    if origin != expected:
        raise EvaluationRunHttpError(
            status.HTTP_403_FORBIDDEN,
            detail=f"cross-origin write rejected: origin {origin!r}",
        )


def create_evaluation_runs_router() -> APIRouter:
    router = APIRouter(
        prefix="/api/v2/pyromind/conversations",
        tags=["Pyromind Evaluation Runs"],
    )

    @router.get("/{conversation_id}/review-tasks/{run_id}")
    async def get_review_task(
        conversation_id: str,
        run_id: str,
        request: Request,
        storage: Annotated[EvaluationStorage, Depends(get_evaluation_storage)],
    ) -> dict[str, Any]:
        return await _execute(
            partial(
                get_product_runtime(request).read_review_task,
                conversation_id,
                run_id,
                evaluation_request_context(request),
                storage,
            )
        )

    @router.get("/{conversation_id}/review-tasks/{run_id}/annotations")
    @router.get(
        "/{conversation_id}/evaluation-runs/{run_id}/annotations",
    )
    async def get_annotations(
        conversation_id: str,
        run_id: str,
        request: Request,
        storage: Annotated[EvaluationStorage, Depends(get_evaluation_storage)],
    ) -> dict[str, Any]:
        try:
            result = await _execute(
                partial(
                    get_product_runtime(request).load_evaluation_annotations,
                    conversation_id,
                    run_id,
                    evaluation_request_context(request),
                    storage,
                ),
            )
        except AnnotationNotInitializedError as exc:
            raise EvaluationRunHttpError(
                status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc
        except AnnotationStateFault as exc:
            raise EvaluationRunHttpError(
                status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc
        except EvaluationAnnotationError as exc:
            logger.error("loading annotations failed for run %s: %s", run_id, exc)
            raise EvaluationRunHttpError(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="loading annotations failed",
            ) from exc
        return result.model_dump(mode="json")

    @router.get("/{conversation_id}/review-tasks/{run_id}/annotations/summary")
    @router.get(
        "/{conversation_id}/evaluation-runs/{run_id}/annotations/summary",
    )
    async def get_annotation_summary(
        conversation_id: str,
        run_id: str,
        request: Request,
        storage: Annotated[EvaluationStorage, Depends(get_evaluation_storage)],
    ) -> dict[str, Any]:
        try:
            result = await _execute(
                partial(
                    get_product_runtime(request).load_evaluation_annotation_summary,
                    conversation_id,
                    run_id,
                    evaluation_request_context(request),
                    storage,
                )
            )
        except AnnotationNotInitializedError as exc:
            raise EvaluationRunHttpError(
                status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc
        except EvaluationAnnotationError as exc:
            logger.error(
                "loading annotation summary failed for run %s: %s", run_id, exc
            )
            raise EvaluationRunHttpError(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="loading annotation summary failed",
            ) from exc
        return result.model_dump(mode="json")

    @router.put("/{conversation_id}/review-tasks/{run_id}/annotations")
    @router.put(
        "/{conversation_id}/evaluation-runs/{run_id}/annotations",
    )
    async def put_annotations(
        conversation_id: str,
        run_id: str,
        request: Request,
        body: SaveAnnotationsBody,
        storage: Annotated[EvaluationStorage, Depends(get_evaluation_storage)],
    ) -> dict[str, Any]:
        _reject_cross_origin_write(request)
        try:
            result = await _execute(
                partial(
                    get_product_runtime(request).save_evaluation_annotations,
                    conversation_id,
                    run_id,
                    evaluation_request_context(request),
                    storage,
                    save_id=body.save_id,
                    base_revision=body.base_revision,
                    records=body.records,
                    reviewer_id=evaluation_request_context(request).user_id,
                ),
            )
        except EvaluationAnnotationError as exc:
            raise _save_http_error(exc) from exc
        return result.model_dump(mode="json")

    @router.patch("/{conversation_id}/review-tasks/{run_id}/annotations")
    @router.patch(
        "/{conversation_id}/evaluation-runs/{run_id}/annotations",
    )
    async def patch_annotations(
        conversation_id: str,
        run_id: str,
        request: Request,
        body: SaveAnnotationChangesBody,
        storage: Annotated[EvaluationStorage, Depends(get_evaluation_storage)],
    ) -> dict[str, Any]:
        _reject_cross_origin_write(request)
        try:
            result = await _execute(
                partial(
                    get_product_runtime(request).save_evaluation_annotation_changes,
                    conversation_id,
                    run_id,
                    evaluation_request_context(request),
                    storage,
                    save_id=body.save_id,
                    base_revision=body.base_revision,
                    changes=body.changes,
                    reviewer_id=evaluation_request_context(request).user_id,
                ),
            )
        except EvaluationAnnotationError as exc:
            raise _save_http_error(exc) from exc
        return result.model_dump(mode="json")

    @router.get("/{conversation_id}/review-tasks/{run_id}/cases")
    @router.get(
        "/{conversation_id}/evaluation-runs/{run_id}/cases",
    )
    async def get_cases(
        conversation_id: str,
        run_id: str,
        request: Request,
        storage: Annotated[EvaluationStorage, Depends(get_evaluation_storage)],
        cursor: str | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        query: Annotated[str, Query(max_length=256)] = "",
        review_status: Literal["all", "reviewed", "in_progress", "unreviewed"] = "all",
    ) -> dict[str, Any]:
        try:
            page = await _execute(
                partial(
                    get_product_runtime(request).list_evaluation_cases,
                    conversation_id,
                    run_id,
                    evaluation_request_context(request),
                    storage,
                    cursor=cursor,
                    limit=limit,
                    query=query,
                    review_status=review_status,
                )
            )
        except EvaluationCaseRequestError as exc:
            raise EvaluationRunHttpError(
                status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        except EvaluationCaseError as exc:
            logger.error("loading evaluation cases failed for run %s: %s", run_id, exc)
            raise EvaluationRunHttpError(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="loading evaluation cases failed",
            ) from exc
        return {
            "items": page.items,
            "next_cursor": page.next_cursor,
            "total": page.total,
            "filtered_total": page.filtered_total,
            "revision": page.revision,
        }

    @router.get("/{conversation_id}/review-tasks/{run_id}/media/{media_id}")
    @router.get(
        "/{conversation_id}/evaluation-runs/{run_id}/media/{media_id}",
    )
    async def get_media(
        conversation_id: str,
        run_id: str,
        media_id: str,
        request: Request,
        storage: Annotated[EvaluationStorage, Depends(get_evaluation_storage)],
    ) -> Response:
        try:
            media = await _execute(
                partial(
                    get_product_runtime(request).read_evaluation_media,
                    conversation_id,
                    run_id,
                    evaluation_request_context(request),
                    storage,
                    media_id,
                )
            )
        except EvaluationMediaNotFoundError as exc:
            raise EvaluationRunHttpError(
                status.HTTP_404_NOT_FOUND, detail="evaluation media not found"
            ) from exc
        except EvaluationCaseError as exc:
            logger.error("loading evaluation media failed for run %s: %s", run_id, exc)
            raise EvaluationRunHttpError(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="loading evaluation media failed",
            ) from exc
        return Response(
            content=media.payload,
            media_type=media.mime_type,
            headers={"Cache-Control": "private, max-age=86400, immutable"},
        )

    @router.get("/{conversation_id}/review-tasks/{run_id}/report")
    @router.get(
        "/{conversation_id}/evaluation-runs/{run_id}/report",
        response_class=HTMLResponse,
    )
    async def get_report(
        conversation_id: str,
        run_id: str,
        request: Request,
        storage: Annotated[EvaluationStorage, Depends(get_evaluation_storage)],
    ) -> Response:
        # _execute owns the failure mapping: confirmed-absent report → 404,
        # every storage failure → sanitized 5xx (never an interpolated error).
        html = await _execute(
            partial(
                get_product_runtime(request).read_evaluation_report,
                conversation_id,
                run_id,
                evaluation_request_context(request),
                storage,
            ),
        )
        paged_cases = await _execute(
            partial(
                get_product_runtime(request).has_evaluation_case_products,
                conversation_id,
                run_id,
                evaluation_request_context(request),
                storage,
            )
        )
        api_config = {"annotations": "annotations", "exports": "exports/"}
        if paged_cases:
            api_config.update(
                {
                    "annotation_summary": "annotations/summary",
                    "cases": "cases",
                    "media": "media/",
                }
            )
        marker = (
            "<script>window.__EVAL_REPORT_API__ = "
            + json.dumps(api_config, separators=(",", ":"))
            + ";</script>"
        )
        document = html.decode("utf-8", errors="replace")
        if "<head>" in document:
            document = document.replace("<head>", f"<head>{marker}", 1)
        else:
            document = marker + document
        return HTMLResponse(
            content=document,
            headers={"Content-Security-Policy": REPORT_CSP},
        )

    @router.get("/{conversation_id}/review-tasks/{run_id}/exports/{filename}")
    @router.get(
        "/{conversation_id}/evaluation-runs/{run_id}/exports/{filename}",
    )
    async def get_export(
        conversation_id: str,
        run_id: str,
        filename: str,
        request: Request,
        storage: Annotated[EvaluationStorage, Depends(get_evaluation_storage)],
    ) -> Response:
        if filename not in EXPORT_NAMES:
            raise EvaluationRunHttpError(
                status.HTTP_404_NOT_FOUND, detail=f"unknown export {filename!r}"
            )
        try:
            payload = await _execute(
                partial(
                    get_product_runtime(request).export_evaluation_annotations,
                    conversation_id,
                    run_id,
                    evaluation_request_context(request),
                    storage,
                    filename,
                ),
            )
        except AnnotationNotInitializedError as exc:
            raise EvaluationRunHttpError(
                status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc
        except EvaluationAnnotationError as exc:
            logger.error("exporting annotations failed for run %s: %s", run_id, exc)
            raise EvaluationRunHttpError(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="exporting annotations failed",
            ) from exc
        media_type = (
            "application/json" if filename.endswith(".json") else "application/x-ndjson"
        )
        return Response(
            content=payload,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return router
