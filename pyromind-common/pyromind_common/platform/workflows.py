"""Prepare workflow definitions and submit them with an authenticated SDK client."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from threading import get_ident
from typing import Any

import httpx
from filelock import FileLock
from pydantic import BaseModel, Field, ValidationError
from pyromind_sdk import PyroMindAPIClient
from pyromind_sdk.client.base import PyroMindAPIError
from pyromind_sdk.client.models import (
    TrainingTaskCreateRequest,
    TrainingTaskCreateResponse,
)
from requests import RequestException
from requests.adapters import HTTPAdapter

from pyromind_common.platform.credentials import materialize_runtime_environment
from pyromind_common.platform.scene_core import (
    prepare_debug_workflow as prepare_debug_workflow,
)
from pyromind_common.platform.workflow_paths import (
    validate_persistent_output_paths,
)


WORKFLOW_ATTEMPT_STATE_KEY = "pyromind_workflow_attempts"


class _SubmissionLogFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.thread = get_ident()

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread != self.thread


@contextmanager
def sdk_request_logging() -> Iterator[None]:
    # SDK errors can include server-echoed request bodies containing worker keys.
    logger = logging.getLogger("pyromind_sdk")
    log_filter = _SubmissionLogFilter()
    logger.addFilter(log_filter)
    try:
        yield
    finally:
        logger.removeFilter(log_filter)


class WorkflowPreparationError(ValueError):
    def __init__(self, stage: str, details: dict[str, Any]) -> None:
        self.stage = stage
        self.details = details
        super().__init__(f"Workflow {stage} validation failed: {details}")


class WorkflowSubmissionUncertain(RuntimeError):
    """The create response did not establish whether the platform accepted it."""


class DebugBudgetPersistenceError(RuntimeError):
    def __init__(self, response: TrainingTaskCreateResponse) -> None:
        self.response = response
        super().__init__(
            f"Platform accepted debug task {response.task_id}, but saving its "
            "submission budget failed. Keep the task ID; do not submit it again."
        )


class WorkflowBatchSubmissionError(RuntimeError):
    def __init__(
        self, accepted: list[TrainingTaskCreateResponse], error: Exception
    ) -> None:
        self.accepted = tuple(accepted)
        self.error = error
        super().__init__(
            f"Batch stopped after {len(accepted)} accepted tasks; "
            "retain their task IDs and reconcile the failed submission"
        )


class _ValidationResult(BaseModel):
    valid: bool
    workflow: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    available_resources: list[dict[str, Any]] = Field(default_factory=list)


def _validated_response(response: httpx.Response, stage: str) -> _ValidationResult:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise WorkflowPreparationError(stage, {"error": "invalid validator response"})
    result = _ValidationResult.model_validate(payload.get("data"))
    if not result.valid or result.errors:
        raise WorkflowPreparationError(stage, result.model_dump(exclude_none=True))
    return result


def prepare_workflow(
    validator: httpx.Client, *, dsl: str, name: str = "workflow"
) -> dict[str, Any]:
    """Use a caller-authenticated client rooted at the platform workflows API.

    This prepares the server's actual graph without creating a platform task.
    The caller supplies validation credentials separately from the SDK client.
    """
    if not dsl.strip() or not name.strip():
        raise ValueError("Workflow DSL and name must not be empty")
    result = _validated_response(
        validator.post("dsl/validate", json={"name": name, "dsl": dsl}), "dsl"
    )
    workflow = result.workflow
    if (
        workflow is None
        or not isinstance(workflow.get("nodes"), list)
        or not isinstance(workflow.get("edges"), list)
    ):
        raise WorkflowPreparationError("dsl", {"error": "validated graph is missing"})
    validate_persistent_output_paths(workflow["nodes"])
    validate_resources(validator, workflow)
    return workflow


def validate_resources(validator: httpx.Client, workflow: dict[str, Any]) -> None:
    """Check the graph that will actually be submitted, without reserving resources."""
    _validated_response(
        validator.post("resources/validate", json={"workflow": workflow}), "resources"
    )


def submit_workflow(
    client: PyroMindAPIClient,
    workflow: dict[str, Any],
    *,
    out_id: str,
    name: str | None = None,
) -> TrainingTaskCreateResponse:
    """Call SDK create once; association and Project updates remain caller-owned."""
    if not out_id.strip():
        raise ValueError("out_id must come from the current execution context")
    workflow_name = name or workflow.get("name")
    if not isinstance(workflow_name, str) or not workflow_name.strip():
        raise ValueError("Workflow name must not be empty")
    adapter = client.studio.session.get_adapter(client.studio.base_url)
    if not isinstance(adapter, HTTPAdapter) or adapter.max_retries.total != 0:
        raise ValueError("Construct the SDK client with max_retries=0 for submission")
    prepared = materialize_runtime_environment(workflow)
    request = TrainingTaskCreateRequest(
        name=workflow_name, workflow=prepared, out_id=out_id
    )
    try:
        with sdk_request_logging():
            response = client.studio.create(request)
    except PyroMindAPIError as exc:
        if exc.status_code is not None and 400 <= exc.status_code < 500:
            raise PyroMindAPIError(
                "Platform rejected the submission; inspect validation and permissions",
                status_code=exc.status_code,
            ) from None
        raise WorkflowSubmissionUncertain(
            "Submission result is unconfirmed; check the platform or callback "
            "before submitting again"
        ) from None
    except (RequestException, ValidationError, ValueError):
        raise WorkflowSubmissionUncertain(
            "Submission result is unconfirmed; check the platform or callback "
            "before submitting again"
        ) from None
    if not response.task_id:
        raise WorkflowSubmissionUncertain("Platform returned no task ID")
    return response


def submit_workflow_batch(
    client: PyroMindAPIClient,
    workflows: Iterable[dict[str, Any]],
    *,
    out_id: str,
) -> list[TrainingTaskCreateResponse]:
    """Stop at the first failed shard and preserve every accepted platform ID."""
    accepted = []
    try:
        for workflow in workflows:
            accepted.append(submit_workflow(client, workflow, out_id=out_id))
    except Exception as exc:
        raise WorkflowBatchSubmissionError(accepted, exc) from exc
    return accepted


@dataclass
class DebugBudget:
    """Round state supplied and persisted by the caller, independent of a Tool."""

    accepted: int = 0
    maximum: int = 10
    state_path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.maximum < 1 or self.accepted < 0:
            raise ValueError("Debug budget must have nonnegative usage and a limit")

    @classmethod
    def load(cls, path: Path, *, maximum: int = 10) -> DebugBudget:
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock"):
            state = _debug_state(path)
        return cls(
            accepted=state.get(WORKFLOW_ATTEMPT_STATE_KEY, 0),
            maximum=maximum,
            state_path=path,
        )

    def save(self) -> None:
        if self.state_path is None:
            raise ValueError("Debug budget has no state file")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.state_path) + ".lock"):
            self._save()

    def _save(self) -> None:
        assert self.state_path is not None
        state = _debug_state(self.state_path)
        state[WORKFLOW_ATTEMPT_STATE_KEY] = self.accepted
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.state_path.parent, delete=False
            ) as destination:
                temporary = Path(destination.name)
                json.dump(state, destination)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, self.state_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _debug_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("Debug state must be a JSON object")
    attempts = state.get(WORKFLOW_ATTEMPT_STATE_KEY, 0)
    if type(attempts) is not int or attempts < 0:
        raise ValueError("Debug submission count must be a nonnegative integer")
    return state


def submit_debug_workflow(
    client: PyroMindAPIClient,
    workflow: dict[str, Any],
    *,
    out_id: str,
    budget: DebugBudget,
    validator: httpx.Client | None = None,
    validated: bool = False,
    name: str | None = None,
) -> TrainingTaskCreateResponse:
    """Submit a fixed test graph and consume budget only after an accepted task."""
    if budget.state_path is not None:
        budget.state_path.parent.mkdir(parents=True, exist_ok=True)
    lock = (
        FileLock(str(budget.state_path) + ".lock")
        if budget.state_path is not None
        else nullcontext()
    )
    with lock:
        if budget.state_path is not None:
            budget.accepted = _debug_state(budget.state_path).get(
                WORKFLOW_ATTEMPT_STATE_KEY, 0
            )
        if budget.accepted >= budget.maximum:
            raise ValueError("Debug submission limit reached for the current round")
        prepared = prepare_debug_workflow(workflow)
        if validator is not None:
            validate_resources(validator, prepared)
        elif not validated or prepared != workflow:
            raise ValueError("Pass the already-validated fixed test graph")
        response = submit_workflow(client, prepared, out_id=out_id, name=name)
        budget.accepted += 1
        if budget.state_path is not None:
            try:
                budget._save()
            except (OSError, ValueError) as exc:
                raise DebugBudgetPersistenceError(response) from exc
        return response
