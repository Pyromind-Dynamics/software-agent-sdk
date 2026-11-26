"""Stop a running DataFlow pipeline task on the Pyromind platform.

When a user previews partial output and decides the run is not what they
want, the agent should stop the platform-side task before adjusting the
pipeline and resubmitting. The task is identified by ``task_id`` directly,
or resolved from a ``run_id`` / ``output_dir`` recorded by
``df_submit_pipeline`` or ``df_submit_evaluation``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast
from urllib.parse import urlsplit

import httpx
from pydantic import Field
from pyromind_sdk import PyroMindAPIError
from pyromind_sdk.client.models import TrainingTaskResponse
from rich.text import Text

from pyromind_common.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from pyromind_common.sdk.utils.deprecation import warn_deprecated
from pyromind_common.tools.data_preparation.platform_submit import (
    TASK_ASSOCIATION_DIRNAME,
    DataPreparationTaskStore,
)
from pyromind_common.tools.pyromind_dataset.definition import (
    _decode_json_response,
    _normalize_headers,
    _resolve_conversation_headers,
    _resolve_secret_headers,
)
from pyromind_common.tools.utils.conversation_dirs import conversation_state_dir
from pyromind_common.tools.utils.platform_env import (
    platform_endpoint,
    platform_headers,
    platform_portal_url,
)
from pyromind_common.tools.utils.pyromind_api_client import _parse_cookie_pairs
from pyromind_common.tools.utils.storage_paths import to_tool_storage_path
from pyromind_common.tools.workflow.task_submission import (
    PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET,
    create_workflow_api_client,
)


if TYPE_CHECKING:
    from pyromind_common.sdk.conversation.base import ToolConversation
    from pyromind_common.sdk.conversation.state import ToolState


PRE_STOP_URL = "https://pre-api-portal.pyromind.ai/std2/studio_api/api/stop_task"
PROD_STOP_URL = "https://api-portal.pyromind.ai/std2/studio_api/api/stop_task"

TOOL_DESCRIPTION = """\
Stop a running DataFlow pipeline task on the platform.

Identify the task with `task_id` (from either submission tool), or pass the
`run_id` / `output_dir` returned by either submission tool and the tool resolves
the task for you. Call this when the user previews partial output, decides
the run is not what they want, and asks to adjust the pipeline — stop the
running task first, then edit and resubmit.
"""


def _default_stop_url(env: str | None = None) -> str:
    return platform_endpoint(PROD_STOP_URL, PRE_STOP_URL, env)


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------


class DfStopTaskAction(Action):
    """Stop a running DataFlow pipeline task."""

    task_id: str | None = Field(
        default=None,
        description=(
            "Platform task id from either submission tool. Preferred when available."
        ),
    )
    run_id: str | None = Field(
        default=None,
        description="Run id from either submission tool, used to look up the task.",
    )
    output_dir: str | None = Field(
        default=None,
        description=(
            "`/workspace/...` output directory returned by either submission "
            "tool, used to look up the task."
        ),
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Stop pipeline task: ", style="bold red")
        target = self.task_id or self.run_id or self.output_dir or "<unknown>"
        content.append(target)
        return content


class DfStopTaskObservation(Observation):
    """Result of a stop-task request."""

    task_id: str | None = Field(
        default=None, description="The resolved platform task id requested to stop."
    )
    stopped: bool = Field(
        default=False, description="Whether the stop request succeeded."
    )
    status: str = Field(
        default="Failed", description="Stopped for an accepted request, or Failed."
    )
    platform_status: str | None = Field(
        default=None,
        description="Platform task status returned after the stop request.",
    )
    terminal_confirmed: bool = Field(
        default=False,
        description="Whether the platform confirmed a terminal task state.",
    )

    @property
    def visualize(self) -> Text:
        text = Text()
        style = "bold green" if self.stopped else "bold red"
        text.append(
            f"Stop request: {'accepted' if self.stopped else 'failed'}\n", style=style
        )
        text.append(self.text)
        return text


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class DfStopTaskExecutor(ToolExecutor[DfStopTaskAction, DfStopTaskObservation]):
    """Resolve the task id and request a stop through the platform SDK."""

    def __init__(
        self,
        *,
        stop_url: str | None = None,
        env: str | None = None,
        cluster: str | None = None,
        headers: dict[str, str] | None = None,
        secret_headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        task_store_dir: str | None = None,
        task_resolver: Callable[[str | None, str | None, str | None], str | None]
        | None = None,
    ) -> None:
        self._stop_url = (stop_url or _default_stop_url(env)).rstrip("/")
        endpoint = urlsplit(self._stop_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.netloc:
            raise ValueError("stop_url must be an absolute HTTP(S) URL")
        self._legacy_stop_url = stop_url is not None and self._stop_url not in {
            PROD_STOP_URL,
            PRE_STOP_URL,
            platform_endpoint(PROD_STOP_URL, PRE_STOP_URL, "pre2"),
        }
        if self._legacy_stop_url:
            warn_deprecated(
                "DfStopTaskExecutor custom stop_url",
                deprecated_in="1.29.3",
                removed_in="1.34.0",
                details="Use env and cluster to stop tasks through the platform SDK.",
            )
        self._env = env
        self._cluster = cluster
        self._headers = dict(headers or {})
        self._secret_headers = dict(secret_headers or {})
        self._timeout = timeout
        self._task_resolver = task_resolver
        self._task_store_dir = (
            Path(task_store_dir) if task_store_dir is not None else None
        )

    def __call__(
        self,
        action: DfStopTaskAction,
        conversation: ToolConversation | None = None,
    ) -> DfStopTaskObservation:
        task_id = self._resolve_task_id(action, conversation)
        if task_id is None:
            return DfStopTaskObservation.from_text(
                text=(
                    "Could not resolve a task to stop. Provide task_id, or the "
                    "run_id / output_dir returned by either submission tool."
                ),
                task_id=None,
                stopped=False,
                status="Failed",
                is_error=True,
            )

        try:
            headers = self._resolved_headers(conversation)
            result = self._stop_task(task_id, headers, conversation)
        except (
            PyroMindAPIError,
            httpx.RequestError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return DfStopTaskObservation.from_text(
                text=f"Failed to stop task {task_id}: {type(exc).__name__}: {exc}",
                task_id=task_id,
                stopped=False,
                status="Failed",
                is_error=True,
            )

        platform_status = result.status if result is not None else None
        return DfStopTaskObservation.from_text(
            text=f"Task {task_id} stop request accepted.",
            task_id=task_id,
            stopped=True,
            status="Stopped",
            platform_status=platform_status,
            terminal_confirmed=platform_status is not None
            and platform_status.strip().lower()
            in {
                "stopped",
                "cancelled",
                "canceled",
                "terminated",
                "completed",
                "succeeded",
                "success",
                "failed",
                "error",
            },
        )

    # -- Resolution ---------------------------------------------------------

    def _resolve_task_id(
        self, action: DfStopTaskAction, conversation: ToolConversation | None
    ) -> str | None:
        output_dir = (
            to_tool_storage_path(action.output_dir, field="output_dir")
            if action.output_dir and action.output_dir.strip()
            else None
        )
        if self._task_resolver is not None:
            resolved = self._task_resolver(action.task_id, action.run_id, output_dir)
            if resolved is not None:
                return resolved
        if action.task_id and action.task_id.strip():
            task_id = action.task_id.strip()
            if conversation is None:
                return task_id
            store = self._task_store(conversation)
            if store is None:
                return None
            association = store.get(task_id)
        else:
            store = self._task_store(conversation)
            if store is None:
                return None
            association = None
            if action.run_id and action.run_id.strip():
                association = store.get_by_run_id(action.run_id.strip())
            if association is None and output_dir is not None:
                association = store.get_by_output_dir(output_dir)
        if association is None:
            return None
        if conversation is not None and str(association.conversation_id) != str(
            conversation.id
        ):
            return None
        return association.task_id

    def _task_store(
        self, conversation: ToolConversation | None
    ) -> DataPreparationTaskStore | None:
        if self._task_store_dir is not None:
            return DataPreparationTaskStore(self._task_store_dir)
        if conversation is None:
            return None
        state_dir = conversation_state_dir(conversation, TASK_ASSOCIATION_DIRNAME)
        if state_dir is None:
            return None
        return DataPreparationTaskStore(state_dir)

    # -- SDK ----------------------------------------------------------------

    def _resolved_headers(
        self, conversation: ToolConversation | None
    ) -> dict[str, str]:
        headers = {"accept": "*/*", "content-type": "application/json", **self._headers}
        headers.update(_resolve_conversation_headers(conversation))
        headers.update(_resolve_secret_headers(conversation, self._secret_headers))
        return headers

    def _stop_task(
        self,
        task_id: str,
        headers: dict[str, str],
        conversation: ToolConversation | None,
    ) -> TrainingTaskResponse | None:
        if self._legacy_stop_url:
            self._stop_legacy_task(task_id, headers)
            return None
        normalized = {name.lower(): value for name, value in headers.items()}
        auth_token = normalized.get("auth_token") or _parse_cookie_pairs(
            normalized.get("cookie", "")
        ).get("auth_token")
        if conversation is not None:
            state = cast("ToolState", conversation.state)
            auth_token = (
                state.secret_registry.get_secret_value(
                    PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET
                )
                or auth_token
            )
        endpoint = urlsplit(platform_portal_url(self._stop_url, headers))
        env = self._env or {
            "pre-api-portal.pyromind.ai": "pre",
            "pre2-api-portal.pyromind.ai": "pre2",
        }.get(endpoint.hostname or "", "prod")
        with create_workflow_api_client(
            env=env,
            cluster=self._cluster,
            auth_token=auth_token,
            headers=headers,
            timeout=int(self._timeout),
        ) as client:
            return client.studio.stop(task_id)

    def _stop_legacy_task(self, task_id: str, headers: dict[str, str]) -> None:
        response = httpx.post(
            platform_portal_url(self._stop_url, headers),
            headers=platform_headers(headers),
            json={"task_id": int(task_id) if task_id.isdigit() else task_id},
            timeout=self._timeout,
        )
        payload = _decode_json_response(response, "Pyromind stop_task API")
        if isinstance(payload, str):
            raise ValueError(payload)
        code = payload.get("code")
        if payload.get("success") is False or (
            isinstance(code, int) and code not in (0, 200)
        ):
            raise ValueError(
                str(payload.get("message") or "stop_task reported failure")
            )


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------


class DfStopTaskTool(ToolDefinition[DfStopTaskAction, DfStopTaskObservation]):
    """Tool definition for stopping a DataFlow pipeline task."""

    @classmethod
    def create(
        cls,
        conv_state: ToolState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        stop_url_value = params.pop("stop_url", None)
        stop_url = str(stop_url_value) if stop_url_value is not None else None
        env = params.pop("env", None)
        cluster = params.pop("cluster", None)
        headers = _normalize_headers(params.pop("headers", None))
        secret_headers = _normalize_headers(
            params.pop("storage_secret_headers", params.pop("secret_headers", None))
        )
        timeout = float(params.pop("timeout", 30.0))
        task_resolver = params.pop("task_resolver", None)
        task_store_dir_value = params.pop("task_store_dir", None)
        task_store_dir = (
            str(task_store_dir_value) if task_store_dir_value is not None else None
        )
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"DfStopTaskTool got unknown params: {names}")
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=DfStopTaskAction,
                observation_type=DfStopTaskObservation,
                executor=DfStopTaskExecutor(
                    stop_url=stop_url,
                    env=env,
                    cluster=cluster,
                    headers=headers,
                    secret_headers=secret_headers,
                    timeout=timeout,
                    task_store_dir=task_store_dir,
                    task_resolver=task_resolver,
                ),
                annotations=ToolAnnotations(
                    title="df_stop_task",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
        ]


register_tool("df_stop_task", DfStopTaskTool)
