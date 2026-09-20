"""Definition of the Label Studio project management tool."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    register_tool,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


DEFAULT_LABEL_STUDIO_BASE_URL = "https://label-studio.pyromind.ai"
LABEL_STUDIO_TOKEN_SECRET = "LABEL_STUDIO_TOKEN"


class LabelStudioProjectAction(Action):
    """Input for Label Studio project lifecycle operations."""

    operation: Literal[
        "create", "get", "update_config", "status", "export", "refresh_media"
    ] = Field(description="The Label Studio project operation to perform.")
    dataset_path: str | None = Field(
        default=None,
        description=(
            "User storage path. Required for operation='create'. A directory of "
            "sample directories for the directory adapters, or one JSON Lines "
            "object for adapter='jsonl' -- a pipeline's own output file is "
            "imported as written, without being reshaped first."
        ),
    )
    label_config_path: str | None = Field(
        default=None,
        description=(
            "Workspace-relative label_config.xml path. Required for "
            "operation='create' and 'update_config'."
        ),
    )
    field_map_path: str | None = Field(
        default=None,
        description=(
            "Optional workspace-relative JSON path declaring how meta fields and "
            "image files bind to the label config's controls. Omit to use the "
            "adapter's built-in bindings; declare one to rename a control, bind "
            "an extra or optional image, or pin a coordinate unit. Only for "
            "operation='create'."
        ),
    )
    project_ref: str | None = Field(
        default=None,
        description=(
            "PyroMind project reference returned by a previous create or get."
        ),
    )
    expected_config_version: int | None = Field(
        default=None,
        description=(
            "Current config_version for operation='update_config'; get the "
            "project first and pass the returned value."
        ),
        ge=1,
    )
    adapter: str = Field(
        default="avi_train",
        description=(
            "Dataset layout used during project creation: 'avi_train' reads "
            "meta_vlm.json (quality/findings) from each sample directory; "
            "'aoi_export' reads meta.json (whole-sample vlm_verdict/note) from "
            "each sample directory; 'jsonl' reads one task per line of a JSON "
            "Lines file, each row carrying its own metadata and image paths."
        ),
    )
    idempotency_key: str | None = Field(
        default=None,
        description=(
            "Optional caller key folded into the deterministic project_ref. "
            "Pass a distinct value to create a separate project over the same "
            "dataset and label config; omit it to resume an existing one."
        ),
    )
    output_path: str | None = Field(
        default=None,
        description=(
            "User storage directory for operation='export'. Defaults to the "
            "project artifact directory."
        ),
    )


class LabelStudioProjectObservation(Observation):
    """Result of a Label Studio project operation."""

    operation: str = Field(description="The operation that produced this result.")
    project_ref: str = Field(default="", description="PyroMind project reference.")
    project_id: int | None = Field(default=None, description="Label Studio project ID.")
    status: str = Field(
        default="",
        description="Persisted project status, or empty for an error result.",
    )
    config_version: int | None = Field(
        default=None, description="Current label-config version."
    )
    task_count: int | None = Field(
        default=None,
        description="Task count from persisted state or Label Studio.",
    )
    imported_count: int | None = Field(
        default=None, description="Number of tasks imported so far."
    )
    next_batch: int | None = Field(
        default=None, description="One-based next import batch."
    )
    last_error: str | None = Field(
        default=None, description="Last persisted project error."
    )
    annotation_count: int | None = Field(
        default=None, description="Number of exported annotated samples."
    )
    manifest_path: str | None = Field(
        default=None, description="User storage path of the task manifest."
    )
    open_url: str | None = Field(
        default=None,
        description=(
            "Portal SSO URL that signs the browser in to Label Studio and lands "
            "on the project. This is the canonical link to give the user. It also "
            "recovers from an expired Label Studio session by returning through "
            "the platform login."
        ),
    )
    project_url: str | None = Field(
        default=None,
        description=(
            "Direct Label Studio project URL. It is also safe to give or save: "
            "an anonymous or expired-session page load is redirected through "
            "portal SSO and returns to this project. open_url remains the "
            "preferred entry point."
        ),
    )
    export_path: str | None = Field(
        default=None, description="User storage path of exported annotations."
    )


TOOL_DESCRIPTION = """Create, inspect, update, or export a Label Studio annotation project.

Use operation='create' to import a user-storage dataset after preview_dataset and
after generating a validated label_config.xml in the workspace. The dataset is a
directory of sample directories, or one JSON Lines file when a pipeline writes
its own dataset rows; either way it is converted deterministically, imported in
batches, and returns a PyroMind project_ref, Label Studio project_id,
manifest_path, open_url, and project_url. open_url is the preferred link to give
the user; project_url is the same project's own Label Studio address. Both
recover from an expired Label Studio session by returning through portal SSO.

Pre-annotations are written through bindings: each adapter has built-in ones, and
an optional field_map_path JSON replaces any of them (control names, image slots,
region sources, coordinate units). What the converter can write is limited, and
this list is the whole of it: a whole-sample Choices verdict; rectangles plus
per-region text, when a sample carries coordinates and a category; and, for
aoi_export, a whole-sample TextArea note. It does not write brushes, polygons, or
keypoints, and it cannot give different samples different control sets. When a
request needs something outside that list, say which capability is missing rather
than probing schema after schema -- every probe re-uploads the media and rebuilds
the project, and a feature the converter does not have will not start working.
The label-studio skill carries the full capability matrix and a runnable example
per adapter.

Use operation='get' to load a project, 'status' to inspect import and task counts,
'update_config' to safely update label XML, and 'export' to convert Label Studio
annotations back to PyroMind samples in user storage.

Use operation='refresh_media' to re-sign the image URLs stored inside task data.
Label Studio keeps whichever URL the import produced and never renews it, but the
portal serves that URL for as long as the account it names can still act, so the
images do not go stale on their own. Refreshing keeps task IDs, annotations, and
predictions, and is safe to run again.

Do not pass user IDs, API tokens, or credentials. Do not call Label Studio REST
APIs with terminal or HTTP tools; this tool owns server-side auth and format
conversion.
"""  # noqa: E501


def _default_storage_base_url() -> str:
    from openhands.tools.pyromind_dataset.definition import _default_storage_base_url

    return _default_storage_base_url()


class LabelStudioProjectTool(
    ToolDefinition[LabelStudioProjectAction, LabelStudioProjectObservation]
):
    """Tool definition for Label Studio project operations."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[ToolDefinition]:
        from openhands.tools.label_studio.executor import LabelStudioProjectExecutor

        ls_base_url = str(
            params.pop(
                "ls_base_url",
                os.getenv("LABEL_STUDIO_BASE_URL") or DEFAULT_LABEL_STUDIO_BASE_URL,
            )
        )
        ls_token_secret = str(params.pop("ls_token_secret", LABEL_STUDIO_TOKEN_SECRET))
        storage_base_url = str(
            params.pop("storage_base_url", None) or _default_storage_base_url()
        )
        # The portal is the browser-facing half of the integration: it owns the
        # SSO entry point and the media route that task data points at.
        portal_base_url = str(
            params.pop("portal_base_url", None)
            or params.pop("sso_base_url", None)
            or os.getenv("LABEL_STUDIO_PORTAL_BASE_URL")
            or os.getenv("LABEL_STUDIO_SSO_BASE_URL", "")
        )
        # Which cluster's storage service signs media downloads. Sent to the
        # portal, which pins it inside the signed media URL. Never defaulted:
        # Storage is replicated per cluster.
        cluster = str(params.pop("cluster", None) or "")
        headers = params.pop("headers", None)
        secret_headers = params.pop("secret_headers", None)
        batch_size = int(params.pop("batch_size", 500))
        timeout = int(params.pop("timeout", 60))

        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"LabelStudioProjectTool got unknown params: {names}")
        if not ls_base_url.strip():
            raise ValueError("ls_base_url must be a non-empty URL.")
        if not ls_token_secret.strip():
            raise ValueError("ls_token_secret must be a non-empty secret name.")
        if not storage_base_url.strip():
            raise ValueError("storage_base_url must be a non-empty URL.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0.")

        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=LabelStudioProjectAction,
                observation_type=LabelStudioProjectObservation,
                executor=LabelStudioProjectExecutor(
                    ls_base_url=ls_base_url.rstrip("/"),
                    ls_token_secret=ls_token_secret,
                    storage_base_url=storage_base_url.rstrip("/"),
                    headers=headers,
                    secret_headers=secret_headers,
                    portal_base_url=portal_base_url.rstrip("/"),
                    cluster=cluster.strip(),
                    batch_size=batch_size,
                    timeout=timeout,
                ),
                annotations=ToolAnnotations(
                    title="label_studio_project",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(LabelStudioProjectTool.name, LabelStudioProjectTool)
