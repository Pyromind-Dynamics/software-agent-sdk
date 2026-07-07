"""Definitions of the preview_dataset and upload_file_to_pyromind tools.

Both tools are MOCKED: the real platform APIs are not implemented yet, so
the executors return fixed values. Replace the executors with real HTTP
calls once the platform endpoints are available (search for "TODO(mock)").
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import Field
from rich.text import Text

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


# ---------------------------------------------------------------------------
# preview_dataset
# ---------------------------------------------------------------------------


class PreviewDatasetAction(Action):
    """Preview a dataset stored on Pyromind storage."""

    dataset_path: str = Field(
        description=(
            "Relative path of the dataset on Pyromind storage, as pasted by "
            "the user (e.g. 'datasets/my_data/' or "
            "'datasets/my_data/train.jsonl'). Directories and single files "
            "are both accepted."
        ),
    )
    max_samples: int = Field(
        default=3,
        description="Maximum number of sample rows to return (1-10).",
        ge=1,
        le=10,
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Preview dataset: ", style="bold blue")
        content.append(self.dataset_path)
        return content


class PreviewDatasetObservation(Observation):
    """Statistics and sample rows of a storage dataset."""

    dataset_path: str = Field(description="The storage path that was previewed.")
    files: list[str] = Field(
        default_factory=list,
        description="Data files found under the path (relative to it).",
    )
    num_rows: int | None = Field(
        default=None, description="Total row count across data files."
    )
    columns: list[str] = Field(
        default_factory=list, description="Top-level fields of each row."
    )
    p95_sequence_length: int | None = Field(
        default=None,
        description="P95 of per-row token length (prompt + response).",
    )
    has_vision: bool | None = Field(
        default=None, description="Whether rows contain image/video fields."
    )
    sample_rows: list[dict[str, Any]] = Field(
        default_factory=list, description="A few raw sample rows."
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Dataset preview: ", style="bold green")
        content.append(self.dataset_path)
        if self.num_rows is not None:
            content.append(f"\nrows={self.num_rows}")
        if self.columns:
            content.append(f"\ncolumns={', '.join(self.columns)}")
        return content


_PREVIEW_DATASET_DESCRIPTION = """Preview a dataset the user uploaded to Pyromind storage.

Given the storage-relative path the user pasted into the chat, this returns:
- the data files found under the path
- total row count and top-level columns/fields
- P95 sequence length (tokens) and whether the data contains images/videos
- a few raw sample rows

Call this BEFORE generating any training workflow that uses a user-provided
dataset, so you can determine the data format (SFT messages / prompt-response
/ DPO chosen-rejected / GRPO prompt-only), pick the right dataset config
builder node, and fill in field-mapping parameters from real field names
instead of guessing.
"""


# TODO(mock): fixed preview payload; replace with a real storage API call.
_MOCK_PREVIEW: dict[str, Any] = {
    "files": ["train.jsonl"],
    "num_rows": 12000,
    "columns": ["messages"],
    "p95_sequence_length": 6800,
    "has_vision": False,
    "sample_rows": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant with access to tools..."
                    ),
                },
                {"role": "user", "content": "Turn on the living room lights."},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\nThe user wants to turn on the living room "
                        "lights...\n</think>"
                    ),
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "control_light",
                                "arguments": {
                                    "room": "living room",
                                    "state": "on",
                                },
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "control_light",
                    "content": "The lights in the living room are now on.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "<think>\nTool succeeded. I can confirm to the user.\n"
                        "</think>\nDone!"
                    ),
                },
            ]
        }
    ],
}


class PreviewDatasetExecutor(
    ToolExecutor[PreviewDatasetAction, PreviewDatasetObservation]
):
    """MOCK executor: always returns the same fixed dataset preview."""

    def __call__(
        self,
        action: PreviewDatasetAction,
        conversation: BaseConversation | None = None,  # noqa: ARG002
    ) -> PreviewDatasetObservation:
        sample_rows = _MOCK_PREVIEW["sample_rows"][: action.max_samples]
        summary = {
            "dataset_path": action.dataset_path,
            "files": _MOCK_PREVIEW["files"],
            "num_rows": _MOCK_PREVIEW["num_rows"],
            "columns": _MOCK_PREVIEW["columns"],
            "p95_sequence_length": _MOCK_PREVIEW["p95_sequence_length"],
            "has_vision": _MOCK_PREVIEW["has_vision"],
            "sample_rows": sample_rows,
        }
        return PreviewDatasetObservation.from_text(
            text=json.dumps(summary, ensure_ascii=False, indent=2),
            dataset_path=action.dataset_path,
            files=_MOCK_PREVIEW["files"],
            num_rows=_MOCK_PREVIEW["num_rows"],
            columns=_MOCK_PREVIEW["columns"],
            p95_sequence_length=_MOCK_PREVIEW["p95_sequence_length"],
            has_vision=_MOCK_PREVIEW["has_vision"],
            sample_rows=sample_rows,
        )


class PreviewDatasetTool(
    ToolDefinition[PreviewDatasetAction, PreviewDatasetObservation]
):
    """Tool for previewing datasets on Pyromind storage (currently mocked)."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[ToolDefinition]:
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"PreviewDatasetTool got unknown params: {names}")
        return [
            cls(
                description=_PREVIEW_DATASET_DESCRIPTION,
                action_type=PreviewDatasetAction,
                observation_type=PreviewDatasetObservation,
                executor=PreviewDatasetExecutor(),
                annotations=ToolAnnotations(
                    title="preview_dataset",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
        ]


# ---------------------------------------------------------------------------
# upload_file_to_pyromind
# ---------------------------------------------------------------------------


class UploadFileToPyromindAction(Action):
    """Upload a local workspace file to Pyromind storage."""

    file_path: str = Field(
        description=(
            "Path of the file in the current conversation workspace to "
            "upload (e.g. 'acc.py')."
        ),
    )
    target_dir: str = Field(
        default="/workspace/script/agent",
        description=(
            "Storage directory to upload into. Defaults to "
            "/workspace/script/agent, which is where custom metric/reward "
            "scripts belong."
        ),
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Upload file to Pyromind: ", style="bold blue")
        content.append(self.file_path)
        return content


class UploadFileToPyromindObservation(Observation):
    """Result of a storage upload."""

    storage_path: str | None = Field(
        default=None,
        description=(
            "Absolute storage path of the uploaded file, usable in node "
            "parameters (e.g. /workspace/script/agent/acc.py)."
        ),
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        if self.is_error:
            content.append("Upload failed", style="bold red")
        else:
            content.append("Uploaded: ", style="bold green")
            content.append(self.storage_path or "")
        return content


_UPLOAD_FILE_DESCRIPTION = """Upload a file from the conversation workspace to Pyromind storage.

Use this when a workflow node needs a server-side file path, most commonly a
custom evaluation metric or reward script for MetricsConfigBuilderCustomNode:
write the Python file locally first, upload it with this tool, then use the
returned storage path in the node's `entry` parameter as
`<storage_path>:<function_name>` (e.g. /workspace/script/agent/acc.py:acc_func).

Returns the absolute storage path of the uploaded file.
"""


class UploadFileToPyromindExecutor(
    ToolExecutor[UploadFileToPyromindAction, UploadFileToPyromindObservation]
):
    """MOCK executor: pretends the upload succeeded and returns the path."""

    def __call__(
        self,
        action: UploadFileToPyromindAction,
        conversation: BaseConversation | None = None,  # noqa: ARG002
    ) -> UploadFileToPyromindObservation:
        # TODO(mock): actually upload the file via the platform API.
        filename = PurePosixPath(action.file_path.replace("\\", "/")).name
        if not filename:
            return UploadFileToPyromindObservation.from_text(
                text=f"Invalid file path: {action.file_path!r}",
                is_error=True,
            )
        storage_path = str(PurePosixPath(action.target_dir) / filename)
        return UploadFileToPyromindObservation.from_text(
            text=f"File uploaded to Pyromind storage: {storage_path}",
            storage_path=storage_path,
        )


class UploadFileToPyromindTool(
    ToolDefinition[UploadFileToPyromindAction, UploadFileToPyromindObservation]
):
    """Tool for uploading workspace files to Pyromind storage (currently mocked)."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[ToolDefinition]:
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(
                f"UploadFileToPyromindTool got unknown params: {names}"
            )
        return [
            cls(
                description=_UPLOAD_FILE_DESCRIPTION,
                action_type=UploadFileToPyromindAction,
                observation_type=UploadFileToPyromindObservation,
                executor=UploadFileToPyromindExecutor(),
                annotations=ToolAnnotations(
                    title="upload_file_to_pyromind",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(PreviewDatasetTool.name, PreviewDatasetTool)
register_tool(UploadFileToPyromindTool.name, UploadFileToPyromindTool)
