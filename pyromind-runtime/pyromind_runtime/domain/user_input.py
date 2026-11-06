from __future__ import annotations

from typing import Literal

from pydantic import Field, model_serializer, model_validator

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import ContentBlock
from pyromind_runtime.domain.execution_policy import CheckpointRef
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload


# Maximum UTF-8 JSON payload size for each request or answer.
# Validate before sending or persisting the data, when constructing the model.
MAX_USER_INPUT_BYTES = 64 * 1024

type UserInputOutcome = Literal["answered", "cancelled", "interrupted"]


def _encoded_size(value: str) -> int:
    return len(value.encode("utf-8"))


class UserInputOption(ContractModel):
    id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1)
    description: str | None = None


class UserInputQuestion(ContractModel):
    id: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1)
    # An empty option list means free text; labels do not identify options.
    options: tuple[UserInputOption, ...] = ()

    @model_validator(mode="after")
    def _unique_option_ids(self) -> UserInputQuestion:
        ids = [option.id for option in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError(f"option ids must be unique within question {self.id!r}")
        return self


class UserInputPresentation(ContractModel):
    """Agent-authored display only; choices and authorization live outside it."""

    title: str | None = Field(default=None, max_length=200)
    content: tuple[ContentBlock, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def _safe_payload(self) -> UserInputPresentation:
        validate_persisted_payload(self.model_dump(mode="json"))
        return self


class UserInputRequest(ContractModel):
    """The ask_user tool arguments: one batch of one to three questions."""

    presentation: UserInputPresentation | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Optional display content: title and Chat content blocks. Use text for "
            "Markdown, tables and file links; images may reference HTTPS URLs or "
            "absolute Storage paths. Organize evidence here, not in UI templates. "
            "Display content never grants authorization."
        ),
    )
    checkpoint: CheckpointRef | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    questions: tuple[UserInputQuestion, ...] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def _validate_batch(self) -> UserInputRequest:
        ids = [question.id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique within the request")
        if _encoded_size(self.model_dump_json()) > MAX_USER_INPUT_BYTES:
            raise ValueError("user input request exceeds the size limit")
        return self

    @classmethod
    def tool_json_schema(cls) -> dict[str, object]:
        """JSON Schema for the ask_user tool parameters, reused by Node."""
        schema = cls.model_json_schema()
        schema.pop("title", None)
        return schema


class UserInputAnswer(ContractModel):
    question_id: str = Field(min_length=1, max_length=128)
    option_id: str | None = Field(default=None, min_length=1, max_length=128)
    text: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _exactly_one_choice(self) -> UserInputAnswer:
        if (self.option_id is None) == (self.text is None):
            raise ValueError(
                f"answer for question {self.question_id!r} must carry exactly "
                "one of option_id or text"
            )
        if self.text is not None and not self.text.strip():
            raise ValueError(
                f"answer text for question {self.question_id!r} must contain "
                "non-whitespace content"
            )
        return self

    @model_serializer(mode="wrap")
    def _omit_unused_choice(self, handler):
        # Serialize only the populated field of this mutually exclusive pair.
        data = handler(self)
        if self.option_id is None:
            data.pop("option_id", None)
        if self.text is None:
            data.pop("text", None)
        return data


class PendingUserInput(ContractModel):
    """A request the product snapshot keeps until it receives a resolution."""

    request_id: str = Field(min_length=1)
    run_id: str | None = None
    operation_id: str | None = None
    task_run_id: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    presentation: UserInputPresentation | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Optional display content: title and Chat content blocks. Use text for "
            "Markdown, tables and file links; images may reference HTTPS URLs or "
            "absolute Storage paths. Organize evidence here, not in UI templates. "
            "Display content never grants authorization."
        ),
    )
    checkpoint: CheckpointRef | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    questions: tuple[UserInputQuestion, ...] = Field(min_length=1, max_length=3)


class UserInputResolution(ContractModel):
    """The single durable outcome recorded for one pending request."""

    request_id: str = Field(min_length=1)
    outcome: UserInputOutcome
    answers: tuple[UserInputAnswer, ...] = ()
    command_id: str | None = None
    reason: str | None = None

    @model_serializer(mode="wrap")
    def _omit_empty(self, handler):
        # Include optional fields only when populated so older clients do not see
        # empty values.
        data = handler(self)
        if not self.answers:
            data.pop("answers", None)
        if self.command_id is None:
            data.pop("command_id", None)
        if self.reason is None:
            data.pop("reason", None)
        return data
