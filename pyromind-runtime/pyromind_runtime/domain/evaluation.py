"""Evaluation annotation save-protocol domain models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from pyromind_runtime.domain.base import AwareDatetime, ContractModel


class AnnotationSaveResult(ContractModel):
    """Outcome of one logical save, replayable via its save_id."""

    revision: int = Field(ge=0)
    reviewed: int = Field(ge=0)
    in_progress: int = Field(ge=0)
    unreviewed: int = Field(ge=0)
    overall_status: Literal["completed", "completed_with_open_items", "in_progress"]
    superseded_by: int | None = Field(
        default=None,
        description="Set on replay when the replayed revision is no longer current.",
    )


class SaveHistoryEntry(ContractModel):
    """One replayable save bound to the exact request content it produced."""

    save_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    content_hash: str = Field(min_length=1)
    result: AnnotationSaveResult


class EvaluationCommitState(ContractModel):
    """The single commit point of the annotation save protocol.

    Stored in the conversation Product Store; Storage only ever holds
    immutable ``*.r{N}`` objects that this state references.
    """

    run_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    annotations_object: str = Field(min_length=1)
    labeled_object: str = Field(min_length=1)
    history: list[SaveHistoryEntry] = Field(default_factory=list, max_length=5)
    updated_at: AwareDatetime


class AnnotationLoadResult(ContractModel):
    """Current committed annotation state as returned by ``load``."""

    revision: int = Field(ge=0)
    records: list[dict[str, Any]]
    reviewed: int = Field(ge=0)
    in_progress: int = Field(ge=0)
    unreviewed: int = Field(ge=0)
    overall_status: Literal["completed", "completed_with_open_items", "in_progress"]


class AnnotationSummaryResult(ContractModel):
    """Lightweight current annotation state without per-case records."""

    revision: int = Field(ge=0)
    reviewed: int = Field(ge=0)
    in_progress: int = Field(ge=0)
    unreviewed: int = Field(ge=0)
    overall_status: Literal["completed", "completed_with_open_items", "in_progress"]
