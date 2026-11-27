from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyromind_agent_server.echomind_storage import (
    canonical_uuid,
    project_directory,
    read_model,
    write_model,
)


AcquisitionStatus = Literal[
    "pending",
    "feeding",
    "waiting_batch",
    "graph_started",
    "succeeded",
    "paused",
    "failed",
    "canceled",
]
AcquisitionSplit = Literal["train", "val"]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class AcquisitionSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    train_uri: str = Field(min_length=1, max_length=4000)
    val_uri: str = Field(min_length=1, max_length=4000)
    image_root: str = Field(min_length=1, max_length=4000)
    image_layout: Literal["pcb_report_directory"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class AcquisitionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    sample_ids: list[str] = Field(min_length=1)
    claimed: bool = False
    input_directory: str | None = None

    @model_validator(mode="after")
    def validate_claim(self) -> AcquisitionBatch:
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("acquisition batch samples must be unique")
        if self.input_directory is not None and not self.claimed:
            raise ValueError("materialized batch must have a confirmed claim")
        return self


class AcquisitionSimulation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    simulation_id: str
    project_id: str
    request_id: str = Field(min_length=1, max_length=256)
    status: AcquisitionStatus = "pending"
    source: AcquisitionSource
    requested_samples: int = Field(gt=0)
    model: str | None = Field(default=None, max_length=256)
    completed_samples: int = Field(default=0, ge=0)
    failed_samples: int = Field(default=0, ge=0)
    requested_rounds: int = Field(gt=0)
    completed_rounds: int = Field(default=0, ge=0)
    splits: list[AcquisitionSplit] = Field(min_length=1)
    auto_start_graph_run: bool = True
    manual_start: bool = False
    tracer_resource_id: str | None = Field(default=None, max_length=128)
    batch_ids: list[str] = Field(default_factory=list)
    batches: dict[str, AcquisitionBatch] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    collecting_batch: AcquisitionBatch | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    graph_run_ids: list[str] = Field(default_factory=list)
    attempted_sample_ids: list[str] = Field(default_factory=list)
    completed_sample_ids: list[str] = Field(default_factory=list)
    failed_sample_ids: list[str] = Field(default_factory=list)
    error: str | None = Field(default=None, max_length=4000)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    finished_at: datetime | None = None

    @model_validator(mode="after")
    def validate_progress(self) -> AcquisitionSimulation:
        if len(set(self.splits)) != len(self.splits):
            raise ValueError("acquisition splits must be unique")
        if len(set(self.batch_ids)) != len(self.batch_ids):
            raise ValueError("acquisition batch IDs must be unique")
        if self.batches and set(self.batch_ids) != self.batches.keys():
            raise ValueError("acquisition batch receipts must match batch IDs")
        if len(set(self.graph_run_ids)) != len(self.graph_run_ids):
            raise ValueError("acquisition GraphRun IDs must be unique")
        for field_name in (
            "attempted_sample_ids",
            "completed_sample_ids",
            "failed_sample_ids",
        ):
            values = getattr(self, field_name)
            if len(set(values)) != len(values):
                raise ValueError(f"acquisition {field_name} must be unique")
        attempted = set(self.attempted_sample_ids)
        completed = set(self.completed_sample_ids)
        failed = set(self.failed_sample_ids)
        if not completed <= attempted:
            raise ValueError("completed samples must have been attempted")
        if not failed <= attempted:
            raise ValueError("failed samples must have been attempted")
        if completed & failed:
            raise ValueError("a sample cannot be both completed and failed")
        if self.completed_samples != len(completed):
            raise ValueError("completed_samples must match completed_sample_ids")
        if self.failed_samples != len(failed):
            raise ValueError("failed_samples must match failed_sample_ids")
        if len(attempted) > self.requested_samples:
            raise ValueError("acquisition attempts exceed requested samples")
        if self.completed_rounds > self.requested_rounds:
            raise ValueError("completed rounds exceed requested rounds")
        if self.status in {"feeding", "waiting_batch", "graph_started"}:
            if not self.tracer_resource_id:
                raise ValueError(
                    f"{self.status} acquisition requires a Tracer resource"
                )
        if self.status == "graph_started" and (
            not self.batch_ids or not self.graph_run_ids
        ):
            raise ValueError("graph_started acquisition requires batch and run IDs")
        if self.status == "succeeded":
            if (
                self.completed_samples != self.requested_samples
                or self.failed_samples
                or len(self.attempted_sample_ids) != self.requested_samples
            ):
                raise ValueError("succeeded acquisition requires all samples")
            if self.completed_rounds != self.requested_rounds:
                raise ValueError("succeeded acquisition requires all rounds")
            if self.finished_at is None:
                raise ValueError("succeeded acquisition requires finished_at")
        elif self.status in {"failed", "canceled"}:
            if self.finished_at is None:
                raise ValueError(f"{self.status} acquisition requires finished_at")
        elif self.finished_at is not None:
            raise ValueError("active acquisition must not have finished_at")
        if self.status == "failed" and not self.error:
            raise ValueError("failed acquisition requires an error")
        return self


class AcquisitionSimulationStore:
    def __init__(self, projects_root: Path | str) -> None:
        self.projects_root = Path(projects_root)

    def create(
        self,
        *,
        project_id: str,
        request_id: str,
        source: AcquisitionSource,
        requested_samples: int,
        requested_rounds: int,
        splits: list[AcquisitionSplit],
        auto_start_graph_run: bool,
        model: str | None = None,
        manual_start: bool = False,
    ) -> AcquisitionSimulation:
        existing = self.find_by_request_id(project_id, request_id)
        if existing is not None:
            return existing
        simulation = AcquisitionSimulation(
            simulation_id=str(uuid4()),
            project_id=project_id,
            request_id=request_id,
            source=source,
            requested_samples=requested_samples,
            requested_rounds=requested_rounds,
            splits=splits,
            auto_start_graph_run=auto_start_graph_run,
            manual_start=manual_start,
            model=model,
        )
        return self.save(simulation)

    def save(self, simulation: AcquisitionSimulation) -> AcquisitionSimulation:
        write_model(
            self._simulation_path(
                simulation.project_id,
                simulation.simulation_id,
            ),
            simulation,
        )
        return simulation

    def load(
        self,
        *,
        project_id: str,
        simulation_id: str,
    ) -> AcquisitionSimulation:
        path = self._simulation_path(project_id, simulation_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"acquisition simulation not found: {simulation_id}"
            )
        simulation = read_model(path, AcquisitionSimulation)
        if canonical_uuid(simulation.project_id, kind="project") != canonical_uuid(
            project_id,
            kind="project",
        ):
            raise ValueError("acquisition simulation belongs to another project")
        return simulation

    def list(self, project_id: str) -> list[AcquisitionSimulation]:
        simulations_root = project_directory(self.projects_root, project_id) / (
            "collection_simulations"
        )
        if not simulations_root.is_dir():
            return []
        simulations = [
            read_model(path, AcquisitionSimulation)
            for path in simulations_root.glob("*.json")
        ]
        return sorted(
            simulations,
            key=lambda simulation: (
                simulation.created_at,
                simulation.simulation_id,
            ),
            reverse=True,
        )

    def find_by_request_id(
        self,
        project_id: str,
        request_id: str,
    ) -> AcquisitionSimulation | None:
        return next(
            (
                simulation
                for simulation in self.list(project_id)
                if simulation.request_id == request_id
            ),
            None,
        )

    def _simulation_path(self, project_id: str, simulation_id: str) -> Path:
        canonical_simulation_id = canonical_uuid(
            simulation_id,
            kind="acquisition simulation",
        )
        return (
            project_directory(self.projects_root, project_id)
            / "collection_simulations"
            / f"{canonical_simulation_id}.json"
        )
