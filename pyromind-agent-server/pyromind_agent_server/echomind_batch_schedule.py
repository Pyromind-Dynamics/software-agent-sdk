"""Adapt platform batch controls and archived GraphRuns to continuous scheduling."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Literal
from uuid import UUID

import httpx
import pyarrow.parquet as parquet
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from pyromind_agent_server.echomind_acquisition_runtime import (
    TracerBatchMaterializer,
    platform_run_inputs,
)
from pyromind_agent_server.echomind_graph_runs import GraphRun, GraphRunStore
from pyromind_agent_server.echomind_projects import ProjectInfo
from pyromind_agent_server.echomind_publication import PublishedGraph
from pyromind_agent_server.echomind_runtime_scheduler import RuntimeGraphScheduler
from pyromind_agent_server.echomind_runtime_slots import RuntimeSlot
from pyromind_agent_server.echomind_storage import canonical_sha256
from pyromind_runtime.domain.batch_schedule import (
    BatchReceipt,
    BatchRunState,
    BatchScheduleConfig,
    ReadyProjectBatch,
)
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.storage_paths import normalize_workspace_path
from pyromind_runtime.ports.batch_schedule import BatchScheduleBlocked
from pyromind_runtime.ports.storage import (
    EvaluationStorage,
    EvaluationStorageMissingError,
)


class ConfirmedBatchSample(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    id: str = Field(min_length=1)
    task_id: str | None = Field(default=None, min_length=1)
    confirmed: bool = Field(strict=True)
    user_prompt: str = Field(min_length=1)
    images: list[str] = Field(min_length=1)
    gt: str | dict[str, JsonValue]

    @field_validator("confirmed")
    @classmethod
    def require_confirmation(cls, value: bool) -> bool:
        if not value:
            raise ValueError("Ground truth must be confirmed before batch training")
        return value

    @field_validator("images")
    @classmethod
    def storage_images(cls, paths: list[str]) -> list[str]:
        return [normalize_workspace_path(path) for path in paths]

    def dataset_row(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json", exclude={"confirmed", "task_id"})


class PlatformBatchScheduleRuntime:
    def __init__(
        self,
        *,
        projects_root: Path,
        project: ProjectInfo,
        published: PublishedGraph,
        slot: RuntimeSlot,
        scheduler: RuntimeGraphScheduler,
        storage: EvaluationStorage | None,
        admin_key: str | None,
        report_archived: Callable[[str], bool],
        http_client: httpx.Client | None = None,
    ) -> None:
        if UUID(project.project.project_id) != UUID(published.project_id) or UUID(
            project.project.project_id
        ) != UUID(slot.project_id):
            raise ValueError("Batch runtime resources belong to another Project")
        self.project = project
        self.published = published
        self.slot = slot
        self.scheduler = scheduler
        self.storage = storage
        self.admin_key = admin_key
        self.report_archived = report_archived
        self.http_client = http_client
        self.runs = GraphRunStore(projects_root)

    @property
    def tracer_resource_id(self) -> str:
        return self.slot.tracer.resource_id or "" if self.slot.tracer else ""

    @property
    def release_id(self) -> str:
        return self.published.release_id

    def preflight(self, config: BatchScheduleConfig) -> None:
        if (
            self.project.project.lifecycle != "frozen"
            or self.published.publication_mode != "production"
            or self.slot.status != "ready"
            or self.slot.active_release_id != self.release_id
            or not self.tracer_resource_id
        ):
            raise BatchScheduleBlocked(
                "Publish the Project and prepare its runtime resources first."
            )
        if self.runs.active(self.project.project.project_id) is not None:
            raise BatchScheduleBlocked(
                "A Project run is already active; waiting for it to finish."
            )
        if not self.admin_key:
            raise BatchScheduleBlocked(
                "Platform Tracer batch control needs its own admin authorization; "
                "the inference proxy key does not grant claim or release access."
            )
        if self.storage is None:
            raise BatchScheduleBlocked(
                "Authorize unattended Storage access for the Project owner first."
            )
        inventory = self._request("GET", "/admin/batches")
        if inventory.get("batch_mode") is not True or not isinstance(
            inventory.get("batches"), list
        ):
            raise BatchScheduleBlocked(
                "This Tracer does not expose a supported batch lifecycle."
            )
        self._samples(config)
        platform_run_inputs(self.published, self.slot)

    def list_ready(self) -> list[ReadyProjectBatch]:
        inventory = self._request("GET", "/admin/batches")
        batches = inventory.get("batches")
        if not isinstance(batches, list):
            raise BatchScheduleBlocked("Tracer returned an invalid batch inventory.")
        ready = []
        for batch in batches:
            if not isinstance(batch, dict) or batch.get("state") != "ready":
                continue
            index = batch.get("index")
            if type(index) is not int or index < 0:
                raise BatchScheduleBlocked("Tracer returned an invalid batch identity.")
            ready.append(ReadyProjectBatch(batch_id=f"batch-{index:08d}", index=index))
        return ready

    def claim(self, receipt: BatchReceipt) -> None:
        response = self._request("POST", f"/admin/batches/{receipt.batch_index}/claim")
        if response.get("ok") is not True or response.get("claimed") is not True:
            raise BatchScheduleBlocked(
                "The batch could not be claimed; it may have another consumer."
            )

    def materialize(
        self, receipt: BatchReceipt, config: BatchScheduleConfig
    ) -> JsonObject:
        storage = self._storage()
        training, validation = self._samples(config)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.tracer_resource_id):
            raise BatchScheduleBlocked("Invalid Tracer resource identity.")
        prefix = f"/EchoMind/{self.tracer_resource_id}/data/spans/{receipt.batch_index}"
        shards = sorted(
            name
            for name in storage.list(prefix)
            if re.fullmatch(r"spans_[A-Za-z0-9_-]+\.parquet", name)
        )
        if not shards:
            raise BatchScheduleBlocked("The sealed batch has no Parquet artifacts.")
        selected: dict[str, ConfirmedBatchSample] = {}
        for shard in shards:
            table = parquet.read_table(
                BytesIO(storage.read(f"{prefix}/{shard}")),
                columns=["name", "rollout_id", "attributes"],
            )
            for row in table.to_pylist():
                if row["name"] != "llm.completion":
                    continue
                task_id = row["rollout_id"]
                sample = training.get(task_id)
                if sample is None:
                    raise BatchScheduleBlocked(
                        "A collected task has no confirmed ground truth "
                        "in this Project."
                    )
                attributes = json.loads(row["attributes"])
                if not isinstance(attributes, dict):
                    raise BatchScheduleBlocked(
                        "The collected task attributes are malformed."
                    )
                for name in ("gt", "task.gt"):
                    if attributes.get(name) not in (None, "") and _ground_truth(
                        attributes[name]
                    ) != _ground_truth(sample.gt):
                        raise BatchScheduleBlocked(
                            "Collected ground truth disagrees with "
                            "the confirmed manifest."
                        )
                selected[task_id] = sample
        if not selected:
            raise BatchScheduleBlocked(
                "The sealed batch has no completed training samples."
            )
        train_rows = [selected[key].dataset_row() for key in sorted(selected)]
        val_rows = [sample.dataset_row() for sample in validation]
        conversation = UUID(self.project.project.conversation_id).hex
        directory = (
            f"/workspace/.pyromind-agent/{conversation}/project-batches/"
            f"{self.tracer_resource_id}/{receipt.batch_index}"
        )
        TracerBatchMaterializer(storage).write(
            directory=directory, rows={"train": train_rows, "val": val_rows}
        )
        source_name, defaults = platform_run_inputs(self.published, self.slot)
        return {
            **defaults,
            source_name: directory,
            "tracer_batch": {
                "resource_id": receipt.tracer_resource_id,
                "batch_index": receipt.batch_index,
                "batch_id": receipt.batch_id,
                "confirmed_source_sha256": canonical_sha256(train_rows),
                "validation_dataset_uri": config.validation_dataset_uri,
                "validation_sha256": canonical_sha256(val_rows),
            },
        }

    def find_run(self, receipt: BatchReceipt) -> BatchRunState | None:
        matches = []
        for run in self.runs.list(self.project.project.project_id):
            if run.execution_mode != "published":
                continue
            source = run.inputs.get("tracer_batch")
            if (
                run.source_tracer_resource_id == receipt.tracer_resource_id
                and run.source_batch_id == receipt.batch_id
            ) or (
                isinstance(source, dict)
                and source.get("resource_id") == receipt.tracer_resource_id
                and type(source.get("batch_index")) is int
                and source.get("batch_index") == receipt.batch_index
            ):
                matches.append(run)
        active = [
            run
            for run in matches
            if run.status not in {"succeeded", "failed", "canceled"}
        ]
        if len(active) > 1:
            raise BatchScheduleBlocked(
                "Multiple active runs refer to this Tracer batch."
            )
        candidates = (
            active or [run for run in matches if run.status == "succeeded"] or matches
        )
        run = max(candidates, key=lambda item: item.created_at) if candidates else None
        return self._run_state(run) if run is not None else None

    def start_run(self, receipt: BatchReceipt, inputs: JsonObject) -> BatchRunState:
        run = self.scheduler.start(
            project=self.project,
            published=self.published,
            request_id=f"continuous:{receipt.tracer_resource_id}:{receipt.batch_id}",
            inputs=inputs,
            trigger="tracer_batch",
            source_tracer_resource_id=receipt.tracer_resource_id,
            source_batch_id=receipt.batch_id,
        )
        return self._run_state(run)

    def release_archived(
        self, receipt: BatchReceipt, verify_archive: Callable[[], None]
    ) -> bool:
        if receipt.tracer_resource_id != self.tracer_resource_id:
            raise BatchScheduleBlocked("Archived batch belongs to another Tracer.")
        inventory = self._request("GET", "/admin/batches")
        batches = inventory.get("batches")
        current = inventory.get("current_batch_index")
        if (
            type(current) is not int
            or not isinstance(batches, list)
            or any(
                not isinstance(batch, dict) or type(batch.get("index")) is not int
                for batch in batches
            )
        ):
            raise BatchScheduleBlocked("Tracer did not return a valid batch inventory.")
        if current <= receipt.batch_index:
            # Releasing the current index reopens it under the same ID.
            return False
        matches = [
            batch
            for batch in batches
            if isinstance(batch, dict) and batch.get("index") == receipt.batch_index
        ]
        if len(matches) > 1 or (
            matches and matches[0].get("state") not in {"ready", "in_training"}
        ):
            raise BatchScheduleBlocked("Tracer did not confirm this batch is sealed.")
        verify_archive()
        if matches:
            self._release_sealed(receipt)
        return True

    def release(self, receipt: BatchReceipt) -> None:
        inventory = self._request("GET", "/admin/batches")
        batches = inventory.get("batches")
        current = inventory.get("current_batch_index")
        if not isinstance(batches, list) or type(current) is not int:
            raise BatchScheduleBlocked(
                "Cannot verify batch release against the inventory."
            )
        if current > receipt.batch_index and all(
            isinstance(batch, dict) and batch.get("index") != receipt.batch_index
            for batch in batches
        ):
            return
        self._release_sealed(receipt)

    def _release_sealed(self, receipt: BatchReceipt) -> None:
        response = self._request(
            "POST", f"/admin/batches/{receipt.batch_index}/release"
        )
        if (
            response.get("ok") is not True
            or response.get("released_batch_index") != receipt.batch_index
        ):
            raise BatchScheduleBlocked(
                "Tracer has not confirmed release of this batch."
            )

    def _run_state(self, run: GraphRun) -> BatchRunState:
        state: Literal["running", "succeeded", "failed", "canceled"] = "running"
        if run.status == "succeeded":
            state = "succeeded"
        elif run.status == "failed":
            state = "failed"
        elif run.status == "canceled":
            state = "canceled"
        return BatchRunState(
            run_id=run.run_id,
            status=state,
            report_archived=run.status == "succeeded"
            and self.report_archived(run.run_id),
        )

    def _samples(
        self, config: BatchScheduleConfig
    ) -> tuple[dict[str, ConfirmedBatchSample], list[ConfirmedBatchSample]]:
        storage = self._storage()
        if not config.source_manifest_uri or not config.validation_dataset_uri:
            raise BatchScheduleBlocked(
                "Confirmed ground truth and fixed holdout paths are required."
            )
        training = _read_samples(storage, config.source_manifest_uri)
        validation = _read_samples(storage, config.validation_dataset_uri)
        if any(sample.task_id is None for sample in training):
            raise BatchScheduleBlocked(
                "Confirmed training samples must declare the Tracer task_id."
            )
        keyed = {
            sample.task_id: sample for sample in training if sample.task_id is not None
        }
        if len(keyed) != len(training):
            raise BatchScheduleBlocked(
                "Confirmed ground truth contains duplicate task IDs."
            )
        train_ids = {sample.id for sample in training}
        val_ids = {sample.id for sample in validation}
        train_images = {path for sample in training for path in sample.images}
        val_images = {path for sample in validation for path in sample.images}
        if train_ids & val_ids or train_images & val_images:
            raise BatchScheduleBlocked(
                "Training and fixed validation data share sample IDs or images."
            )
        return keyed, validation

    def _storage(self) -> EvaluationStorage:
        if self.storage is None:
            raise BatchScheduleBlocked(
                "The owner's unattended Storage authorization is unavailable."
            )
        return self.storage

    def _request(self, method: str, path: str) -> dict[str, JsonValue]:
        tracer = self.slot.tracer
        if tracer is None or not tracer.endpoint or not self.admin_key:
            raise BatchScheduleBlocked(
                "Tracer batch control authorization is unavailable."
            )
        client = self.http_client or httpx.Client(timeout=15)
        try:
            response = client.request(
                method,
                tracer.endpoint.rstrip("/") + path,
                headers={"Authorization": f"Bearer {self.admin_key}"},
                json={} if method == "POST" else None,
            )
            if response.status_code in {401, 403}:
                raise BatchScheduleBlocked(
                    "Tracer rejected batch-control authorization; "
                    "reconnect its admin access."
                )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise BatchScheduleBlocked(
                    "Tracer returned an invalid batch-control response."
                )
            return data
        finally:
            if self.http_client is None:
                client.close()


def _read_samples(storage: EvaluationStorage, path: str) -> list[ConfirmedBatchSample]:
    try:
        samples = [
            ConfirmedBatchSample.model_validate_json(line)
            for line in storage.read(path).decode().splitlines()
            if line.strip()
        ]
    except (ValueError, EvaluationStorageMissingError):
        raise BatchScheduleBlocked(
            "The confirmed source or holdout JSONL is missing or invalid."
        ) from None
    if not samples or len({sample.id for sample in samples}) != len(samples):
        raise BatchScheduleBlocked(
            "Confirmed datasets need non-empty, unique sample IDs."
        )
    return samples


def _ground_truth(value: JsonValue) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return value.strip()
    return json.dumps(value, sort_keys=True, ensure_ascii=False)
