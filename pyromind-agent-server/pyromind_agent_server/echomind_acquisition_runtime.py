from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import mimetypes
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any, Protocol

import httpx
import pyarrow.parquet as parquet
from pydantic import BaseModel, ConfigDict, Field

from pyromind_agent_server.echomind_acquisition import (
    AcquisitionBatch,
    AcquisitionSimulation,
    AcquisitionSimulationStore,
    AcquisitionSource,
    AcquisitionSplit,
)
from pyromind_agent_server.echomind_graph_runs import GraphRunStore
from pyromind_agent_server.echomind_projects import ProjectInfo, ProjectNode
from pyromind_agent_server.echomind_publication import PublishedGraph
from pyromind_agent_server.echomind_runtime_scheduler import (
    ReadyBatch,
    RuntimeGraphScheduler,
)
from pyromind_agent_server.echomind_runtime_slots import RuntimeSlot
from pyromind_agent_server.echomind_storage import canonical_uuid
from pyromind_agent_server.echomind_tracer_batch_admin import (
    is_tracer_batch_admin_response,
)
from pyromind_agent_server.echomind_train_runtime import (
    TrainRuntimeError,
    _json_path_tokens,
)
from pyromind_runtime.domain.storage_paths import (
    from_storage_api_path,
    to_storage_api_path,
)
from pyromind_runtime.ports.storage import (
    EvaluationStorage,
    EvaluationStorageMissingError,
)


logger = logging.getLogger(__name__)


class AcquisitionRuntimeError(RuntimeError):
    pass


class AcquisitionPaused(AcquisitionRuntimeError):
    pass


class TracerAdminUnavailable(AcquisitionRuntimeError):
    pass


class AcquisitionStorageReader(Protocol):
    def read(self, path: str) -> bytes: ...


class AcquisitionSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1, max_length=512)
    split: AcquisitionSplit
    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    images: list[str] = Field(default_factory=list)
    ground_truth: str | None = Field(min_length=1)
    image_sha256: list[str] = Field(default_factory=list, exclude_if=lambda v: not v)


class TracerCollectionStatus(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    capacity: int | None = Field(default=None, ge=0)
    held: int = Field(default=0, ge=0)
    buffer_size: int = Field(default=0, ge=0)
    collected_data_size: int = Field(default=0, ge=0)
    current_batch_index: int = Field(default=0, ge=0)
    ready_count: int = Field(default=0, ge=0)
    inflight_count: int = Field(default=0, ge=0)
    batch_mode: bool = False


class TracerAcquisitionTransport(Protocol):
    def check_batch_admin(self, *, endpoint: str, admin_key: str) -> None: ...

    def status(
        self,
        *,
        endpoint: str,
        access_key: str,
    ) -> TracerCollectionStatus: ...

    def models(self, *, endpoint: str, access_key: str) -> list[str]: ...

    def chat_completion(
        self,
        *,
        endpoint: str,
        access_key: str,
        model: str,
        task_id: str,
        messages: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...

    def end_task(
        self,
        *,
        endpoint: str,
        access_key: str,
        task_id: str,
        final_reward: float | None,
        ground_truth: str | None,
    ) -> Mapping[str, Any]: ...

    def seal_batch(
        self,
        *,
        endpoint: str,
        admin_key: str,
    ) -> int | None: ...

    def claim_batch(
        self, *, endpoint: str, admin_key: str, batch_index: int
    ) -> bool: ...


class HttpTracerAcquisitionTransport:
    def __init__(
        self,
        *,
        timeout_seconds: float = 300,
        client: httpx.Client | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._client = client

    def status(
        self,
        *,
        endpoint: str,
        access_key: str,
    ) -> TracerCollectionStatus:
        return TracerCollectionStatus.model_validate(
            self._request_json(
                "GET",
                endpoint,
                "/status",
                access_key=access_key,
            )
        )

    def models(self, *, endpoint: str, access_key: str) -> list[str]:
        payload = self._request_json(
            "GET",
            endpoint,
            "/v1/models",
            access_key=access_key,
        )
        data = payload.get("data")
        if not isinstance(data, list):
            return []
        return [
            str(item["id"])
            for item in data
            if isinstance(item, dict) and item.get("id")
        ]

    def chat_completion(
        self,
        *,
        endpoint: str,
        access_key: str,
        model: str,
        task_id: str,
        messages: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        return self._request_json(
            "POST",
            endpoint,
            "/v1/chat/completions",
            access_key=access_key,
            headers={"X-Task-ID": task_id, "X-Task-Trace": "true"},
            body={
                "model": model,
                "messages": list(messages),
                "temperature": 0.0,
            },
        )

    def end_task(
        self,
        *,
        endpoint: str,
        access_key: str,
        task_id: str,
        final_reward: float | None,
        ground_truth: str | None,
    ) -> Mapping[str, Any]:
        return self._request_json(
            "POST",
            endpoint,
            "/end_task",
            access_key=access_key,
            body={
                "task_id": task_id,
                **({"final_reward": final_reward} if final_reward is not None else {}),
                **({"gt": ground_truth} if ground_truth is not None else {}),
            },
        )

    def seal_batch(
        self,
        *,
        endpoint: str,
        admin_key: str,
    ) -> int | None:
        try:
            payload = self._request_json(
                "POST",
                endpoint,
                "/admin/flush",
                access_key=admin_key,
                body={"seal_batch": True, "trigger_workflow": False},
            )
        except TracerAdminUnavailable:
            raise
        except AcquisitionRuntimeError as exc:
            raise TracerAdminUnavailable(
                "Tracer batch sealing API is unavailable"
            ) from exc
        sealed = payload.get("sealed_batch_index")
        return int(sealed) if sealed is not None else None

    def claim_batch(self, *, endpoint: str, admin_key: str, batch_index: int) -> bool:
        payload = self._request_json(
            "POST",
            endpoint,
            f"/admin/batches/{batch_index}/claim",
            access_key=admin_key,
            body={},
        )
        return payload.get("ok") is True and payload.get("claimed") is True

    def check_batch_admin(self, *, endpoint: str, admin_key: str) -> None:
        try:
            payload = self._request_json(
                "GET", endpoint, "/admin/batches", access_key=admin_key
            )
        except AcquisitionRuntimeError as exc:
            raise TracerAdminUnavailable(
                "Tracer batch admin API is unavailable"
            ) from exc
        if not is_tracer_batch_admin_response(payload):
            raise TracerAdminUnavailable("Tracer batch admin API is unavailable")

    def _request_json(
        self,
        method: str,
        endpoint: str,
        path: str,
        *,
        access_key: str,
        headers: Mapping[str, str] | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        request_headers = {
            "Authorization": f"Bearer {access_key}",
            **(headers or {}),
        }
        client = self._client or httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        try:
            response = client.request(
                method,
                f"{endpoint.rstrip('/')}{path}",
                headers=request_headers,
                json=body,
            )
            if response.status_code in {404, 503} and path == "/admin/flush":
                raise TracerAdminUnavailable(
                    f"Tracer batch sealing API returned HTTP {response.status_code}"
                )
            response.raise_for_status()
            payload = response.json()
        except TracerAdminUnavailable:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise AcquisitionRuntimeError(
                f"Tracer {method} {path} failed: {type(exc).__name__}"
            ) from exc
        finally:
            if self._client is None:
                client.close()
        if not isinstance(payload, dict):
            raise AcquisitionRuntimeError(
                f"Tracer {method} {path} returned a non-object payload"
            )
        return payload


class JsonlAcquisitionSampleSource:
    def __init__(self, reader: AcquisitionStorageReader) -> None:
        self.reader = reader

    def load(
        self,
        *,
        source: AcquisitionSource,
        splits: Sequence[AcquisitionSplit],
        limit: int,
        balance_splits: bool = False,
        dataset_labels: bool = False,
    ) -> list[AcquisitionSample]:
        if limit <= 0:
            return []
        samples: list[AcquisitionSample] = []
        seen: set[str] = set()
        for split_index, split in enumerate(splits):
            split_limit = (
                math.ceil((limit - len(samples)) / (len(splits) - split_index))
                if balance_splits
                else limit - len(samples)
            )
            if split_limit <= 0:
                break
            loaded = 0
            uri = source.train_uri if split == "train" else source.val_uri
            try:
                raw = self.reader.read(_storage_path(uri))
            except Exception as exc:
                raise AcquisitionRuntimeError(
                    f"Could not read acquisition source: {uri}"
                ) from exc
            for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AcquisitionRuntimeError(
                        f"Invalid JSONL at {uri}:{line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise AcquisitionRuntimeError(
                        f"Acquisition row must be an object at {uri}:{line_number}"
                    )
                sample = _sample_from_row(
                    row=row,
                    split=split,
                    source=source,
                    location=f"{uri}:{line_number}",
                    dataset_labels=dataset_labels,
                )
                if sample.sample_id in seen:
                    continue
                seen.add(sample.sample_id)
                samples.append(sample)
                loaded += 1
                if loaded == split_limit:
                    break
            if balance_splits and loaded != split_limit:
                raise AcquisitionRuntimeError(
                    f"Acquisition split {split} needs {split_limit} unique rows"
                )
            if len(samples) == limit:
                return samples
        raise AcquisitionRuntimeError(
            f"Acquisition requested {limit} samples but source only has "
            f"{len(samples)} unique rows"
        )


class TracerBatchMaterializer:
    def __init__(self, storage: EvaluationStorage) -> None:
        self.storage = storage

    @staticmethod
    def _tracer_data_path(tracer_resource_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", tracer_resource_id):
            raise AcquisitionRuntimeError(
                "Tracer resource ID cannot name a Storage path"
            )
        return f"/EchoMind/{tracer_resource_id}/data"

    def latest_platform_batch_index(self, tracer_resource_id: str) -> int | None:
        prefix = self._tracer_data_path(tracer_resource_id)
        try:
            names = self.storage.list(f"{prefix}/spans")
        except EvaluationStorageMissingError:
            return None
        return max(
            (int(name) for name in names if re.fullmatch(r"0|[1-9][0-9]*", name)),
            default=None,
        )

    def platform_batch_ready(
        self, tracer_resource_id: str, batch_index: int, *, expected_samples: int
    ) -> bool:
        prefix = self._tracer_data_path(tracer_resource_id)
        try:
            marker = self.storage.read(f"{prefix}/project-batches/{batch_index}.json")
        except EvaluationStorageMissingError:
            # A full buffer can prevent opening the next slot before the hook runs.
            try:
                metadata = json.loads(
                    self.storage.read(f"{prefix}/spans/{batch_index}/batch.json")
                )
            except (EvaluationStorageMissingError, ValueError):
                return False
            return (
                isinstance(metadata, dict)
                and type(metadata.get("index")) is int
                and metadata["index"] == batch_index
                and metadata.get("state") in ("ready", "in_training")
                and type(metadata.get("collected")) is int
                and metadata["collected"] == expected_samples
            )
        try:
            payload = json.loads(marker)
        except ValueError as exc:
            raise AcquisitionPaused(
                f"Platform Tracer batch {batch_index} has an invalid trigger marker"
            ) from exc
        if (
            not isinstance(payload, dict)
            or type(payload.get("batch_index")) is not int
            or payload != {"batch_index": batch_index}
        ):
            raise AcquisitionPaused(
                f"Platform Tracer batch {batch_index} has an invalid trigger marker"
            )
        return True

    def rows(
        self,
        *,
        tracer_resource_id: str,
        simulation_id: str,
        batch: AcquisitionBatch,
        samples: Sequence[AcquisitionSample],
        completed_sample_ids: set[str] | None = None,
        fixed_validation: bool = False,
    ) -> dict[AcquisitionSplit, list[dict[str, Any]]]:
        expected = {
            (
                _platform_task_id(simulation_id, sample)
                if completed_sample_ids is not None
                else _task_id(simulation_id, sample.sample_id)
            ): sample
            for sample in samples
            if sample.sample_id in batch.sample_ids
        }
        if len(expected) != len(batch.sample_ids):
            raise AcquisitionPaused(
                "The sealed batch no longer matches its source rows"
            )
        prefix = f"{self._tracer_data_path(tracer_resource_id)}/spans/{batch.index}"
        shards = sorted(
            name
            for name in self.storage.list(prefix)
            if re.fullmatch(r"spans_[A-Za-z0-9_-]+\.parquet", name)
        )
        traces: dict[str, str | None] = {}
        if not shards:
            raise AcquisitionPaused(f"Tracer batch {batch.index} has no Parquet shards")
        for name in shards:
            try:
                table = parquet.read_table(
                    BytesIO(self.storage.read(f"{prefix}/{name}")),
                    columns=["name", "rollout_id", "attributes"],
                )
            except (OSError, ValueError) as exc:
                raise AcquisitionPaused(
                    f"Tracer batch {batch.index} contains an unreadable shard"
                ) from exc
            for record in table.to_pylist():
                if record["name"] != "llm.completion":
                    continue
                task_id = record["rollout_id"]
                if task_id not in expected:
                    mismatch = (
                        "another producer's data or a changed source row"
                        if completed_sample_ids is not None
                        else "another producer's data"
                    )
                    raise AcquisitionPaused(
                        f"Tracer batch {batch.index} contains {mismatch}"
                    )
                try:
                    attributes = json.loads(record["attributes"])
                except (TypeError, ValueError) as exc:
                    raise AcquisitionPaused(
                        "Tracer span attributes are malformed"
                    ) from exc
                if not isinstance(attributes, dict):
                    raise AcquisitionPaused("Tracer span attributes are not an object")
                sample = expected[task_id]
                if sample.ground_truth is None:
                    if (
                        completed_sample_ids is None
                        or sample.sample_id not in completed_sample_ids
                    ):
                        raise AcquisitionPaused("Missing finalized raw sample receipt")
                    traces[task_id] = None
                    continue
                ground_truths = [
                    attributes[key]
                    for key in ("gt", "task.gt")
                    if key in attributes and attributes[key] not in (None, "")
                ]
                if (
                    not ground_truths
                    and completed_sample_ids is not None
                    and sample.sample_id in completed_sample_ids
                ):
                    ground_truths = [sample.ground_truth]
                if not ground_truths or any(
                    not isinstance(gt, str) for gt in ground_truths
                ):
                    raise AcquisitionPaused("Tracer span has no finalized ground truth")
                ground_truth = ground_truths[0]
                if any(
                    _extract_json_result(gt) != _extract_json_result(ground_truth)
                    for gt in ground_truths
                ):
                    raise AcquisitionPaused("Tracer spans disagree on ground truth")
                if task_id in traces and _extract_json_result(
                    traces[task_id] or ""
                ) != _extract_json_result(ground_truth):
                    raise AcquisitionPaused("Tracer spans disagree on ground truth")
                traces[task_id] = ground_truth
        if traces.keys() != expected.keys():
            raise AcquisitionPaused(
                f"Tracer batch {batch.index} is missing completed simulation samples"
            )
        rows: dict[AcquisitionSplit, list[dict[str, Any]]] = {"train": [], "val": []}
        for task_id, sample in expected.items():
            image_paths = [from_storage_api_path(f"/{path}") for path in sample.images]
            row = {
                "id": sample.sample_id,
                "images": image_paths,
                "system_prompt": sample.system_prompt,
                "user_prompt": sample.user_prompt,
            }
            if sample.ground_truth is None:
                if sample.image_sha256:
                    row["image_sha256"] = sample.image_sha256
                rows[sample.split].append(row)
                continue
            gt = _extract_json_result(traces[task_id] or "")
            if gt is None or gt != _extract_json_result(sample.ground_truth):
                raise AcquisitionPaused(
                    "Tracer ground truth differs from the source row"
                )
            result = gt.get("result")
            if result not in ("pass", "defect", "fail", "真点", "假点"):
                raise AcquisitionPaused(
                    "Tracer row does not have a supported SFT label"
                )
            boxes = gt.get("boxes")
            rows[sample.split].append(
                {
                    **row,
                    "label": result,
                    "category": _result_category(gt),
                    "boxes": boxes if isinstance(boxes, list) else [],
                    "gt": sample.ground_truth,
                    # Some published Dataset parsers require result as the first key.
                    "assistant_output": json.dumps(
                        {"result": result, **gt}, ensure_ascii=False
                    ),
                }
            )
        if fixed_validation:
            if not rows["train"] or rows["val"]:
                raise AcquisitionPaused("This batch must supply training images only")
        elif not all(rows.values()):
            raise AcquisitionPaused("A platform SFT batch requires both source splits")
        return rows

    def write(
        self,
        *,
        directory: str,
        rows: Mapping[AcquisitionSplit, Sequence[Mapping[str, Any]]],
    ) -> None:
        prefix = to_storage_api_path(directory)
        for split, records in rows.items():
            payload = "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            ).encode("utf-8")
            # Sandbox uploads require at least one chunk, even for an empty split.
            payload = payload or b"\n"
            path = f"{prefix}/{split}.jsonl"
            try:
                existing = self.storage.read(path)
            except EvaluationStorageMissingError:
                self.storage.write(path, payload)
            else:
                if existing != payload and not (existing == b"" and not records):
                    raise AcquisitionPaused(
                        f"Batch input differs from its receipt: {path}"
                    )


class AcquisitionSimulationRunner:
    def __init__(
        self,
        *,
        projects_root: str,
        project: ProjectInfo,
        published: PublishedGraph,
        slot: RuntimeSlot,
        tracer_secret: str,
        sample_source: JsonlAcquisitionSampleSource,
        transport: TracerAcquisitionTransport | None = None,
        scheduler: RuntimeGraphScheduler | None = None,
        batch_materializer: TracerBatchMaterializer | None = None,
        tracer_admin_secret: str | None = None,
        platform_batch_trigger: bool = False,
        sleep: Callable[[float], None] = time.sleep,
        poll_seconds: float = 3,
        backpressure_timeout_seconds: float = 300,
        dispatch_timeout_seconds: float = 3600,
        max_inflight_batches: int = 2,
    ) -> None:
        if slot.tracer is None or not slot.tracer.endpoint:
            raise AcquisitionRuntimeError("RuntimeSlot has no Tracer endpoint")
        if platform_batch_trigger and batch_materializer is None:
            raise AcquisitionRuntimeError("Platform Tracer requires Project Storage")
        self.projects_root = projects_root
        self.project = project
        self.published = published
        self.slot = slot
        self.tracer_secret = tracer_secret
        self.sample_source = sample_source
        self.transport = transport or HttpTracerAcquisitionTransport()
        self.scheduler = scheduler or RuntimeGraphScheduler(projects_root=projects_root)
        self.batch_materializer = batch_materializer
        self.tracer_admin_secret = tracer_admin_secret or tracer_secret
        self.platform_batch_trigger = platform_batch_trigger
        self.sleep = sleep
        self.poll_seconds = poll_seconds
        self.backpressure_timeout_seconds = backpressure_timeout_seconds
        self.dispatch_timeout_seconds = dispatch_timeout_seconds
        self.max_inflight_batches = max_inflight_batches
        self.store = AcquisitionSimulationStore(projects_root)
        self.graph_run_store = GraphRunStore(projects_root)

    def run(self, simulation: AcquisitionSimulation) -> AcquisitionSimulation:
        current = simulation
        try:
            current = self._activate(simulation)
            if current.status in {"succeeded", "failed", "canceled"}:
                return current
            return self._run_active(current)
        except AcquisitionPaused as exc:
            return self._pause_latest(current, str(exc))
        except Exception as exc:
            logger.exception(
                "EchoMind acquisition simulation failed: %s",
                current.simulation_id,
            )
            return self._pause_latest(
                current,
                f"{type(exc).__name__}: {exc}"[:4000],
            )

    def _activate(self, simulation: AcquisitionSimulation) -> AcquisitionSimulation:
        if simulation.status in {"succeeded", "failed", "canceled"}:
            return simulation
        tracer = self.slot.tracer
        assert tracer is not None
        assert tracer.resource_id
        status = self._collection_status(
            endpoint=tracer.endpoint or "",
            access_key=self.tracer_secret,
        )
        if not status.batch_mode or status.capacity is None:
            raise AcquisitionPaused("Tracer batch mode is not enabled")
        if self.platform_batch_trigger and (
            not status.buffer_size
            or simulation.requested_samples
            != simulation.requested_rounds * status.buffer_size
        ):
            raise AcquisitionPaused(
                f"Platform Tracer requires {status.buffer_size} samples per round"
            )
        if (
            simulation.tracer_resource_id is not None
            and simulation.tracer_resource_id != tracer.resource_id
        ):
            raise AcquisitionPaused("The simulation belongs to another Tracer resource")
        model = simulation.model
        if not model:
            models = self.transport.models(
                endpoint=tracer.endpoint or "",
                access_key=self.tracer_secret,
            )
            if not models:
                raise AcquisitionPaused("Tracer did not expose a model for acquisition")
            model = models[0]
        return self.store.save(
            simulation.model_copy(
                update={
                    "status": "feeding",
                    "tracer_resource_id": tracer.resource_id,
                    "model": model,
                    "error": None,
                    "updated_at": _utc_now(),
                }
            )
        )

    def _run_active(self, simulation: AcquisitionSimulation) -> AcquisitionSimulation:
        tracer = self.slot.tracer
        assert tracer is not None
        assert tracer.resource_id
        endpoint = tracer.endpoint or ""
        current = simulation
        materialize = (
            current.auto_start_graph_run and self.batch_materializer is not None
        )
        fixed_validation = self.platform_batch_trigger and fixed_batch_validation(
            self.published
        )
        required_splits = {"train"} if fixed_validation else {"train", "val"}
        if materialize and (
            set(current.splits) != required_splits
            or current.requested_samples
            < len(required_splits) * current.requested_rounds
        ):
            if fixed_validation:
                raise AcquisitionPaused("This release requires train-only batch input")
            raise AcquisitionPaused(
                "Every platform SFT batch needs train and val samples"
            )
        if materialize:
            self._platform_inputs()
            if set(current.batch_ids) != current.batches.keys():
                raise AcquisitionPaused(
                    "Existing sealed batches have no ownership receipt"
                )
            if not self.platform_batch_trigger and (
                current.completed_rounds < current.requested_rounds
                or any(not batch.claimed for batch in current.batches.values())
            ):
                self.transport.check_batch_admin(
                    endpoint=endpoint, admin_key=self.tracer_admin_secret
                )
        samples = self.sample_source.load(
            source=current.source,
            splits=current.splits,
            limit=current.requested_samples,
            balance_splits=materialize,
            dataset_labels=self.platform_batch_trigger
            and dataset_labels_batch(self.published),
        )
        attempted = set(current.attempted_sample_ids)
        completed = set(current.completed_sample_ids)
        for sample in samples:
            if sample.sample_id in attempted and sample.sample_id not in completed:
                current = self._mark_failed(
                    current,
                    sample.sample_id,
                    "previous attempt ended without a confirmed Tracer completion",
                )

        if materialize:
            for batch_id in current.batch_ids:
                current = self._materialize_batch(
                    current, batch_id=batch_id, samples=samples
                )

        while current.completed_rounds < current.requested_rounds:
            if current.failed_sample_ids:
                raise AcquisitionPaused(
                    "Acquisition stopped after a Tracer sample failed"
                )
            pending = current.collecting_batch
            remaining_rounds = current.requested_rounds - current.completed_rounds
            remaining_samples = current.requested_samples - current.completed_samples
            round_size = math.ceil(remaining_samples / remaining_rounds)
            unprocessed = [
                sample
                for sample in samples
                if sample.sample_id not in set(current.attempted_sample_ids)
            ]
            round_samples = unprocessed[:round_size]
            if pending is not None:
                round_samples = [
                    sample
                    for sample in samples
                    if sample.sample_id in pending.sample_ids
                ]
                if len(round_samples) != len(pending.sample_ids):
                    raise AcquisitionPaused(
                        "The collecting batch no longer matches its source"
                    )
            elif len(round_samples) < round_size:
                raise AcquisitionPaused(
                    "Acquisition source does not contain enough unprocessed samples"
                )
            elif materialize and not fixed_validation:
                train = [sample for sample in unprocessed if sample.split == "train"]
                val = [sample for sample in unprocessed if sample.split == "val"]
                train_count = min(
                    len(train) - remaining_rounds + 1,
                    max(1, round_size - math.ceil(len(val) / remaining_rounds)),
                )
                val_count = round_size - train_count
                if (
                    train_count < 1
                    or not 1 <= val_count <= len(val) - remaining_rounds + 1
                ):
                    raise AcquisitionPaused(
                        "Could not preserve both splits for every batch"
                    )
                round_samples = train[:train_count] + val[:val_count]
            if pending is None:
                before_status = self._wait_for_capacity(
                    endpoint=endpoint,
                    access_key=self.tracer_secret,
                    round_size=len(round_samples),
                )
            else:
                before_status = self._collection_status(
                    endpoint=endpoint,
                    access_key=self.tracer_secret,
                )
                if not before_status.batch_mode:
                    raise AcquisitionPaused("Tracer batch mode is not enabled")
            if pending is None or (
                pending.index == before_status.current_batch_index + 1
                and set(pending.sample_ids).isdisjoint(current.attempted_sample_ids)
            ):
                before_status = self._next_collection_status(
                    before_status, round_size=len(round_samples)
                )
            if (
                pending is not None
                and before_status.current_batch_index != pending.index
                and not set(pending.sample_ids).issubset(current.completed_sample_ids)
            ):
                raise AcquisitionPaused(
                    "Tracer collecting batch changed before its samples completed"
                )
            if (materialize or self.platform_batch_trigger) and pending is None:
                if before_status.collected_data_size:
                    raise AcquisitionPaused(
                        "The Tracer collecting batch already contains data"
                    )
                if self.platform_batch_trigger:
                    if len(round_samples) != before_status.buffer_size:
                        raise AcquisitionPaused(
                            f"Platform Tracer requires {before_status.buffer_size}"
                            " samples per round"
                        )
                elif len(round_samples) >= before_status.buffer_size:
                    raise AcquisitionPaused(
                        "Simulation round must be smaller than Tracer auto-seal size"
                    )
                pending = AcquisitionBatch(
                    index=before_status.current_batch_index,
                    sample_ids=[sample.sample_id for sample in round_samples],
                )
                current = self.store.save(
                    current.model_copy(update={"collecting_batch": pending})
                )
            for sample in round_samples:
                if sample.sample_id in current.completed_sample_ids:
                    continue
                current = self._attempt_sample(
                    current,
                    sample,
                    endpoint=endpoint,
                )
                if current.failed_sample_ids:
                    raise AcquisitionPaused(
                        current.error or "A Tracer sample did not complete"
                    )
            if self.platform_batch_trigger:
                assert pending is not None
                sealed_index = self._wait_for_platform_batch(
                    tracer.resource_id, pending
                )
            else:
                sealed_index = self._seal_batch(
                    endpoint=endpoint,
                    before=before_status,
                    expected_index=pending.index if pending else None,
                )
            batch_id = _batch_id(sealed_index)
            current = self.store.save(
                current.model_copy(
                    update={
                        "status": "waiting_batch",
                        "completed_rounds": current.completed_rounds + 1,
                        "batch_ids": [*current.batch_ids, batch_id],
                        "batches": (
                            {**current.batches, batch_id: pending}
                            if pending is not None
                            else current.batches
                        ),
                        "collecting_batch": None,
                        "error": None,
                        "updated_at": _utc_now(),
                    }
                )
            )
            if current.failed_sample_ids:
                return self._pause(
                    current,
                    "Acquisition batch contains failed samples and was not dispatched",
                )
            if current.auto_start_graph_run:
                if materialize:
                    current = self._materialize_batch(
                        current, batch_id=batch_id, samples=samples
                    )
                current = self._dispatch(
                    current,
                    ready_batches=[self._ready_batch(current, batch_id)],
                )

        if current.auto_start_graph_run:
            current = self._wait_for_dispatches(current)
            return self.store.save(
                current.model_copy(
                    update={
                        "status": "succeeded",
                        "finished_at": _utc_now(),
                        "error": None,
                        "updated_at": _utc_now(),
                    }
                )
            )
        return self.store.save(
            current.model_copy(
                update={
                    "status": "waiting_batch",
                    "error": None,
                    "updated_at": _utc_now(),
                }
            )
        )

    def _attempt_sample(
        self,
        simulation: AcquisitionSimulation,
        sample: AcquisitionSample,
        *,
        endpoint: str,
    ) -> AcquisitionSimulation:
        current = self.store.save(
            simulation.model_copy(
                update={
                    "attempted_sample_ids": [
                        *simulation.attempted_sample_ids,
                        sample.sample_id,
                    ],
                    "updated_at": _utc_now(),
                }
            )
        )
        task_id = (
            _platform_task_id(current.simulation_id, sample)
            if self.platform_batch_trigger
            else _task_id(current.simulation_id, sample.sample_id)
        )
        try:
            messages = _messages(sample=sample, reader=self.sample_source.reader)
            response = self.transport.chat_completion(
                endpoint=endpoint,
                access_key=self.tracer_secret,
                model=current.model or "",
                task_id=task_id,
                messages=messages,
            )
            assistant_text = _assistant_text(response)
            reward = None
            if sample.ground_truth is not None:
                expected = _extract_json_result(sample.ground_truth)
                reward = _score_prediction(
                    expected, _extract_json_result(assistant_text)
                )
            elif not assistant_text.strip():
                raise AcquisitionRuntimeError(
                    "Tracer returned no prediction for the image"
                )
            receipt = self.transport.end_task(
                endpoint=endpoint,
                access_key=self.tracer_secret,
                task_id=task_id,
                final_reward=reward,
                ground_truth=sample.ground_truth,
            )
            if self.platform_batch_trigger and receipt.get("task_id") != task_id:
                raise AcquisitionRuntimeError(
                    "Tracer completion receipt does not match the submitted task"
                )
        except Exception as exc:
            return self._mark_failed(
                current,
                sample.sample_id,
                f"{type(exc).__name__}: {exc}",
            )
        return self.store.save(
            current.model_copy(
                update={
                    "completed_samples": current.completed_samples + 1,
                    "completed_sample_ids": [
                        *current.completed_sample_ids,
                        sample.sample_id,
                    ],
                    "updated_at": _utc_now(),
                }
            )
        )

    def _mark_failed(
        self,
        simulation: AcquisitionSimulation,
        sample_id: str,
        error: str,
    ) -> AcquisitionSimulation:
        return self.store.save(
            simulation.model_copy(
                update={
                    "failed_samples": simulation.failed_samples + 1,
                    "failed_sample_ids": [
                        *simulation.failed_sample_ids,
                        sample_id,
                    ],
                    "status": "paused",
                    "error": error[:4000],
                    "updated_at": _utc_now(),
                }
            )
        )

    def _collection_status(
        self, *, endpoint: str, access_key: str
    ) -> TracerCollectionStatus:
        status = self.transport.status(endpoint=endpoint, access_key=access_key)
        if (
            not self.platform_batch_trigger
            or "batch_mode" in status.model_fields_set
            or "collected_data_size" not in status.model_fields_set
        ):
            return status
        tracer = self.slot.tracer
        materializer = self.batch_materializer
        assert tracer is not None and tracer.resource_id and materializer is not None
        buffer_size = status.buffer_size or tracer.buffer_config.batch_size
        collected = status.collected_data_size
        latest_index = materializer.latest_platform_batch_index(tracer.resource_id)
        if collected > buffer_size or (latest_index is None and collected):
            raise AcquisitionPaused("Platform Tracer status contradicts batch Storage")
        current_index = latest_index if latest_index is not None else 0
        # Legacy Tracer advances a full head when it stores the next span.
        if collected == buffer_size:
            current_index += 1
            collected = 0
        return status.model_copy(
            update={
                "batch_mode": True,
                "capacity": tracer.buffer_config.trajectory_buffer_size,
                "buffer_size": buffer_size,
                "collected_data_size": collected,
                "current_batch_index": current_index,
            }
        )

    def _next_collection_status(
        self, status: TracerCollectionStatus, *, round_size: int
    ) -> TracerCollectionStatus:
        if (
            not self.platform_batch_trigger
            or status.capacity is None
            or status.capacity - status.held < round_size
            or status.buffer_size == 0
            or status.collected_data_size != status.buffer_size
        ):
            return status
        tracer = self.slot.tracer
        materializer = self.batch_materializer
        assert tracer is not None and tracer.resource_id and materializer is not None
        if not materializer.platform_batch_ready(
            tracer.resource_id,
            status.current_batch_index,
            expected_samples=status.buffer_size,
        ):
            return status
        # Tracer opens the next index when it receives the first span after release.
        return status.model_copy(
            update={
                "collected_data_size": 0,
                "current_batch_index": status.current_batch_index + 1,
            }
        )

    def _wait_for_capacity(
        self,
        *,
        endpoint: str,
        access_key: str,
        round_size: int,
    ) -> TracerCollectionStatus:
        deadline = time.monotonic() + self.backpressure_timeout_seconds
        while True:
            status = self._collection_status(
                endpoint=endpoint,
                access_key=access_key,
            )
            if not status.batch_mode or status.capacity is None:
                raise AcquisitionPaused("Tracer batch mode is not enabled")
            free_slots = max(0, status.capacity - status.held)
            if self.platform_batch_trigger:
                # Sealed platform batches remain in the inventory after GraphRun ends.
                can_collect = free_slots >= round_size
            else:
                can_collect = (
                    free_slots > 0 and status.inflight_count < self.max_inflight_batches
                )
            if can_collect:
                return status
            if time.monotonic() >= deadline:
                if self.platform_batch_trigger:
                    raise AcquisitionPaused(
                        f"Tracer has {free_slots} free sample slots; "
                        f"this round needs {round_size}"
                    )
                raise AcquisitionPaused(
                    "Tracer backpressure did not clear before the timeout"
                )
            self.sleep(self.poll_seconds)

    def _wait_for_platform_batch(
        self, tracer_resource_id: str, batch: AcquisitionBatch
    ) -> int:
        materializer = self.batch_materializer
        assert materializer is not None
        deadline = time.monotonic() + self.backpressure_timeout_seconds
        while not materializer.platform_batch_ready(
            tracer_resource_id, batch.index, expected_samples=len(batch.sample_ids)
        ):
            if time.monotonic() >= deadline:
                raise AcquisitionPaused(
                    f"Platform Tracer batch {batch.index} did not publish its trigger "
                    "marker or sealed batch metadata before the timeout"
                )
            self.sleep(self.poll_seconds)
        return batch.index

    def _seal_batch(
        self,
        *,
        endpoint: str,
        before: TracerCollectionStatus,
        expected_index: int | None = None,
    ) -> int:
        if expected_index is not None:
            after = self.transport.status(
                endpoint=endpoint, access_key=self.tracer_secret
            )
            if after.current_batch_index > expected_index:
                return expected_index
            if after.current_batch_index != expected_index:
                raise AcquisitionPaused("The Tracer collecting batch has changed")
        try:
            sealed = self.transport.seal_batch(
                endpoint=endpoint,
                admin_key=self.tracer_admin_secret,
            )
        except TracerAdminUnavailable:
            after = self.transport.status(
                endpoint=endpoint,
                access_key=self.tracer_secret,
            )
            if (
                after.current_batch_index > before.current_batch_index
                and after.ready_count > before.ready_count
            ):
                return before.current_batch_index
            raise AcquisitionPaused(
                "Tracer did not seal a batch and the admin API is unavailable"
            ) from None
        if sealed is None:
            raise AcquisitionPaused("Tracer did not seal a batch")
        if expected_index is not None and sealed != expected_index:
            raise AcquisitionPaused("Tracer sealed a different batch")
        return sealed

    def _materialize_batch(
        self,
        simulation: AcquisitionSimulation,
        *,
        batch_id: str,
        samples: Sequence[AcquisitionSample],
    ) -> AcquisitionSimulation:
        materializer = self.batch_materializer
        tracer = self.slot.tracer
        assert materializer is not None and tracer is not None and tracer.resource_id
        batch = simulation.batches[batch_id]
        if batch.input_directory is not None:
            return simulation
        completed_sample_ids = None
        if self.platform_batch_trigger:
            completed_sample_ids = set(simulation.completed_sample_ids)
            if not completed_sample_ids.issuperset(batch.sample_ids):
                raise AcquisitionPaused("Platform Tracer batch has unconfirmed samples")
        rows = materializer.rows(
            tracer_resource_id=tracer.resource_id,
            simulation_id=simulation.simulation_id,
            batch=batch,
            samples=samples,
            completed_sample_ids=completed_sample_ids,
            fixed_validation=self.platform_batch_trigger
            and fixed_batch_validation(self.published),
        )
        if not batch.claimed:
            if not self.platform_batch_trigger and not self.transport.claim_batch(
                endpoint=tracer.endpoint or "",
                admin_key=self.tracer_admin_secret,
                batch_index=batch.index,
            ):
                raise AcquisitionPaused("Tracer batch is already claimed elsewhere")
            batch = batch.model_copy(update={"claimed": True})
            simulation = self._save_batch(simulation, batch_id, batch)
        conversation_id = canonical_uuid(
            self.project.project.conversation_id, kind="conversation"
        ).replace("-", "")
        simulation_id = canonical_uuid(simulation.simulation_id, kind="simulation")
        directory = (
            f"/workspace/.pyromind-agent/{conversation_id}"
            f"/tracer-batches/{simulation_id}/{batch.index}"
        )
        materializer.write(directory=directory, rows=rows)
        return self._save_batch(
            simulation,
            batch_id,
            batch.model_copy(update={"input_directory": directory}),
        )

    def _save_batch(
        self, simulation: AcquisitionSimulation, batch_id: str, batch: AcquisitionBatch
    ) -> AcquisitionSimulation:
        return self.store.save(
            simulation.model_copy(
                update={
                    "batches": {**simulation.batches, batch_id: batch},
                    "updated_at": _utc_now(),
                }
            )
        )

    def _platform_inputs(self) -> tuple[str, dict[str, str]]:
        return platform_run_inputs(self.published, self.slot)

    def _ready_batch(
        self, simulation: AcquisitionSimulation, batch_id: str
    ) -> ReadyBatch:
        tracer = self.slot.tracer
        assert tracer is not None and tracer.resource_id
        ready = _ready_batch(
            simulation=simulation,
            tracer_resource_id=tracer.resource_id,
            batch_id=batch_id,
            batch_index=_batch_index(batch_id),
        )
        if self.batch_materializer is None:
            return ready
        batch = simulation.batches[batch_id]
        assert batch.input_directory is not None
        source_name, inputs = self._platform_inputs()
        return ready.model_copy(
            update={
                "inputs": {
                    **inputs,
                    **ready.inputs,
                    source_name: batch.input_directory,
                }
            }
        )

    def _dispatch(
        self,
        simulation: AcquisitionSimulation,
        *,
        ready_batches: Sequence[ReadyBatch],
    ) -> AcquisitionSimulation:
        tracer = self.slot.tracer
        assert tracer is not None
        assert tracer.resource_id
        started = self.scheduler.dispatch_ready_batches(
            project=self.project,
            published=self.published,
            source=StaticReadyBatchSource(ready_batches),
            tracer_resource_id=tracer.resource_id,
        )
        run_ids = [*simulation.graph_run_ids]
        for batch in ready_batches:
            existing = self.graph_run_store.find_by_batch_key(
                project_id=self.project.project.project_id,
                tracer_resource_id=batch.tracer_resource_id,
                batch_id=batch.batch_id,
            )
            if existing is not None and existing.run_id not in run_ids:
                run_ids.append(existing.run_id)
        for run in started:
            if run.run_id not in run_ids:
                run_ids.append(run.run_id)
        return self.store.save(
            simulation.model_copy(
                update={
                    "status": "graph_started" if run_ids else "waiting_batch",
                    "graph_run_ids": run_ids,
                    "updated_at": _utc_now(),
                }
            )
        )

    def _wait_for_dispatches(
        self,
        simulation: AcquisitionSimulation,
    ) -> AcquisitionSimulation:
        tracer = self.slot.tracer
        assert tracer is not None
        assert tracer.resource_id
        current = simulation
        batches = [
            self._ready_batch(current, batch_id) for batch_id in current.batch_ids
        ]
        deadline = time.monotonic() + self.dispatch_timeout_seconds
        while True:
            current = self._dispatch(current, ready_batches=batches)
            if len(current.graph_run_ids) >= len(current.batch_ids):
                return current
            if time.monotonic() >= deadline:
                raise AcquisitionPaused(
                    "Acquisition batches are ready but GraphRun dispatch is blocked"
                )
            active = self.graph_run_store.active(self.project.project.project_id)
            if active is not None and active.status != "paused":
                self.scheduler.recover(project=self.project, run_id=active.run_id)
            self.sleep(self.poll_seconds)

    def _pause(
        self,
        simulation: AcquisitionSimulation,
        error: str,
    ) -> AcquisitionSimulation:
        return self.store.save(
            simulation.model_copy(
                update={
                    "status": "paused",
                    "error": error[:4000],
                    "updated_at": _utc_now(),
                }
            )
        )

    def _pause_latest(
        self, simulation: AcquisitionSimulation, error: str
    ) -> AcquisitionSimulation:
        latest = self.store.load(
            project_id=simulation.project_id, simulation_id=simulation.simulation_id
        )
        return self._pause(latest, error)


class StaticReadyBatchSource:
    def __init__(self, batches: Sequence[ReadyBatch]) -> None:
        self.batches = list(batches)

    def list_ready_batches(
        self,
        *,
        project_id: str,
        tracer_resource_id: str,
    ) -> list[ReadyBatch]:
        del project_id
        return [
            batch
            for batch in self.batches
            if batch.tracer_resource_id == tracer_resource_id
        ]


def _sample_from_row(
    *,
    row: Mapping[str, Any],
    split: AcquisitionSplit,
    source: AcquisitionSource,
    location: str,
    dataset_labels: bool = False,
) -> AcquisitionSample:
    sample_id = str(row.get("id") or "").strip()
    user_prompt = str(row.get("user_prompt") or row.get("prompt") or "").strip()
    system_prompt = str(
        row.get("system_prompt") or "You are a careful visual inspection assistant."
    ).strip()
    raw_gt = row.get("gt", row.get("ground_truth"))
    if not sample_id:
        raise AcquisitionRuntimeError(f"Acquisition sample has no id at {location}")
    if not user_prompt:
        raise AcquisitionRuntimeError(f"Acquisition sample has no prompt at {location}")
    if not dataset_labels and (raw_gt is None or not str(raw_gt).strip()):
        raise AcquisitionRuntimeError(
            f"Acquisition sample has no ground truth at {location}"
        )
    image_values = row.get("images")
    if image_values is None:
        image_values = []
    if not isinstance(image_values, list):
        raise AcquisitionRuntimeError(
            f"Acquisition sample images must be a list at {location}"
        )
    images = [
        _image_storage_path(str(value), image_root=source.image_root)
        for value in image_values
        if str(value).strip()
    ]
    if not images:
        raise AcquisitionRuntimeError(f"Acquisition sample has no images at {location}")
    ground_truth = None
    if not dataset_labels:
        ground_truth = (
            raw_gt
            if isinstance(raw_gt, str)
            else json.dumps(raw_gt, ensure_ascii=False)
        )
    hashes = row.get("image_sha256", []) if dataset_labels else []
    if not isinstance(hashes, list) or (
        hashes
        and (
            len(hashes) != len(images)
            or not all(
                isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes
            )
        )
    ):
        raise AcquisitionRuntimeError(f"Invalid image fingerprints at {location}")
    return AcquisitionSample(
        sample_id=sample_id,
        split=split,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
        ground_truth=ground_truth,
        image_sha256=hashes,
    )


def dataset_labels_batch(published: PublishedGraph) -> bool:
    dataset = next((node for node in published.nodes if node.type == "dataset"), None)
    return dataset is not None and dataset.config.get("batch_label_source") == "dataset"


def fixed_batch_validation(published: PublishedGraph) -> bool:
    dataset = next((node for node in published.nodes if node.type == "dataset"), None)
    return dataset is not None and dataset.config.get("batch_validation_source") in (
        "fixed",
        "none",
    )


def platform_run_inputs(
    published: PublishedGraph, slot: RuntimeSlot
) -> tuple[str, dict[str, str]]:
    dataset = next((node for node in published.nodes if node.type == "dataset"), None)
    if dataset is None or "dataset_runtime" not in dataset.config:
        raise AcquisitionPaused("The active release needs a platform Dataset workflow")
    port = dataset.inputs.get("source_batch")
    if port is None or port.binding.kind != "run_input" or not port.binding.name:
        raise AcquisitionPaused("Dataset source_batch must bind to a run input")
    inputs: dict[str, str] = {}
    defaults: dict[str, str] = {}
    for node in published.nodes:
        declared = node.config.get("runtime_input_defaults")
        if isinstance(declared, dict):
            defaults.update(
                {
                    name: value
                    for name, value in declared.items()
                    if isinstance(value, str)
                }
            )
    serving = slot.serving_inference
    frozen_models: set[str] | None = None
    for node in published.nodes:
        for name, input_port in node.inputs.items():
            binding = input_port.binding
            if (
                binding.kind != "run_input"
                or binding.name is None
                or binding.name == port.binding.name
            ):
                continue
            value = defaults.get(binding.name)
            if not value and name == "base_model":
                if frozen_models is None:
                    frozen_models = {
                        model
                        for node in published.nodes
                        if node.type in {"train", "evaluator"}
                        for model in _workflow_base_models(node)
                    }
                if len(frozen_models) > 1:
                    raise AcquisitionPaused(
                        "Release Train and Eval bind different base models"
                    )
                value = next(
                    iter(frozen_models), serving.model_uri if serving else None
                )
            if value:
                inputs[binding.name] = value
            elif input_port.required:
                raise AcquisitionPaused(
                    f"Missing runtime input default: {binding.name}"
                )
    return port.binding.name, inputs


def _workflow_base_models(node: ProjectNode) -> set[str]:
    config = node.config.get(node.executor)
    if not isinstance(config, dict):
        config = node.config
    workflow = config.get("workflow")
    bindings = config.get("workflow_bindings")
    if not isinstance(workflow, dict) or not isinstance(bindings, dict):
        return set()
    models: set[str] = set()
    for path, port in bindings.items():
        if port != "base_model" or not isinstance(path, str):
            continue
        try:
            model: Any = workflow
            for token in _json_path_tokens(path):
                model = model[token]
        except (IndexError, KeyError, TypeError, TrainRuntimeError) as exc:
            raise AcquisitionPaused("Release base model binding is invalid") from exc
        if isinstance(model, str) and model.startswith(("/workspace/", "storage://")):
            models.add(model)
    return models


def _messages(
    *,
    sample: AcquisitionSample,
    reader: AcquisitionStorageReader,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": sample.user_prompt}]
    for index, path in enumerate(sample.images):
        try:
            payload = reader.read(path)
        except Exception as exc:
            raise AcquisitionRuntimeError(
                f"Could not read acquisition image: {path}"
            ) from exc
        if (
            sample.image_sha256
            and hashlib.sha256(payload).hexdigest() != sample.image_sha256[index]
        ):
            raise AcquisitionRuntimeError(
                f"Acquisition image changed since selection: {path}"
            )
        mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
        encoded = base64.b64encode(payload).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            }
        )
    return [
        {"role": "system", "content": sample.system_prompt},
        {"role": "user", "content": content},
    ]


def _assistant_text(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _extract_json_result(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if (
            isinstance(value, dict)
            and "result" in value
            and ("reason" in value or "category" in value)
        ):
            candidates.append(value)
        index = end
    return candidates[-1] if candidates else None


def _result_category(value: dict[str, Any]) -> str:
    if value.get("result") in ("真点", "假点") and "category" in value:
        return str(value["category"]).strip()
    return str(value.get("reason") or value.get("category") or "").strip()


def _score_prediction(
    expected: dict[str, Any] | None,
    prediction: dict[str, Any] | None,
) -> float:
    if expected is None or prediction is None:
        return 0.0
    if (
        str(expected.get("result", "")).strip()
        != str(prediction.get("result", "")).strip()
    ):
        return 0.0
    if _result_category(expected) != _result_category(prediction):
        return 0.0
    return 1.0


def _storage_path(uri: str) -> str:
    if uri.startswith("storage://"):
        uri = uri.removeprefix("storage://")
    if "://" in uri:
        raise AcquisitionRuntimeError(f"Unsupported acquisition URI: {uri}")
    return to_storage_api_path(f"/{uri.lstrip('/')}").lstrip("/")


def _image_storage_path(value: str, *, image_root: str) -> str:
    if value.startswith("storage://"):
        return _storage_path(value)
    path = PurePosixPath(value)
    if path.is_absolute():
        return _storage_path(str(path))
    return _storage_path(str(PurePosixPath(_storage_path(image_root)) / path))


def _task_id(simulation_id: str, sample_id: str) -> str:
    digest = hashlib.sha256(f"{simulation_id}:{sample_id}".encode()).hexdigest()
    return f"echomind-{digest}"


def _platform_task_id(simulation_id: str, sample: AcquisitionSample) -> str:
    # A legacy span may lack GT, so a resumed round must use the original source row.
    return _task_id(simulation_id, sample.model_dump_json())


def _batch_id(index: int) -> str:
    return f"batch-{index:08d}"


def _batch_index(batch_id: str) -> int:
    return int(batch_id.removeprefix("batch-"))


def _ready_batch(
    *,
    simulation: AcquisitionSimulation,
    tracer_resource_id: str,
    batch_id: str,
    batch_index: int,
) -> ReadyBatch:
    return ReadyBatch(
        tracer_resource_id=tracer_resource_id,
        batch_id=batch_id,
        inputs={
            "tracer_batch": {
                "batch_id": batch_id,
                "batch_index": batch_index,
                "resource_id": tracer_resource_id,
                "uri": f"tracer://{tracer_resource_id}/batches/{batch_index}",
                "simulation_id": simulation.simulation_id,
            }
        },
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)
