"""Bounded platform/W&B reads; credentials remain local to each request."""

import ast
import hashlib
import json
import os
import re
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from pyromind_common.tools.utils.platform_env import platform_portal_url
from pyromind_common.tools.utils.training_source import (
    PYROMIND_WANDB_BASE_URL,
    WANDB_BUILDER_NODE_TYPE,
    finite_number,
    node_id,
    node_type,
    wandb_builder_for_training,
    wandb_environment,
    wandb_run_reference,
    worb_history_point,
    worb_run_api_url,
    worb_run_lookup_url,
)
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.observations import (
    MetricConfig,
    MetricPoint,
    RunBinding,
    SourcePage,
)
from pyromind_runtime.ports.observation_source import (
    ObservationForbiddenError,
    ObservationUnavailableError,
)


MAX_BYTES = 1024 * 1024
HISTORY_QUERY = """
query ProjectTrainingHistory($entity: String!, $project: String!, $run: String!,
                             $specs: [JSONString!]!) {
  project(name: $project, entityName: $entity) {
    run(name: $run) { historyKeys sampledHistory(specs: $specs) }
  }
}
"""


@dataclass
class ReadBudget:
    remaining: int = MAX_BYTES


class ObservationPageTooLarge(ObservationUnavailableError):
    pass


async def bounded_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    budget: ReadBudget | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    budget = budget or ReadBudget()
    async with client.stream(method, url, **kwargs) as response:
        if response.status_code in {401, 403}:
            raise ObservationForbiddenError("Observation source authorization required")
        response.raise_for_status()
        content = bytearray()
        async for chunk in response.aiter_bytes():
            budget.remaining -= len(chunk)
            if budget.remaining < 0:
                raise ObservationPageTooLarge(
                    "Observation response exceeded its byte budget"
                )
            content.extend(chunk)
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ObservationUnavailableError("Invalid observation response")
        return value


def find_nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("nodes", "node_list", "node_infos", "data"):
        value = payload.get(key)
        if (
            isinstance(value, list)
            and value
            and all(isinstance(v, dict) for v in value)
        ):
            return value
    for key in ("workflow", "data"):
        nested = payload.get(key)
        if isinstance(nested, dict) and (nodes := find_nodes(nested)):
            return nodes
    return []


async def bounded_lines(
    response: httpx.Response, budget: ReadBudget
) -> AsyncIterator[bytes]:
    pending = b""
    async for chunk in response.aiter_bytes(chunk_size=4096):
        budget.remaining -= len(chunk)
        if budget.remaining < 0:
            raise ObservationPageTooLarge("W&B history exceeded its byte budget")
        lines = (pending + chunk).split(b"\n")
        pending = lines.pop()
        for line in lines:
            if line.strip():
                yield line
    if pending.strip():
        yield pending


async def worb_scalars(
    client: httpx.AsyncClient,
    url: str,
    auth: tuple[str, str],
    params: dict[str, str],
    keys: set[str],
    budget: ReadBudget,
) -> AsyncGenerator[tuple[str, int, float, float]]:
    async with client.stream(
        "GET", url, auth=auth, params=params, follow_redirects=False
    ) as response:
        response.raise_for_status()
        async for line in bounded_lines(response, budget):
            point = worb_history_point(json.loads(line))
            if point[0] not in keys:
                raise ValueError("Unexpected W&B metric")
            yield point


def log_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(log_text(item) for item in value)
    if isinstance(value, dict):
        for key in (
            "entries",
            "log",
            "logs",
            "content",
            "data",
            "stdout",
            "text",
            "raw",
            "m",
        ):
            if key in value:
                return log_text(value[key])
    return ""


# Studio returns raw trainer output, which may use Python dicts or JSON.
LOG_RECORD = re.compile(r"\{[^{}\r\n]{1,4096}\}")


@dataclass
class TaskLogRows:
    steps: dict[int, dict[str, float]] = field(default_factory=dict)
    epochs: dict[int, dict[str, float]] = field(default_factory=dict)
    gap: bool = False

    def add(self, logs: str, metrics: tuple[MetricConfig, ...]) -> None:
        keys = (
            {metric.key for metric in metrics}
            | {"_timestamp", "epoch"}
            | {metric.axis_key for metric in metrics if metric.axis_key}
        )
        for match in LOG_RECORD.finditer(logs):
            text = match.group()
            if not re.search(r"""["'](?:step|epoch)["']\s*:""", text):
                continue
            try:
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    row = ast.literal_eval(text)
            except (ValueError, TypeError, SyntaxError, RecursionError):
                continue
            if not isinstance(row, dict):
                continue
            values = {
                key: value
                for key in keys
                if (value := log_number(row.get(key))) is not None
            }
            step = row.get("step")
            use_epoch = "step" not in row and 0 <= values.get("epoch", -1) < 2**32
            if any(
                metric_value_available(metric, values, use_epoch) for metric in metrics
            ):
                # An epoch coordinate stays stable when consecutive log windows
                # overlap. Loss and validation at the same coordinate are sparse keys.
                if use_epoch:
                    self.epochs.setdefault(
                        round(values["epoch"] * 1_000_000), {}
                    ).update(values)
                elif type(step) is int and 0 <= step < 2**53:
                    self.steps.setdefault(step, {}).update(values)
        for rows in (self.steps, self.epochs):
            if len(rows) > 2048:
                self.gap = True
                for step in sorted(rows)[:-2048]:
                    del rows[step]


def history_points(
    rows: dict[int, dict[str, Any]],
    metrics: tuple[MetricConfig, ...],
    source_run: str,
    *,
    require_axis: bool = False,
) -> tuple[MetricPoint, ...]:
    points = []
    for step, row in sorted(rows.items()):
        for metric in metrics:
            if require_axis and not metric_value_available(metric, row, False):
                continue
            value = finite_number(row.get(metric.key))
            if value is not None:
                points.append(
                    MetricPoint(
                        key=metric.key,
                        value=value,
                        unit=metric.unit,
                        source_point_id=f"{source_run}:{step}:{metric.key}",
                        source_run_id=source_run,
                        log_step=step,
                        training_step=finite_number(row.get(metric.axis_key))
                        if metric.axis_key
                        else None,
                        time=finite_number(row.get("_timestamp")),
                    )
                )
    return tuple(points)


def metric_value_available(
    metric: MetricConfig, values: dict[str, float], use_epoch: bool
) -> bool:
    return (
        metric.key in values
        and (
            metric.axis != "training_step"
            or metric.axis_key in values
            or (use_epoch and metric.axis_key == "step")
        )
        and (metric.axis != "time" or "_timestamp" in values)
    )


def task_log_history(
    logs: str,
    binding: RunBinding,
    platform_node: str,
    metrics: tuple[MetricConfig, ...],
    start: int,
    count: int,
    *,
    retained: TaskLogRows | None = None,
    previous_source: str = "",
) -> SourcePage | None:
    retained = retained or TaskLogRows()
    retained.add(logs, metrics)
    use_epoch = not retained.steps and bool(retained.epochs)
    rows = retained.epochs if use_epoch else retained.steps
    if not rows:
        return None
    source = "task-log-epoch-v2" if use_epoch else "task-log"
    source_run = f"{source}:{binding.run_id}:{binding.task_id}:{platform_node}"
    if previous_source and source_run != previous_source:
        start = 0
    steps = sorted(step for step in rows if step >= start)
    points = tuple(
        MetricPoint(
            key=metric.key,
            value=row[metric.key],
            unit=metric.unit,
            source_point_id=f"{source_run}:{step}:{metric.key}",
            source_run_id=source_run,
            log_step=step,
            training_step=row.get(metric.axis_key) if metric.axis_key else None,
            epoch=row["epoch"] if use_epoch else None,
            time=row.get("_timestamp"),
        )
        for step in steps[:count]
        for row in (rows[step],)
        for metric in metrics
        if metric_value_available(metric, row, use_epoch)
    )
    has_more = len(steps) > count
    next_step = steps[count] if has_more else steps[-1] if steps else start
    return SourcePage(
        points=points,
        source_run_id=source_run,
        next_step=next_step,
        has_more=has_more,
        gap=retained.gap and start < min(rows),
    )


def log_number(value: object) -> float | None:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    return finite_number(value)


@dataclass
class TrainingLogState:
    node: str
    rows: TaskLogRows = field(default_factory=TaskLogRows)
    timestamp: str = ""
    wandb: tuple[str, str, str] | None = None
    scroll_supported: bool = True
    worb_run: tuple[str, str] | None = None
    worb_viewer: str = ""


def scroll_timestamp(payload: dict[str, Any]) -> str:
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return ""
    for entry in reversed(entries):
        if not isinstance(entry, dict) or not entry.get("m"):
            continue
        timestamp = entry.get("t")
        if not isinstance(timestamp, str) or not timestamp:
            continue
        try:
            observed = datetime.fromisoformat(timestamp)
        except ValueError:
            continue
        if observed.tzinfo is None:
            continue
        # Studio advances the supplied cursor by one second. Overlap the last
        # second so records emitted after the preceding request are included.
        return (observed.astimezone(UTC) - timedelta(seconds=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    return ""


class PlatformTrainingMetricSource:
    def __init__(self) -> None:
        self._logs: OrderedDict[tuple, TrainingLogState] = OrderedDict()

    async def read(
        self,
        binding: RunBinding,
        metrics: tuple[MetricConfig, ...],
        start_step: int,
        step_count: int,
        context: RequestContext,
        *,
        previous_source: str = "",
    ) -> SourcePage:
        if not binding.task_id.isdigit():
            raise ObservationUnavailableError("Binding has no numeric platform task ID")
        if not context.cookie and not context.authorization:
            raise ObservationForbiddenError("Platform credentials required")
        headers = {"x-cluster": context.x_cluster or ""}
        if context.cookie:
            headers["cookie"] = context.cookie
        if context.authorization:
            headers["authorization"] = context.authorization
        base = platform_portal_url(
            "https://api-portal.pyromind.ai/std2/studio_api/", headers
        )
        identity = (
            context.user_id,
            context.profile_id,
            context.x_cluster,
            hashlib.sha256(
                json.dumps([context.cookie, context.authorization]).encode()
            ).digest(),
            binding.run_id,
            binding.task_id,
            binding.platform_node_id,
            tuple((m.key, m.axis, m.axis_key) for m in metrics),
        )
        state = self._logs.get(identity)
        try:
            async with httpx.AsyncClient(timeout=4, follow_redirects=False) as client:
                payload: dict[str, Any] = {}
                nodes: list[dict[str, Any]] | None = None

                async def workflow_nodes() -> list[dict[str, Any]]:
                    nonlocal nodes, payload
                    if nodes is None:
                        payload = await bounded_json(
                            client,
                            "GET",
                            base + "api/task_workflow_result",
                            params={"task_id": binding.task_id},
                            headers=headers,
                        )
                        nodes = find_nodes(payload)
                    return nodes

                if state is None:
                    candidates = [
                        node_id(node)
                        for node in await workflow_nodes()
                        if "config" not in node_type(node).lower()
                        and any(
                            k in node_type(node).lower()
                            for k in ("train", "sft", "grpo")
                        )
                        and (
                            not binding.platform_node_id
                            or node_id(node) == binding.platform_node_id
                        )
                    ]
                    if len(candidates) != 1 or not candidates[0]:
                        raise ObservationUnavailableError(
                            "Bind one real platform training node"
                        )
                    state = TrainingLogState(candidates[0])

                async def wandb_page(ref: tuple[str, str, str]) -> SourcePage | None:
                    try:
                        run_nodes = await workflow_nodes()
                    except (
                        ObservationForbiddenError,
                        ObservationUnavailableError,
                        httpx.HTTPError,
                    ):
                        return None
                    return await self._wandb_page(
                        client,
                        base,
                        headers,
                        binding,
                        run_nodes,
                        payload,
                        state,
                        ref,
                        metrics,
                        start_step,
                        step_count,
                        previous_source,
                    )

                continuing_wandb = bool(
                    previous_source
                ) and not previous_source.startswith(
                    ("task-log:", "task-log-epoch:", "task-log-epoch-v2:")
                )
                tried_wandb = bool(state.wandb and continuing_wandb)
                if state.wandb and tried_wandb:
                    page = await wandb_page(state.wandb)
                    if page is not None and (
                        page.points
                        or page.has_more
                        or page.source_run_id == previous_source
                    ):
                        return page

                log_page = await self._log_page(client, base, headers, binding, state)
                logs = log_text(log_page)
                state.timestamp = max(state.timestamp, scroll_timestamp(log_page))
                ref = wandb_run_reference(logs)
                if ref and (ref[0] or not state.wandb or state.wandb[2] != ref[2]):
                    state.wandb = ref
                studio = task_log_history(
                    logs,
                    binding,
                    state.node,
                    metrics,
                    start_step,
                    step_count,
                    retained=state.rows,
                    previous_source=previous_source,
                )
                self._logs[identity] = state
                self._logs.move_to_end(identity)
                if len(self._logs) > 64:
                    self._logs.popitem(last=False)

                if (
                    state.wandb
                    and not tried_wandb
                    and (not previous_source or studio is None or continuing_wandb)
                ):
                    page = await wandb_page(state.wandb)
                    if page is not None and (
                        page.points
                        or page.has_more
                        or page.source_run_id == previous_source
                    ):
                        return page
                if continuing_wandb:
                    raise ObservationUnavailableError(
                        "W&B history temporarily unavailable"
                    )
                if studio is not None:
                    return studio
                return SourcePage(
                    points=(),
                    next_step=0,
                    source_run_id=f"task-log:{binding.run_id}:{binding.task_id}:{state.node}",
                )
        except ObservationForbiddenError:
            self._logs.pop(identity, None)
            raise
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ObservationUnavailableError("Training source read failed") from exc

    async def _log_page(
        self,
        client: httpx.AsyncClient,
        base: str,
        headers: dict[str, str],
        binding: RunBinding,
        state: TrainingLogState,
    ) -> dict[str, Any]:
        if state.scroll_supported:
            counts = (500, 125) if state.timestamp else (2000, 500, 125)
            for count in counts:
                try:
                    payload = await bounded_json(
                        client,
                        "POST",
                        base + "internal/logs/scroll",
                        headers=headers,
                        params={
                            "action": "scroll_down",
                            "ac_type": "node",
                            "timestamp": state.timestamp,
                            "expect_count": count,
                        },
                        json={"task_id": binding.task_id, "node_code": state.node},
                    )
                    entries = payload.get("entries")
                    if isinstance(entries, list) and len(entries) > count:
                        state.rows.gap = True
                    return payload
                except ObservationPageTooLarge:
                    if count == counts[-1]:
                        raise
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code not in {404, 405}:
                        raise
                    state.scroll_supported = False
                    break
        return await bounded_json(
            client,
            "GET",
            base + "internal/logs/node/raw",
            headers=headers,
            params={"nodeId": state.node, "taskId": binding.task_id},
            budget=ReadBudget(8 * MAX_BYTES),
        )

    async def _wandb_page(
        self,
        client: httpx.AsyncClient,
        base: str,
        headers: dict[str, str],
        binding: RunBinding,
        nodes: list[dict[str, Any]],
        workflow: dict[str, Any],
        state: TrainingLogState,
        ref: tuple[str, str, str],
        metrics: tuple[MetricConfig, ...],
        start: int,
        count: int,
        previous_source: str,
    ) -> SourcePage | None:
        builder = wandb_builder_for_training(nodes, workflow, state.node)
        if not builder and any("WandbConfigBuilderNode" in node_type(n) for n in nodes):
            return None
        credentials: dict[str, str] = {}
        for node in nodes:
            if node is not builder and node_id(node) != state.node:
                continue
            data = node.get("data", {})
            if isinstance(data, dict) and isinstance(data.get("config"), dict):
                config = data["config"]
                if node is builder and node_type(node) == WANDB_BUILDER_NODE_TYPE:
                    config = {k: v for k, v in config.items() if k != "wandb_api_key"}
                credentials.update(wandb_environment(config))
        try:
            if not credentials.get("WANDB_API_KEY") and builder:
                output = await bounded_json(
                    client,
                    "GET",
                    base + "internal/output/node/raw",
                    params={"node_code": node_id(builder), "task_id": binding.task_id},
                    headers=headers,
                )
                data = output.get("data", output)
                if isinstance(data, dict):
                    credentials.update(wandb_environment(data))
            key = credentials.get("WANDB_API_KEY")
            entity = ref[0] or credentials.get("WANDB_ENTITY", "")
            project = ref[1] or credentials.get("WANDB_PROJECT", "")
            if not key or not project:
                return None
            managed = (
                builder is not None and node_type(builder) == WANDB_BUILDER_NODE_TYPE
            )
            if managed and not entity:
                if not state.worb_viewer:
                    viewer = await bounded_json(
                        client,
                        "GET",
                        PYROMIND_WANDB_BASE_URL + "/api/v1/viewer",
                        auth=("api", key),
                        follow_redirects=False,
                    )
                    if isinstance(viewer.get("entity"), str):
                        state.worb_viewer = viewer["entity"]
                entity = state.worb_viewer
            if not entity:
                return None
            if previous_source and previous_source != f"{entity}/{project}/{ref[2]}":
                start = 0
            if managed:
                return await self._worb_history(
                    client, state, entity, project, ref[2], key, metrics, start, count
                )
            return await self._history(
                client,
                entity,
                project,
                ref[2],
                key,
                metrics,
                start,
                count,
                ReadBudget(),
            )
        except (
            ObservationForbiddenError,
            ObservationUnavailableError,
            httpx.HTTPError,
            ValueError,
            KeyError,
            TypeError,
        ):
            return None

    @staticmethod
    async def _worb_history(
        client: httpx.AsyncClient,
        state: TrainingLogState,
        entity: str,
        project: str,
        run: str,
        api_key: str,
        metrics: tuple[MetricConfig, ...],
        start: int,
        count: int,
    ) -> SourcePage:
        source_run = f"{entity}/{project}/{run}"
        auth = ("api", api_key)
        if not state.worb_run or state.worb_run[0] != source_run:
            async with client.stream(
                "GET",
                worb_run_lookup_url(entity, project, run),
                auth=auth,
                follow_redirects=False,
            ) as response:
                if response.status_code != 302:
                    raise ObservationUnavailableError("W&B run lookup unavailable")
                url = worb_run_api_url(response.headers.get("location", ""))
            state.worb_run = source_run, url
        url = state.worb_run[1] + "/history"
        keys = {m.key for m in metrics}
        keys.update(m.axis_key for m in metrics if m.axis_key)
        keys.discard("step")
        if any("," in key for key in keys) or not keys:
            raise ObservationUnavailableError("Unsupported W&B metric selection")
        params = {"keys": ",".join(sorted(keys))}
        budget = ReadBudget()
        end = start + count
        rows: dict[int, dict[str, Any]] = {}
        async for key, index, step, value in worb_scalars(
            client,
            url,
            auth,
            {**params, "xMin": str(start), "xMax": str(end - 1)},
            keys,
            budget,
        ):
            if not start <= index < end:
                raise ValueError("W&B metric outside the requested window")
            row = rows.setdefault(index, {})
            row[key] = value
            # WORB coalesces an absent training step to the log index.
            if index != step:
                row["step"] = step

        # Line count is not a high-water mark: uploads can omit scalar rows.
        async with aclosing(
            worb_scalars(client, url, auth, {**params, "xMin": str(end)}, keys, budget)
        ) as future:
            later = await anext(future, None)
        if later is not None and later[1] < end:
            raise ValueError("W&B metric outside the requested window")
        return SourcePage(
            points=history_points(rows, metrics, source_run, require_axis=True),
            source_run_id=source_run,
            next_step=end if later else max(rows, default=start),
            has_more=later is not None,
        )

    async def _history(
        self,
        client: httpx.AsyncClient,
        entity: str,
        project: str,
        run: str,
        api_key: str,
        metrics: tuple[MetricConfig, ...],
        start: int,
        count: int,
        budget: ReadBudget,
    ) -> SourcePage:
        # Separate specs retain sparse keys; a shared key list takes intersections.
        keys = list(
            dict.fromkeys(
                [m.key for m in metrics]
                + [m.axis_key for m in metrics if m.axis_key]
                + ["_timestamp"]
            )
        )
        end = start + count
        response = await bounded_json(
            client,
            "POST",
            os.getenv(
                "PYROMIND_OBSERVATIONS_WANDB_URL", "https://api.wandb.ai/graphql"
            ),
            budget=budget,
            auth=("api", api_key),
            json={
                "query": HISTORY_QUERY,
                "variables": {
                    "entity": entity,
                    "project": project,
                    "run": run,
                    "specs": [
                        json.dumps(
                            {
                                "keys": ["_step", key],
                                "minStep": start,
                                "maxStep": end,
                                "samples": count,
                            }
                        )
                        for key in keys
                    ],
                },
            },
        )
        if response.get("errors"):
            raise ObservationUnavailableError("W&B history query rejected")
        data = response["data"]["project"]["run"]
        if not isinstance(data, dict) or not isinstance(data.get("historyKeys"), dict):
            raise ObservationUnavailableError("W&B history metadata unavailable")
        histories = data["sampledHistory"]
        if not isinstance(histories, list) or len(histories) != len(keys):
            raise ObservationUnavailableError("W&B history shape changed")
        rows: dict[int, dict[str, Any]] = {}
        for history in histories:
            if not isinstance(history, list) or len(history) > count:
                raise ObservationUnavailableError(
                    "W&B history page exceeded row budget"
                )
            for item in history:
                if not isinstance(item, dict):
                    raise ObservationUnavailableError("Invalid W&B metric row")
                step = item.get("_step")
                if type(step) is not int or not start <= step < end:
                    raise ObservationUnavailableError("Invalid source step window")
                rows.setdefault(step, {}).update(item)
        source_run = f"{entity}/{project}/{run}"
        points = history_points(rows, metrics, source_run)
        last_step = data.get("historyKeys", {}).get("lastStep", -1)
        if type(last_step) is not int or last_step < -1:
            raise ObservationUnavailableError("Invalid W&B history step")
        # Like Studio's overlapping log window, revisit the newest W&B step
        # so a metric added to that step after the previous read is visible.
        next_step = min(end, max(start, last_step))
        return SourcePage(
            points=tuple(points),
            source_run_id=source_run,
            next_step=next_step,
            has_more=last_step >= end,
        )
