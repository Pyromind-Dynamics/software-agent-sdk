import json

import httpx
import pytest

from pyromind_agent_server import training_metric_source
from pyromind_agent_server.training_metric_source import (
    MAX_BYTES,
    PlatformTrainingMetricSource,
    ReadBudget,
    task_log_history,
)
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.observations import MetricConfig, RunBinding


async def test_wandb_polling_revisits_last_step_for_late_metrics():
    metrics = (
        MetricConfig(key="loss", title="Loss", axis="log_step"),
        MetricConfig(key="eval_loss", title="Eval loss", axis="log_step"),
    )
    history_calls = 0

    def route(request: httpx.Request) -> httpx.Response:
        nonlocal history_calls
        history_calls += 1
        specs = [
            json.loads(item)
            for item in json.loads(request.content)["variables"]["specs"]
        ]
        histories = []
        for spec in specs:
            metric = spec["keys"][-1]
            if spec["minStep"] > 3 or metric == "_timestamp":
                histories.append([])
            elif metric == "loss":
                histories.append([{"_step": 3, "loss": 0.8}])
            else:
                histories.append(
                    [{"_step": 3, "eval_loss": 0.7}] if history_calls > 1 else []
                )
        return httpx.Response(
            200,
            json={
                "data": {
                    "project": {
                        "run": {
                            "historyKeys": {"lastStep": 3},
                            "sampledHistory": histories,
                        }
                    }
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(route)) as client:
        source = PlatformTrainingMetricSource()
        first = await source._history(
            client, "entity", "project", "run", "fixture", metrics, 0, 10, ReadBudget()
        )
        second = await source._history(
            client,
            "entity",
            "project",
            "run",
            "fixture",
            metrics,
            first.next_step,
            10,
            ReadBudget(),
        )
    assert [p.key for p in first.points] == ["loss"]
    assert [p.key for p in second.points] == ["loss", "eval_loss"]
    assert second.points[0].source_point_id == first.points[0].source_point_id


def test_task_log_history_recovers_epoch_without_inventing_training_steps():
    binding = RunBinding(
        project_node_id="train",
        run_id="project:train:1",
        task_id="1274",
        business_status="succeeded",
    )
    metrics = tuple(
        MetricConfig(key=key, title=key, axis="training_step", axis_key="step")
        for key in ("loss", "eval_loss", "learning_rate")
    )
    logs = "\n".join(
        [
            "{'loss': '0.81', 'learning_rate': 0.0001, 'epoch': 0.5}",
            '{"eval_loss": 0.76, "epoch": 0.5}',
            "{'loss': 0.65, 'learning_rate': 0.00005, 'epoch': 1.0}",
        ]
    )
    first = task_log_history(logs, binding, "11", metrics, 0, 1)
    assert first is not None and first.has_more
    assert [(p.key, p.value, p.epoch, p.training_step) for p in first.points] == [
        ("loss", 0.81, 0.5, None),
        ("eval_loss", 0.76, 0.5, None),
        ("learning_rate", 0.0001, 0.5, None),
    ]
    last = task_log_history(logs, binding, "11", metrics, first.next_step, 2)
    assert last is not None and not last.has_more
    assert last.source_run_id == first.source_run_id
    assert [(p.key, p.epoch) for p in last.points] == [
        ("loss", 1.0),
        ("learning_rate", 1.0),
    ]

    explicit = task_log_history(
        logs + "\n" + '{"step": 10, "loss": 0.65, "epoch": 1}',
        binding,
        "11",
        metrics,
        0,
        2,
    )
    assert explicit is not None
    assert explicit.source_run_id != first.source_run_id
    assert [(p.training_step, p.epoch) for p in explicit.points] == [(10, None)]


@pytest.mark.parametrize("history_status", [401, 200])
async def test_long_studio_logs_scroll_and_keep_points_when_wandb_is_empty_or_denied(
    monkeypatch, history_status
):
    context = RequestContext(user_id="fixture", cookie="fixture-cookie")
    binding = RunBinding(
        project_node_id="train",
        run_id="project:train:1",
        task_id="1274",
        business_status="running",
    )
    metrics = (
        MetricConfig(key="loss", title="Loss", axis="training_step", axis_key="step"),
    )
    client = httpx.AsyncClient
    calls: list[str] = []
    cursors: list[str] = []

    def route(request):
        path = request.url.path
        calls.append(path)
        if path.endswith("/task_workflow_result"):
            return httpx.Response(
                200,
                json={
                    "nodes": [
                        {
                            "id": "11",
                            "type": "ModelTrainSFTNode",
                            "data": {"config": {"wandb_api_key": "fixture"}},
                        }
                    ]
                },
            )
        if path.endswith("/internal/logs/scroll"):
            timestamp = request.url.params["timestamp"]
            cursors.append(timestamp)
            if request.url.params["expect_count"] == "2000":
                return httpx.Response(200, json={"entries": [{"m": "x" * MAX_BYTES}]})
            entries = [
                {
                    "t": "2026-11-23T10:00:08.125Z",
                    "m": "https://wandb.ai/owner/train/runs/test\n",
                },
                {"t": "", "m": '{"epoch": 0.5, "loss": 0.8}\n'},
            ]
            if timestamp:
                entries.append({"t": "", "m": '{"epoch": 1.0, "loss": 0.6}\n'})
            entries.append({"t": "2026-11-23T10:00:45Z", "m": ""})
            return httpx.Response(200, json={"entries": entries})
        assert path == "/graphql"
        return httpx.Response(
            history_status,
            json={
                "data": {
                    "project": {
                        "run": {
                            "historyKeys": {"lastStep": 20},
                            "sampledHistory": [[], [], []],
                        }
                    }
                }
            },
        )

    monkeypatch.setattr(
        training_metric_source.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(route), **kw),
    )
    source = PlatformTrainingMetricSource()
    first = await source.read(binding, metrics, 0, 200, context)
    second = await source.read(
        binding,
        metrics,
        first.next_step,
        200,
        context,
        previous_source=first.source_run_id,
    )
    assert [point.value for point in first.points] == [0.8]
    assert [point.value for point in second.points] == [0.8, 0.6]
    assert second.points[0].source_point_id == first.points[0].source_point_id
    assert cursors == ["", "", "2026-11-23T10:00:07Z"]
    assert calls.count("/graphql") == 1
    assert sum(path.endswith("/task_workflow_result") for path in calls) == 1
