import base64
import importlib
import json
from functools import partial
from pathlib import Path

import httpx
import pytest

from pyromind_common.tools.utils.training_source import (
    PYROMIND_WANDB_BASE_URL,
    WANDB_BUILDER_NODE_TYPE,
)


SCRIPTS = (
    Path(__file__).resolve().parents[2] / ".agents/skills/training-analysis/scripts"
)
UUID = "d24fec2b-2399-40ea-9593-c497aaaca244"


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("training_analysis_worker")


@pytest.fixture
def mixed_workflow(worker, monkeypatch):
    graph = {
        "nodes": [
            {
                "id": "other",
                "data": {
                    "nodeType": "WandbConfigBuilderNode",
                    "config": {"WANDB_API_KEY": "public-fixture"},
                },
            },
            {
                "id": "worb",
                "data": {
                    "nodeType": WANDB_BUILDER_NODE_TYPE,
                    "config": {"wandb_api_key": "secret-name"},
                },
            },
            {"id": "train", "data": {"nodeType": "ModelTrainSFTNode"}},
        ],
        "edges": [
            {"source": "worb", "target": "train", "targetHandle": "wandb_config"}
        ],
    }
    platform = worker.train_analysis.PlatformClient
    monkeypatch.setattr(
        platform,
        "task_workflow_result",
        lambda self, task: {"data": {"workflow": graph}},
    )
    monkeypatch.setattr(
        platform,
        "node_log",
        lambda self, node, task: (
            f"{PYROMIND_WANDB_BASE_URL}/owner/private/runs/test-run"
        ),
    )

    def output(self, node, task):
        assert node == "worb"
        return {
            "data": {
                "wandb_config": {
                    "WANDB_API_KEY": "private-fixture",
                    "WANDB_BASE_URL": "https://untrusted.invalid",
                }
            }
        }

    monkeypatch.setattr(platform, "node_output", output)
    return graph


def test_bound_worb_worker_resolves_probes_analyzes_and_reports(
    worker, mixed_workflow, monkeypatch, tmp_path
):
    run_path = f"/api/runs/{UUID}"
    metadata = {
        "ID": UUID,
        "DisplayName": "demo",
        "State": "finished",
        "Summary": {"loss": 0.25},
        "Config": {"lr": {"value": 0.01}},
        "HistoryLineCount": 3,
    }
    visited = []

    def handle(request):
        assert str(request.url).startswith(PYROMIND_WANDB_BASE_URL + "/")
        assert (
            request.headers["authorization"]
            == "Basic " + base64.b64encode(b"api:private-fixture").decode()
        )
        visited.append(request.url.path)
        if request.url.path == "/owner/private/runs/test-run":
            return httpx.Response(302, headers={"Location": f"/runs/{UUID}"})
        if request.url.path == run_path:
            return httpx.Response(200, json=metadata)
        if request.url.path == run_path + "/history/keys":
            return httpx.Response(200, json=["loss", "global_step"])
        if request.url.path == run_path + "/history":
            assert "maxPoints" not in request.url.params
            assert {"global_step", "loss"}.issubset(
                request.url.params["keys"].split(",")
            )
            points = [
                {"i": 4, "s": 4, "k": "loss", "v": 0.25},
                {"i": 3, "s": 30, "k": "loss", "v": 0.5},
                {"i": 2, "s": 22, "k": "loss", "v": 0.75},
                {"i": 2, "s": 22, "k": "global_step", "v": 20},
            ]
            return httpx.Response(200, text="\n".join(map(json.dumps, points)))
        if request.url.path == "/api/projects":
            return httpx.Response(
                200, json=[{"Entity": "owner", "Name": "private", "ID": UUID}]
            )
        if request.url.path == f"/api/projects/{UUID}/runs":
            return httpx.Response(200, json=[metadata])
        raise AssertionError(request.url)

    monkeypatch.setattr(
        httpx, "Client", partial(httpx.Client, transport=httpx.MockTransport(handle))
    )
    monkeypatch.setenv("WANDB_API_KEY", "unrelated-public-key")
    creds = tmp_path / "creds.json"
    target = worker._resolve_from_task({"task_id": "42"}, creds)
    assert target["env"]["WANDB_BASE_URL"] == PYROMIND_WANDB_BASE_URL
    assert not visited
    assert "private-fixture" not in str(target)
    probe = worker._operation({"operation": "probe"}, target, creds)
    assert probe["run_summary_keys"] == ["loss"]
    assert probe["sample_metric_keys"] == ["global_step", "loss"]
    service = worker.train_analysis.TrainingAnalysisService()
    project = service.probe(creds_file=creds, entity_project="owner/private")
    assert project["has_step_history"]
    analysis = worker._operation(
        {"operation": "analyze", "include_history": True}, target, creds
    )
    assert analysis["config"] == {"lr": 0.01}
    assert [
        (p["log_step"], p["step"], p["step_key"])
        for p in analysis["metric_series"]["loss"]
    ] == [(2, 20, "global_step"), (3, 30, "step"), (4, None, None)]
    assert analysis["metric_stats"]["final"] == 0.25
    report = tmp_path / "report.md"
    worker._operation(
        {"operation": "report", "output_path": str(report)}, target, creds
    )
    assert "0.25" in report.read_text()
    assert run_path + "/history" in visited


def test_private_builder_requires_own_credentials_and_trusted_redirect(
    worker, mixed_workflow, monkeypatch, tmp_path
):
    monkeypatch.setenv("WANDB_API_KEY", "unrelated-cloud-key")
    monkeypatch.setattr(
        worker.train_analysis.PlatformClient, "node_output", lambda *args: {}
    )
    service = worker.train_analysis.TrainingAnalysisService()
    creds = tmp_path / "creds.json"
    target = worker._resolve_from_task({"task_id": "42"}, creds)
    assert json.loads(creds.read_text())["WANDB_API_KEY"] == ""
    with pytest.raises(ValueError, match="credential missing"):
        worker._operation({"operation": "probe"}, target, creds)

    calls = []

    def redirect(request):
        calls.append(request.url.host)
        return httpx.Response(
            302, headers={"Location": f"https://untrusted.invalid/runs/{UUID}"}
        )

    monkeypatch.setattr(
        httpx, "Client", partial(httpx.Client, transport=httpx.MockTransport(redirect))
    )
    service.resolve_target(
        task_id="42",
        api_key="private-fixture",
        creds_out=str(creds),
        run_url="https://untrusted.invalid/owner/private/runs/test-run",
    )
    with pytest.raises(ValueError, match="redirect"):
        service.probe(
            entity_project="owner/private", run_id="test-run", creds_file=creds
        )
    assert calls == [httpx.URL(PYROMIND_WANDB_BASE_URL).host]

    mixed_workflow["edges"] = []
    with pytest.raises(worker.train_analysis.WandbAnalysisError, match="bound"):
        service.resolve_target(task_id="42")


def test_legacy_builder_keeps_wandb_sdk_in_mixed_graph(
    worker, mixed_workflow, monkeypatch, tmp_path
):
    mixed_workflow["edges"][0]["source"] = "other"
    legacy = importlib.import_module("data_sources.wandb").WandbDataSource
    monkeypatch.setattr(legacy, "connect", lambda self, creds: None)
    creds = tmp_path / "creds.json"
    service = worker.train_analysis.TrainingAnalysisService()
    service.resolve_target(
        task_id="42",
        run_url="https://wandb.ai/owner/public/runs/old",
        creds_out=str(creds),
    )
    source, config = service._source(creds, "wandb")
    assert type(source) is legacy
    assert config["WANDB_API_KEY"] == "public-fixture"
    assert "wandb_builder_type" not in config
