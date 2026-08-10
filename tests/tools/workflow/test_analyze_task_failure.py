"""Tests for the ``analyze_task_failure`` tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import SecretStr

from openhands.agent_server.pyromind_router import (
    PYROMIND_VALIDATE_AUTH_COOKIE_SECRET,
    PYROMIND_VALIDATE_HEADERS_STATE_KEY,
)
from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.secret import StaticSecret
from openhands.sdk.tool import Tool
from openhands.sdk.tool.registry import resolve_tool
from openhands.tools.workflow import (
    AnalyzeTaskFailureAction,
    AnalyzeTaskFailureExecutor,
    AnalyzeTaskFailureTool,
)
from openhands.tools.workflow.analyze_task_failure import (
    PRE_API_BASE,
    PROD_API_BASE,
    USER_AGENT,
)


class _FakeWorkspace:
    def __init__(self, working_dir: Path) -> None:
        self.working_dir = str(working_dir)


class _Response:
    def __init__(self, status_code: int, payload: Any, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _fake_conversation(
    *,
    secret_registry: SecretRegistry | None = None,
    agent_state: dict[str, Any] | None = None,
) -> Any:
    return type(
        "FakeConversation",
        (),
        {
            "workspace": _FakeWorkspace(Path("/tmp")),
            "state": type(
                "FakeState",
                (),
                {
                    "secret_registry": secret_registry or SecretRegistry(),
                    "agent_state": agent_state or {},
                },
            )(),
        },
    )()


def _cookie_registry() -> SecretRegistry:
    registry = SecretRegistry()
    registry.update_secrets(
        {
            PYROMIND_VALIDATE_AUTH_COOKIE_SECRET: StaticSecret(
                value=SecretStr("session=abc")
            )
        }
    )
    return registry


def _task_result_payload(
    *,
    task_status: str = "failed",
    nodes: list[dict[str, Any]] | None = None,
    nested: bool = True,
) -> dict[str, Any]:
    # Real calibrated XYFlow shape: top-level ``type`` is a render hint, the
    # business type lives in ``data.nodeType`` and the run status in
    # ``properties.dystatus``.
    default_nodes = [
        {
            "id": "1",
            "type": "default",
            "data": {
                "nodeType": "ModelTrainSFTNode",
                "display_name": "SFT Training",
            },
            "properties": {"dystatus": "Failed"},
        },
        {
            "id": "2",
            "type": "default",
            "data": {
                "nodeType": "LoadDatasetNode",
                "display_name": "Load Dataset",
            },
            "properties": {"dystatus": "Succeeded"},
        },
    ]
    node_list = nodes if nodes is not None else default_nodes
    if nested:
        return {
            "path": "/workflows/demo",
            "task_status": task_status,
            "workflow": {"nodes": node_list},
        }
    return {"path": "/workflows/demo", "task_status": task_status, "nodes": node_list}


def _log_payload(lines: list[str]) -> dict[str, Any]:
    return {
        "size": {"cols": 200, "row": len(lines)},
        "entries": [{"t": 0, "m": line} for line in lines],
    }


def _install_router(
    monkeypatch, routes: dict[str, Any]
) -> list[tuple[str, dict[str, str], dict[str, str]]]:
    calls: list[tuple[str, dict[str, str], dict[str, str]]] = []

    def fake_get(url, *, params, headers, timeout):
        calls.append((url, params, headers))
        for prefix, response in routes.items():
            if prefix in url:
                return response
        return _Response(404, {"message": "not found"})

    monkeypatch.setattr(httpx, "get", fake_get)
    return calls


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


def test_api_base_defaults_to_pre_for_local_and_pre(monkeypatch):
    routes = {
        "task_workflow_result": _Response(200, _task_result_payload()),
        "logs/node/raw": _Response(200, _log_payload(["line 1", "line 2"])),
    }
    calls = _install_router(monkeypatch, routes)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("PYROMIND_API_BASE", raising=False)

    executor = AnalyzeTaskFailureExecutor()
    observation = executor(AnalyzeTaskFailureAction(task_id="7758"))
    assert not observation.is_error
    assert calls[0][0] == f"{PRE_API_BASE}api/task_workflow_result"
    assert calls[0][1] == {"task_id": "7758"}

    monkeypatch.setenv("APP_ENV", "pre")
    observation = AnalyzeTaskFailureExecutor()(AnalyzeTaskFailureAction(task_id="7758"))
    assert not observation.is_error
    assert calls[-2][0].startswith(PRE_API_BASE)


def test_api_base_defaults_to_prod_for_online_envs(monkeypatch):
    routes = {
        "task_workflow_result": _Response(200, _task_result_payload()),
        "logs/node/raw": _Response(200, _log_payload(["ok"])),
    }
    calls = _install_router(monkeypatch, routes)
    for app_env in ("prod", "production", "online"):
        monkeypatch.setenv("APP_ENV", app_env)
        observation = AnalyzeTaskFailureExecutor()(
            AnalyzeTaskFailureAction(task_id="7758")
        )
        assert not observation.is_error
        assert calls[-2][0] == f"{PROD_API_BASE}api/task_workflow_result"


def test_api_base_override_env(monkeypatch):
    routes = {
        "task_workflow_result": _Response(200, _task_result_payload()),
        "logs/node/raw": _Response(200, _log_payload(["ok"])),
    }
    calls = _install_router(monkeypatch, routes)
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("PYROMIND_API_BASE", "https://legacy.test/std2/studio_api/")
    observation = AnalyzeTaskFailureExecutor()(AnalyzeTaskFailureAction(task_id="7758"))
    assert not observation.is_error
    assert calls[0][0] == "https://legacy.test/std2/studio_api/api/task_workflow_result"


def test_api_base_override_constructor(monkeypatch):
    routes = {
        "task_workflow_result": _Response(200, _task_result_payload()),
        "logs/node/raw": _Response(200, _log_payload(["ok"])),
    }
    calls = _install_router(monkeypatch, routes)
    executor = AnalyzeTaskFailureExecutor(api_base="https://custom.test/studio/")
    observation = executor(AnalyzeTaskFailureAction(task_id="7758"))
    assert not observation.is_error
    assert calls[0][0] == "https://custom.test/studio/api/task_workflow_result"


# ---------------------------------------------------------------------------
# Node parsing and failed-node detection
# ---------------------------------------------------------------------------


def test_detects_failed_nodes_from_workflow_nodes(monkeypatch):
    routes = {
        "task_workflow_result": _Response(200, _task_result_payload()),
        "logs/node/raw": _Response(
            200,
            _log_payload(["Traceback (most recent call last):", "RuntimeError: boom"]),
        ),
    }
    calls = _install_router(monkeypatch, routes)
    observation = AnalyzeTaskFailureExecutor()(AnalyzeTaskFailureAction(task_id="7758"))

    assert not observation.is_error
    assert observation.task_status == "failed"
    assert observation.source == "status"
    assert [node.node_id for node in observation.failed_nodes] == ["1"]
    assert observation.nodes[0].node_type == "ModelTrainSFTNode"
    # Only the failed node's log is fetched.
    assert calls[1][0].endswith("internal/logs/node/raw")
    assert calls[1][1] == {"nodeId": "1", "taskId": "7758"}
    assert "RuntimeError: boom" in observation.logs["1"]


def test_parses_top_level_nodes_and_alternate_status_keys(monkeypatch):
    nodes = [
        {
            "node_code": "n1",
            "node_type": "WandbConfigBuilderNode",
            "runStatus": "ERROR",
        },
        {
            "id": "n2",
            "data": {"nodeType": "LoadDatasetNode", "nodeStatus": "exception"},
        },
        {
            "id": "n3",
            "nodeType": "FilterNode",
            "state": {"message": "running"},
        },
    ]
    routes = {
        "task_workflow_result": _Response(
            200, _task_result_payload(nodes=nodes, nested=False)
        ),
        "logs/node/raw": _Response(200, _log_payload(["log"])),
    }
    _install_router(monkeypatch, routes)
    observation = AnalyzeTaskFailureExecutor()(AnalyzeTaskFailureAction(task_id="7758"))

    assert not observation.is_error
    assert [node.node_id for node in observation.failed_nodes] == ["n1", "n2"]
    assert observation.nodes[1].status == "exception"
    assert observation.nodes[2].status is None


def test_xyflow_shape_ignores_render_type_and_reads_dystatus(monkeypatch):
    # Mirrors the real task_workflow_result payload (task 7758): top-level
    # ``type: "default"`` must not shadow ``data.nodeType``, and the run status
    # lives in ``properties.dystatus``.
    nodes = [
        {
            "id": "19",
            "type": "default",
            "data": {"nodeType": "ModelMergeLoraNode", "display_name": "Merge LoRA"},
            "properties": {"dystatus": "Succeeded"},
        },
        {
            "id": "20",
            "type": "default",
            "data": {"nodeType": "VLLMInference", "display_name": "Inference (VLLM)"},
            "properties": {"dystatus": "Failed"},
        },
        {
            "id": "21",
            "type": "default",
            "data": {"nodeType": "TestLLMNode", "display_name": "LLM Test"},
            "properties": {"dystatus": "Omitted"},
        },
    ]
    routes = {
        "task_workflow_result": _Response(
            200, _task_result_payload(task_status="Error", nodes=nodes)
        ),
        "logs/node/raw": _Response(200, _log_payload(["Traceback", "boom"])),
    }
    calls = _install_router(monkeypatch, routes)
    observation = AnalyzeTaskFailureExecutor()(AnalyzeTaskFailureAction(task_id="7758"))

    assert not observation.is_error
    assert observation.task_status == "Error"
    assert observation.source == "status"
    assert [node.node_id for node in observation.failed_nodes] == ["20"]
    assert [node.node_type for node in observation.nodes] == [
        "ModelMergeLoraNode",
        "VLLMInference",
        "TestLLMNode",
    ]
    assert [node.node_name for node in observation.nodes] == [
        "Merge LoRA",
        "Inference (VLLM)",
        "LLM Test",
    ]
    # Only the failed node's log is fetched.
    assert calls[1][1] == {"nodeId": "20", "taskId": "7758"}
    assert observation.logs["20"]


def test_all_source_lists_available_nodes(monkeypatch):
    nodes = [
        {
            "id": "1",
            "type": "default",
            "data": {"nodeType": "ModelTrainSFTNode", "display_name": "SFT Training"},
        },
        {
            "id": "2",
            "type": "default",
            "data": {"nodeType": "LoadDatasetNode", "display_name": "Load Dataset"},
        },
    ]
    routes = {
        "task_workflow_result": _Response(
            200, _task_result_payload(task_status="Error", nodes=nodes)
        ),
    }
    _install_router(monkeypatch, routes)
    observation = AnalyzeTaskFailureExecutor()(AnalyzeTaskFailureAction(task_id="7758"))

    assert not observation.is_error
    assert observation.source == "all"
    assert observation.logs == {}
    assert "Available nodes:" in observation.text
    assert "- 1: SFT Training" in observation.text
    assert "- 2: Load Dataset" in observation.text


def test_status_without_failure_markers_is_not_failed(monkeypatch):
    nodes = [
        {"id": "1", "data": {"nodeType": "A", "status": "running"}},
        {"id": "2", "data": {"nodeType": "B", "status": "SUCCEEDED"}},
        {"id": "3", "data": {"nodeType": "C", "status": "failed: user cancelled"}},
    ]
    routes = {
        "task_workflow_result": _Response(200, _task_result_payload(nodes=nodes)),
        "logs/node/raw": _Response(200, _log_payload(["log"])),
    }
    _install_router(monkeypatch, routes)
    observation = AnalyzeTaskFailureExecutor()(AnalyzeTaskFailureAction(task_id="7758"))

    assert not observation.is_error
    assert [node.node_id for node in observation.failed_nodes] == ["3"]


def test_explicit_node_id_skips_status_detection(monkeypatch):
    routes = {
        "task_workflow_result": _Response(200, _task_result_payload()),
        "logs/node/raw": _Response(200, _log_payload(["detail line"])),
    }
    calls = _install_router(monkeypatch, routes)
    observation = AnalyzeTaskFailureExecutor()(
        AnalyzeTaskFailureAction(task_id="7758", node_id="2")
    )

    assert not observation.is_error
    assert observation.source == "explicit"
    assert observation.logs == {"2": "detail line"}
    assert calls[1][1] == {"nodeId": "2", "taskId": "7758"}


def test_no_identifiable_failed_node_returns_all_source(monkeypatch):
    nodes = [
        {"id": "1", "data": {"nodeType": "A"}},
        {"id": "2", "data": {"nodeType": "B"}},
    ]
    routes = {
        "task_workflow_result": _Response(
            200, _task_result_payload(task_status="running", nodes=nodes)
        ),
    }
    _install_router(monkeypatch, routes)
    observation = AnalyzeTaskFailureExecutor()(AnalyzeTaskFailureAction(task_id="7758"))

    assert not observation.is_error
    assert observation.source == "all"
    assert observation.logs == {}
    assert "Pass `node_id`" in observation.text


# ---------------------------------------------------------------------------
# Log tail handling
# ---------------------------------------------------------------------------


def test_log_tail_keeps_last_lines_only(monkeypatch):
    lines = [f"line {i}" for i in range(150)]
    routes = {
        "task_workflow_result": _Response(200, _task_result_payload()),
        "logs/node/raw": _Response(200, _log_payload(lines)),
    }
    _install_router(monkeypatch, routes)
    observation = AnalyzeTaskFailureExecutor()(
        AnalyzeTaskFailureAction(task_id="7758", tail_lines=100)
    )
    assert "line 49" not in observation.logs["1"]
    assert "line 50" in observation.logs["1"]
    assert len(observation.logs["1"].splitlines()) == 100


def test_log_max_chars_trims_tail(monkeypatch):
    lines = ["x" * 90, "y" * 90, "z" * 90]
    routes = {
        "task_workflow_result": _Response(200, _task_result_payload()),
        "logs/node/raw": _Response(200, _log_payload(lines)),
    }
    _install_router(monkeypatch, routes)
    observation = AnalyzeTaskFailureExecutor()(
        AnalyzeTaskFailureAction(task_id="7758", max_log_chars=100)
    )
    assert len(observation.logs["1"]) == 100
    assert observation.logs["1"].endswith("z" * 90)


def test_empty_log_reports_placeholder(monkeypatch):
    routes = {
        "task_workflow_result": _Response(200, _task_result_payload()),
        "logs/node/raw": _Response(200, {"size": {"cols": 0, "row": 0}}),
    }
    _install_router(monkeypatch, routes)
    observation = AnalyzeTaskFailureExecutor()(AnalyzeTaskFailureAction(task_id="7758"))
    assert observation.logs["1"] == "<empty node log>"


# ---------------------------------------------------------------------------
# Auth headers
# ---------------------------------------------------------------------------


def test_forwarded_headers_and_browser_ua(monkeypatch):
    routes = {
        "task_workflow_result": _Response(200, _task_result_payload()),
        "logs/node/raw": _Response(200, _log_payload(["ok"])),
    }
    calls = _install_router(monkeypatch, routes)
    registry = _cookie_registry()
    registry.update_secrets(
        {"PYROMIND_AUTHORIZATION": StaticSecret(value=SecretStr("Bearer token"))}
    )
    conversation = cast(
        Any,
        _fake_conversation(
            secret_registry=registry,
            agent_state={
                PYROMIND_VALIDATE_HEADERS_STATE_KEY: {"x-cluster": "us-west-1"}
            },
        ),
    )
    executor = AnalyzeTaskFailureExecutor(
        headers={"accept-language": "zh-CN"},
        secret_headers={"authorization": "PYROMIND_AUTHORIZATION"},
    )
    observation = executor(AnalyzeTaskFailureAction(task_id="7758"), conversation)

    assert not observation.is_error
    _, _, headers = calls[0]
    assert headers["user-agent"] == USER_AGENT
    assert headers["cookie"] == "session=abc"
    assert headers["x-cluster"] == "us-west-1"
    assert headers["accept-language"] == "zh-CN"
    assert headers["authorization"] == "Bearer token"


def test_missing_cookie_secret_fails_cleanly(monkeypatch):
    routes = {
        "task_workflow_result": _Response(200, _task_result_payload()),
        "logs/node/raw": _Response(200, _log_payload(["ok"])),
    }
    calls = _install_router(monkeypatch, routes)
    conversation = cast(Any, _fake_conversation(secret_registry=SecretRegistry()))
    observation = AnalyzeTaskFailureExecutor(
        secret_headers={"cookie": "MISSING_SECRET"}
    )(AnalyzeTaskFailureAction(task_id="7758"), conversation)
    assert observation.is_error
    assert observation.error_message is not None
    assert not calls


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_http_401_returns_error_with_no_retry_guidance(monkeypatch):
    routes = {
        "task_workflow_result": _Response(
            401, {"message": "unauthorized"}, text="unauthorized"
        ),
    }
    _install_router(monkeypatch, routes)
    observation = AnalyzeTaskFailureExecutor()(AnalyzeTaskFailureAction(task_id="7758"))
    assert observation.is_error
    assert "HTTP 401" in observation.text
    assert "Do not retry" in observation.text


def test_invalid_json_returns_error(monkeypatch):
    routes = {
        "task_workflow_result": _Response(200, ValueError("bad json")),
    }
    _install_router(monkeypatch, routes)
    observation = AnalyzeTaskFailureExecutor()(AnalyzeTaskFailureAction(task_id="7758"))
    assert observation.is_error
    assert "invalid JSON" in observation.text


def test_empty_node_list_returns_error(monkeypatch):
    routes = {
        "task_workflow_result": _Response(
            200, {"task_status": "failed", "workflow": {"nodes": []}}
        ),
    }
    _install_router(monkeypatch, routes)
    observation = AnalyzeTaskFailureExecutor()(AnalyzeTaskFailureAction(task_id="7758"))
    assert observation.is_error
    assert "no workflow nodes" in observation.text


def test_log_fetch_failure_reports_placeholder(monkeypatch):
    routes = {
        "task_workflow_result": _Response(200, _task_result_payload()),
        "logs/node/raw": _Response(500, {"message": "boom"}, text="boom"),
    }
    _install_router(monkeypatch, routes)
    observation = AnalyzeTaskFailureExecutor()(AnalyzeTaskFailureAction(task_id="7758"))
    assert not observation.is_error
    assert "<log fetch failed:" in observation.logs["1"]


# ---------------------------------------------------------------------------
# Operator source integration (get_node_function_signature API)
# ---------------------------------------------------------------------------


def test_include_source_fetches_operator_source(monkeypatch):
    routes = {
        "task_workflow_result": _Response(200, _task_result_payload()),
        "logs/node/raw": _Response(
            200, _log_payload(["Traceback", "RuntimeError: boom"])
        ),
    }
    _install_router(monkeypatch, routes)
    post_calls: list[tuple[str, dict, dict]] = []

    def fake_post(url, *, json, headers, timeout):
        post_calls.append((url, json, headers))
        return _Response(
            200, {"success": True, "data": {"source_code": "def train(...):\n    pass"}}
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    registry = _cookie_registry()
    registry.update_secrets({"auth_token": StaticSecret(value=SecretStr("token-123"))})
    conversation = cast(Any, _fake_conversation(secret_registry=registry))

    observation = AnalyzeTaskFailureExecutor()(
        AnalyzeTaskFailureAction(task_id="7758"), conversation
    )

    assert not observation.is_error
    assert "RuntimeError: boom" in observation.logs["1"]
    assert observation.node_sources["1"] == "def train(...):\n    pass"
    assert post_calls[0][0].endswith("api/agent/nodes/function_signature")
    assert post_calls[0][1] == {
        "node_name": "ModelTrainSFTNode",
        "node_type": None,
        "include_source": True,
        "max_source_lines": 200,
    }
    assert post_calls[0][2]["auth_token"] == "token-123"
    # Source block is rendered into the observation text.
    assert "operator source" in observation.text


def test_include_source_false_skips_source_fetch(monkeypatch):
    routes = {
        "task_workflow_result": _Response(200, _task_result_payload()),
        "logs/node/raw": _Response(200, _log_payload(["Traceback", "boom"])),
    }
    _install_router(monkeypatch, routes)
    post_calls: list = []

    def fake_post(url, *, json, headers, timeout):
        post_calls.append(url)
        return _Response(200, {"success": True, "data": {}})

    monkeypatch.setattr(httpx, "post", fake_post)
    registry = _cookie_registry()
    registry.update_secrets({"auth_token": StaticSecret(value=SecretStr("token-123"))})
    conversation = cast(Any, _fake_conversation(secret_registry=registry))

    observation = AnalyzeTaskFailureExecutor()(
        AnalyzeTaskFailureAction(task_id="7758", include_source=False), conversation
    )

    assert not observation.is_error
    assert observation.node_sources == {}
    assert post_calls == []


def test_source_fetch_failure_keeps_logs(monkeypatch):
    routes = {
        "task_workflow_result": _Response(200, _task_result_payload()),
        "logs/node/raw": _Response(200, _log_payload(["Traceback", "boom"])),
    }
    _install_router(monkeypatch, routes)

    def fake_post(url, *, json, headers, timeout):
        return _Response(404, {"message": "not found"})

    monkeypatch.setattr(httpx, "post", fake_post)
    registry = _cookie_registry()
    registry.update_secrets({"auth_token": StaticSecret(value=SecretStr("token-123"))})
    conversation = cast(Any, _fake_conversation(secret_registry=registry))

    observation = AnalyzeTaskFailureExecutor()(
        AnalyzeTaskFailureAction(task_id="7758"), conversation
    )

    assert not observation.is_error
    assert "RuntimeError" not in observation.text or "boom" in observation.logs["1"]
    assert observation.logs["1"]  # logs survive a failed source fetch
    assert observation.node_sources == {}


# ---------------------------------------------------------------------------
# Registration and creation
# ---------------------------------------------------------------------------


def test_tool_is_registered_and_creates_executor():
    resolved = resolve_tool(Tool(name=AnalyzeTaskFailureTool.name), cast(Any, None))
    assert isinstance(resolved[0], AnalyzeTaskFailureTool)
    instances = AnalyzeTaskFailureTool.create(headers={"x-cluster": "us-west-1"})
    assert len(instances) == 1
    assert isinstance(instances[0].executor, AnalyzeTaskFailureExecutor)


def test_create_rejects_unknown_params():
    with pytest.raises(ValueError, match="unknown params"):
        AnalyzeTaskFailureTool.create(bogus_param=1)
