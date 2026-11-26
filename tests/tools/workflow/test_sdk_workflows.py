from __future__ import annotations

import json
import subprocess

import httpx
import pytest
from pyromind_sdk import PyroMindAPIClient
from pyromind_sdk.client.base import PyroMindAPIError

from pyromind_common.platform.credentials import (
    materialize_runtime_environment,
    runtime_environment_prefix,
)
from pyromind_common.platform.workflows import (
    DebugBudget,
    DebugBudgetPersistenceError,
    WorkflowBatchSubmissionError,
    WorkflowPreparationError,
    WorkflowSubmissionUncertain,
    prepare_debug_workflow,
    prepare_workflow,
    submit_debug_workflow,
    submit_workflow,
    submit_workflow_batch,
)


def test_sdk_binds_worker_values_only_in_transport(
    studio_http, direct_sdk, monkeypatch
):
    key = "synthetic-worker-'key;$()"
    monkeypatch.setenv("LLM_AUTH_TOKEN", key)
    declaration = runtime_environment_prefix(("LLM_AUTH_TOKEN",))
    graph = {
        "name": "worker",
        "nodes": [{"data": {"config": {"command": declaration + " python runner.py"}}}],
        "edges": [],
    }
    frozen = json.dumps(graph)
    studio_http.responses.append(
        (200, {"data": {"task_id": "111", "name": "worker", "status": "Pending"}})
    )
    response = submit_workflow(direct_sdk, graph, out_id="signed-context")
    assert response.task_id == "111"
    assert json.dumps(graph) == frozen
    assert key not in frozen
    command = studio_http.requests[0][1]["workflow"]["nodes"][0]["data"]["config"][
        "command"
    ]
    assert "{{pyromind.runtime:" not in command
    assert "export LLM_AUTH_TOKEN=" in command


def test_sdk_submission_error_cannot_echo_worker_key(
    studio_http, direct_sdk, monkeypatch, caplog, capsys
):
    key = "synthetic-secret-returned-without-key-label"
    monkeypatch.setenv("LLM_AUTH_TOKEN", key)
    graph = {
        "name": "worker",
        "nodes": [
            {
                "data": {
                    "config": {
                        "command": runtime_environment_prefix(("LLM_AUTH_TOKEN",))
                        + " python runner.py"
                    }
                }
            }
        ],
    }
    studio_http.responses.append((422, {"detail": key}))
    with caplog.at_level("DEBUG", logger="pyromind_sdk"):
        with pytest.raises(PyroMindAPIError) as failure:
            submit_workflow(direct_sdk, graph, out_id="signed-context")
    assert failure.value.status_code == 422
    assert failure.value.response is None
    assert key not in str(failure.value)
    assert key not in caplog.text
    captured = capsys.readouterr()
    assert key not in captured.out + captured.err
    assert len(studio_http.requests) == 1


def test_worker_environment_preserves_shell_failure_and_quotes_values():
    key = "synthetic-'key;$(printf injected)\nvalue"
    prefix = runtime_environment_prefix(("LLM_AUTH_TOKEN",))

    def command(source):
        graph = {"nodes": [{"data": {"config": {"command": source}}}]}
        prepared = materialize_runtime_environment(graph, {"LLM_AUTH_TOKEN": key})
        return prepared["nodes"][0]["data"]["config"]["command"]

    success = subprocess.run(
        ["/bin/sh", "-c", command(prefix + ' printf %s "$LLM_AUTH_TOKEN"')],
        capture_output=True,
        text=True,
        check=False,
    )
    assert success.returncode == 0
    assert success.stdout == key
    failed = subprocess.run(
        ["/bin/sh", "-c", command("false && " + prefix + " printf ran")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode != 0
    assert failed.stdout == ""


def test_missing_worker_environment_fails_before_sdk_create(
    studio_http, direct_sdk, monkeypatch
):
    monkeypatch.delenv("LLM_AUTH_TOKEN", raising=False)
    graph = {
        "name": "worker",
        "nodes": [
            {
                "data": {
                    "config": {
                        "command": runtime_environment_prefix(("LLM_AUTH_TOKEN",))
                    }
                }
            }
        ],
    }
    with pytest.raises(ValueError, match="worker environment is unavailable"):
        submit_workflow(direct_sdk, graph, out_id="signed-context")
    assert not studio_http.requests


def test_model_configuration_is_frozen_until_submission(monkeypatch):
    profile = {
        "DF_API_URL": "https://model.invalid/v1/chat/completions",
        "DF_MODEL_NAME": "one",
    }
    monkeypatch.setenv("PYROMIND_DATAFLOW_PROFILES", json.dumps({"text": profile}))
    graph = {
        "nodes": [
            {
                "data": {
                    "config": {
                        "command": runtime_environment_prefix(
                            tuple(profile), profile="text"
                        )
                        + ' printf "%s|%s" "${DF_API_KEY-unset}" "$DF_MODEL_NAME"'
                    }
                }
            }
        ]
    }
    prepared = materialize_runtime_environment(graph)
    executed = subprocess.run(
        ["/bin/sh", "-c", prepared["nodes"][0]["data"]["config"]["command"]],
        env={"DF_API_KEY": "unrelated-worker-key", "DF_MODEL_NAME": "worker-model"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert executed.returncode == 0
    assert executed.stdout == "unset|one"
    profile["DF_MODEL_NAME"] = "two"
    with pytest.raises(ValueError, match="Model configuration changed"):
        materialize_runtime_environment(
            graph, {"PYROMIND_DATAFLOW_PROFILES": json.dumps({"text": profile})}
        )


def test_preparation_and_sdk_use_server_graph_without_product_registration(
    studio_http, direct_sdk
):
    graph = {"name": "server-graph", "nodes": [], "edges": []}
    studio_http.responses.extend(
        [
            (200, {"success": True, "data": {"valid": True, "workflow": graph}}),
            (200, {"success": True, "data": {"valid": True}}),
            (
                200,
                {
                    "success": True,
                    "data": {
                        "task_id": "101",
                        "name": "server-graph",
                        "status": "Pending",
                    },
                },
            ),
        ]
    )
    with httpx.Client(
        base_url=studio_http.url + "/workflows/", trust_env=False
    ) as validator:
        prepared = prepare_workflow(validator, dsl="source = Node(id=1)")
    response = submit_workflow(direct_sdk, prepared, out_id="signed-context")

    assert response.task_id == "101"
    assert [path for path, _ in studio_http.requests] == [
        "/workflows/dsl/validate",
        "/workflows/resources/validate",
        "/studio/tasks",
    ]
    assert studio_http.requests[1][1] == {"workflow": graph}
    assert studio_http.requests[2][1] == {
        "name": "server-graph",
        "workflow": graph,
        "out_id": "signed-context",
    }


@pytest.mark.parametrize("status", [403, 503])
def test_create_failure_does_not_retry(studio_http, direct_sdk, status):
    studio_http.responses.append((status, {"message": "rejected"}))
    error = PyroMindAPIError if status == 403 else WorkflowSubmissionUncertain
    with pytest.raises(error):
        submit_workflow(
            direct_sdk, {"name": "one", "nodes": [], "edges": []}, out_id="ctx"
        )
    assert len(studio_http.requests) == 1


def test_submit_rejects_default_sdk_retry_configuration(studio_http):
    with PyroMindAPIClient(api_key="key", base_url=studio_http.url) as client:
        with pytest.raises(ValueError, match="max_retries=0"):
            submit_workflow(client, {"name": "one"}, out_id="ctx")
    assert not studio_http.requests


def test_failed_batch_preserves_accepted_tasks_and_stops(studio_http, direct_sdk):
    studio_http.responses.extend(
        [
            (
                200,
                {
                    "success": True,
                    "data": {"task_id": "101", "name": "a", "status": "Pending"},
                },
            ),
            (503, {"message": "unavailable"}),
        ]
    )
    with pytest.raises(WorkflowBatchSubmissionError) as error:
        submit_workflow_batch(
            direct_sdk, [{"name": name} for name in "abc"], out_id="ctx"
        )
    assert [item.task_id for item in error.value.accepted] == ["101"]
    assert isinstance(error.value.error, WorkflowSubmissionUncertain)
    assert len(studio_http.requests) == 2


def test_resource_rejection_prevents_submission(studio_http):
    graph = {"nodes": [], "edges": []}
    studio_http.responses.extend(
        [
            (200, {"success": True, "data": {"valid": True, "workflow": graph}}),
            (
                200,
                {
                    "success": True,
                    "data": {"valid": False, "errors": [{"code": "NO_CAPACITY"}]},
                },
            ),
        ]
    )
    with httpx.Client(base_url=studio_http.url + "/", trust_env=False) as validator:
        with pytest.raises(WorkflowPreparationError) as error:
            prepare_workflow(validator, dsl="source = Node(id=1)")
    assert error.value.stage == "resources"
    assert error.value.details["errors"] == [{"code": "NO_CAPACITY"}]
    assert len(studio_http.requests) == 2


def test_debug_forces_test_mode_without_mutating_source():
    workflow = {"execution_argos": [{"execution_mode": "prod"}, {"other": "value"}]}
    prepared = prepare_debug_workflow(workflow)
    assert workflow["execution_argos"][0] == {"execution_mode": "prod"}
    assert prepared["execution_argos"] == [
        {"other": "value"},
        {"execution_mode": "test"},
    ]


def test_debug_budget_counts_only_accepted_submissions(studio_http, direct_sdk):
    seen = []

    def resource_validation(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"success": True, "data": {"valid": True}})

    studio_http.responses.extend(
        [
            (403, {"message": "rejected"}),
            (
                200,
                {
                    "success": True,
                    "data": {"task_id": "101", "name": "debug", "status": "Pending"},
                },
            ),
        ]
    )
    budget = DebugBudget(maximum=1)
    with httpx.Client(
        base_url="https://validation.test/",
        transport=httpx.MockTransport(resource_validation),
    ) as validator:
        with pytest.raises(PyroMindAPIError):
            submit_debug_workflow(
                direct_sdk,
                {"name": "debug"},
                out_id="ctx",
                budget=budget,
                validator=validator,
            )
        assert budget.accepted == 0
        submit_debug_workflow(
            direct_sdk,
            {"name": "debug"},
            out_id="ctx",
            budget=budget,
            validator=validator,
        )
        assert budget.accepted == 1
        with pytest.raises(ValueError, match="limit reached"):
            submit_debug_workflow(
                direct_sdk,
                {"name": "debug"},
                out_id="ctx",
                budget=budget,
                validator=validator,
            )
    assert len(studio_http.requests) == 2
    assert all(
        item["workflow"]["execution_argos"] == [{"execution_mode": "test"}]
        for item in seen
    )


def test_debug_budget_reload_preserves_legacy_count_and_other_state(
    tmp_path, studio_http, direct_sdk
):
    path = tmp_path / "business-state.json"
    path.write_text(json.dumps({"pyromind_workflow_attempts": 9, "other": "kept"}))
    reloaded = DebugBudget.load(path)
    studio_http.responses.append(
        (
            200,
            {
                "success": True,
                "data": {"task_id": "101", "name": "debug", "status": "Pending"},
            },
        )
    )
    graph = prepare_debug_workflow({"name": "debug"})
    submit_debug_workflow(
        direct_sdk, graph, out_id="ctx", budget=DebugBudget.load(path), validated=True
    )
    assert json.loads(path.read_text()) == {
        "pyromind_workflow_attempts": 10,
        "other": "kept",
    }
    with pytest.raises(ValueError, match="limit reached"):
        submit_debug_workflow(
            direct_sdk, graph, out_id="ctx", budget=reloaded, validated=True
        )
    assert len(studio_http.requests) == 1


def test_debug_budget_save_failure_keeps_real_accepted_response(
    tmp_path, studio_http, direct_sdk, monkeypatch
):
    studio_http.responses.append(
        (
            200,
            {
                "success": True,
                "data": {"task_id": "101", "name": "debug", "status": "Pending"},
            },
        )
    )
    budget = DebugBudget.load(tmp_path / "business-state.json")

    def fail_save(self):
        raise OSError("disk full")

    monkeypatch.setattr(DebugBudget, "_save", fail_save)
    with pytest.raises(DebugBudgetPersistenceError) as error:
        submit_debug_workflow(
            direct_sdk,
            prepare_debug_workflow({"name": "debug"}),
            out_id="ctx",
            budget=budget,
            validated=True,
        )
    assert error.value.response.task_id == "101"
    assert budget.accepted == 1
    assert len(studio_http.requests) == 1


def test_debug_requires_the_already_validated_test_graph(studio_http, direct_sdk):
    with pytest.raises(ValueError, match="already-validated fixed test graph"):
        submit_debug_workflow(
            direct_sdk,
            {"name": "production"},
            out_id="ctx",
            budget=DebugBudget(),
            validated=True,
        )
    assert not studio_http.requests
