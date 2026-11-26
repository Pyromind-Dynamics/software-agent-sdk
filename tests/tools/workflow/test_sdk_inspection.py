"""Inspect Studio facts through the SDK's native HTTP transport."""

import pytest

from pyromind_common.platform.inspection import (
    WorkflowInspectionError,
    inspect_workflow_task,
    workflow_execution_state,
)


@pytest.fixture
def inspection_payload():
    return {
        "success": True,
        "data": {
            "task_id": "1304",
            "status": "Failed",
            "nodes": [
                {
                    "node_id": 43050,
                    "node_code": "train",
                    "node_name": "Train",
                    "status": "Failed",
                },
                {
                    "node_id": 43051,
                    "node_name": "Merge",
                    "start_at": "2026-11-24T10:00:00",
                    "end_at": "2026-11-24T10:05:00",
                },
            ],
        },
    }


def test_sdk_preserves_raw_execution_state_without_guessing_from_dates(
    studio_http, direct_sdk, inspection_payload
):
    studio_http.responses.append((200, inspection_payload))
    result = inspect_workflow_task(direct_sdk, "1304")

    assert result.task_state == "failed"
    assert [node.node_id for node in result.nodes] == ["43050", "43051"]
    assert [node.state for node in result.nodes] == ["failed", "unknown"]
    assert result.nodes[0].raw_status == "Failed"
    assert result.nodes[1].raw_status is None
    assert result.node_status_available is False
    assert [node.node_id for node in result.failed_nodes] == ["43050"]
    assert result.log is None and result.selected_node is None
    assert studio_http.requests == [("/studio/tasks/1304", {})]
    assert studio_http.headers[0]["Authorization"] == "Bearer test-key"
    assert "Cookie" not in studio_http.headers[0]


def test_sdk_selects_node_code_but_reads_log_with_database_id_and_true_tail(
    studio_http, direct_sdk, inspection_payload
):
    studio_http.responses.extend(
        [
            (200, inspection_payload),
            (200, {"data": {"logs": ["first\nsecond\n", "third\nfourth"]}}),
        ]
    )
    result = inspect_workflow_task(
        direct_sdk,
        "1304",
        node_id="train",
        include_log=True,
        tail_lines=2,
        max_log_chars=10,
    )

    assert result.selected_node is not None
    assert result.selected_node.node_id == "43050"
    assert result.log == "ird\nfourth"
    assert result.node_log_available and result.log_error is None
    assert studio_http.requests == [
        ("/studio/tasks/1304", {}),
        ("/studio/tasks/1304/nodes/43050/logs", {}),
    ]


@pytest.mark.parametrize("logs", [[], [{"message": "unexpected SDK log entry"}]])
def test_empty_or_malformed_sdk_logs_preserve_task_facts(
    studio_http, direct_sdk, inspection_payload, logs
):
    studio_http.responses.extend(
        [(200, inspection_payload), (200, {"data": {"logs": logs}})]
    )
    result = inspect_workflow_task(
        direct_sdk, "1304", node_id="43050", include_log=True
    )
    assert result.task_state == "failed"
    assert result.selected_node is not None
    assert result.selected_node.state == "failed"
    assert not result.node_log_available
    assert bool(result.log_error) == bool(logs)


@pytest.mark.parametrize("stage", ["task", "log"])
def test_sdk_inspection_errors_do_not_echo_response_body(
    studio_http, direct_sdk, inspection_payload, stage, caplog, capsys
):
    secret = "synthetic-unlabelled-key-in-server-response"
    if stage == "log":
        studio_http.responses.append((200, inspection_payload))
    studio_http.responses.append((401 if stage == "task" else 503, {"detail": secret}))

    with caplog.at_level("DEBUG", logger="pyromind_sdk"):
        if stage == "task":
            with pytest.raises(WorkflowInspectionError) as failure:
                inspect_workflow_task(direct_sdk, "1304")
            assert failure.value.status_code == 401
            output = str(failure.value)
        else:
            result = inspect_workflow_task(
                direct_sdk, "1304", node_id="train", include_log=True
            )
            assert result.task_state == "failed"
            assert result.log_error and not result.node_log_available
            output = result.model_dump_json()

    captured = capsys.readouterr()
    assert secret not in output + captured.out + captured.err + caplog.text


@pytest.mark.parametrize("nodes", [None, []])
def test_empty_node_lists_do_not_infer_node_completion(
    studio_http, direct_sdk, inspection_payload, nodes
):
    inspection_payload["data"].update(status="Succeeded", nodes=nodes)
    studio_http.responses.append((200, inspection_payload))
    result = inspect_workflow_task(direct_sdk, "1304")
    assert result.task_state == "succeeded"
    assert result.nodes == result.failed_nodes == []
    assert not result.node_status_available


def test_explicit_code_links_definition_state_and_identity_without_name_matching(
    studio_http, direct_sdk, inspection_payload
):
    del inspection_payload["data"]["nodes"][0]["status"]
    inspection_payload["data"]["workflow"] = {
        "nodes": [
            {
                "id": "train",
                "name": "Train",
                "node_type": "Sft",
                "properties": {"node_status": "Running"},
            },
            {"id": "merge", "name": "Merge", "properties": {"dystatus": "Succeeded"}},
        ]
    }
    studio_http.responses.append((200, inspection_payload))
    result = inspect_workflow_task(direct_sdk, "1304")

    assert result.nodes[0].node_id == "43050"
    assert result.nodes[0].node_type == "Sft"
    assert result.nodes[0].state == "running"
    assert result.nodes[1].state == "unknown"

    inspection_payload["data"]["nodes"] = []
    studio_http.responses.append((200, inspection_payload))
    result = inspect_workflow_task(direct_sdk, "1304")
    assert [node.node_id for node in result.nodes] == ["train", "merge"]
    assert [node.state for node in result.nodes] == ["running", "succeeded"]
    assert result.node_status_available


def test_ambiguous_definition_codes_do_not_supply_a_guessed_state(
    studio_http, direct_sdk, inspection_payload
):
    del inspection_payload["data"]["nodes"][0]["status"]
    inspection_payload["data"]["workflow"] = {
        "nodes": [
            {"id": "train", "status": "Succeeded"},
            {"id": "train", "status": "Failed"},
        ]
    }
    studio_http.responses.append((200, inspection_payload))
    result = inspect_workflow_task(direct_sdk, "1304")
    assert result.nodes[0].state == "unknown"


@pytest.mark.parametrize("node_id", ["missing", "Merge"])
def test_unknown_node_does_not_request_a_guessed_log(
    studio_http, direct_sdk, inspection_payload, node_id
):
    studio_http.responses.append((200, inspection_payload))
    with pytest.raises(ValueError, match="exactly one node"):
        inspect_workflow_task(direct_sdk, "1304", node_id=node_id, include_log=True)
    assert len(studio_http.requests) == 1


def test_invalid_selection_fails_before_http(studio_http, direct_sdk):
    with pytest.raises(ValueError, match="node_id is required"):
        inspect_workflow_task(direct_sdk, "1304", include_log=True)
    for task_id in ("../1304", ".", ".."):
        with pytest.raises(ValueError):
            inspect_workflow_task(direct_sdk, task_id)
    with pytest.raises(ValueError):
        inspect_workflow_task(direct_sdk, "1304", node_id="..")
    assert not studio_http.requests


@pytest.mark.parametrize("mismatch", ["wrong-task", "invalid-node"])
def test_mismatched_or_malformed_task_response_is_rejected(
    studio_http, direct_sdk, inspection_payload, mismatch
):
    if mismatch == "wrong-task":
        inspection_payload["data"]["task_id"] = "1305"
    else:
        inspection_payload["data"]["nodes"] = [{"node_name": "No identity"}]
    studio_http.responses.append((200, inspection_payload))
    with pytest.raises(WorkflowInspectionError):
        inspect_workflow_task(direct_sdk, "1304")


@pytest.mark.parametrize(
    ("raw", "state"),
    [
        ("Failed", "failed"),
        ("not_started", "pending"),
        ("Timeout", "failed"),
        ("Stopped", "cancelled"),
        ("no error", "unknown"),
        ("platform-state", "unknown"),
    ],
)
def test_normalization_does_not_guess_from_status_fragments(raw, state):
    assert workflow_execution_state(raw) == state
