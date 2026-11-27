from dataclasses import replace
from unittest.mock import Mock

import pytest

from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.graph_repair import GRAPH_REPAIR_PROPOSAL
from pyromind_runtime.domain.snapshot import ConversationSnapshot, ExternalTaskState
from pyromind_runtime.domain.task_correlation import TaskOrigin
from pyromind_runtime.ports.graph_repair import GraphRepairSubmitter

from .fake_adapter import FakeAdapter


@pytest.fixture
def repair(tmp_path):
    context = RequestContext(user_id="owner", profile_id="echomind", x_cluster="east")
    service = Mock(spec=GraphRepairSubmitter)
    runtime = ConversationRuntime(
        tmp_path, FakeAdapter(), graph_repair_submitter=service
    )
    store = runtime._store("original")
    store.conversation_dir.mkdir()
    record = ExternalTaskState(
        task_id="repair:one",
        kind="echomind_debug_repair",
        submitted_at="2026-01-01",
        updated_at="2026-01-01",
        status="failed",
        completion_result={
            "project_id": "project",
            "handoff_id": "one",
            "run_id": "run",
            "stage_log_excerpt": "reader failed",
        },
    )
    revision = record.model_copy(
        update={
            "task_id": "repair:one:revision",
            "completion_result": {
                "handoff_id": "one",
                "proposal_validation_error": "missing asset",
            },
        }
    )
    store.create(
        ConversationSnapshot(
            conversation_id="original",
            capabilities=HarnessCapabilities(),
        ),
        user_id=context.user_id,
        profile_id=context.profile_id,
        tool_allowlist=(GRAPH_REPAIR_PROPOSAL,),
    )
    for task in (record, revision):
        store.append(
            ProductEvent(
                conversation_id="original",
                type="external_task.submitted",
                payload=task.model_dump(mode="json"),
            )
        )
    return runtime, service, context


async def test_repair_inspect_survives_followup_and_retains_original_evidence(repair):
    runtime, service, context = repair
    service.inspect.return_value = {"nodes": [{"node_id": "dataset"}]}
    result = await runtime.graph_repair_proposal(
        "original", {"action": "inspect"}, context
    )
    service.inspect.assert_called_once_with(
        conversation_id="original",
        project_id="project",
        run_id="run",
        handoff_id="one",
        context=context,
    )
    assert result["evidence"]["stage_log_excerpt"] == "reader failed"
    assert result["evidence"]["proposal_validation_error"] == "missing asset"
    assert result["evidence"]["nodes"] == [{"node_id": "dataset"}]
    service.submit.assert_not_called()
    for other in (
        replace(context, user_id="other"),
        replace(context, task_origin=TaskOrigin(type="subagent", subagent_id="eval")),
    ):
        with pytest.raises(PermissionError):
            await runtime.graph_repair_proposal(
                "original", {"action": "inspect"}, other
            )
    assert service.inspect.call_count == 1


async def test_submit_returns_same_validator_feedback_and_review_receipt(repair):
    runtime, service, context = repair
    proposal = {
        "release_digest": "sha256:" + "a" * 64,
        "node_id": "dataset",
        "reason": "Verified reader with five rows",
        "runtime_config": {},
    }
    replacement = {"old_sha256": "b" * 64, "new_sha256": "c" * 64}
    args = {
        "action": "submit",
        "handoff_id": "another",
        "proposal": proposal,
        "recipe_asset_replacement": replacement,
    }
    assert (await runtime.graph_repair_proposal("original", args, context))[
        "status"
    ] == "invalid"
    service.submit.assert_not_called()
    args["handoff_id"] = "one"
    malformed = await runtime.graph_repair_proposal(
        "original", {**args, "proposal": {**proposal, "reason": ""}}, context
    )
    assert malformed["status"] == "invalid"
    assert "proposal.reason" in malformed["errors"][0]
    service.submit.assert_not_called()
    service.submit.side_effect = ValueError("Frozen replacement digest does not match")
    result = await runtime.graph_repair_proposal("original", args, context)
    assert result == {
        "status": "invalid",
        "error": "Frozen replacement digest does not match",
    }
    service.submit.side_effect = None
    service.submit.return_value = {
        "status": "reviewable",
        "proposal_sha256": "verified",
    }
    assert (
        await runtime.graph_repair_proposal("original", args, context)
        == service.submit.return_value
    )
    service.submit.assert_called_with(
        conversation_id="original",
        project_id="project",
        run_id="run",
        handoff_id="one",
        proposal={**proposal, "recipe_asset_replacement": replacement},
        read_asset=None,
        context=context,
    )
