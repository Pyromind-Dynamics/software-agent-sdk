import json
from datetime import timedelta

from pyromind_runtime.application.consolidation_repairs import plan_repair
from pyromind_runtime.domain.base import utc_now
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.commands import CommandReceipt
from pyromind_runtime.domain.content import JsonObject, TextContent
from pyromind_runtime.domain.snapshot import (
    ConversationSnapshot,
    TimelineOperation,
)


def test_repair_waits_for_author_then_reports_persistent_gap():
    snapshot = ConversationSnapshot(
        conversation_id="test", capabilities=HarnessCapabilities()
    )

    def next_repair(*, state=snapshot, receipts=(), revision="", retry=None):
        return plan_repair(
            stage="dataset",
            evidence="sha256:" + "3" * 64,
            facts={"stage": "dataset", "development_task": {"task_id": "1086"}},
            issues=["dataset_sft_export_missing"],
            revision=revision,
            receipts=receipts,
            snapshot=state,
            retry_command_id=retry,
        )

    first, command = next_repair()
    assert first.status == "working" and command
    assert "主会话直接补齐" in command.content[0].model_dump_json()
    assert next_repair()[1] == command
    assert "1086" in command.content[0].model_dump_json()
    assert "prepare_pipeline" in command.content[0].model_dump_json()
    assert "df_submit_pipeline" not in command.content[0].model_dump_json()
    permission = snapshot.model_copy(update={"status": "waiting_for_confirmation"})
    assert "待授权" in next_repair(state=permission)[0].message
    active = snapshot.model_copy(update={"status": "running"})
    assert "平台" not in next_repair(state=active)[0].message
    old = CommandReceipt(
        command_id=first.command_id,
        status="completed",
        recorded_at=utc_now() - timedelta(minutes=10),
    )
    child = TimelineOperation(
        item_id="child", started_seq=1, name="dataset", category="subagent"
    )
    pending = snapshot.model_copy(update={"timeline": (child,)})
    assert next_repair(state=pending, receipts=(old,))[0].status == "working"
    user = snapshot.model_copy(update={"status": "waiting_for_input"})
    assert next_repair(state=user, receipts=(old,))[0].status == "needs_input"
    stopped, second = next_repair(receipts=(old,))
    assert stopped.status == "blocked" and second is None
    assert next_repair(receipts=(old,), retry="different")[1] is None
    _, retry = next_repair(receipts=(old,), retry=old.command_id)
    assert retry and retry.command_id != old.command_id
    fixed, patch = next_repair(receipts=(old,), revision="new-patch")
    assert patch and fixed.command_id not in {old.command_id, retry.command_id}


def test_dataset_repair_checks_export_against_accepted_train_fields():
    fields: JsonObject = {"image_field": "images", "assistant_response_field": "gt"}
    _, command = plan_repair(
        stage="dataset",
        evidence="sha256:" + "4" * 64,
        facts={"train_input_fields": fields},
        issues=["dataset_train_fields_missing"],
        revision="",
        receipts=(),
        snapshot=ConversationSnapshot(
            conversation_id="test", capabilities=HarnessCapabilities()
        ),
        retry_command_id=None,
    )
    assert command and isinstance(command.content[0], TextContent)
    guidance, facts = command.content[0].text.rsplit("\n", 1)
    assert json.loads(facts)["train_input_fields"] == fields
    assert "提交小样前在本地导出一行，对照 train_input_fields" in guidance
    assert "修正 Dataset 导出" in guidance
