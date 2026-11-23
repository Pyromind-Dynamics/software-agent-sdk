import asyncio
import json
from hashlib import sha256

from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.echomind_chat import completed_chat_turns
from pyromind_runtime.application.echomind_notices import public_summary_excerpt
from pyromind_runtime.application.plan_execution import PLAN_PATH, next_approved_task
from pyromind_runtime.application.preparation import Preparations
from pyromind_runtime.domain.commands import CreateProjectCommand, UserMessageCommand
from pyromind_runtime.domain.content import JsonObject, TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.echomind_chat import CHAT_PROFILE
from pyromind_runtime.domain.events import HarnessEvent, ProductEvent, ProductEventType
from pyromind_runtime.domain.preparation import ProjectView, SetPreparationPlan
from pyromind_runtime.domain.snapshot import (
    ConversationSnapshot,
    ExternalTaskState,
    TimelineMessage,
)
from pyromind_runtime.ports.harness import DelegatedRole, SessionSpec

from .fake_adapter import FakeAdapter
from .test_preparation import preparation_setup as preparation_setup


async def wait_for(check):
    async with asyncio.timeout(8):
        while not check():
            await asyncio.sleep(0.02)


def plan_request():
    return SetPreparationPlan.model_validate(
        {
            "entry_task_id": "data",
            "plan": {
                "goal": "Prepare data and use it in a baseline",
                "resources": [
                    {
                        "id": "source",
                        "name": "source",
                        "storage_path": "/workspace/source.jsonl",
                    }
                ],
                "stages": [{"id": "stage", "kind": "measure", "name": "Test"}],
                "tasks": [
                    {
                        "id": "data",
                        "stage_id": "stage",
                        "kind": "score",
                        "name": "Prepare",
                        "inputs": [{"name": "dataset", "resource_id": "source"}],
                        "outputs": [{"name": "samples"}],
                    },
                    {
                        "id": "baseline",
                        "stage_id": "stage",
                        "kind": "score",
                        "name": "Baseline",
                        "inputs": [
                            {
                                "name": "dataset",
                                "task_id": "data",
                                "output_name": "samples",
                            }
                        ],
                    },
                ],
            },
        }
    )


def local_completion():
    return {
        "id": "data",
        "stage_id": "stage",
        "business_status": "completed",
        "verified_outputs": {"samples": "/workspace/prepared.jsonl"},
    }


def test_readable_fallback_only_uses_the_final_summary_section():
    summary = (
        "## Working notes\n- These notes must stay private.\n"
        "## 中文总结\n- 共 15 张图片，约 1 MB。\n"
        "- 内部指令包含私有内容。\n## Raw output\n- Ignore this section."
    )
    assert public_summary_excerpt(summary) == "## 中文总结\n- 共 15 张图片，约 1 MB。"
    assert public_summary_excerpt("## Working notes\n- Nothing public.") is None


def test_final_result_accepts_report_prose_without_requiring_fixed_headings():
    result = "共 5 条记录，其中训练 3 条、验证 2 条，共 15 张图片。"
    assert public_summary_excerpt(result) == result
    assert (
        public_summary_excerpt(
            f"## 目录概况\n45 个文件。\n## 简单总结\n{result}\n- 清洗通过 5 条。"
        )
        == f"## 目录概况\n45 个文件。\n## 简单总结\n{result}\n- 清洗通过 5 条。"
    )
    assert (
        public_summary_excerpt(f"~~~md\n## 总结\n- private fixture\n~~~\n{result}")
        == result
    )
    assert public_summary_excerpt("## 内部推理结果\nprivate fixture") is None
    assert public_summary_excerpt("Working notes: private fixture") is None


def test_arbitrary_report_sections_keep_findings_and_exclude_private_subsections():
    evidence = (
        "## 目录概览\n**规模**\n45 个文件、13 个目录，15 张图片。\n"
        "## 工作笔记\n### 样本\nprivate fixture\n"
        "## 数据用途\n适合做流程验证。\n"
        "## 文件清单\n| 类型 | 数量 |\n| JPG | 15 |"
    )
    result = public_summary_excerpt(evidence)
    assert result and all(
        fact in result
        for fact in ("45 个文件", "15 张图片", "流程验证", "| JPG | 15 |")
    )
    assert "private fixture" not in result


def test_approved_continuation_requires_proven_local_outputs():
    project = ProjectView(
        project_id="project",
        conversation_id="session",
        plan_sequence=1,
        creation_command_id="create",
        plan=plan_request().plan,
    )
    done = local_completion()
    ready = next_approved_task(project, [done], "data")
    assert ready is not None and ready[0].id == "baseline"
    assert ready[1] == {"dataset": "/workspace/prepared.jsonl"}
    assert (
        next_approved_task(
            project, [{**done, "verified_outputs": {"other": "x"}}], "data"
        )
        is None
    )
    assert (
        next_approved_task(
            project, [{**done, "platform_task_id": "already-submitted"}], "data"
        )
        is None
    )
    assert (
        next_approved_task(
            project, [done, {"id": "baseline", "business_status": "blocked"}], "data"
        )
        is None
    )


async def test_local_project_continuation_precedes_a_later_requirement(
    preparation_setup, tmp_path
):
    preparations, owner, old_policy = preparation_setup
    policy = old_policy.model_copy(
        update={
            "allowed_roles": (),
            "workflow_nodes": tuple(
                node.model_copy(update={"roles": ()})
                for node in old_policy.workflow_nodes
            ),
        }
    )
    cid = owner.conversation_id
    context = RequestContext(
        user_id=owner.user_id, profile_id=owner.profile_id, x_cluster="test"
    )
    adapter = FakeAdapter()
    runtime = ConversationRuntime(tmp_path, adapter, preparation_store=preparations)
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=cid,
                user_id=context.user_id,
                profile_id=context.profile_id,
                workspace_root=str(tmp_path / cid),
                preparation_policy=policy,
                command_allowlist=("create_project", "enqueue_task"),
            ),
            context,
        )
        plan = Preparations(preparations).set_plan(owner, plan_request(), policy, 0)
        receipt = await runtime.submit_command(
            cid,
            CreateProjectCommand(
                command_id="create",
                expected_preparation_revision=plan.revision,
            ),
            context,
        )
        await wait_for(lambda: bool(adapter.external_task_notifications))
        first = adapter.external_task_notifications[0][1]
        assert first.kind == "project_start"
        started = HarnessEvent(
            session_id=cid,
            type="status.changed",
            run_id="first-run",
            task_id=first.task_id,
            payload={"status": "running"},
        )
        adapter.queues[cid].put_nowait(started)
        store = runtime._store(cid)
        await wait_for(lambda: store.load_snapshot().status == "running")
        runtime._queue_task(
            store, cid, "requirement:later", "requirement", "Inspect something new"
        )
        (tmp_path / cid / PLAN_PATH).write_text(
            json.dumps({"tasks": [local_completion()]})
        )
        adapter.queues[cid].put_nowait(
            HarnessEvent(
                session_id=cid,
                type="run.finished",
                run_id="first-run",
                task_id=first.task_id,
                payload={"status": "idle"},
            )
        )
        await wait_for(lambda: len(adapter.external_task_notifications) >= 2)
        continuation = adapter.external_task_notifications[1][1]
        assert (
            continuation.task_id
            == f"project-continue:{receipt.response['project_id']}:baseline"
        )
        assert "/workspace/prepared.jsonl" in continuation.hidden_text
        assert adapter.external_task_notifications[1][1].kind == "project_continue"
        await wait_for(lambda: len(adapter.external_task_notifications) >= 3)
        assert adapter.external_task_notifications[2][1].task_id == "requirement:later"
    finally:
        await runtime.close()


async def test_adapter_queued_receipt_keeps_oldest_requirement_at_head(tmp_path):
    class Adapter(FakeAdapter):
        ready_to_start = False

        async def notify_external_task(
            self, handle, notification, context
        ) -> JsonObject:
            await super().notify_external_task(handle, notification, context)
            return {"accepted": True, "queued": not self.ready_to_start}

    adapter = Adapter()
    runtime = ConversationRuntime(tmp_path, adapter)
    cid = "queue-order"
    context = RequestContext(user_id="owner")
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=cid, user_id="owner", workspace_root=str(tmp_path / cid)
            ),
            context,
        )
        store = runtime._store(cid)
        active = runtime._active[cid]
        for task in ("older", "newer"):
            runtime._queue_task(store, cid, task, "requirement", task)
        await runtime._dispatch_queued_tasks(active, store)
        await runtime._dispatch_queued_tasks(active, store)
        assert [n.task_id for _, n, _ in adapter.external_task_notifications] == [
            "older",
            "older",
        ]
        assert [
            item["task_id"] for item in store.load_snapshot().experiment_queue or ()
        ] == ["older", "newer"]
        adapter.ready_to_start = True
        await runtime._dispatch_queued_tasks(active, store)
        assert [
            item["task_id"] for item in store.load_snapshot().experiment_queue or ()
        ] == ["newer"]
    finally:
        await runtime.close()


async def test_chat_accepts_messages_durably_and_starts_one_at_a_time(tmp_path):
    cid = "echomind-chat-order"
    context = RequestContext(user_id="owner", profile_id=CHAT_PROFILE)
    adapter = FakeAdapter()
    runtime = ConversationRuntime(tmp_path, adapter)
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=cid,
                user_id="owner",
                profile_id=CHAT_PROFILE,
                workspace_root=str(tmp_path / cid),
                command_allowlist=("user_message",),
            ),
            context,
        )
        store = runtime._store(cid)
        store.append(
            ProductEvent(
                conversation_id=cid,
                type="execution.policy.updated",
                payload={"automatic_turns_suspended": True},
            )
        )
        for request in ("A", "B", "C"):
            receipt = await runtime.submit_command(
                cid,
                UserMessageCommand(
                    command_id=request, content=(TextContent(text=request),)
                ),
                context,
            )
            assert receipt.response["queued"]
        store = runtime._store(cid)
        assert [cmd.command_id for _, cmd, _ in adapter.sent] == ["A"]
        assert not store.load_snapshot().automatic_turns_suspended
        assert [
            item["task_id"] for item in store.load_snapshot().experiment_queue or ()
        ] == [
            "chat-message:A",
            "chat-message:B",
            "chat-message:C",
        ]
        for index, request in enumerate(("A", "B", "C"), 1):
            await wait_for(lambda: len(adapter.sent) == index)
            adapter.emit(
                cid,
                "status.changed",
                {"status": "running"},
                event_id=f"{request}:start",
                run_id=request,
            )
            adapter.emit(
                cid,
                "message.started",
                {
                    "role": "user",
                    "command_id": request,
                    "message_id": f"message:{request}",
                },
                event_id=f"{request}:message",
                run_id=request,
            )
            await wait_for(
                lambda: all(
                    item["task_id"] != f"chat-message:{request}"
                    for item in store.load_snapshot().experiment_queue or ()
                )
            )
            assert len(adapter.sent) == index
            adapter.emit(
                cid,
                "run.finished",
                {"status": "idle"},
                event_id=f"{request}:done",
                run_id=request,
            )
        assert [cmd.command_id for _, cmd, _ in adapter.sent] == ["A", "B", "C"]
    finally:
        await runtime.close()


async def test_chat_recovery_uses_recorded_user_message_as_dispatch_ack(tmp_path):
    cid = "echomind-chat-recovery"
    context = RequestContext(user_id="owner", profile_id=CHAT_PROFILE)
    adapter = FakeAdapter()
    runtime = ConversationRuntime(tmp_path, adapter)
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=cid,
                user_id="owner",
                profile_id=CHAT_PROFILE,
                workspace_root=str(tmp_path / cid),
                command_allowlist=("user_message",),
            ),
            context,
        )
        for request in ("A", "B"):
            await runtime.submit_command(
                cid,
                UserMessageCommand(
                    command_id=request, content=(TextContent(text=request),)
                ),
                context,
            )
        store = runtime._store(cid)
        store.append(
            ProductEvent(
                conversation_id=cid,
                type="message.completed",
                run_id="A",
                turn_id="A",
                payload={
                    "role": "user",
                    "message_id": "message:A",
                    "command_id": "A",
                    "content": [{"type": "text", "text": "A"}],
                },
            )
        )
    finally:
        await runtime.close()

    restored_adapter = FakeAdapter()
    restored = ConversationRuntime(tmp_path, restored_adapter)
    try:
        await restored.get_snapshot(cid, context)
        await wait_for(lambda: bool(restored_adapter.sent))
        assert [cmd.command_id for _, cmd, _ in restored_adapter.sent] == ["B"]
        assert [
            item["task_id"] for item in store.load_snapshot().experiment_queue or ()
        ] == ["chat-message:B"]
    finally:
        await restored.close()


async def test_unacknowledged_platform_callback_is_retried_before_new_requirement(
    tmp_path,
):
    class Adapter(FakeAdapter):
        available = False

        async def notify_external_task(
            self, handle, notification, context
        ) -> JsonObject:
            await super().notify_external_task(handle, notification, context)
            return {"accepted": self.available}

    adapter = Adapter()
    runtime = ConversationRuntime(tmp_path, adapter)
    cid = "callback-order"
    context = RequestContext(user_id="owner")
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=cid, user_id="owner", workspace_root=str(tmp_path / cid)
            ),
            context,
        )
        store = runtime._store(cid)
        task = ExternalTaskState(
            task_id="platform-result",
            kind="workflow_run",
            status="succeeded",
            submitted_at="2026-11-23T00:00:00Z",
            updated_at="2026-11-23T00:00:01Z",
            resume_pending=True,
            resume_auto_run=True,
        )
        store.append(
            ProductEvent(
                conversation_id=cid,
                type="external_task.submitted",
                payload=task.model_copy(
                    update={"status": "pending", "resume_pending": False}
                ).model_dump(mode="json"),
            )
        )
        store.append(
            ProductEvent(
                conversation_id=cid,
                type="external_task.completed",
                payload=task.model_dump(mode="json"),
            )
        )
        runtime._queue_task(store, cid, "newer", "requirement", "Run later")
        active = runtime._active[cid]
        await runtime._resume_notifications(active, store)
        assert [n.task_id for _, n, _ in adapter.external_task_notifications] == [
            "platform-result"
        ]
        adapter.available = True
        await runtime._resume_notifications(active, store)
        assert [n.task_id for _, n, _ in adapter.external_task_notifications] == [
            "platform-result",
            "platform-result",
            "newer",
        ]
    finally:
        await runtime.close()


async def test_ordinary_execution_completion_delivers_one_readable_result(
    preparation_setup, tmp_path
):
    preparations, owner, policy = preparation_setup
    cid = owner.conversation_id
    context = RequestContext(user_id=owner.user_id, profile_id="echomind")
    outcomes = []

    class Adapter(FakeAdapter):
        async def compose_execution_notice(self, handle, facts):
            outcomes.append(facts)
            return '{"message":null}'

    adapter = Adapter()
    runtime = ConversationRuntime(tmp_path, adapter, preparation_store=preparations)
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=cid,
                user_id=context.user_id,
                profile_id=context.profile_id,
                workspace_root=str(tmp_path / cid),
                preparation_policy=policy,
                delegated_roles=(DelegatedRole(name="eval"),),
            ),
            context,
        )
        chat = await runtime.echomind_chat.open(cid, {}, context)
        manager = runtime._store(cid)
        chat_store = runtime._store(chat.conversation_id)
        for index, evidence in enumerate(
            (
                "发现 15 张图片。token=fixture-value",
                "<think>internal marker",
                "## 目录概况\n有图片。\n## 简单总结\n共 5 条记录、15 张图片。",
                "<think>internal marker</think>3 条训练记录、2 条验证记录。",
            ),
            1,
        ):
            task = f"requirement:chat:{index}"
            run = f"analysis-{index}"
            runtime._queue_task(
                manager,
                cid,
                task,
                "requirement",
                "Inspect files",
                origin_turn_id="user-turn",
            )
            events: tuple[tuple[ProductEventType, JsonObject], ...] = (
                (
                    "message.completed",
                    {
                        "role": "assistant",
                        "message_id": run,
                        "content": [{"type": "text", "text": evidence}],
                    },
                ),
                ("status.changed", {"status": "idle"}),
            )
            for kind, payload in events:
                event, _ = manager.append(
                    ProductEvent(
                        conversation_id=cid,
                        type=kind,
                        run_id=run,
                        task_id=task,
                        payload=payload,
                    )
                )
                runtime._publish(event)
                runtime._publish(event)
            await wait_for(lambda: not runtime.echomind_notices.workers)
        run = "historical-run"
        origin = "historical-turn"
        task = "requirement:chat:historical"
        runtime._queue_task(
            manager, cid, task, "requirement", "Inspect", origin_turn_id=origin
        )
        manager.append(
            ProductEvent(
                conversation_id=cid,
                task_id=task,
                run_id=run,
                type="message.completed",
                payload={
                    "role": "assistant",
                    "message_id": run,
                    "content": [
                        {"type": "text", "text": "## 目录概览\n图片共有 15 张。"}
                    ],
                },
            )
        )
        ended, _ = manager.append(
            ProductEvent(
                conversation_id=cid,
                task_id=task,
                run_id=run,
                type="status.changed",
                payload={"status": "idle"},
            )
        )
        identity = "execution-notice:" + sha256(f"round:{run}".encode()).hexdigest()
        chat_store.append(
            ProductEvent(
                conversation_id=chat.conversation_id,
                event_id=identity,
                run_id=run,
                type="message.completed",
                payload={
                    "role": "assistant",
                    "message_id": identity,
                    "content": [
                        {
                            "type": "text",
                            "text": "本轮执行已结束。可查看原任务中的执行详情。",
                        }
                    ],
                    "execution_notice": {"origin_turn_id": origin},
                },
            )
        )
        chat_store.save_execution_notice_cursor(ended.seq)
        for _ in range(2):
            await runtime.echomind_chat.snapshot(cid, context)
            await wait_for(lambda: not runtime.echomind_notices.workers)
        messages = [
            item
            for item in chat_store.load_snapshot().timeline
            if isinstance(item, TimelineMessage)
        ]
        assert len(messages) == 5
        assert messages[-1].item_id == identity
        assert messages[-1].content == (
            TextContent(text="## 目录概览\n图片共有 15 张。"),
        )
        assert (
            messages[-1].execution_notice
            and messages[-1].execution_notice.origin_turn_id == origin
        )
        assert all(
            item.execution_notice
            and item.execution_notice.origin_turn_id == "user-turn"
            for item in messages[:-1]
        )
        assert "15 张图片" in messages[0].content[0].model_dump()["text"]
        assert "internal marker" not in json.dumps(outcomes, ensure_ascii=False)
        assert "fixture-value" not in str(messages)
        assert "没有可展示的结果正文" in messages[1].content[0].model_dump()["text"]
        assert not outcomes
        assert "共 5 条记录、15 张图片。" in messages[2].content[0].model_dump()["text"]
        assert messages[3].content == (
            TextContent(text="3 条训练记录、2 条验证记录。"),
        )
    finally:
        await runtime.close()


def test_chat_model_pairs_late_results_with_original_questions():
    rows = []

    def message(role, turn, text, *, run=None, notice=None, identity=None):
        seq = len(rows) + 1
        rows.append(
            dict(
                kind="message",
                item_id=identity or str(seq),
                turn_id=turn,
                run_id=run or turn,
                role=role,
                content=[{"type": "text", "text": text}],
                started_seq=seq,
                completed_seq=seq,
                status="completed",
                execution_notice=notice,
            )
        )

    message("user", "a", "Inspect A")
    rows.append(
        dict(
            kind="operation",
            item_id="queue",
            turn_id="a",
            started_seq=2,
            completed_seq=2,
            name="queue_echomind_requirement",
            category="tool",
            status="completed",
        )
    )
    message("assistant", "a", "Queue receipt")
    message("user", "b", "Explain B")
    message("assistant", "b", "Answer B")
    message(
        "assistant",
        None,
        "Old A",
        run="old",
        notice={"origin_turn_id": "a"},
        identity="execution-notice:" + sha256(b"round:old").hexdigest(),
    )
    message(
        "assistant",
        None,
        "A result",
        notice={"origin_turn_id": "a", "has_result": True},
    )
    message(
        "assistant",
        None,
        "Status only",
        notice={"origin_turn_id": "a", "has_result": False},
    )
    message("assistant", None, "Orphan", notice={"has_result": True})
    message("user", "c", "Current question")
    message("assistant", "c", "Current intermediate step")
    snapshot = ConversationSnapshot.model_validate(
        {"conversation_id": "chat", "capabilities": {}, "timeline": rows}
    )
    assert completed_chat_turns(snapshot, "c") == [
        {"question": "Inspect A", "answer": "Old A\n\nA result"},
        {"question": "Explain B", "answer": "Answer B"},
    ]
