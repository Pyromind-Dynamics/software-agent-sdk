import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from harness_adapter.pi_adapter import PiAdapter
from pyromind_agent_server.api.echomind_chat import router
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.preparation import Preparations
from pyromind_runtime.application.versions import Versions
from pyromind_runtime.domain.commands import (
    CreateProjectCommand,
    EnqueueTaskCommand,
    ReviewPreparationCommand,
    UserMessageCommand,
)
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.echomind_chat import (
    CHAT_PROFILE,
    CHAT_TOOLS,
    ENQUEUE,
    GET_STATUS,
    READ_REPORT,
)
from pyromind_runtime.domain.errors import ProductRuntimeError
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.experiment_execution import ExperimentExecutionView
from pyromind_runtime.domain.snapshot import TimelineMessage
from pyromind_runtime.domain.versions import VersionToolRequest
from pyromind_runtime.infrastructure.file_artifact_snapshots import (
    FileArtifactSnapshots,
)
from pyromind_runtime.infrastructure.sqlite_version_store import SQLiteVersionStore
from pyromind_runtime.ports.harness import DelegatedRole, SessionSpec

from .fake_adapter import FakeAdapter
from .test_native_preparation import stream
from .test_preparation import (
    preparation_setup as preparation_setup,
    ready_request as ready_request,
)
from .test_product_api import _app


async def test_standalone_saved_report_delivers_readable_link_once(
    preparation_setup, tmp_path
):
    preparation, owner, policy = preparation_setup
    manager_id = owner.conversation_id
    context = RequestContext(user_id=owner.user_id, profile_id="echomind")
    versions = Versions(
        SQLiteVersionStore(tmp_path / "versions.db"),
        FileArtifactSnapshots(tmp_path / "storage"),
    )
    root = tmp_path / "conversations"
    runtime = ConversationRuntime(
        root, FakeAdapter(), preparation_store=preparation, versions=versions
    )
    api = _app(tmp_path, runtime)
    api.include_router(router)
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=manager_id,
                user_id=context.user_id,
                workspace_root=str(root / manager_id),
                profile_id="echomind",
                preparation_policy=policy,
                delegated_roles=(DelegatedRole(name="eval"),),
                command_allowlist=("enqueue_task",),
            ),
            context,
        )
        chat = await runtime.echomind_chat.open(manager_id, {}, context)
        manager = runtime._store(manager_id)
        runtime._queue_task(
            manager,
            manager_id,
            "summary-task",
            "requirement",
            "Create a report",
            origin_turn_id="request-turn",
        )
        started, _ = manager.append(
            ProductEvent(
                conversation_id=manager_id,
                type="status.changed",
                run_id="summary-run",
                task_id="summary-task",
                payload={"status": "running"},
            )
        )
        runtime._publish(started)
        request = VersionToolRequest(
            action="save",
            arguments={
                "kind": "report",
                "logical_key": "summary",
                "source": {
                    "text": "# Dataset summary\n\nFrozen report",
                    "filename": "summary.md",
                },
            },
        )
        saved = await runtime.version_call(manager_id, request, "save-report", context)
        artifact_id = saved["id"]
        async with asyncio.timeout(5):
            while runtime.echomind_notices.workers:
                await asyncio.sleep(0.02)
        chat_store = runtime._store(chat.conversation_id)
        first = chat_store.load_snapshot().timeline[-1]
        assert isinstance(first, TimelineMessage) and first.execution_notice
        assert first.execution_notice.origin_turn_id == "request-turn"
        assert first.execution_notice.action == "report"
        assert first.execution_notice.target_id == artifact_id
        runtime._record_execution(
            manager,
            ExperimentExecutionView(
                run_id="business-run",
                project_id="project",
                task_id="summary",
                task_name="Summary",
                role_id="main",
                state="reported",
                report_state="ready",
                artifacts=({"artifact_id": artifact_id},),
            ),
        )
        async with asyncio.timeout(5):
            while runtime.echomind_notices.workers:
                await asyncio.sleep(0.02)
        assert chat_store.load_snapshot().timeline[-1] == first
        assert (
            sum(
                isinstance(item, TimelineMessage)
                for item in chat_store.load_snapshot().timeline
            )
            == 1
        )

        url = f"/api/v2/echomind/conversations/{manager_id}/chat/reports/{artifact_id}"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api),
            base_url="http://test",
            headers={"x-pyromind-debug-user-id": context.user_id},
        ) as client:
            response = await client.get(url)
            assert response.status_code == 200
            assert response.json()["text"] == "# Dataset summary\n\nFrozen report"
            assert response.headers["cache-control"] == "no-store"
            denied = await client.get(url, headers={"x-pyromind-debug-user-id": "99"})
            assert denied.status_code == 404
    finally:
        await runtime.close()


async def test_execution_notice_writes_after_chat_replies_once_and_links_original_turn(
    preparation_setup, tmp_path
):
    preparation, owner, policy = preparation_setup
    manager_id = owner.conversation_id
    context = RequestContext(user_id=owner.user_id, profile_id="echomind")
    composing = asyncio.Event()
    reply = asyncio.Event()

    class Adapter(FakeAdapter):
        async def compose_execution_notice(self, handle, facts):
            if facts["milestone"] == "report_ready":
                composing.set()
                await reply.wait()
                return '{"message":"报告已就绪，可以查看。"}'
            return '{"message":null}'

    runtime = ConversationRuntime(tmp_path, Adapter(), preparation_store=preparation)
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=manager_id,
                user_id=context.user_id,
                workspace_root=str(tmp_path / manager_id),
                profile_id="echomind",
                preparation_policy=policy,
                delegated_roles=(DelegatedRole(name="eval"),),
            ),
            context,
        )
        chat = await runtime.echomind_chat.open(manager_id, {}, context)
        manager = runtime._store(manager_id)
        chat_store = runtime._store(chat.conversation_id)

        def write_chat(message_id, role, turn_id):
            event, _ = chat_store.append(
                ProductEvent(
                    conversation_id=chat.conversation_id,
                    type="message.completed",
                    turn_id=turn_id,
                    payload={
                        "message_id": message_id,
                        "role": role,
                        "content": [{"type": "text", "text": "用户正在对话。"}],
                    },
                )
            )
            runtime._publish(event)

        write_chat("first", "user", "origin")
        runtime._queue_task(
            manager,
            manager_id,
            "linked-task",
            "requirement",
            "review",
            origin_turn_id="origin",
        )
        execution = ExperimentExecutionView(
            run_id="run",
            project_id="project",
            task_id="task",
            task_name="evaluation",
            role_id="main",
            state="waiting_analysis",
            report_state="not_started",
            platform_task_id="linked-task",
        )
        runtime._record_execution(manager, execution)
        result = execution.model_copy(
            update={"state": "reported", "report_state": "ready"}
        )
        runtime._record_execution(manager, result)
        async with asyncio.timeout(5):
            await composing.wait()
        write_chat("second", "user", "later")
        write_chat("reply", "assistant", "later")
        reply.set()
        # Related projections can repeat the same verified milestone.
        runtime._record_execution(
            manager, result.model_copy(update={"summary": "updated"})
        )
        async with asyncio.timeout(5):
            while runtime.echomind_notices.workers:
                await asyncio.sleep(0.02)
        messages = [
            item
            for item in chat_store.load_snapshot().timeline
            if isinstance(item, TimelineMessage)
        ]
        assert [item.item_id for item in messages[:-1]] == ["first", "second", "reply"]
        notice = messages[-1]
        assert notice.execution_notice is not None
        assert notice.execution_notice.origin_turn_id == "origin"
        assert notice.content == (TextContent(text="报告已就绪，可以查看。"),)
        assert (
            sum(
                event.type == "chat.notification.skipped"
                for event in chat_store.replay()
            )
            == 1
        )
        pending, _ = manager.append(
            ProductEvent(
                conversation_id=manager_id,
                type="input.requested",
                task_id="linked-task",
                payload={
                    "request_id": "question",
                    "questions": [{"id": "review", "prompt": "是否扩大测试范围？"}],
                },
            )
        )
        runtime._publish(pending)
        async with asyncio.timeout(5):
            while runtime.echomind_notices.workers:
                await asyncio.sleep(0.02)
        last = chat_store.load_snapshot().timeline[-1]
        assert isinstance(last, TimelineMessage) and last.execution_notice is not None
        assert last.execution_notice.action == "confirmation"
        assert last.execution_notice.target_id == "question"
        assert last.execution_notice.origin_turn_id == "origin"
        assert last.content == (TextContent(text="执行需要你回复，请查看待确认事项。"),)
    finally:
        reply.set()
        await runtime.close()


async def test_chat_handles_new_requirements_before_busy_main_turn_finishes(
    preparation_setup, tmp_path, monkeypatch
):
    preparation, owner, policy = preparation_setup
    root = tmp_path / "conversations"
    manager_id = owner.conversation_id
    context = RequestContext(
        user_id=owner.user_id, profile_id="echomind", x_cluster="test"
    )
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    adapter = PiAdapter(root, terminal_backend="os-sandbox")
    versions = Versions(
        SQLiteVersionStore(tmp_path / "versions.db"),
        FileArtifactSnapshots(tmp_path / "storage"),
    )
    runtime = ConversationRuntime(
        root, adapter, preparation_store=preparation, versions=versions
    )
    adapter.set_echo_chat_handler(runtime.echomind_chat.call)
    first_main_model = asyncio.Event()
    release_main = asyncio.Event()
    main_requests = []
    tool_receipts = []

    async def main_model(request):
        main_requests.append(await request.json())
        if not first_main_model.is_set():
            first_main_model.set()
            await release_main.wait()
        return stream()

    async def chat_model(request):
        body = await request.json()
        assert {t["function"]["name"] for t in body["tools"]} == set(CHAT_TOOLS)
        last = body["messages"][-1]
        if last["role"] == "tool":
            tool_receipts.append(json.loads(last["content"]))
            return stream()
        content = last["content"]
        if not isinstance(content, str):
            content = content[0]["text"]
        if content == "Where are we?":
            return stream(GET_STATUS, {}, call="reused-tool-id")
        return stream(ENQUEUE, {"instruction": content}, call="reused-tool-id")

    models = web.Application()
    models.router.add_post("/main/chat/completions", main_model)
    models.router.add_post("/chat/chat/completions", chat_model)
    api = _app(tmp_path, runtime)
    api.include_router(router)

    async def wait_for(predicate):
        async with asyncio.timeout(12):
            while not predicate():
                await asyncio.sleep(0.02)

    try:
        async with TestServer(models) as server:
            await runtime.create_conversation(
                SessionSpec(
                    conversation_id=manager_id,
                    user_id=context.user_id,
                    workspace_root=str(root / manager_id),
                    profile_id=context.profile_id,
                    preparation_policy=policy,
                    delegated_roles=(DelegatedRole(name="eval"),),
                    command_allowlist=("user_message", "enqueue_task"),
                    tool_allowlist=(),
                    model_configuration={
                        "model": "test",
                        "base_url": str(server.make_url("/main")),
                    },
                ),
                context,
            )
            await runtime.submit_command(
                manager_id,
                UserMessageCommand(
                    command_id="old-turn",
                    content=(TextContent(text="Existing preparation"),),
                ),
                context,
            )
            await asyncio.wait_for(first_main_model.wait(), 5)
            url = f"/api/v2/echomind/conversations/{manager_id}/chat"
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api),
                base_url="http://test",
                headers={"x-pyromind-debug-user-id": "42"},
            ) as client:
                assert (await client.get(f"{url}/snapshot")).json() is None
                model = {"model": "test", "base_url": str(server.make_url("/chat"))}
                opened = await asyncio.gather(
                    client.post(url, json=model), client.post(url, json=model)
                )
                assert [response.status_code for response in opened] == [200, 200]
                chat_id = opened[0].json()["conversation_id"]
                assert (
                    chat_id != manager_id
                    and opened[1].json()["conversation_id"] == chat_id
                )
                assert (
                    await client.get(
                        f"{url}/snapshot", headers={"x-pyromind-debug-user-id": "99"}
                    )
                ).status_code == 404
                denied = await client.post(
                    f"{url}/commands",
                    headers={"x-pyromind-debug-user-id": "99"},
                    json={
                        "type": "user_message",
                        "command_id": "cross-user",
                        "content": [{"type": "text", "text": "unauthorized"}],
                    },
                )
                assert denied.status_code == 404
                store = runtime._store(manager_id)
                chat = runtime._store(chat_id)

                for index, text in enumerate(
                    (
                        "Where are we?",
                        "Inspect sample coverage",
                        "Check evaluation labels",
                    ),
                    1,
                ):
                    response = await client.post(
                        f"{url}/commands",
                        json={
                            "type": "user_message",
                            "command_id": f"chat-message-{index}",
                            "content": [{"type": "text", "text": text}],
                        },
                    )
                    assert response.status_code == 202, response.text
                    await wait_for(
                        lambda: (
                            len(tool_receipts) == index
                            and chat.load_snapshot().status == "idle"
                        )
                    )

                assert tool_receipts[0]["status"] == "running"
                assert tool_receipts[0]["project_id"] is None
                queued = [
                    item["task_id"]
                    for item in (store.load_snapshot().experiment_queue or ())
                ]
                assert queued == [item["task_id"] for item in tool_receipts[1:]]
                assert len(set(queued)) == 2
                bindings = dict(
                    zip(queued, ("chat-message-2", "chat-message-3"), strict=True)
                )
                assert store.load_snapshot().task_origins == bindings
                calls = [
                    item
                    for item in chat.load_snapshot().timeline
                    if item.kind == "operation"
                ]
                assert [item.turn_id for item in calls] == [
                    f"chat-message-{i}" for i in (1, 2, 3)
                ]
                assert len({item.item_id for item in calls}) == 3
                with pytest.raises(ValueError, match="No ready report"):
                    await runtime.echomind_chat.call(
                        chat_id,
                        READ_REPORT,
                        {"artifact_id": "missing"},
                        "early-report",
                        context,
                        None,
                    )
                retry = await runtime.echomind_chat.call(
                    chat_id,
                    ENQUEUE,
                    {"instruction": "Inspect sample coverage"},
                    "reused-tool-id",
                    replace(context, profile_id=CHAT_PROFILE),
                    "chat-message-2",
                )
                assert retry == tool_receipts[1]
                assert len(store.load_snapshot().experiment_queue or ()) == 2
                report = VersionToolRequest(
                    action="save",
                    arguments={
                        "kind": "report",
                        "logical_key": "report",
                        "source": {
                            "text": "Frozen report text",
                            "filename": "report.md",
                        },
                    },
                )
                versions.call(manager_id, report, "report-1", context)
                versions.call("another-project", report, "report-1", context)
                (local,) = runtime._artifact_versions(manager_id, context)
                (foreign,) = runtime._artifact_versions("another-project", context)
                result = await runtime.echomind_chat.call(
                    chat_id,
                    READ_REPORT,
                    {"artifact_id": local.id},
                    "report",
                    context,
                    None,
                )
                assert result["text"] == "Frozen report text"
                with pytest.raises(ValueError, match="current project"):
                    await runtime.echomind_chat.call(
                        chat_id,
                        READ_REPORT,
                        {"artifact_id": foreign.id},
                        "report",
                        context,
                        None,
                    )
                assert len(main_requests) == 1
                release_main.set()
                await wait_for(
                    lambda: (
                        not store.load_snapshot().experiment_queue
                        and len(main_requests) >= 3
                    )
                )
                assert "Inspect sample coverage" in json.dumps(main_requests[1])
                assert "Check evaluation labels" in json.dumps(main_requests[2])
                await wait_for(
                    lambda: {
                        item.task_id
                        for item in store.load_snapshot().timeline
                        if item.kind == "message" and item.status == "completed"
                    }.issuperset(queued)
                )
                await wait_for(
                    lambda: {
                        run.task_id
                        for run in (store.load_snapshot().task_runs or {}).values()
                        if run.status == "idle"
                    }.issuperset(queued)
                )
                assert store.load_snapshot().task_origins == bindings
    finally:
        release_main.set()
        await runtime.close()


async def test_preparation_requirement_survives_restart_and_guards_creation(
    preparation_setup, ready_request, tmp_path
):
    preparation, owner, policy = preparation_setup
    owner = owner.model_copy(update={"profile_id": "echomind"})
    manager_id = owner.conversation_id
    context = RequestContext(
        user_id=owner.user_id, profile_id=owner.profile_id, x_cluster="test"
    )
    root = tmp_path / "conversations"
    adapter = FakeAdapter()
    runtime = ConversationRuntime(root, adapter, preparation_store=preparation)
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=manager_id,
                user_id=context.user_id,
                workspace_root=str(root / manager_id),
                profile_id=context.profile_id,
                preparation_policy=policy,
                tool_allowlist=("set_preparation_plan",),
                delegated_roles=(DelegatedRole(name="eval"),),
                command_allowlist=(
                    "user_message",
                    "create_project",
                    "enqueue_task",
                    "review_preparation",
                ),
            ),
            context,
        )
        plan = Preparations(preparation).set_plan(owner, ready_request, policy, 0)
        chat = await runtime.echomind_chat.open(manager_id, {}, context)
        receipt = await runtime.echomind_chat.call(
            chat.conversation_id,
            ENQUEUE,
            {"instruction": "Update the review dataset"},
            "tool-1",
            context,
            "user-1",
        )
        assert receipt["status"] == "queued"
        assert preparation.get(owner) == plan
        with pytest.raises(ValueError, match="wait for queued requirements"):
            await runtime.submit_command(
                manager_id,
                CreateProjectCommand(
                    command_id="create", expected_preparation_revision=plan.revision
                ),
                context,
            )
    finally:
        await runtime.close()

    restored_adapter = FakeAdapter()
    restored = ConversationRuntime(
        root, restored_adapter, preparation_store=preparation
    )
    try:
        await restored.get_snapshot(manager_id, context)
        async with asyncio.timeout(8):
            while not restored_adapter.external_task_notifications:
                await asyncio.sleep(0.02)
        assert (
            restored_adapter.external_task_notifications[0][1].task_id
            == receipt["task_id"]
        )
        assert restored._store(manager_id).load_snapshot().task_origins == {
            str(receipt["task_id"]): "user-1"
        }
        invalidated = preparation.get(owner)
        assert invalidated.review_pending and invalidated.revision == plan.revision + 1
        assert "including when the saved plan is unchanged" in (
            restored_adapter.external_task_notifications[0][1].hidden_text
        )
        with pytest.raises(ValueError, match="preparation is not ready"):
            await restored.submit_command(
                manager_id,
                CreateProjectCommand(
                    command_id="stale-draft",
                    expected_preparation_revision=invalidated.revision,
                ),
                context,
            )
        review = ReviewPreparationCommand(
            command_id="review", expected_preparation_revision=invalidated.revision
        )
        with pytest.raises(ProductRuntimeError, match="plan review changed"):
            await restored.submit_command(
                manager_id,
                review.model_copy(
                    update={
                        "command_id": "obsolete-review",
                        "expected_preparation_revision": plan.revision,
                    }
                ),
                context,
            )
        accepted = await restored.submit_command(manager_id, review, context)
        assert await restored.submit_command(manager_id, review, context) == accepted
        with pytest.raises(ProductRuntimeError, match="plan review changed"):
            await restored.submit_command(
                manager_id,
                review.model_copy(update={"command_id": "duplicate"}),
                context,
            )
        async with asyncio.timeout(8):
            while len(restored_adapter.external_task_notifications) < 2:
                await asyncio.sleep(0.02)
        _, notification, _ = restored_adapter.external_task_notifications[1]
        assert notification.task_id == accepted.response["task_id"]
        assert "Saved draft for review:" in notification.hidden_text
        review_context = replace(context, preparation_revision=invalidated.revision)
        result = await restored.preparation_call(
            manager_id,
            "set_preparation_plan",
            ready_request.model_dump(mode="json"),
            review_context,
        )
        assert result["phase"] == "ready"
        assert not preparation.get(owner).review_pending
        assert preparation.project(owner) is None
    finally:
        await restored.close()


async def test_slow_requirement_dispatch_does_not_block_another_main_agent(
    preparation_setup, tmp_path
):
    preparation, owner, policy = preparation_setup
    started_first = asyncio.Event()
    dispatched_second = asyncio.Event()
    release_first = asyncio.Event()

    class Adapter(FakeAdapter):
        async def notify_external_task(self, handle, notification, context):
            if handle.session_id == "first":
                started_first.set()
                await release_first.wait()
            else:
                dispatched_second.set()
            return await super().notify_external_task(handle, notification, context)

    context = RequestContext(user_id=owner.user_id, profile_id="echomind")
    runtime = ConversationRuntime(
        tmp_path / "conversations", Adapter(), preparation_store=preparation
    )
    try:
        for name in ("first", "second"):
            await runtime.create_conversation(
                SessionSpec(
                    conversation_id=name,
                    user_id=context.user_id,
                    profile_id=context.profile_id,
                    workspace_root=str(runtime.conversation_root / name),
                    preparation_policy=policy,
                    delegated_roles=(DelegatedRole(name="eval"),),
                    command_allowlist=("enqueue_task",),
                ),
                context,
            )
        await runtime.enqueue_chat_requirement(
            "first",
            EnqueueTaskCommand(command_id="chat:first", instruction="Review samples"),
            context,
        )
        await asyncio.wait_for(started_first.wait(), timeout=6)
        await runtime.enqueue_chat_requirement(
            "second",
            EnqueueTaskCommand(command_id="chat:second", instruction="Review samples"),
            context,
        )
        await asyncio.wait_for(dispatched_second.wait(), timeout=6)
        assert runtime._store("first").load_snapshot().experiment_queue
        assert not runtime._store("second").load_snapshot().experiment_queue
    finally:
        release_first.set()
        await runtime.close()
