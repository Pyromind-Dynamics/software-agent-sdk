from __future__ import annotations

import asyncio
import base64
import json
import os
import signal
from pathlib import Path
from uuid import uuid4

import pytest
from harness_adapter.agentgenome_bridge import AgentGenomeBackend, ExecutionHost
from pyromind_runtime.application.reusable_workflows import ReusableWorkflows


@pytest.fixture
async def setup_workflow(tmp_path):
    package = tmp_path / "source"
    package.mkdir()
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "id": "sample",
                "version": "1.0.0",
                "name": "Sample",
                "description": "Read input and create a verified report",
                "parameters": {"data": {"type": "path"}},
            }
        )
    )
    (package / "graph.yaml").write_text("""version: gt/1.0
node:
  id: report
  input:
    data: {type: path, from: params.data}
  do:
    run: python3 report.py {input.data} > {artifact}
  output:
    report: {type: json}
  verify:
    - run: test -s {artifact}
    - metric: report.rows > 0
""")
    (package / "report.py").write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "print(json.dumps({'rows': len(Path(sys.argv[1]).read_text().splitlines())}))\n"
    )
    binary = bytes(range(256)) * 4097
    (package / "reference.bin").write_bytes(binary)
    workspace = tmp_path / "remote"
    workspace.mkdir()
    (workspace / "public_data").mkdir()
    data = workspace / "public_data/input.txt"
    data.write_text("one\ntwo\n")
    sinks = {}
    processes = {}
    calls = []

    async def request(method, params):
        assert len(json.dumps(params).encode()) < 1024 * 1024
        calls.append(method)
        if method == "execution.resolve":
            path = (workspace / params["path"]).resolve()
            if not path.is_relative_to(workspace / "public_data"):
                raise ValueError("PATH_SCOPE_ERROR")
            return {"path": str(path)}
        if method == "execution.write":
            path = Path(params["path"])
            assert path.is_relative_to(workspace / "public_data")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(params["content"]))
            return {"path": str(path)}
        if method == "execution.cancel":
            os.killpg(processes[params["id"]].pid, signal.SIGTERM)
            return {"requested": True}
        assert method == "execution.run"
        assert "base64" not in params["command"]
        assert "python3 -c" not in params["command"]
        process = await asyncio.create_subprocess_shell(
            params["command"],
            cwd=params["cwd"] or workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        processes[params["id"]] = process
        output, _ = await process.communicate()
        processes.pop(params["id"])
        if params["stream"]:
            await sinks[params["id"]]({"text": output.decode()})
        return {"rc": process.returncode, "stdout": output.decode(), "truncated": False}

    def subscribe(execution_id, sink):
        if sink is None:
            sinks.pop(execution_id, None)
        else:
            sinks[execution_id] = sink

    backend = AgentGenomeBackend(
        tmp_path / "registry", lambda _scope: ExecutionHost(request, subscribe)
    )
    backend.service.import_asset(package)
    state = backend.service.submit(
        "validation",
        "validate",
        "sample",
        "1.0.0",
        {"data": "public_data/input.txt"},
        allow_draft=True,
    )

    async def ignore(_event):
        pass

    result = await backend.execute("validation", state["id"], ignore)
    assert result["status"] == "succeeded", result
    copied = workspace / "public_data/workflow-runs" / state["id"] / "package"
    assert (copied / "reference.bin").read_bytes() == binary
    backend.service.publish("sample", "1.0.0")
    return backend, workspace, calls


async def test_sdk_executes_artifacts_through_host_only(setup_workflow):
    backend, workspace, calls = setup_workflow
    for rows in (3, 1):
        (workspace / "public_data/input.txt").write_text("row\n" * rows)
        state = await backend.invoke(
            "session",
            "run",
            {
                "asset_id": "sample",
                "version": "1.0.0",
                "params": {"data": "public_data/input.txt"},
            },
            uuid4().hex,
        )
        events = []

        async def emit(event):
            events.append(event)

        result = await backend.execute("session", state["id"], emit)
        assert result["status"] == "succeeded", result
        report = result["result"]["outputs"]["report"]
        assert json.loads(Path(report["execution_path"]).read_text())["rows"] == rows
        assert not (backend.service.root / "runs" / state["id"] / "artifacts").exists()
        assert any(e["event"] == "check" for e in events)
    assert "execution.resolve" in calls
    assert "execution.write" in calls
    with pytest.raises(ValueError, match="conversation"):
        await backend.invoke(
            "other-session", "status", {"run_id": state["id"]}, "query"
        )


async def test_workflow_waits_for_turn_and_cancels_queued(setup_workflow):
    backend, _, _ = setup_workflow
    events = []

    async def publish(*event):
        events.append(event)

    workflows = ReusableWorkflows(backend, publish)
    arguments = {
        "asset_id": "sample",
        "version": "1.0.0",
        "params": {"data": "public_data/input.txt"},
    }
    state = await workflows.invoke(
        "session", "run", arguments, "request", "origin-turn"
    )
    assert (await workflows.invoke("session", "run", arguments, "request"))[
        "id"
    ] == state["id"]
    await asyncio.sleep(0)
    assert not events
    assert workflows.busy("session")
    workflows.release("session", "older-turn")
    await asyncio.sleep(0)
    assert not events
    workflows.release("session", "origin-turn")
    await asyncio.wait_for(workflows.wait_idle("session"), 10)
    assert [e[2] for e in events].count("operation.completed") == 1
    state = await workflows.invoke("session", "run", arguments, "second")
    await workflows.cancel("session")
    await workflows.wait_idle("session")
    assert backend.service.status("session", state["id"])["status"] == "cancelled"
    await workflows.close()


async def test_missing_input_stops_and_history_survives_restart(setup_workflow):
    backend, _, _ = setup_workflow
    state = await backend.invoke(
        "session",
        "run",
        {
            "asset_id": "sample",
            "version": "1.0.0",
            "params": {"data": "public_data/missing.txt"},
        },
        "failed",
    )

    async def emit(_event):
        pass

    result = await backend.execute("session", state["id"], emit)
    assert result["status"] == "failed"
    pending = await backend.invoke(
        "session",
        "run",
        {
            "asset_id": "sample",
            "version": "1.0.0",
            "params": {"data": "public_data/input.txt"},
        },
        "pending",
    )
    backend.service.recover()
    assert backend.service.status("session", pending["id"])["status"] == "interrupted"
    assert backend.service.status("session", state["id"])["status"] == "failed"


async def test_runtime_persists_workflow_progress_and_authorizes_calls(
    setup_workflow, tmp_path
):
    from pyromind_runtime.application.conversation_runtime import ConversationRuntime
    from pyromind_runtime.domain.context import RequestContext
    from pyromind_runtime.domain.snapshot import TimelineOperation
    from pyromind_runtime.ports.harness import SessionSpec

    from .fake_adapter import FakeAdapter

    backend, _, _ = setup_workflow
    adapter = FakeAdapter()
    runtime = ConversationRuntime(
        tmp_path / "conversations", adapter, workflows=backend
    )
    context = RequestContext(user_id="owner")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="session",
            user_id="owner",
            workspace_root=str(tmp_path / "conversations"),
        ),
        context,
    )
    try:
        with pytest.raises(Exception, match="does not belong"):
            await runtime.workflow_call(
                "session", "list", {}, "bad", RequestContext(user_id="other")
            )
        await runtime.workflow_call(
            "session",
            "run",
            {
                "asset_id": "sample",
                "version": "1.0.0",
                "params": {"data": "public_data/input.txt"},
            },
            "call",
            context,
        )
        assert runtime._workflows is not None
        assert not runtime._is_releasable(runtime._active["session"])
        from pyromind_runtime.domain.commands import UserMessageCommand
        from pyromind_runtime.domain.content import TextContent

        receipt = await runtime.submit_command(
            "session",
            UserMessageCommand(
                command_id="next", content=(TextContent(text="next task"),)
            ),
            context,
        )
        assert receipt.status == "accepted"
        assert not adapter.sent
        adapter.emit(
            "session",
            "run.finished",
            {"outcome": {"status": "completed"}},
            run_id="origin-turn",
        )
        await asyncio.wait_for(runtime._workflows.wait_idle("session"), 10)
        await asyncio.gather(*runtime._workflows.deferred)
        assert adapter.sent[0][1].command_id == "next"
        snapshot = await runtime.get_snapshot("session", context)
        operations = [
            item for item in snapshot.timeline if isinstance(item, TimelineOperation)
        ]
        assert len(operations) == 1
        assert operations[0].status == "completed"
        assert operations[0].output[0].type == "text"
    finally:
        await runtime.close()


async def test_cancellation_stops_inflight_script(setup_workflow, tmp_path):
    backend, _, _ = setup_workflow
    package = tmp_path / "source"
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["version"] = "2.0.0"
    (package / "manifest.json").write_text(json.dumps(manifest))
    (package / "report.py").write_text("import time\ntime.sleep(30)\n")
    backend.service.import_asset(package)
    state = backend.service.submit(
        "session",
        "slow",
        "sample",
        "2.0.0",
        {
            "data": "public_data/input.txt",
        },
        allow_draft=True,
    )
    started = asyncio.Event()

    async def emit(event):
        if event.get("event") == "command_started":
            started.set()

    execution = asyncio.create_task(backend.execute("session", state["id"], emit))
    await asyncio.wait_for(started.wait(), 5)
    await asyncio.sleep(0.05)
    await backend.invoke("session", "cancel", {"run_id": state["id"]}, "cancel")
    result = await asyncio.wait_for(execution, 5)
    assert result["status"] == "cancelled"
    await backend.close()
