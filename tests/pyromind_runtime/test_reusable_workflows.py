from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import signal
from pathlib import Path
from uuid import uuid4

import pytest
from harness_adapter.agentgenome_bridge import AgentGenomeBackend, ExecutionHost
from pyromind_runtime.application.reusable_workflows import ReusableWorkflows
from pyromind_runtime.domain.content import JsonObject, JsonValue


@pytest.mark.parametrize(
    "cancelled,stopped", [(False, False), (True, False), (True, True)]
)
async def test_execution_host_reports_preparation_stop_evidence(cancelled, stopped):
    from agentgenome.service import RunCancelled

    evidence = {
        "execution_id": "remote-1",
        "phase": "environment",
        "started": False,
        "stopped": stopped,
        "error_code": "startup_timeout",
    }

    async def request(method, payload):
        return {
            "rc": None,
            "stopped": stopped,
            "error": "startup timeout",
            "execution": evidence,
        }

    events = []
    subscriptions = []
    host = ExecutionHost(request, lambda key, callback: subscriptions.append(callback))
    host.emit = events.append
    if cancelled:
        host.cancel.set()
    expected = RunCancelled if cancelled and stopped else RuntimeError
    with pytest.raises(expected):
        await host._run_async("mkdir run", ".", stream=False)
    assert events[0]["execution"]["phase"] == "preparing"
    assert events[-1]["execution"] == evidence
    assert subscriptions[-1] is None


async def test_execution_host_mirrors_host_local_exports(tmp_path):
    conversation = tmp_path / "conversation"
    export = conversation / "public_data/agentgenome-traces/trace.json"
    export.parent.mkdir(parents=True)
    export.write_text('{"trace_id": "trace"}')
    remote: dict[str, bytes] = {}
    calls: list[str] = []

    async def request(method, payload):
        calls.append(method)
        if method == "execution.write":
            remote[payload["path"]] = base64.b64decode(payload["content"])
            return {"path": payload["path"]}
        if method == "execution.resolve":
            return {"path": f"/sandbox/{payload['path']}"}
        assert method == "execution.read"
        content = remote.get(str(payload["path"]).removeprefix("/sandbox/"))
        if content is None:
            raise FileNotFoundError(payload["path"])
        return {"content": base64.b64encode(content).decode("ascii")}

    host = ExecutionHost(request, lambda *_: None, conversation)
    # The bridge runs the genome service in a worker thread, so blocking on the
    # loop is only safe away from the loop thread.
    assert (
        await asyncio.to_thread(host.read, "public_data/agentgenome-traces/trace.json")
        == b'{"trace_id": "trace"}'
    )
    assert calls == ["execution.write", "execution.resolve", "execution.read"]

    remote["public_data/data/raw.csv"] = b"a,b\n"
    calls.clear()
    assert await asyncio.to_thread(host.read, "public_data/data/raw.csv") == b"a,b\n"
    assert calls == ["execution.resolve", "execution.read"]


async def test_execution_host_reads_a_bounded_head(tmp_path):
    conversation = tmp_path / "conversation"
    export = conversation / "public_data/raw.csv"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"header\n" + b"x" * 100)
    calls: list[str] = []

    async def request(method, payload):
        calls.append(method)
        if method == "execution.resolve":
            return {"path": f"/sandbox/{payload['path']}"}
        assert method == "execution.run"
        return {"rc": 0, "stdout": base64.b64encode(b"header\n").decode("ascii")}

    host = ExecutionHost(request, lambda *_: None, conversation)
    # A host-local export is cut locally; the execution plane is never asked to
    # move a whole dataset just so capture can look at its first rows.
    assert await asyncio.to_thread(host.head, "public_data/raw.csv", 7) == b"header\n"
    assert calls == []

    calls.clear()
    assert (
        await asyncio.to_thread(host.head, "public_data/remote.csv", 7) == b"header\n"
    )
    assert calls == ["execution.resolve", "execution.run"]


async def test_capture_sampling_reads_a_remote_input_through_the_execution_host(
    tmp_path,
):
    from agentgenome.capture.sample import sample_inputs

    body = b"text,label\none,a\ntwo,b\n"

    async def request(method, payload):
        if method == "execution.resolve":
            return {"path": f"/sandbox/{payload['path']}"}
        assert method == "execution.run"
        window = int(payload["command"].split("head -c ")[1].split(" ")[0])
        return {"rc": 0, "stdout": base64.b64encode(body[:window]).decode("ascii")}

    host = ExecutionHost(request, lambda *_: None, tmp_path / "conversation")
    directory = tmp_path / "samples"
    index = await asyncio.to_thread(
        sample_inputs, host, {"raw": "public_data/raw.csv"}, directory
    )
    entry = index["files"][0]
    assert entry["format"] == "csv"
    assert entry["schema"]["columns"] == ["text", "label"]
    assert entry["rows"] == 2
    assert (directory / entry["name"]).read_text() == "text,label\none,a\ntwo,b\n"


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
        if method == "execution.capabilities":
            return {"file_read": True, "revision": True}
        if method == "execution.read":
            path = Path(params["path"])
            assert path.is_relative_to(workspace / "public_data")
            content = path.read_bytes()
            assert len(content) <= params["limit"]
            return {"content": base64.b64encode(content).decode()}
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


async def test_legacy_revision_needs_declarations_and_successful_run_can_be_handed_over(
    setup_workflow,
):
    backend, _, calls = setup_workflow
    asset = await backend.invoke(
        "session", "get", {"asset_id": "sample", "version": "1.0.0"}, "inspect"
    )
    assert asset["revision_availability"]["available"]
    state = await backend.invoke(
        "session",
        "run",
        {
            "asset_id": "sample",
            "version": "1.0.0",
            "params": {"data": "public_data/input.txt"},
        },
        "submit",
    )

    async def emit(_event):
        pass

    result = await backend.execute("session", state["id"], emit)
    assert result["status"] == "succeeded"
    with pytest.raises(ValueError, match="legacy package requires"):
        await backend.invoke(
            "session", "prepare_revision", {"run_id": state["id"]}, "draft"
        )
    before = calls.count("execution.run")
    taken = await backend.invoke(
        "session",
        "takeover",
        {"run_id": state["id"], "reason": "outputs do not meet the task"},
        "takeover",
    )
    assert taken["taken_over"] and taken["status"] == "succeeded"
    assert taken["result"] == result["result"]
    status = await backend.invoke(
        "session", "status", {"run_id": state["id"]}, "inspect-run"
    )
    assert not status["recovery"]["revision"]["available"]
    assert status["recovery"]["takeover"]["taken_over"]
    assert calls.count("execution.run") == before


PRELABEL_SCRIPT = """import csv, json, sys

prefix = json.load(open(sys.argv[2]))["prefix"]
with open(sys.argv[1], newline="") as stream:
    for row in csv.DictReader(stream):
        print(json.dumps({"text": prefix + row["text"], "label": row["label"]}))
"""

CLEAN_SCRIPT = """import csv, json, sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
writer = csv.DictWriter(sys.stdout, fieldnames=["text", "label"])
writer.writeheader()
for row in rows:
    writer.writerow(row)
"""


def write_capture_workspace(workspace: Path) -> list[JsonValue]:
    """Scripts, data and exported traces the way the Pi extension leaves them."""
    public = workspace / "public_data"
    for name, source in (("prelabel", PRELABEL_SCRIPT), ("clean", CLEAN_SCRIPT)):
        path = public / "scripts" / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    (public / "rules.json").write_text('{"prefix": "p-"}')
    for directory in ("data", "out", "agentgenome-traces"):
        (public / directory).mkdir(parents=True, exist_ok=True)
    (public / "data/raw.csv").write_text("text,label\none,a\n")
    (public / "data/raw2.csv").write_text("text,label\nthree,c\n")
    (public / "out/review.jsonl").write_text('{"text": "p-one", "label": "a"}\n')
    (public / "out/review2.jsonl").write_text('{"text": "p-three", "label": "c"}\n')
    commands = [
        "python3 public_data/scripts/prelabel.py public_data/data/raw.csv "
        "public_data/rules.json > public_data/out/prelabeled.jsonl",
        "python3 public_data/scripts/clean.py public_data/out/review.jsonl "
        "> public_data/out/clean.csv",
    ]
    refs: list[JsonValue] = []
    for trace_id in ("s1", "s2"):
        steps = [
            {
                "kind": "tool",
                "entry_id": f"{trace_id}-{index}",
                "tool": "bash",
                "call_id": f"{trace_id}-{index}",
                "timestamp": "2026-01-01T00:00:01Z",
                "args": {"command": command},
                "result": {"is_error": False, "text": "ok"},
            }
            for index, command in enumerate(commands, start=1)
        ]
        relative = f"public_data/agentgenome-traces/{trace_id}.json"
        (workspace / relative).write_text(
            json.dumps(
                {
                    "trace_version": 1,
                    "trace_id": trace_id,
                    "session_id": trace_id,
                    "step_count": len(steps),
                    "steps": steps,
                }
            )
        )
        refs.append({"trace_id": trace_id, "path": relative})
    return refs


async def test_capture_freezes_a_graph_and_publishes_after_its_replay(setup_workflow):
    backend, workspace, _ = setup_workflow
    refs = write_capture_workspace(workspace)
    events = []

    async def publish(*event):
        events.append(event)

    workflows = ReusableWorkflows(backend, publish)
    runs = backend.service.root / "runs"
    executed = {path.name for path in runs.iterdir()} if runs.exists() else set()
    plan = await workflows.invoke(
        "session",
        "capture",
        {
            "trace_refs": refs,
            "label": "预打标到清洗",
            "asset_id": "prelabel-clean",
            "new_params": {
                "raw": "public_data/data/raw2.csv",
                "review": "public_data/out/review2.jsonl",
                "rules": "public_data/rules.json",
            },
        },
        "capture-1",
        "origin-turn",
    )
    assert plan["reused"] is False
    assert plan["run_id"]
    # The capture call only freezes the package and queues replay A.
    assert {path.name for path in runs.iterdir()} == executed
    assert workflows.busy("session")
    await asyncio.sleep(0)
    assert not events
    workflows.release("session", "origin-turn")
    await asyncio.wait_for(workflows.wait_idle("session"), 30)

    assert [event[2] for event in events].count("operation.completed") == 1
    capture = backend.service.captures.status(plan["capture_id"])
    assert capture["stage"] == "published"
    assert capture["generalized"] is True
    assert capture["run"]["status"] == "succeeded"
    package = plan["package"]
    assert isinstance(package, str)
    assert json.loads(Path(package, "manifest.json").read_text())["parameters"]

    again = await workflows.invoke(
        "session",
        "capture",
        {"trace_refs": refs, "label": "预打标到清洗", "asset_id": "prelabel-clean"},
        "capture-2",
        "origin-turn",
    )
    assert again["reused"] is True
    assert again["run_id"] is None
    await workflows.close()


async def test_distill_normalizes_a_chain_and_publishes_after_its_replay(
    setup_workflow,
):
    backend, workspace, _ = setup_workflow
    write_capture_workspace(workspace)
    events = []

    async def publish(*event):
        events.append(event)

    workflows = ReusableWorkflows(backend, publish)
    arguments = {
        "label": "预打标到清洗",
        "asset_id": "prelabel-clean-distilled",
        "final_artifact": "public_data/out/clean.csv",
        "steps": [
            {
                "run": "python3 public_data/scripts/prelabel.py "
                "public_data/data/raw.csv public_data/rules.json "
                "> public_data/out/prelabeled.jsonl",
                "from": "s1:1",
            },
            {
                "run": "python3 public_data/scripts/clean.py "
                "public_data/out/prelabeled.jsonl > public_data/out/clean.csv",
                "from": "s1:2",
            },
        ],
        "new_params": {
            "raw": "public_data/data/raw2.csv",
            "rules": "public_data/rules.json",
        },
    }
    plan = await workflows.invoke(
        "session", "distill", arguments, "distill-1", "origin-turn"
    )
    assert plan["reused"] is False
    assert plan["run_id"]
    distilled = plan["distilled"]
    assert isinstance(distilled, dict)
    trace_id = distilled["trace_id"]
    assert isinstance(trace_id, str)
    assert trace_id.startswith("distilled-")
    nodes = distilled["nodes"]
    assert isinstance(nodes, list)
    assert [node["id"] for node in nodes if isinstance(node, dict)] == [
        "prelabel",
        "clean",
    ]
    assert distilled["auxiliary"] == []
    # The distill call only freezes the package and queues replay A.
    assert workflows.busy("session")
    workflows.release("session", "origin-turn")
    await asyncio.wait_for(workflows.wait_idle("session"), 30)

    assert [event[2] for event in events].count("operation.completed") == 1
    capture = backend.service.captures.status(plan["capture_id"])
    assert capture["stage"] == "published"
    assert capture["generalized"] is True
    assert capture["run"]["status"] == "succeeded"

    again = await workflows.invoke(
        "session", "distill", arguments, "distill-2", "origin-turn"
    )
    assert again["reused"] is True
    assert again["run_id"] is None
    with pytest.raises(ValueError, match="内联代码"):
        await workflows.invoke(
            "session",
            "distill",
            {
                **arguments,
                "steps": [{"run": "python3 -c 'print(1)' > public_data/out/x.csv"}],
            },
            "distill-3",
            "origin-turn",
        )
    await workflows.close()


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
        assert receipt.status == "completed"
        assert adapter.sent[0][1].command_id == "next"
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
        assert len(adapter.external_task_notifications) == 1
        notification = adapter.external_task_notifications[0][1]
        assert notification.trigger_turn is True
        assert notification.kind == "historical_experience"
        assert "execution_path" in notification.hidden_text
        assert "storage/" not in notification.hidden_text
        await runtime.deliver_external_task_status(
            "session", task_id=notification.task_id, status="succeeded"
        )
        assert len(adapter.external_task_notifications) == 1
        assert snapshot.external_tasks[0].resume_pending is False
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


async def test_stage_wait_does_not_block_status_and_cancel(setup_workflow):
    backend, _, _ = setup_workflow
    started = asyncio.Event()
    stopped = asyncio.Event()
    events = []

    async def agent_task(scope: str, request: JsonObject) -> JsonObject:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        raise AssertionError("a cancelled stage must not resume the agent task")

    class StageBackend:
        async def invoke(
            self,
            scope: str,
            action: str,
            arguments: JsonObject,
            request_id: str,
        ) -> JsonObject:
            return {
                "id": "stage-run",
                "status": "queued" if action == "run" else "cancelling",
            }

        async def execute(self, scope, run_id, emit):
            result = await emit(
                {
                    "event": "waiting_agent",
                    "node": "report",
                    "request": {"request_id": "stage-request"},
                }
            )
            assert result["status"] == "cancelled"
            return {"id": run_id, "status": "cancelled"}

        async def close(self):
            pass

    async def publish(*event):
        events.append(event)

    workflows = ReusableWorkflows(StageBackend(), publish, agent_task=agent_task)
    await workflows.invoke("session", "run", {}, "start", "origin")
    workflows.release("session", "origin")
    await asyncio.wait_for(started.wait(), 2)
    await workflows.invoke("session", "status", {}, "status")
    await workflows.cancel("session")
    await asyncio.wait_for(workflows.wait_idle("session"), 2)
    assert stopped.is_set()
    assert events[-1][2] == "operation.completed"
    await workflows.close()


@pytest.mark.parametrize("fix", ["parameter", "script"])
async def test_remote_prompt_revision_executes_without_protection(setup_workflow, fix):
    backend, workspace, calls = setup_workflow
    source = backend.service.root / "revision-source"
    shutil.copytree(backend.service.root / "assets/sample/1.0.0", source)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest.update(
        version="1.1.0",
        revision={
            "safe_to_rerun": True,
            "editable_scripts": ["report.py"],
            "verification_resources": [],
        },
    )
    (source / "manifest.json").write_text(json.dumps(manifest))
    backend.service.import_asset(source)
    params = {
        "data": "public_data/input.txt"
        if fix == "script"
        else "public_data/missing.txt"
    }
    state = backend.service.submit(
        "session", "initial", "sample", "1.1.0", params, allow_draft=True
    )

    async def emit(_event):
        pass

    result = await backend.execute("session", state["id"], emit)
    assert result["status"] == ("succeeded" if fix == "script" else "failed")
    # Publish against a separate validation run for the failed-input case.
    if fix == "parameter":
        valid = backend.service.submit(
            "validation",
            "new-version",
            "sample",
            "1.1.0",
            {"data": "public_data/input.txt"},
            allow_draft=True,
        )
        assert (await backend.execute("validation", valid["id"], emit))[
            "status"
        ] == "succeeded"
    backend.service.publish("sample", "1.1.0")
    arguments = {"run_id": state["id"]}
    if fix == "script":
        arguments["reason"] = "count unique lines, not all lines"
    draft = await backend.invoke("session", "prepare_revision", arguments, "draft")
    assert draft["verification_protection"] == "prompt"
    assert "不要修改验收脚本" in draft["instruction"]
    assert (
        await backend.invoke("session", "prepare_revision", arguments, "draft") == draft
    )
    if fix == "script":
        (workspace / "public_data/input.txt").write_text("one\none\ntwo\n")
        script = Path(draft["editable_dir"]) / "report.py"
        script.write_text(
            script.read_text().replace(
                "len(Path(sys.argv[1]).read_text().splitlines())",
                "len(set(Path(sys.argv[1]).read_text().splitlines()))",
            )
        )
    original_graph = (source / "graph.yaml").read_bytes()
    (Path(draft["editable_dir"]) / "graph.yaml").write_text("bypass all checks")
    arguments = {
        "revision_id": draft["revision_id"],
        "params": {"data": "public_data/input.txt"},
        "change_summary": fix,
    }
    if fix == "script":

        def sample(case_id, text):
            return {
                "id": case_id,
                "files": {"input.txt": text},
                "params": {"data": "fixture:input.txt"},
                "assertions": [
                    {
                        "node": "report",
                        "output": "report",
                        "kind": "json",
                        "expected": {"rows": 2},
                    }
                ],
            }

        arguments.update(
            baseline_cases=[sample("old", "one\ntwo\n")],
            regression_cases=[sample("new", "one\none\ntwo\n")],
        )
    child = await backend.invoke("session", "run", arguments, "rerun")
    assert (await backend.invoke("session", "run", arguments, "rerun"))["id"] == child[
        "id"
    ]
    result = await backend.execute("session", child["id"], emit)
    assert result["status"] == "succeeded", result
    report = json.loads(
        Path(result["result"]["outputs"]["report"]["execution_path"]).read_text()
    )
    assert report["rows"] == 2
    revision = backend.service.revisions.for_run(child["id"])
    assert (Path(revision["package"]) / "graph.yaml").read_bytes() == original_graph
    assert "execution.protect" not in calls
    assert "execution.read" in calls and "execution.write" in calls
