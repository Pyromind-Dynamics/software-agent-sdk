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
from core.ports import PortError, ToolResult
from harness_adapter.agentgenome_bridge import (
    AgentGenomeBackend,
    BridgeTool,
    ExecutionHost,
)
from harness_adapter.pi_adapter.runner import PiRunnerError
from pyromind_runtime.application.reusable_workflows import (
    ReusableWorkflows,
    started_run,
)
from pyromind_runtime.domain.content import JsonObject, JsonValue


def missing_file_error(path: Path | str) -> PiRunnerError:
    """The sandbox file API's wording for a file that is not there."""
    return PiRunnerError(
        "request_failed: sandbox file request failed: GET "
        "/api/v1/sandboxes/sb-1/files/read -> 400 "
        '{"success":false,"error":{"code":"FILE_READ_ERROR","message":'
        f"\"Cannot read file '{path}': ls: cannot access '{path}': "
        'No such file or directory"}}'
    )


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


async def test_execution_host_reads_host_local_exports_without_the_plane(tmp_path):
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
    relative = "public_data/agentgenome-traces/trace.json"
    assert await asyncio.to_thread(host.read, relative) == b'{"trace_id": "trace"}'
    # The bytes are on this host already, so neither a single read nor a batch
    # read pays an upload and a read-back for them.
    assert calls == []
    assert await asyncio.to_thread(host.read_many, [relative]) == {
        relative: b'{"trace_id": "trace"}'
    }
    assert calls == []

    remote["public_data/data/raw.csv"] = b"a,b\n"
    assert await asyncio.to_thread(host.read, "public_data/data/raw.csv") == b"a,b\n"
    assert calls == ["execution.resolve", "execution.read"]


async def test_execution_host_batches_reads_and_reuses_answers(tmp_path):
    pages = [
        {
            "files": [
                {"path": "scripts/a.py", "content": base64.b64encode(b"a").decode()},
                {"path": "scripts/ghost.py", "content": None},
            ],
            "remaining": ["scripts/b.py"],
        },
        {
            "files": [
                {"path": "scripts/b.py", "content": base64.b64encode(b"b").decode()},
            ],
            "remaining": [],
        },
    ]
    calls: list[tuple[str, list[str]]] = []

    async def request(method, payload):
        calls.append((method, list(payload["paths"])))
        assert method == "execution.read_many"
        return pages.pop(0)

    host = ExecutionHost(request, lambda *_: None, tmp_path / "conversation")
    wanted = ["scripts/a.py", "scripts/ghost.py", "scripts/b.py"]
    answers = await asyncio.to_thread(host.read_many, wanted)
    assert answers == {
        "scripts/a.py": b"a",
        "scripts/ghost.py": None,
        "scripts/b.py": b"b",
    }
    assert calls == [
        ("execution.read_many", wanted),
        ("execution.read_many", ["scripts/b.py"]),
    ]
    calls.clear()
    # Everything the host already answered is served without touching the plane.
    assert await asyncio.to_thread(host.read_many, wanted) == answers
    assert calls == []


async def test_execution_host_bounds_a_wedged_read(tmp_path, monkeypatch):
    monkeypatch.setenv("PYROMIND_AGENTGENOME_CALL_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("PYROMIND_AGENTGENOME_SLOW_CALL_TIMEOUT_SECONDS", "0.05")

    async def request(method, payload):
        await asyncio.sleep(60)

    host = ExecutionHost(request, lambda *_: None, tmp_path / "conversation")
    with pytest.raises(TimeoutError, match="did not answer"):
        await asyncio.to_thread(host.read, "scripts/clean.py")
    with pytest.raises(TimeoutError, match="did not answer"):
        await asyncio.to_thread(host.read_many, ["scripts/clean.py"])
    await asyncio.sleep(0.01)


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


async def test_execution_host_reports_a_missing_remote_file_as_not_found(tmp_path):
    """Capture probes for optional files, so a "no such file" answer is an `OSError`.

    The sandbox file API words that answer as a 400 request error; callers that probe
    for a script's runtime siblings must not have to know that.
    """

    async def request(method, payload):
        if method == "execution.resolve":
            return {"path": f"/sandbox/{payload['path']}"}
        assert method == "execution.read"
        raise missing_file_error("/sandbox/scripts/__init__.py")

    host = ExecutionHost(request, lambda *_: None, tmp_path / "conversation")
    with pytest.raises(FileNotFoundError):
        await asyncio.to_thread(host.read, "scripts/__init__.py")


async def test_execution_host_keeps_other_runner_errors(tmp_path):
    async def request(method, payload):
        if method == "execution.resolve":
            return {"path": f"/sandbox/{payload['path']}"}
        raise PiRunnerError(
            "request_failed: sandbox file request failed: GET /x -> 500 boom"
        )

    host = ExecutionHost(request, lambda *_: None, tmp_path / "conversation")
    with pytest.raises(PiRunnerError):
        await asyncio.to_thread(host.read, "scripts/clean.py")


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
            return {"file_read": True, "revision": True, "tool": True}
        if method == "execution.read":
            path = Path(params["path"])
            assert path.is_relative_to(workspace / "public_data")
            if not path.is_file():
                # The real file API reports a missing file as a request error.
                raise missing_file_error(path)
            content = path.read_bytes()
            assert len(content) <= params["limit"]
            return {"content": base64.b64encode(content).decode()}
        if method == "execution.read_many":
            files = []
            for raw in params["paths"]:
                path = (workspace / raw).resolve()
                if (
                    not path.is_relative_to(workspace / "public_data")
                    or not path.is_file()
                ):
                    files.append({"path": raw, "content": None})
                    continue
                content = path.read_bytes()
                assert len(content) <= params["limit"]
                files.append(
                    {"path": raw, "content": base64.b64encode(content).decode()}
                )
            return {"files": files, "remaining": []}
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
        if method == "execution.tool":
            if params["name"] == "df_submit_pipeline":
                return {
                    "is_error": False,
                    "content": [{"type": "text", "text": "submitted task-1"}],
                    "details": {"task_id": "task-1"},
                }
            return {
                "is_error": True,
                "content": [{"type": "text", "text": f"unknown tool {params['name']}"}],
                "details": {"code": "TASK_REJECTED"},
            }
        assert method == "execution.run"
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


async def prepare_confirmed_run(
    backend: AgentGenomeBackend,
    scope: str,
    arguments: JsonObject,
    request_id: str,
) -> str:
    plan = await backend.invoke(
        scope,
        "prepare_run",
        {
            **arguments,
            "execution_scope": {
                "mode": "full",
                "description": "测试明确确认全量执行",
            },
        },
        request_id,
    )
    assert plan["status"] == "planned"
    assert plan["requires_confirmation"] is True
    assert plan["run_id"] is None
    confirmed = await backend.invoke(
        scope,
        "confirm_run",
        {"plan_id": plan["plan_id"], "digest": plan["digest"]},
        f"{request_id}-confirm",
    )
    run = confirmed["run"]
    assert isinstance(run, dict)
    run_id = run.get("id")
    assert isinstance(run_id, str)
    return run_id


async def test_sdk_executes_artifacts_through_host_only(setup_workflow):
    backend, workspace, calls = setup_workflow
    last_run_id: str | None = None
    for index, rows in enumerate((3, 1)):
        (workspace / "public_data/input.txt").write_text("row\n" * rows)
        run_id = await prepare_confirmed_run(
            backend,
            "session",
            {
                "asset_id": "sample",
                "version": "1.0.0",
                "params": {"data": "public_data/input.txt"},
            },
            f"run-{index}",
        )
        last_run_id = run_id
        events = []

        async def emit(event):
            events.append(event)

        result = await backend.execute("session", run_id, emit)
        assert result["status"] == "succeeded", result
        report = result["result"]["outputs"]["report"]
        assert json.loads(Path(report["execution_path"]).read_text())["rows"] == rows
        assert not (backend.service.root / "runs" / run_id / "artifacts").exists()
        assert any(e["event"] == "check" for e in events)
    assert "execution.resolve" in calls
    assert "execution.write" in calls
    assert last_run_id is not None
    with pytest.raises(ValueError, match="conversation"):
        await backend.invoke(
            "other-session", "status", {"run_id": last_run_id}, "query"
        )


async def test_bridge_tool_calls_the_execution_plane_tool_port():
    calls: list[tuple[str, JsonObject]] = []

    async def request(method: str, payload: JsonObject) -> JsonValue:
        calls.append((method, payload))
        return {
            "is_error": False,
            "content": [
                {"type": "text", "text": "submitted"},
                {"type": "text", "text": "task-1"},
            ],
            "details": {"task_id": "task-1"},
        }

    result = await asyncio.to_thread(
        BridgeTool(ExecutionHost(request, lambda *_: None)).call,
        "df_submit_pipeline",
        {"dataset": "storage/a.csv"},
        execution_id="node-submit:1:call-1",
    )
    assert result == ToolResult(False, {"task_id": "task-1"}, "submitted\ntask-1")
    method, payload = calls[0]
    assert method == "execution.tool"
    assert payload["name"] == "df_submit_pipeline"
    assert payload["arguments"] == {"dataset": "storage/a.csv"}
    assert payload["tool_call_id"] == "node-submit:1:call-1"

    async def broken(_method: str, _payload: JsonObject) -> JsonValue:
        return {"accepted": True}

    with pytest.raises(PortError, match="invalid response"):
        await asyncio.to_thread(
            BridgeTool(ExecutionHost(broken, lambda *_: None)).call,
            "df_submit_pipeline",
            {},
            execution_id="node-submit:1:call-2",
        )


TOOL_GRAPH = """version: "gt/1.0"
node:
  id: pretag-sop
  mode: sequence
  input:
    dataset: {type: path, from: params.dataset}
  output:
    submission: {type: json, compose: [submit.output.submission]}
  verify:
    - metric: "submission.task_id"
  children:
    - id: submit
      input:
        dataset: {type: path, from: pretag-sop.input.dataset}
      do:
        tool:
          name: __TOOL__
          args:
            dataset: "{input.dataset}"
            task_name: pretag
            retries: 2
      output:
        submission: {type: json}
      verify:
        - metric: "submission.task_id"
"""


def write_tool_package(root: Path, asset_id: str, tool_name: str) -> Path:
    """The package distill emits for one fixed host tool call."""
    package = root / asset_id
    package.mkdir()
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "id": asset_id,
                "version": "1.0.0",
                "name": "Platform submission",
                "description": "Submit a dataset through a host tool",
                "parameters": {"dataset": {"type": "path"}},
            }
        )
    )
    (package / "graph.yaml").write_text(TOOL_GRAPH.replace("__TOOL__", tool_name))
    return package


def read_ledger(backend: AgentGenomeBackend, run_id: str) -> list[dict[str, object]]:
    path = backend.service.root / "runs" / run_id / "ledger.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def test_tool_nodes_run_through_the_host_tool_port(setup_workflow, tmp_path):
    backend, workspace, _ = setup_workflow
    backend.service.import_asset(
        write_tool_package(tmp_path, "pretag-submit", "df_submit_pipeline")
    )
    (workspace / "public_data/pretag.csv").write_text("question\nhow?\n")
    params: JsonObject = {"dataset": "public_data/pretag.csv"}

    async def ignore(_event):
        pass

    draft = backend.service.submit(
        "validation", uuid4().hex, "pretag-submit", "1.0.0", params, allow_draft=True
    )
    replay = await backend.execute("validation", draft["id"], ignore)
    assert replay["status"] == "succeeded", replay
    backend.service.publish("pretag-submit", "1.0.0")
    run_id = await prepare_confirmed_run(
        backend,
        "session",
        {"asset_id": "pretag-submit", "version": "1.0.0", "params": params},
        "run-tool",
    )
    result = await backend.execute("session", run_id, ignore)
    assert result["status"] == "succeeded", result
    submission = result["result"]["outputs"]["submission"]
    assert json.loads(Path(submission["execution_path"]).read_text()) == {
        "task_id": "task-1"
    }
    call = next(
        item for item in read_ledger(backend, run_id) if item["event"] == "call_tool"
    )
    assert call["tool"] == "df_submit_pipeline"
    assert call["is_error"] is False
    # The graph slot carries the execution plane's own path for the parameter,
    # not the workspace-relative string the caller submitted.
    assert call["arguments"] == {
        "dataset": str((workspace / "public_data/pretag.csv").resolve()),
        "task_name": "pretag",
        "retries": 2,
    }


async def test_tool_node_errors_fail_the_node_and_keep_its_artifact(
    setup_workflow, tmp_path
):
    backend, workspace, _ = setup_workflow
    backend.service.import_asset(
        write_tool_package(tmp_path, "label-import", "df_import_labels")
    )
    (workspace / "public_data/pretag.csv").write_text("question\nhow?\n")
    state = backend.service.submit(
        "session",
        uuid4().hex,
        "label-import",
        "1.0.0",
        {"dataset": "public_data/pretag.csv"},
        allow_draft=True,
    )

    async def ignore(_event):
        pass

    result = await backend.execute("session", state["id"], ignore)
    assert result["status"] == "failed", result
    assert result["result"]["failure"]["stage"] == "do"
    call = next(
        item
        for item in read_ledger(backend, state["id"])
        if item["event"] == "call_tool"
    )
    assert call["is_error"] is True
    artifact = (
        workspace
        / "public_data/workflow-runs"
        / state["id"]
        / "artifacts/pretag-sop--submit.out"
    )
    assert json.loads(artifact.read_text()) == {"code": "TASK_REJECTED"}


async def test_legacy_revision_needs_declarations_and_successful_run_can_be_handed_over(
    setup_workflow,
):
    backend, _, calls = setup_workflow
    asset = await backend.invoke(
        "session", "get", {"asset_id": "sample", "version": "1.0.0"}, "inspect"
    )
    assert asset["revision_availability"]["available"]
    run_id = await prepare_confirmed_run(
        backend,
        "session",
        {
            "asset_id": "sample",
            "version": "1.0.0",
            "params": {"data": "public_data/input.txt"},
        },
        "submit",
    )

    async def emit(_event):
        pass

    result = await backend.execute("session", run_id, emit)
    assert result["status"] == "succeeded"
    with pytest.raises(ValueError, match="legacy package requires"):
        await backend.invoke("session", "prepare_revision", {"run_id": run_id}, "draft")
    before = calls.count("execution.run")
    taken = await backend.invoke(
        "session",
        "takeover",
        {"run_id": run_id, "reason": "outputs do not meet the task"},
        "takeover",
    )
    assert taken["taken_over"] and taken["status"] == "succeeded"
    assert taken["result"] == result["result"]
    status = await backend.invoke(
        "session", "status", {"run_id": run_id}, "inspect-run"
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
        "execution_scope": {"mode": "sample", "description": "定向验证小样"},
        "validation_params": {
            "raw": "public_data/data/raw.csv",
            "rules": "public_data/rules.json",
        },
    }
    prepared = await workflows.invoke(
        "session", "distill", arguments, "distill-1", "origin-turn"
    )
    assert prepared["status"] == "planned"
    review = prepared["review"]
    assert isinstance(review, dict)
    validation = review["validation"]
    assert isinstance(validation, dict)
    assert validation["ready"] is True
    plan = await workflows.invoke(
        "session",
        "confirm_distill",
        {"plan_id": prepared["plan_id"], "digest": prepared["digest"]},
        "distill-confirm-1",
        "origin-turn",
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

    again_prepared = await workflows.invoke(
        "session", "distill", arguments, "distill-2", "origin-turn"
    )
    again = await workflows.invoke(
        "session",
        "confirm_distill",
        {
            "plan_id": again_prepared["plan_id"],
            "digest": again_prepared["digest"],
        },
        "distill-confirm-2",
        "origin-turn",
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


async def test_distill_pairs_a_placeholder_write_with_a_literal_read(
    setup_workflow,
):
    """The plan may spell one file two ways, and a script may name absent helpers.

    The plan writes `{filtered_jsonl}` but reads the path that placeholder binds to;
    the capture must link those as one file instead of calling the read an input the
    plan dropped. The script also names runtime helpers that are not on disk, which a
    real sandbox answers with a request error rather than `FileNotFoundError`.
    """
    backend, workspace, _ = setup_workflow
    refs = write_capture_workspace(workspace)
    clean = workspace / "public_data/scripts/clean.py"
    clean.write_text(
        CLEAN_SCRIPT + "\n# serving helpers: dataflow/serving/__init__.py, "
        "serving/api_llm_serving_request.py\n"
    )
    events = []

    async def publish(*event):
        events.append(event)

    workflows = ReusableWorkflows(backend, publish)
    arguments = {
        "label": "预打标到清洗",
        "asset_id": "prelabel-clean-tokens",
        "trace_refs": refs,
        "final_artifact": "public_data/out/clean.csv",
        "steps": [
            {
                "run": "python3 public_data/scripts/prelabel.py "
                "public_data/data/raw.csv public_data/rules.json "
                "> {filtered_jsonl}",
                "from": "s1:1",
            },
            {
                "run": "python3 public_data/scripts/clean.py "
                "public_data/out/prelabeled.jsonl > {sft_messages}",
                "from": "s1:2",
            },
        ],
        "params": {
            "raw": "public_data/data/raw.csv",
            "rules": "public_data/rules.json",
            "filtered_jsonl": "public_data/out/prelabeled.jsonl",
            "sft_messages": "public_data/out/clean.csv",
        },
        "new_params": {
            "raw": "public_data/data/raw2.csv",
            "rules": "public_data/rules.json",
        },
        "execution_scope": {"mode": "sample", "description": "定向验证小样"},
        "validation_params": {
            "raw": "public_data/data/raw.csv",
            "rules": "public_data/rules.json",
        },
    }
    prepared = await workflows.invoke(
        "session", "distill", arguments, "distill-1", "origin-turn"
    )
    assert prepared["status"] == "planned"
    plan = await workflows.invoke(
        "session",
        "confirm_distill",
        {"plan_id": prepared["plan_id"], "digest": prepared["digest"]},
        "distill-confirm-1",
        "origin-turn",
    )
    assert plan["reused"] is False
    assert plan["run_id"]
    distilled = plan["distilled"]
    assert isinstance(distilled, dict)
    assert distilled["chain"] == {
        "checked": True,
        "traces": ["s1", "s2"],
        "boundaries": [],
    }
    params = distilled["params"]
    assert isinstance(params, list)
    assert [item["key"] for item in params if isinstance(item, dict)] == [
        "raw",
        "rules",
    ]
    scripts = distilled["scripts"]
    assert isinstance(scripts, list)
    assert [item["package"] for item in scripts if isinstance(item, dict)] == [
        "scripts/clean.py",
        "scripts/prelabel.py",
    ]
    nodes = distilled["nodes"]
    assert isinstance(nodes, list)
    assert [node["id"] for node in nodes if isinstance(node, dict)] == [
        "prelabel",
        "clean",
    ]

    workflows.release("session", "origin-turn")
    await asyncio.wait_for(workflows.wait_idle("session"), 30)

    assert [event[2] for event in events].count("operation.completed") == 1
    capture = backend.service.captures.status(plan["capture_id"])
    assert capture["stage"] == "published"
    assert capture["run"]["status"] == "succeeded"
    assert capture["generalized"] is True
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
    plan = await workflows.invoke(
        "session",
        "prepare_run",
        {
            **arguments,
            "execution_scope": {"mode": "full", "description": "用户确认全量"},
        },
        "prepare",
    )
    assert plan["status"] == "planned"
    confirmed = await workflows.invoke(
        "session",
        "confirm_run",
        {"plan_id": plan["plan_id"], "digest": plan["digest"]},
        "confirm",
        "origin-turn",
    )
    state = confirmed["run"]
    assert isinstance(state, dict)
    duplicate = await workflows.invoke(
        "session",
        "confirm_run",
        {"plan_id": plan["plan_id"], "digest": plan["digest"]},
        "confirm-duplicate",
    )
    duplicate_run = duplicate["run"]
    assert isinstance(duplicate_run, dict)
    assert duplicate_run["id"] == state["id"]
    await asyncio.sleep(0)
    assert not events
    assert workflows.busy("session")
    workflows.release("session", "older-turn")
    await asyncio.sleep(0)
    assert not events
    workflows.release("session", "origin-turn")
    await asyncio.wait_for(workflows.wait_idle("session"), 10)
    assert [e[2] for e in events].count("operation.completed") == 1
    plan = await workflows.invoke(
        "session",
        "prepare_run",
        {
            **arguments,
            "execution_scope": {"mode": "full", "description": "用户确认全量"},
        },
        "second-prepare",
    )
    confirmed = await workflows.invoke(
        "session",
        "confirm_run",
        {"plan_id": plan["plan_id"], "digest": plan["digest"]},
        "second-confirm",
        "origin-turn",
    )
    state = confirmed["run"]
    assert isinstance(state, dict)
    await workflows.cancel("session")
    await workflows.wait_idle("session")
    assert backend.service.status("session", state["id"])["status"] == "cancelled"
    await workflows.close()


async def test_missing_input_stops_and_history_survives_restart(setup_workflow):
    backend, _, _ = setup_workflow
    run_id = await prepare_confirmed_run(
        backend,
        "session",
        {
            "asset_id": "sample",
            "version": "1.0.0",
            "params": {"data": "public_data/missing.txt"},
        },
        "failed",
    )

    async def emit(_event):
        pass

    result = await backend.execute("session", run_id, emit)
    assert result["status"] == "failed"
    pending_run_id = await prepare_confirmed_run(
        backend,
        "session",
        {
            "asset_id": "sample",
            "version": "1.0.0",
            "params": {"data": "public_data/input.txt"},
        },
        "pending",
    )
    backend.service.recover()
    assert backend.service.status("session", pending_run_id)["status"] == "interrupted"
    assert backend.service.status("session", run_id)["status"] == "failed"


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
        plan = await runtime.workflow_call(
            "session",
            "prepare_run",
            {
                "asset_id": "sample",
                "version": "1.0.0",
                "params": {"data": "public_data/input.txt"},
                "execution_scope": {
                    "mode": "full",
                    "description": "用户明确确认全量执行",
                },
            },
            "prepare",
            context,
        )
        assert plan["status"] == "planned"
        assert plan["run_id"] is None
        assert runtime._workflows is not None
        assert not runtime._workflows.busy("session")
        await runtime.workflow_call(
            "session",
            "confirm_run",
            {"plan_id": plan["plan_id"], "digest": plan["digest"]},
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


def test_started_run_accepts_a_resumed_gate():
    assert started_run("run", {"id": "r1", "status": "queued"}) == "r1"
    confirmed: JsonObject = {"run": {"id": "r-confirmed", "status": "queued"}}
    assert started_run("confirm_run", confirmed) == "r-confirmed"
    assert started_run("run", confirmed) == "r-confirmed"
    assert started_run("resume", {"id": "r1", "status": "queued"}) == "r1"
    assert started_run("resume", {"id": "r1", "status": "waiting"}) is None
    # A requeued replay is a fresh attempt on the same frozen draft, so the new
    # run id has to be followed the same way a plain run is.
    assert started_run("replay_requeue", {"id": "r2", "status": "queued"}) == "r2"
    assert started_run("replay_requeue", {"id": "r2", "status": "failed"}) is None


async def test_parked_run_keeps_the_operation_open_until_resume():
    events = []
    executions = []

    class ParkedBackend:
        async def invoke(
            self,
            scope: str,
            action: str,
            arguments: JsonObject,
            request_id: str,
        ) -> JsonObject:
            if action == "run":
                return {"id": "parked-run", "status": "queued"}
            assert action == "resume"
            return {"id": arguments["run_id"], "status": "queued"}

        async def execute(self, scope, run_id, emit):
            executions.append(run_id)
            if len(executions) == 1:
                return {
                    "id": run_id,
                    "status": "waiting",
                    "result": {
                        "waiting": {
                            "node": "review",
                            "question": "请到 Label Studio 复核",
                            "artifact": f"/runs/{run_id}/review.out",
                        }
                    },
                }
            return {"id": run_id, "status": "succeeded", "result": {"outputs": {}}}

        async def close(self):
            pass

    async def publish(*event):
        events.append(event)

    workflows = ReusableWorkflows(ParkedBackend(), publish)
    queued = await workflows.invoke(
        "session",
        "run",
        {"asset_id": "gate", "version": "1.0.0", "params": {}},
        "run-1",
        "origin-turn",
    )
    assert queued["status"] == "queued"
    workflows.release("session", "origin-turn")
    await asyncio.wait_for(workflows.wait_idle("session"), 5)
    # A parked run is a pause, not a verdict: the operation stays open.
    assert [event[2] for event in events] == ["operation.started"]

    resumed = await workflows.invoke(
        "session", "resume", {"run_id": "parked-run"}, "resume-1", "origin-turn"
    )
    assert resumed["status"] == "queued"
    workflows.release("session", "origin-turn")
    await asyncio.wait_for(workflows.wait_idle("session"), 5)
    assert executions == ["parked-run", "parked-run"]
    assert [event[2] for event in events][-1] == "operation.completed"
    assert json.loads(events[-1][3]["output"][0]["text"])["status"] == "succeeded"
    await workflows.close()


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
