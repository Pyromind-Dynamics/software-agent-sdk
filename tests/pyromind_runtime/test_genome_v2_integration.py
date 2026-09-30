"""Real packaged plugin + Pi session + host RPC; deterministic model responses."""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import pytest
from harness_adapter.agentgenome_bridge import AgentGenomeBackend, ExecutionHost
from harness_adapter.pi_adapter.runner import PiRunnerProcess


@pytest.mark.asyncio
async def test_packaged_stage_and_judge_in_real_pi_session(tmp_path):
    workspace = tmp_path / "conversation"
    (workspace / "public_data").mkdir(parents=True)
    (workspace / "skills").mkdir()
    (workspace / "pi/terminal-output").mkdir(parents=True)
    source = tmp_path / "package"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "id": "report",
                "version": "1.0.0",
                "name": "report",
                "description": "test",
                "parameters": {
                    "data_file": {"type": "path"},
                    "delimiter": {"type": "string"},
                },
                "requires": ["agent_task", "model_judge"],
                "revision": {
                    "safe_to_rerun": True,
                    "editable_scripts": ["clean.py"],
                    "verification_resources": [
                        "judge.py",
                        "criterion.txt",
                        "verify.py",
                    ],
                },
            }
        )
    )
    (source / "graph.yaml").write_text("""version: gt/1.0
node:
  id: pipeline
  mode: sequence
  input:
    data_file: {type: path, from: params.data_file}
    delimiter: {type: text, from: params.delimiter}
  output:
    report: {type: json, compose: [report.output.report]}
  verify:
    - metric: "report.rows > 0"
  children:
    - id: clean
      input:
        data: {type: path, from: pipeline.input.data_file}
        delimiter: {type: text, from: pipeline.input.delimiter}
      do: {run: "python3 clean.py {input.data} {input.delimiter} > {artifact}"}
      output:
        cleaned: {type: path}
      verify:
        - run: "test -s {artifact}"
    - id: report
      input:
        cleaned: {type: path, from: clean.output.cleaned}
      do: {llm: "Write a JSON report to {artifact}, then submit the stage receipt."}
      output:
        report: {type: json}
      verify:
        - run: "python3 verify.py {artifact} {input.cleaned}"
        - run: "python3 judge.py criterion.txt {artifact}"
""")
    clean_source = """import csv, sys
from pathlib import Path
assert Path(sys.argv[1]).suffix == '.csv', 'unsupported filename'
with open(sys.argv[1]) as file:
    reader = csv.DictReader(file, delimiter=sys.argv[2])
    assert len(reader.fieldnames) == 2, 'check delimiter'
    writer = csv.DictWriter(sys.stdout, fieldnames=reader.fieldnames)
    writer.writeheader()
    seen = set()
    for row in reader:
        values = tuple(row.values())
        if all(values) and values not in seen:
            seen.add(values)
            writer.writerow(row)
"""
    (source / "clean.py").write_text(clean_source)
    (source / "verify.py").write_text(
        "import csv,json,sys\nfrom pathlib import Path\n"
        "report=json.loads(Path(sys.argv[1]).read_text())\n"
        "with open(sys.argv[2]) as f: rows=list(csv.DictReader(f))\n"
        "assert report['rows']==len(rows)\n"
    )
    # Use the installed Python distribution, not cross-repository source imports.
    from agentgenome import judge

    (source / "judge.py").write_bytes(Path(judge.__file__).read_bytes())
    (source / "criterion.txt").write_text("The report must explain its row count.")
    stage_request = {}
    current_round = None
    model_calls = []
    stage_calls = 0

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            nonlocal stage_calls
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            model_calls.append(body)
            judging = "Evaluate the evidence" in json.dumps(body.get("messages", []))
            tool = None
            if judging:
                text = json.dumps({"passed": True, "reason": "row count is explained"})
            else:
                stage_calls += 1
                if stage_calls % 3 == 1:
                    tool = (
                        "write",
                        {
                            "path": stage_request["expected_outputs"][0],
                            "content": json.dumps(
                                {
                                    "rows": len(
                                        list(
                                            csv.DictReader(
                                                Path(stage_request["inputs"]["cleaned"])
                                                .read_text()
                                                .splitlines()
                                            )
                                        )
                                    ),
                                    "summary": "Valid rows after cleaning",
                                }
                            ),
                        },
                    )
                elif stage_calls % 3 == 2:
                    tool = (
                        "genome_step_result",
                        {
                            "request_id": stage_request["request_id"],
                            "outcome": "completed",
                            "summary": "report written",
                        },
                    )
                text = "Stage submitted."
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            delta = {"role": "assistant"}
            if tool:
                delta["tool_calls"] = [
                    {
                        "index": 0,
                        "id": uuid4().hex,
                        "type": "function",
                        "function": {"name": tool[0], "arguments": json.dumps(tool[1])},
                    }
                ]
            else:
                delta["content"] = text
            for chunk, finish in [
                (delta, None),
                ({}, "tool_calls" if tool else "stop"),
            ]:
                payload = {
                    "id": "test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "test",
                    "choices": [{"index": 0, "delta": chunk, "finish_reason": finish}],
                }
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    sinks = {}

    async def on_request(method, arguments):
        if method == "workflow.invoke":
            assert arguments["action"] == "step_result"
            return await backend.invoke(
                "session",
                "step_result",
                {**arguments["arguments"], "execution_id": current_round},
                arguments["request_id"],
            )
        if method == "execution.output":
            if sink := sinks.get(arguments["id"]):
                await sink(arguments)
            return {"accepted": True}
        if method == "terminal.permission":
            return {"allow": True}
        raise AssertionError(method)

    async def on_event(event):
        nonlocal current_round
        if event.get("kind") == "agent.started":
            current_round = event["runId"]

    async def on_exit(_event):
        pass

    runner = PiRunnerProcess(
        request_handler=on_request, event_handler=on_event, exit_handler=on_exit
    )

    def subscribe(key, sink):
        if sink is None:
            sinks.pop(key, None)
        else:
            sinks[key] = sink

    backend = AgentGenomeBackend(
        tmp_path / "registry", lambda _: ExecutionHost(runner.request, subscribe)
    )

    async def emit(event):
        if event["event"] == "waiting_agent":
            stage_request.clear()
            stage_request.update(event["request"])
            return await runner.request("stage.execute", event["request"])
        return None

    try:
        await runner.start(
            {
                "session_id": "session",
                "workspace_root": str(workspace),
                "terminal_backend": "os-sandbox",
                "session_path": str(workspace / "pi/session.jsonl"),
                "skills_directory": str(workspace / "skills"),
                "skill_roots": [],
                "tools": [],
                "workflows_enabled": True,
                "model": {
                    "provider": "openai",
                    "id": "test",
                    "api_key": "unused",
                    "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                },
            }
        )
        backend.service.import_asset(source)
        for batch, expected_rows in [(0, 2), (1, 1)]:
            data = workspace / f"public_data/input-{batch}.csv"
            data.write_text(
                "name,age\nAda,36\nAda,36\nBob,\n" + ("Cy,12\n" if batch == 0 else "")
            )
            state = backend.service.submit(
                "session",
                f"batch-{batch}",
                "report",
                "1.0.0",
                {"data_file": str(data), "delimiter": ","},
                allow_draft=True,
            )
            result = await asyncio.wait_for(
                backend.execute("session", state["id"], emit), 30
            )
            assert result["status"] == "succeeded", result
            assert (
                json.loads(
                    Path(
                        result["result"]["outputs"]["report"]["execution_path"]
                    ).read_text()
                )["rows"]
                == expected_rows
            )
        backend.service.publish("report", "1.0.0")
        for case in ["parameter", "script", "takeover"]:
            data = workspace / f"public_data/{case}.csv"
            data.write_text("name;age\nAda;36\n")
            if case == "script":
                data = data.rename(data.with_suffix(".txt"))
            state = await backend.invoke(
                "session",
                "run",
                {
                    "asset_id": "report",
                    "version": "1.0.0",
                    "params": {"data_file": str(data), "delimiter": ","},
                },
                case,
            )
            failed = await asyncio.wait_for(
                backend.execute("session", state["id"], emit), 30
            )
            assert failed["status"] == "failed", failed
            if case == "takeover":
                taken = await backend.invoke(
                    "session",
                    "takeover",
                    {"run_id": state["id"], "reason": "method does not apply"},
                    "takeover",
                )
                assert taken["taken_over"] and taken["status"] == "failed"
                continue
            draft = await backend.invoke(
                "session", "prepare_revision", {"run_id": state["id"]}, f"draft-{case}"
            )
            with pytest.raises(Exception, match="PATH_SCOPE_ERROR"):
                await runner.request(
                    "execution.write",
                    {"path": f"{draft['verification_dir']}/verify.py", "content": ""},
                )
            if case == "script":
                fixed = clean_source.replace(
                    "assert Path(sys.argv[1]).suffix == '.csv', 'unsupported filename'",
                    "",
                )
                await runner.request(
                    "execution.write",
                    {
                        "path": f"{draft['editable_dir']}/clean.py",
                        "content": base64.b64encode(fixed.encode()).decode(),
                    },
                )
            child = await backend.invoke(
                "session",
                "run",
                {
                    "revision_id": draft["revision_id"],
                    "params": {"data_file": str(data), "delimiter": ";"},
                    "change_summary": case,
                    **(
                        {
                            "baseline_cases": [
                                {
                                    "id": "csv",
                                    "files": {"input.csv": "name,age\nAda,36\n"},
                                    "params": {
                                        "data_file": "fixture:input.csv",
                                        "delimiter": ",",
                                    },
                                    "assertions": [
                                        {
                                            "node": "pipeline",
                                            "output": "report",
                                            "kind": "json",
                                            "expected": {"rows": 1},
                                        }
                                    ],
                                }
                            ],
                            "regression_cases": [
                                {
                                    "id": "txt",
                                    "files": {"input.txt": "name;age\nAda;36\n"},
                                    "params": {
                                        "data_file": "fixture:input.txt",
                                        "delimiter": ";",
                                    },
                                    "assertions": [
                                        {
                                            "node": "pipeline",
                                            "output": "report",
                                            "kind": "json",
                                            "expected": {"rows": 1},
                                        }
                                    ],
                                }
                            ],
                        }
                        if case == "script"
                        else {}
                    ),
                },
                f"rerun-{case}",
            )
            result = await asyncio.wait_for(
                backend.execute("session", child["id"], emit), 30
            )
            assert result["status"] == "succeeded", result
            assert result["parent_run_id"] == state["id"]
        assert len(model_calls) == 28
        assert all(
            not call.get("tools")
            for call in model_calls
            if "Evaluate the evidence" in json.dumps(call["messages"])
        )
    finally:
        await runner.close()
        await backend.close()
        server.shutdown()
        server.server_close()
