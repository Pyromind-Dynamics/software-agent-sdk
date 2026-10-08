"""Validate a copied data-cleaning asset using the SDK's actual Pi OS sandbox.

No model request is sent. The Pi session loads the plugin and exposes its normal
execution environment; AgentGenome drives scripts through the SDK bridge.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
from uuid import uuid4

from harness_adapter.agentgenome_bridge import AgentGenomeBackend, ExecutionHost
from harness_adapter.pi_adapter.runner import PiRunnerProcess


async def validate(
    template: Path, home: Path, workspace: Path, test_revisions: bool = False
) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    workspace = workspace.resolve()
    for name in ("public_data", "pi/terminal-output", "skills"):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    sinks = {}

    async def request(method, params):
        if method == "execution.output":
            await sinks[params["id"]](params)
            return {}
        raise ValueError(f"Unexpected runner request: {method}")

    async def ignore(_event):
        pass

    def subscribe(execution_id, sink):
        if sink is None:
            sinks.pop(execution_id, None)
        else:
            sinks[execution_id] = sink

    runner = PiRunnerProcess(
        request_handler=request, event_handler=ignore, exit_handler=ignore
    )
    backend = AgentGenomeBackend(
        home, lambda _scope: ExecutionHost(runner.request, subscribe)
    )
    try:
        await runner.start(
            {
                "session_id": "genome-validation",
                "workspace_root": str(workspace),
                "terminal_backend": "os-sandbox",
                "session_path": str(workspace / "pi/session.jsonl"),
                "skills_directory": str(workspace / "skills"),
                "skill_roots": [],
                "tools": [],
                "workflows_enabled": True,
                "model": {
                    "provider": "validation",
                    "id": "no-model-calls",
                    "api_key": "unused",
                    "base_url": "http://127.0.0.1:1/v1",
                    "context_window": 32000,
                },
            }
        )
        asset = backend.service.import_asset(template)
        scope = f"sdk-validation-{uuid4().hex}"
        delimiter_params = (
            {"delimiter": ","} if "delimiter" in asset["parameters"] else {}
        )
        for index, (csv, expected_rows) in enumerate(
            [
                ("name,age\nAda,36\nAda,36\nBob,\nCy,12\n", 2),
                ("name,age\nX,1\nX,1\n", 1),
            ]
        ):
            name = f"public_data/batch-{index}.csv"
            (workspace / name).write_text(csv)
            state = backend.service.submit(
                scope,
                str(index),
                asset["id"],
                asset["version"],
                {"data_file": name, **delimiter_params},
                allow_draft=True,
            )
            result = await backend.execute(scope, state["id"], ignore)
            if result["status"] != "succeeded":
                raise RuntimeError(json.dumps(result, ensure_ascii=False))
            details = result["result"]
            assert isinstance(details, dict)
            outputs = details["outputs"]
            assert isinstance(outputs, dict)
            report = outputs["report"]
            assert isinstance(report, dict)
            execution_path = report["execution_path"]
            assert isinstance(execution_path, str)
            assert json.loads(Path(execution_path).read_text())["rows"] == expected_rows
            print(
                json.dumps(
                    {
                        "batch": index,
                        "run_id": state["id"],
                        "status": result["status"],
                        "report": report["execution_path"],
                    },
                    ensure_ascii=False,
                )
            )
        if test_revisions:
            if asset["id"] != "data-cleaning":
                raise ValueError(
                    "revision validation requires the CSV cleaning template"
                )
            name = "public_data/semicolon.csv"
            (workspace / name).write_text("name;age\nAda;36\nAda;36\nBob;\nCy;12\n")
            backend.service.publish(asset["id"], asset["version"])
            for fix in ("parameter", "script") if delimiter_params else ("script",):
                original = backend.service.submit(
                    scope,
                    f"wrong-{fix}",
                    asset["id"],
                    asset["version"],
                    {"data_file": name, **delimiter_params},
                    allow_draft=True,
                )
                wrong = await backend.execute(scope, original["id"], ignore)
                assert wrong["status"] == "succeeded", wrong
                report = json.loads(
                    Path(
                        wrong["result"]["outputs"]["report"]["execution_path"]
                    ).read_text()
                )
                assert report["rows"] == 3 and report["columns"] == ["name;age"]
                draft = await backend.invoke(
                    scope,
                    "prepare_revision",
                    {
                        "run_id": original["id"],
                        "reason": "semicolon input was parsed as a single column",
                        **(
                            {
                                "editable_scripts": ["scripts/clean.py"],
                                "verification_resources": ["scripts/verify_report.py"],
                            }
                            if "revision" not in asset
                            else {}
                        ),
                    },
                    f"draft-{fix}",
                )
                params = {
                    "data_file": name,
                    **(
                        {"delimiter": ";" if fix == "parameter" else ","}
                        if delimiter_params
                        else {}
                    ),
                }
                if fix == "script":
                    script = (template / "scripts/clean.py").read_text()
                    script = script.replace(
                        "rows = list(csv.reader(f, delimiter=sys.argv[2]))",
                        "rows = list(csv.reader(f))",
                    )
                    script = script.replace(
                        "rows = list(csv.reader(f))",
                        "sample=f.read(4096); f.seek(0)\n"
                        "        rows = list(csv.reader(f, delimiter="
                        "csv.Sniffer().sniff(sample, delimiters=',;').delimiter))",
                    )
                    await runner.request(
                        "execution.write",
                        {
                            "path": f"{draft['editable_dir']}/scripts/clean.py",
                            "content": base64.b64encode(script.encode()).decode(),
                        },
                    )
                request = {
                    "revision_id": draft["revision_id"],
                    "params": params,
                    "change_summary": fix,
                }
                if fix == "script":

                    def case(case_id, sep):
                        return {
                            "id": case_id,
                            "files": {
                                "input.csv": (
                                    f"name{sep}age\nAda{sep}36\nAda{sep}36\nBob{sep}\n"
                                )
                            },
                            "params": {
                                "data_file": "fixture:input.csv",
                                **delimiter_params,
                            },
                            "assertions": [
                                {
                                    "node": "clean-sop",
                                    "output": "report",
                                    "kind": "json",
                                    "expected": {"rows": 1, "columns": ["name", "age"]},
                                },
                                {
                                    "node": "clean-sop.clean",
                                    "output": "cleaned",
                                    "kind": "csv",
                                    "expected": [["name", "age"], ["Ada", "36"]],
                                },
                            ],
                        }

                    request.update(
                        baseline_cases=[case("comma", ",")],
                        regression_cases=[case("semicolon", ";")],
                    )
                child = await backend.invoke(scope, "run", request, f"rerun-{fix}")
                assert (await backend.invoke(scope, "run", request, f"rerun-{fix}"))[
                    "id"
                ] == child["id"]
                revised = await backend.execute(scope, child["id"], ignore)
                assert revised["status"] == "succeeded", revised
                report = json.loads(
                    Path(
                        revised["result"]["outputs"]["report"]["execution_path"]
                    ).read_text()
                )
                assert report["rows"] == 2 and report["columns"] == ["name", "age"]
                if fix == "script":
                    publication = revised["result"]["publication"]
                    assert publication["status"] == "published", publication
                    assert (
                        backend.service.list()[0]["version"] == publication["version"]
                    )
                print(
                    json.dumps(
                        {
                            "revision": fix,
                            "run_id": child["id"],
                            "status": revised["status"],
                        }
                    )
                )
        backend.service.publish(asset["id"], asset["version"])
        print(json.dumps({"published": backend.service.list()}, ensure_ascii=False))
    finally:
        await backend.close()
        await runner.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--home", type=Path, default=Path("workspace/agentgenome"))
    parser.add_argument(
        "--workspace", type=Path, default=Path("workspace/genome-validation")
    )
    parser.add_argument("--test-revisions", action="store_true")
    args = parser.parse_args()
    asyncio.run(validate(args.template, args.home, args.workspace, args.test_revisions))


if __name__ == "__main__":
    main()
