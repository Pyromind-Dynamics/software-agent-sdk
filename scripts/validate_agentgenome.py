"""Validate a copied data-cleaning asset using the SDK's actual Pi OS sandbox.

No model request is sent. The Pi session loads the plugin and exposes its normal
execution environment; AgentGenome drives scripts through the SDK bridge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from harness_adapter.agentgenome_bridge import AgentGenomeBackend, ExecutionHost
from harness_adapter.pi_adapter.runner import PiRunnerProcess


async def validate(template: Path, home: Path, workspace: Path) -> None:
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
                {"data_file": name},
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
    args = parser.parse_args()
    asyncio.run(validate(args.template, args.home, args.workspace))


if __name__ == "__main__":
    main()
