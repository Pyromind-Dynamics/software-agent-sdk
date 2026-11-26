"""Submit from a deployed SDK process and resume Main from a trusted callback."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pyromind_sdk.client.workflow import DslConverter

from harness_adapter.pi_adapter import PiAdapter
from harness_adapter.pi_adapter.business_tool_host import PyromindBusinessToolHost
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from harness_adapter.pi_adapter.platform_helper_bundle import prepare_host_sdk
from pyromind_agent_server.workflow_status_dispatcher import WorkflowStatusDispatcher
from pyromind_common.tools.utils import pyromind_api_client
from pyromind_common.tools.workflow import (
    task_submission,
    validate_workflow_dsl as validation_module,
)
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.commands import UserMessageCommand
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.task_correlation import TaskCorrelation
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.harness import SessionSpec

from .test_native_preparation import stream


SDK_KEY = "direct-submission-sdk-access-key"
REQUEST_TOKEN = "direct-submission-request-token"
CLUSTER = "cn-east-1#pre2"
MIGRATED_SUBMISSIONS = {
    "submit_workflow",
    "workflow_debug",
    "run_dataset_cleaning",
    "edp_render",
    "edp_aggregate",
    "df_submit_evaluation",
    "df_submit_pipeline",
    "edp_submit",
}


@pytest.mark.parametrize("profile", ["pyromind", "echomind"])
async def test_sdk_submission_adopts_early_callback_and_resumes_original_main(
    tmp_path: Path, monkeypatch, profile: str
) -> None:
    root = tmp_path / "conversations"
    conversation_id = f"sdk-submission-{profile}"
    adapter = PiAdapter(root, terminal_backend="os-sandbox")
    runtime = ConversationRuntime(root, adapter, default_harness_id="pi")
    adapter.set_sdk_context_provider(runtime.platform_sdk_context)
    dispatcher = WorkflowStatusDispatcher(runtime)
    store = FileProductStore(root / conversation_id)
    model_requests = []
    create_requests = []
    credential_requests = []
    correlations: list[TaskCorrelation] = []
    forbidden_gateway = AsyncMock(
        side_effect=AssertionError("SDK must submit directly")
    )
    monkeypatch.setattr(adapter._business_tools, "execute", forbidden_gateway)
    adapter.set_workflow_preparer(forbidden_gateway)
    script = "\n".join(
        (
            "import importlib.metadata, json, os",
            "from pyromind_sdk import PyroMindAPIClient",
            "from pyromind_common.platform.workflows import submit_workflow",
            "print('SDK-VERSION:' + importlib.metadata.version('pyromind-sdk'))",
            "print('HELPER-VERSION:' + "
            "importlib.metadata.version('pyromind-platform-helpers'))",
            "print('SDK-PROCESS:' + str(os.getpid()))",
            "client = PyroMindAPIClient(max_retries=0)",
            "response = submit_workflow(client, "
            "{'name': 'sdk-direct-e2e', 'nodes': [], 'edges': []}, "
            "out_id=json.loads(os.environ['PYROMIND_TASK_CONTEXTS'])['workflow_run'])",
            "print('SDK-TASK:' + response.task_id)",
            "print(os.environ['PYROMIND_API_KEY'])",
            "client.close()",
        )
    )

    async def model(request: web.Request) -> web.Response:
        model_requests.append(await request.json())
        return (
            stream(
                "terminal",
                {
                    "command": '"$PYROMIND_SDK_PYTHON" -c ' + shlex.quote(script),
                    "timeout": 60,
                },
                call="sdk-direct-submit",
            )
            if len(model_requests) == 1
            else stream()
        )

    async def create(request: web.Request) -> web.Response:
        payload = await request.json()
        create_requests.append(
            (
                request.headers.get("authorization"),
                request.headers.get("x-cluster"),
                request.headers.get("cookie"),
                payload,
            )
        )
        assert store.load_snapshot().external_tasks == ()
        address = TaskCorrelation.model_validate_json(payload["out_id"])
        assert address.signature is not None
        assert address.execution is None
        assert address.conversation_id == conversation_id
        assert address.task_kind == "workflow_run"
        assert address.type == "main"
        assert address.cluster == CLUSTER
        assert store.verify_task_correlation(address)
        correlations.append(address)
        delivered = await dispatcher.dispatch(
            task_id="native-sdk-task",
            status="Succeeded",
            correlation=address,
            trusted_platform=True,
        )
        assert delivered.outcome == "delivered_async"
        return web.json_response(
            {
                "success": True,
                "data": {
                    "task_id": "native-sdk-task",
                    "name": "sdk-direct-e2e",
                    "status": "Pending",
                },
            }
        )

    def access_key(**kwargs) -> str:
        credential_requests.append(kwargs)
        return SDK_KEY

    app = web.Application()
    app.router.add_post("/v1/chat/completions", model)
    app.router.add_post("/api/v1/studio/tasks", create)
    monkeypatch.setenv("LLM_API_KEY", "local-sdk-model-key")
    monkeypatch.setattr(task_submission, "get_api_key", access_key)
    context = RequestContext(
        user_id="42",
        profile_id=profile,
        cookie=f"auth_token={REQUEST_TOKEN}",
        x_cluster=CLUSTER,
    )
    try:
        async with TestServer(app) as server:
            monkeypatch.setattr(
                pyromind_api_client,
                "_api_url",
                lambda env: str(server.make_url("/api/v1")),
            )
            await runtime.create_conversation(
                SessionSpec(
                    conversation_id=conversation_id,
                    user_id="42",
                    profile_id=profile,
                    workspace_root=str(root / conversation_id),
                    tool_allowlist=(
                        ("read", "write", "edit", "terminal")
                        if profile == "echomind"
                        else None
                    ),
                    delegated_roles=(),
                    model_configuration={
                        "model": "sdk-local-model",
                        "base_url": str(server.make_url("/v1")),
                        "api_key": "local-sdk-model-key",
                    },
                ),
                context,
            )
            await runtime.submit_command(
                conversation_id,
                UserMessageCommand(
                    command_id="native-sdk-submit",
                    content=(TextContent(text="Submit the prepared workflow."),),
                ),
                context,
            )
            log_path = adapter._sessions[conversation_id].files.session_log_path
            async with asyncio.timeout(90):
                while True:
                    snapshot = await runtime.get_snapshot(conversation_id, context)
                    callback_finished = any(
                        (entry := json.loads(line)).get("customType")
                        == "pyromind.notification.finished"
                        and entry["data"]["runId"]
                        == "callback:native-sdk-task:succeeded"
                        for line in log_path.read_text().splitlines()
                    )
                    if (
                        len(model_requests) >= 3
                        and snapshot.status != "running"
                        and snapshot.external_tasks
                        and not snapshot.external_tasks[0].resume_pending
                        and callback_finished
                    ):
                        break
                    await asyncio.sleep(0.02)

            assert len(create_requests) == 1
            authorization, cluster, cookie, payload = create_requests[0]
            assert (authorization, cluster, cookie) == (
                f"Bearer {SDK_KEY}",
                CLUSTER,
                None,
            )
            assert payload["workflow"] == {
                "name": "sdk-direct-e2e",
                "nodes": [],
                "edges": [],
            }
            assert len(snapshot.external_tasks) == 1
            assert snapshot.external_tasks[0].task_id == "native-sdk-task"
            assert snapshot.external_tasks[0].status == "succeeded"
            assert snapshot.external_tasks[0].correlation == correlations[0]
            assert credential_requests
            assert all(
                call["auth_token"] == REQUEST_TOKEN
                and call["env"] == "pre2"
                and call["origin_headers"]["x-cluster"] == CLUSTER
                for call in credential_requests
            )
            results = json.dumps(
                [
                    message
                    for body in model_requests
                    for message in body["messages"]
                    if message["role"] == "tool"
                ]
            )
            assert "SDK-VERSION:0.1.17" in results, results
            assert "HELPER-VERSION:" in results, results
            assert "SDK-TASK:native-sdk-task" in results, results
            processes = re.findall(r"SDK-PROCESS:(\d+)", results)
            assert processes and all(int(pid) != os.getpid() for pid in processes)
            assert all(
                not (MIGRATED_SUBMISSIONS | {"execute_business_plan"})
                & {tool["function"]["name"] for tool in body["tools"]}
                for body in model_requests
            )
            forbidden_gateway.assert_not_awaited()

            before_duplicates = log_path.read_text()
            notifications = [
                entry
                for line in before_duplicates.splitlines()
                if (entry := json.loads(line)).get("customType")
                == "pyromind.notification"
            ]
            assert len(notifications) == 1
            assert notifications[0]["data"]["runId"] == (
                "callback:native-sdk-task:succeeded"
            )
            for _ in range(2):
                await dispatcher.dispatch(
                    task_id="native-sdk-task",
                    status="Succeeded",
                    correlation=correlations[0],
                    trusted_platform=True,
                )
            assert log_path.read_text() == before_duplicates
            assert len(create_requests) == 1
            completed = [
                event
                for event in store.replay()
                if event.type == "external_task.completed"
            ]
            assert len(completed) == 1
            serialized = json.dumps(model_requests) + "".join(
                event.model_dump_json() for event in store.replay()
            )
            assert SDK_KEY not in serialized
            assert REQUEST_TOKEN not in serialized
    finally:
        await runtime.close()

    for path in root.rglob("*"):
        if path.is_file():
            persisted = path.read_bytes()
            assert SDK_KEY.encode() not in persisted, str(path.relative_to(root))
            assert REQUEST_TOKEN.encode() not in persisted, str(path.relative_to(root))


async def test_sdk_debug_budget_survives_new_processes_and_legacy_rollback(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "conversations"
    conversation_id = "sdk-debug-budget"
    workspace = root / conversation_id
    adapter = PiAdapter(root, terminal_backend="os-sandbox")
    runtime = ConversationRuntime(root, adapter, default_harness_id="pi")
    adapter.set_sdk_context_provider(runtime.platform_sdk_context)
    adapter.set_workflow_preparer(AsyncMock())
    model_requests = []
    create_requests = []
    script = "\n".join(
        (
            "import json, os",
            "from pathlib import Path",
            "from pyromind_sdk import PyroMindAPIClient",
            "from pyromind_common.platform.workflows import "
            "DebugBudget, submit_debug_workflow",
            "budget = DebugBudget.load(Path(os.environ['PYROMIND_DEBUG_STATE_PATH']))",
            "print('DEBUG-PROCESS:' + str(os.getpid()), flush=True)",
            "print('DEBUG-BEFORE:' + str(budget.accepted), flush=True)",
            "client = PyroMindAPIClient(max_retries=0)",
            "response = submit_debug_workflow(client, "
            "{'name': 'sdk-debug-e2e', 'nodes': [], 'edges': [], "
            "'execution_argos': [{'execution_mode': 'test'}]}, "
            "out_id=json.loads(os.environ['PYROMIND_TASK_CONTEXTS'])"
            "['workflow_debug'], "
            "budget=budget, validated=True)",
            "print('DEBUG-AFTER:' + str(budget.accepted))",
            "print('DEBUG-TASK:' + response.task_id)",
            "client.close()",
        )
    )
    terminal: tuple[str, dict[str, object]] = (
        "terminal",
        {
            "command": '"$PYROMIND_SDK_PYTHON" -c ' + shlex.quote(script),
            "timeout": 60,
        },
    )
    actions: list[tuple[str, dict[str, object]]] = [terminal, terminal]

    async def model(request: web.Request) -> web.Response:
        model_requests.append(await request.json())
        return (
            stream(*actions.pop(0), call=f"debug-call-{len(model_requests)}")
            if actions
            else stream()
        )

    async def create(request: web.Request) -> web.Response:
        payload = await request.json()
        create_requests.append(payload)
        assert request.headers.get("authorization") == f"Bearer {SDK_KEY}"
        assert request.headers.get("x-cluster") == CLUSTER
        assert request.headers.get("cookie") is None
        assert payload["workflow"]["execution_argos"] == [{"execution_mode": "test"}]
        address = TaskCorrelation.model_validate_json(payload["out_id"])
        assert address.task_kind == "workflow_debug"
        assert FileProductStore(workspace).verify_task_correlation(address)
        return web.json_response(
            {
                "success": True,
                "data": {
                    "task_id": "sdk-debug-task",
                    "name": "sdk-debug-e2e",
                    "status": "Pending",
                },
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", model)
    app.router.add_post("/api/v1/studio/tasks", create)
    monkeypatch.setenv("LLM_API_KEY", "local-sdk-model-key")
    monkeypatch.setattr(task_submission, "get_api_key", lambda **kwargs: SDK_KEY)
    context = RequestContext(
        user_id="42", cookie=f"auth_token={REQUEST_TOKEN}", x_cluster=CLUSTER
    )
    try:
        async with TestServer(app) as server:
            monkeypatch.setattr(
                pyromind_api_client,
                "_api_url",
                lambda env: str(server.make_url("/api/v1")),
            )
            await runtime.create_conversation(
                SessionSpec(
                    conversation_id=conversation_id,
                    user_id="42",
                    workspace_root=str(workspace),
                    delegated_roles=(),
                    model_configuration={
                        "model": "sdk-local-model",
                        "base_url": str(server.make_url("/v1")),
                        "api_key": "local-sdk-model-key",
                    },
                ),
                context,
            )
            files = PiSessionFiles(workspace)
            files.save_business_state({"pyromind_workflow_attempts": 9})
            workflow_path = workspace / "public_data/workflow_canvas/workflow.py"
            workflow_path.parent.mkdir(parents=True, exist_ok=True)
            workflow_path.write_text("workflow = InputNode()\n")
            await runtime.submit_command(
                conversation_id,
                UserMessageCommand(
                    command_id="debug-to-limit",
                    content=(TextContent(text="Run the authorized debug round."),),
                ),
                context,
            )
            async with asyncio.timeout(90):
                while True:
                    snapshot = await runtime.get_snapshot(conversation_id, context)
                    if len(model_requests) >= 3 and snapshot.status != "running":
                        break
                    await asyncio.sleep(0.02)
            results = json.dumps(model_requests[-1]["messages"])
            assert "DEBUG-BEFORE:9" in results, results
            assert "DEBUG-AFTER:10" in results, results
            assert "DEBUG-BEFORE:10" in results, results
            assert "Debug submission limit reached" in results, results
            processes = set(re.findall(r"DEBUG-PROCESS:(\d+)", results))
            assert len(processes) == 2 and str(os.getpid()) not in processes
            assert len(create_requests) == 1
            for body in model_requests:
                assert MIGRATED_SUBMISSIONS.isdisjoint(
                    tool["function"]["name"] for tool in body["tools"]
                )
            await runtime.close()

            adapter = PiAdapter(
                root, terminal_backend="os-sandbox", platform_sdk_enabled=False
            )
            adapter.set_workflow_preparer(AsyncMock())
            runtime = ConversationRuntime(root, adapter, default_harness_id="pi")
            actions.append(("workflow_debug", {}))
            await runtime.submit_command(
                conversation_id,
                UserMessageCommand(
                    command_id="legacy-after-rollback",
                    content=(TextContent(text="Check the current debug round."),),
                ),
                context,
            )
            async with asyncio.timeout(40):
                while True:
                    snapshot = await runtime.get_snapshot(conversation_id, context)
                    if len(model_requests) >= 5 and snapshot.status != "running":
                        break
                    await asyncio.sleep(0.02)
            assert MIGRATED_SUBMISSIONS <= {
                tool["function"]["name"] for tool in model_requests[-1]["tools"]
            }
            legacy_result = json.dumps(
                [
                    message
                    for message in model_requests[-1]["messages"]
                    if message["role"] == "tool"
                ]
            )
            assert "maximum of 10 accepted submissions" in legacy_result, legacy_result
            assert len(create_requests) == 1
            assert files.load_business_state() == {"pyromind_workflow_attempts": 10}
    finally:
        await runtime.close()

    path = PyromindBusinessToolHost.prepare_debug_budget(workspace)
    interpreter = await asyncio.to_thread(prepare_host_sdk)
    reset = await asyncio.create_subprocess_exec(
        interpreter,
        "-c",
        "import sys; from pathlib import Path; "
        "from pyromind_common.platform.workflows import DebugBudget; "
        "budget = DebugBudget.load(Path(sys.argv[1])); "
        "budget.accepted = 0; budget.save()",
        str(path),
        env={"PATH": os.environ.get("PATH", "")},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await reset.communicate()
    assert reset.returncode == 0, (stdout, stderr)
    PyromindBusinessToolHost.prepare_debug_budget(workspace)
    assert files.load_business_state() == {"pyromind_workflow_attempts": 0}
    for persisted_path in root.rglob("*"):
        if persisted_path.is_file():
            persisted = persisted_path.read_bytes()
            assert SDK_KEY.encode() not in persisted
            assert REQUEST_TOKEN.encode() not in persisted


async def test_staged_evaluation_validates_with_host_identity_then_submits_sdk_graph(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "conversations"
    conversation_id = "sdk-evaluation"
    workspace = root / conversation_id
    scripts = Path(__file__).resolve().parents[2] / (
        ".agents/skills/inference-evaluation/scripts"
    )
    adapter = PiAdapter(root, terminal_backend="os-sandbox")
    runtime = ConversationRuntime(root, adapter, default_harness_id="pi")
    adapter.set_sdk_context_provider(runtime.platform_sdk_context)
    legacy_submit = AsyncMock(side_effect=AssertionError("SDK must submit directly"))
    adapter.set_workflow_preparer(legacy_submit)
    dispatcher = WorkflowStatusDispatcher(runtime)
    store = FileProductStore(workspace)
    model_requests = []
    validation_requests = []
    validated_graphs = []
    resource_graphs = []
    create_requests = []
    stage_script = "\n".join(
        (
            "import json, os",
            "from pathlib import Path",
            "from uuid import UUID",
            "from pyromind_common.platform.evaluation import stage_evaluation",
            "from pyromind_common.platform.evaluation_core import EvaluationOptions",
            "from pyromind_common.platform.preparation import LocalFiles, "
            "StagingDirectory",
            "assert not any('COOKIE' in key.upper() for key in os.environ)",
            "inputs = Path('public_data/evaluation-inputs')",
            "inputs.mkdir(parents=True, exist_ok=True)",
            "(inputs / 'manifest.json').write_text(json.dumps({"
            "'datasets': {'train': {'path': '/train', 'sha256': 'a' * 64}, "
            "'validation': {'path': '/val', 'sha256': 'b' * 64}}, "
            "'source': {'kind': 'import', 'path': '/badcases', 'sha256': 'c' * 64}, "
            "'profile': {'path': '/profile', 'sha256': 'd' * 64}}))",
            "staged = stage_evaluation(EvaluationOptions(operation='report', "
            "input_path='/workspace/badcases', "
            "manifest_path='public_data/evaluation-inputs/manifest.json'), "
            "files=LocalFiles(Path.cwd()), "
            "staging=StagingDirectory(Path('public_data/evaluation'), "
            "'/workspace/jobs/native-evaluation'), "
            f"runtime_dir=Path({str(scripts)!r}), "
            "run_id=UUID('11111111-1111-1111-1111-111111111111'))",
            "assert staged.workflow_path.is_file()",
            "assert staged.frozen_files and "
            "all(p.is_file() for p in staged.frozen_files)",
            "assert not staged.requires_inference",
            "print('EVAL-STAGED:' + str(staged.workflow_path))",
        )
    )

    def terminal(script: str, call: str) -> web.Response:
        return stream(
            "terminal",
            {
                "command": '"$PYROMIND_SDK_PYTHON" -c ' + shlex.quote(script),
                "timeout": 60,
            },
            call=call,
        )

    async def model(request: web.Request) -> web.Response:
        body = await request.json()
        model_requests.append(body)
        if len(model_requests) == 1:
            return terminal(stage_script, "eval-stage")
        if len(model_requests) == 2:
            return stream(
                "validate_workflow_dsl",
                {
                    "dsl_path": "public_data/evaluation/workflow.py",
                    "name": "native-evaluation",
                },
                call="eval-validate",
            )
        if len(model_requests) == 3:
            tool_result = next(
                message["content"]
                for message in body["messages"]
                if message.get("tool_call_id") == "eval-validate"
            )
            document = json.loads(
                tool_result[tool_result.index('{"validated_workflow_path"') :]
            )
            path = document["validated_workflow_path"]
            assert path.startswith("public_data/"), path
            return terminal(
                "\n".join(
                    (
                        "import json, os",
                        "from pathlib import Path",
                        "from pyromind_sdk import PyroMindAPIClient",
                        "from pyromind_common.platform.workflows "
                        "import submit_workflow",
                        "assert not any('COOKIE' in key.upper() for key in os.environ)",
                        f"workflow = json.loads(Path({path!r}).read_text())",
                        "client = PyroMindAPIClient(max_retries=0)",
                        "result = submit_workflow(client, workflow, "
                        "out_id=json.loads(os.environ['PYROMIND_TASK_CONTEXTS'])"
                        "['evaluation'])",
                        "print('EVAL-SDK-TASK:' + result.task_id)",
                        "client.close()",
                    )
                ),
                "eval-sdk-submit",
            )
        return stream()

    async def validate(request: web.Request) -> web.Response:
        payload = await request.json()
        validation_requests.append(
            (request.headers.get("cookie"), request.headers.get("x-cluster"), payload)
        )
        graph = DslConverter().from_python(payload["dsl"], name=payload["name"])
        worker = next(
            node
            for node in graph["nodes"]
            if node["data"]["nodeType"] == "CustomCommandCPUNode"
        )
        assert worker["data"]["config"]["cpu"] == 4
        worker["data"]["config"]["cpu"] = 12
        validated_graphs.append(graph)
        return web.json_response(
            {"success": True, "data": {"valid": True, "workflow": graph}}
        )

    async def resources(request: web.Request) -> web.Response:
        assert request.headers.get("cookie") == f"auth_token={REQUEST_TOKEN}"
        assert request.headers.get("x-cluster") == CLUSTER
        resource_graphs.append((await request.json())["workflow"])
        return web.json_response({"success": True, "data": {"valid": True}})

    async def create(request: web.Request) -> web.Response:
        payload = await request.json()
        create_requests.append(payload)
        assert request.headers.get("authorization") == f"Bearer {SDK_KEY}"
        assert request.headers.get("cookie") is None
        assert request.headers.get("x-cluster") == CLUSTER
        assert store.load_snapshot().external_tasks == ()
        assert payload["workflow"] == validated_graphs[0] == resource_graphs[0]
        address = TaskCorrelation.model_validate_json(payload["out_id"])
        assert address.task_kind == "evaluation"
        assert store.verify_task_correlation(address)
        await dispatcher.dispatch(
            task_id="native-evaluation-task",
            status="Succeeded",
            correlation=address,
            trusted_platform=True,
        )
        return web.json_response(
            {
                "success": True,
                "data": {
                    "task_id": "native-evaluation-task",
                    "name": "native-evaluation",
                    "status": "Pending",
                },
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", model)
    app.router.add_post("/workflows/dsl/validate", validate)
    app.router.add_post("/workflows/resources/validate", resources)
    app.router.add_post("/api/v1/studio/tasks", create)
    monkeypatch.setenv("LLM_API_KEY", "local-sdk-model-key")
    monkeypatch.setattr(task_submission, "get_api_key", lambda **kwargs: SDK_KEY)
    context = RequestContext(
        user_id="42", cookie=f"auth_token={REQUEST_TOKEN}", x_cluster=CLUSTER
    )
    try:
        async with TestServer(app) as server:
            monkeypatch.setattr(
                pyromind_api_client,
                "_api_url",
                lambda env: str(server.make_url("/api/v1")),
            )
            monkeypatch.setattr(
                validation_module,
                "_default_validate_url",
                lambda env=None: str(server.make_url("/workflows/dsl/validate")),
            )
            await runtime.create_conversation(
                SessionSpec(
                    conversation_id=conversation_id,
                    user_id="42",
                    workspace_root=str(workspace),
                    delegated_roles=(),
                    model_configuration={
                        "model": "sdk-local-model",
                        "base_url": str(server.make_url("/v1")),
                        "api_key": "local-sdk-model-key",
                    },
                ),
                context,
            )
            await runtime.submit_command(
                conversation_id,
                UserMessageCommand(
                    command_id="stage-and-submit-evaluation",
                    content=(
                        TextContent(text="Prepare and run the evaluation report."),
                    ),
                ),
                context,
            )
            log_path = adapter._sessions[conversation_id].files.session_log_path
            async with asyncio.timeout(90):
                while True:
                    snapshot = await runtime.get_snapshot(conversation_id, context)
                    if (
                        len(model_requests) >= 4
                        and snapshot.status != "running"
                        and not snapshot.external_tasks
                    ):
                        break
                    if (
                        len(model_requests) >= 5
                        and snapshot.status != "running"
                        and '"customType":"pyromind.notification.finished"'
                        in log_path.read_text()
                    ):
                        break
                    await asyncio.sleep(0.02)
            assert len(validation_requests) == len(resource_graphs) == 1
            cookie, cluster, payload = validation_requests[0]
            assert (cookie, cluster) == (f"auth_token={REQUEST_TOKEN}", CLUSTER)
            assert (
                payload["dsl"]
                == (workspace / "public_data/evaluation/workflow.py").read_text()
            )
            assert len(create_requests) == 1
            assert snapshot.external_tasks[0].task_id == "native-evaluation-task"
            assert snapshot.external_tasks[0].status == "succeeded"
            legacy_submit.assert_not_awaited()
            assert "EVAL-SDK-TASK:native-evaluation-task" in json.dumps(model_requests)
            assert all(
                MIGRATED_SUBMISSIONS.isdisjoint(
                    tool["function"]["name"] for tool in body["tools"]
                )
                for body in model_requests
            )
            assert SDK_KEY not in json.dumps(model_requests)
            assert REQUEST_TOKEN not in json.dumps(model_requests)
    finally:
        await runtime.close()
    for path in root.rglob("*"):
        if path.is_file():
            persisted = path.read_bytes()
            assert SDK_KEY.encode() not in persisted, str(path.relative_to(root))
            assert REQUEST_TOKEN.encode() not in persisted, str(path.relative_to(root))
