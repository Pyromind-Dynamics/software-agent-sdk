"""Exercise the default remote execution backend with the real Pi host."""

import httpx
import pytest
from harness_adapter.pi_adapter import PiAdapter
from pyromind_agent_server.api.router import create_product_router
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.application.conversation_runtime import ConversationRuntime

from . import test_standalone_experiments as storage_fixtures
from .test_product_api import _app


experiment = storage_fixtures.experiment


@pytest.mark.parametrize("backend", ["sandbox", "os-sandbox"])
async def test_actual_manager_and_eval_host_create_through_api(
    tmp_path, monkeypatch, experiment, backend
):
    service, _, _ = experiment
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-without-model-request")
    adapter = PiAdapter(tmp_path / "conversations", terminal_backend=backend)
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        {"pi": adapter},
        default_harness_id="pi",
        experiment_plan_store=service.plans,
        experiment_execution_store=service.store,
        execution_roles=("eval",),
    )
    adapter.set_plan_handler(runtime.experiment_plan_call)
    adapter.set_role_handler(runtime.execution_role_call)
    app = _app(tmp_path, runtime)
    app.include_router(create_product_router(ECHOMIND_PRODUCT_PROFILE))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"x-pyromind-debug-user-id": "42"},
        ) as client:
            created = await client.post(
                ECHOMIND_PRODUCT_PROFILE.route_prefix,
                json={
                    "llm": {
                        "model": "gpt-5",
                        "api_key": "test-key-without-model-request",
                    }
                },
            )
            assert created.status_code == 201, created.text
            assert created.json()["conversation_id"]
            role = ECHOMIND_PRODUCT_PROFILE.delegated_roles[0]
            business = {
                item["name"]
                for item in adapter._business_tools.specs(sandbox=backend == "sandbox")
            }
            native = {"read", "write", "edit", "terminal"}
            runtime_tools = {"record_experiment_result", "get_experiment_execution"}
            assert set(role.tools) <= business | native | runtime_tools
            assert "upload_file_to_pyromind" not in role.tools
            session = adapter._sessions[created.json()["conversation_id"]]
            delegate = next(
                item
                for item in adapter._tool_specs(session)
                if item["name"] == "request_experiment_task"
            )["input_schema"]
            assert "task" in delegate["required"]
            assert "task_id" not in delegate["properties"]
    finally:
        await runtime.close()
