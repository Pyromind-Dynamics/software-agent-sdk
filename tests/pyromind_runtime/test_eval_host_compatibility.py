"""Exercise the default remote execution backend with the real Pi host."""

from unittest.mock import AsyncMock

import httpx
import pytest
from harness_adapter.pi_adapter import PiAdapter
from pyromind_agent_server.api.router import create_product_router
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.infrastructure.sql_preparation import SQLPreparationStore

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
        preparation_store=SQLPreparationStore(service.plans.engine),
        experiment_execution_store=service.store,
        execution_roles=tuple(
            role.name for role in ECHOMIND_PRODUCT_PROFILE.delegated_roles or ()
        ),
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
            assert ECHOMIND_PRODUCT_PROFILE.delegated_roles is not None
            role = ECHOMIND_PRODUCT_PROFILE.delegated_roles[0]
            business = {
                item["name"]
                for item in adapter._business_tools.specs(sandbox=backend == "sandbox")
            }
            native = {"read", "write", "edit", "terminal"}
            runtime_tools = {"get_experiment_execution"}
            assert set(role.tools) <= business | native | runtime_tools
            assert "upload_file_to_pyromind" not in role.tools
            from pyromind_runtime.domain.experiment_execution import ROLE_TOOLS

            for definition in ECHOMIND_PRODUCT_PROFILE.delegated_roles:
                assert set(definition.tools) <= business | native | set(ROLE_TOOLS)
            roles = {
                definition.name: definition
                for definition in ECHOMIND_PRODUCT_PROFILE.delegated_roles
            }
            assert set(roles) == {"init", "dataset", "train", "eval"}
            assert roles["init"].tools == roles["eval"].tools
            assert roles["init"].skills == roles["eval"].skills
            assert "df_submit_pipeline" in roles["dataset"].tools
            assert "df_submit_evaluation" not in roles["dataset"].tools
            assert "submit_workflow" in roles["train"].tools
            assert "df_submit_evaluation" not in roles["train"].tools
            session = adapter._sessions[created.json()["conversation_id"]]
            assert created.json()["preparation"]["phase"] == "collecting"
            assert {"read", "write", "edit", "terminal", "grep", "find", "ls"} <= set(
                session.config["tool_allowlist"]
            )
            ensure = AsyncMock(return_value={"sandbox_id": "preparation-sandbox"})
            monkeypatch.setattr(adapter._sandbox, "ensure", ensure)
            assert await adapter._runner_request(session, "sandbox.ensure", {}) == {
                "sandbox_id": "preparation-sandbox"
            }
            ensure.assert_awaited_once()
            assert "request_experiment_task" not in {
                item["name"] for item in adapter._tool_specs(session)
            }
            from pyromind_runtime.domain.context import RequestContext

            await runtime._activate_development_policy(
                session.session_id,
                runtime._store(session.session_id),
                RequestContext(
                    user_id="42", profile_id=ECHOMIND_PRODUCT_PROFILE.profile_id
                ),
            )
            delegate = next(
                item
                for item in adapter._tool_specs(session)
                if item["name"] == "request_experiment_task"
            )["input_schema"]
            assert "task" in delegate["required"]
            assert "task_id" not in delegate["properties"]
            delegate_description = next(
                item["description"]
                for item in adapter._tool_specs(session)
                if item["name"] == "request_experiment_task"
            )
            assert "get_experiment_context" in delegate_description
            assert "valid_targets" in delegate_description
            assert "do not repeat the same invalid arguments" in delegate_description
    finally:
        await runtime.close()
