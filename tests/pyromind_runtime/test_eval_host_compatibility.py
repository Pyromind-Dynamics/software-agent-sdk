"""Exercise the default remote execution backend with the real Pi host."""

from unittest.mock import AsyncMock

import httpx
import pytest
from harness_adapter.pi_adapter import PiAdapter
from pyromind_agent_server.api.router import create_product_router
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.application.conversation_runtime import ConversationRuntime

from .test_preparation import preparation_setup as preparation_setup
from .test_product_api import _app


@pytest.mark.parametrize("backend", ["sandbox", "os-sandbox"])
async def test_actual_manager_and_eval_host_create_through_api(
    tmp_path, monkeypatch, preparation_setup, backend
):
    store, _, _ = preparation_setup
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-without-model-request")
    adapter = PiAdapter(tmp_path / "conversations", terminal_backend=backend)
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        {"pi": adapter},
        default_harness_id="pi",
        preparation_store=store,
    )
    adapter.set_preparation_handler(
        runtime.preparation_call, runtime.preparation_revision
    )
    adapter.set_workflow_preparer(runtime.prepare_workflow_execution)
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

            for definition in ECHOMIND_PRODUCT_PROFILE.delegated_roles:
                assert set(definition.tools) <= business | native
            roles = {
                definition.name: definition
                for definition in ECHOMIND_PRODUCT_PROFILE.delegated_roles
            }
            assert set(roles) == {"init", "dataset", "train", "eval"}
            assert roles["init"].tools == roles["eval"].tools
            assert roles["init"].skills == roles["eval"].skills
            assert "df_submit_pipeline" in roles["dataset"].tools
            assert "df_submit_evaluation" not in roles["dataset"].tools
            assert {
                "analyze_task_failure",
                "training_analysis",
                "submit_workflow",
            } <= set(roles["train"].tools)
            assert "df_submit_evaluation" not in roles["train"].tools
            session = adapter._sessions[created.json()["conversation_id"]]
            assert created.json()["preparation"]["phase"] == "collecting"
            assert {"read", "write", "edit", "terminal"} <= set(
                session.config["tool_allowlist"]
            )
            assert not {"grep", "find", "ls", "subagent"}.intersection(
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
            assert set(session.config["tool_allowlist"]) == {
                "read",
                "write",
                "edit",
                "terminal",
                "subagent",
                "ask_user",
                "get_storage_url",
            }
            assert not session.config["skill_allowlist"]
            assert {item["name"] for item in adapter._tool_specs(session)} == {
                "get_storage_url",
            }

            execute = AsyncMock(
                return_value={
                    "is_error": False,
                    "content": [],
                    "details": {},
                    "signals": [],
                }
            )
            monkeypatch.setattr(adapter._business_tools, "execute", execute)
            from harness_adapter.pi_adapter.business_tool_host import (
                ToolExecutionContext,
            )

            monkeypatch.setattr(
                adapter,
                "_tool_context",
                AsyncMock(
                    return_value=ToolExecutionContext(
                        conversation_id=session.session_id,
                        workspace_root=session.workspace_root,
                        request_context=session.context,
                        model_configuration={},
                    )
                ),
            )
            for child, tool in (
                ("init", "df_submit_evaluation"),
                ("dataset", "df_submit_pipeline"),
                ("train", "training_analysis"),
                ("eval", "df_submit_evaluation"),
            ):
                await adapter._runner_request(
                    session,
                    "role.tool.execute",
                    {
                        "role": child,
                        "tool_name": tool,
                        "arguments": {},
                        "origin": {"type": "subagent", "subagent_id": child},
                        "tool_call_id": f"native:{child}:call",
                    },
                )
                assert execute.call_args.args[0] == tool
                assert (
                    execute.call_args.args[2].request_context.task_origin.subagent_id
                    == child
                )
            assert execute.await_count == 4
            with pytest.raises(PermissionError):
                await adapter._runner_request(
                    session,
                    "role.tool.execute",
                    {
                        "role": "dataset",
                        "tool_name": "df_submit_evaluation",
                        "arguments": {},
                        "origin": {"type": "subagent", "subagent_id": "dataset"},
                    },
                )
            with pytest.raises(PermissionError):
                await adapter._runner_request(
                    session,
                    "tool.execute",
                    {
                        "tool_name": "df_submit_evaluation",
                        "arguments": {},
                    },
                )
            assert execute.await_count == 4
    finally:
        await runtime.close()
