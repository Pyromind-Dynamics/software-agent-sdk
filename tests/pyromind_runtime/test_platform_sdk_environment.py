import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from pyromind_sdk import PyroMindAPIClient

import harness_adapter.pi_adapter.business_tool_host as host_module
from harness_adapter.pi_adapter.adapter import PI_CAPABILITIES, PiAdapter, _PiSession
from harness_adapter.pi_adapter.business_tool_host import (
    PLATFORM_SDK_TOOL_NAMES,
    ToolExecutionContext,
)
from harness_adapter.pi_adapter.fixed_actor import FixedActorExecutionManager
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from harness_adapter.pi_adapter.sdk_environment import (
    SDK_ATOMIC_TOOL_NAMES,
    SDK_REMOTE_STORAGE_TOOL_NAMES,
)
from pyromind_agent_server.agent_profiles.pyromind import PYROMIND_TOOL_ALLOWLIST
from pyromind_common.tools.data_preparation.platform_submit import (
    TASK_ASSOCIATION_DIRNAME,
    DataPreparationTaskAssociation,
    DataPreparationTaskStore,
)
from pyromind_common.tools.sandbox.definition import default_sandbox_image
from pyromind_runtime.application.pipeline_runs import PipelineRuns
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.snapshot import ConversationSnapshot, ExternalTaskState
from pyromind_runtime.domain.task_correlation import TaskOrigin


_LEGACY_SUBMISSION_TOOLS = frozenset(
    {
        "df_submit_pipeline",
        "df_submit_evaluation",
        "submit_workflow",
        "workflow_debug",
        "run_dataset_cleaning",
        "edp_render",
        "edp_submit",
        "edp_aggregate",
    }
)
_MIGRATED_SUBMISSION_TOOLS = frozenset(
    {
        "df_submit_pipeline",
        "submit_workflow",
        "workflow_debug",
        "run_dataset_cleaning",
        "edp_render",
        "edp_submit",
        "edp_aggregate",
        "df_submit_evaluation",
    }
)


@pytest.fixture
def sdk_session(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.adapter.prepare_host_sdk",
        lambda: "/prepared/sdk/bin/python",
    )
    adapter = PiAdapter(tmp_path, terminal_backend="os-sandbox")
    files = PiSessionFiles(tmp_path)
    config = {"profile_id": "pyromind", "tool_allowlist": None}
    files.initialize(config)
    session = _PiSession(
        "sdk-conversation",
        tmp_path,
        files,
        config,
        RequestContext(
            user_id="42",
            profile_id="pyromind",
            cookie="auth_token=request-token",
            x_cluster="cn-east-1#pre2",
        ),
    )
    return adapter, session, adapter._business_tools


@pytest.fixture
def historical_sdk_session(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "local-model-key")

    async def attach(task, *, profile="pyromind", allowlist=None):
        conversation_id = "historical-sdk-session"
        workspace = tmp_path / conversation_id
        files = PiSessionFiles(workspace)
        files.initialize(
            {
                "profile_id": profile,
                "tool_allowlist": allowlist,
                "delegated_roles": [],
                "model": {"id": "local-model"},
            }
        )
        snapshot = ConversationSnapshot(
            conversation_id=conversation_id,
            capabilities=PI_CAPABILITIES,
            external_tasks=(ExternalTaskState.model_validate(task),) if task else (),
        )
        product = workspace / "product"
        product.mkdir()
        (product / "snapshot.json").write_text(snapshot.model_dump_json())
        context = RequestContext(
            user_id="42", profile_id=profile, cookie="auth_token=current-request-token"
        )
        adapter = PiAdapter(tmp_path, terminal_backend="os-sandbox")
        adapter.set_sdk_context_provider(AsyncMock())
        adapter.set_workflow_preparer(AsyncMock())
        monkeypatch.setattr(adapter, "_start_runner", AsyncMock())

        await adapter.attach_session(conversation_id, context)

        return adapter, adapter._sessions[conversation_id]

    return attach


@pytest.mark.parametrize(
    ("route", "env", "cluster"),
    [
        (None, "prod", "us-west-1"),
        ("cn-east-1#pre2", "pre2", "cn-east-1"),
        ("us-west-2#pre", "pre", "us-west-2"),
    ],
)
async def test_sdk_environment_uses_request_identity_and_explicit_route(
    tmp_path, monkeypatch, sdk_session, route, env, cluster
):
    _, session, host = sdk_session
    prefix = "" if env == "prod" else f"{env}-"
    base_url = f"https://{prefix}api-portal.pyromind.ai/api/v1"
    calls = []

    def client_factory(**params: Any) -> PyroMindAPIClient:
        calls.append(params)
        return PyroMindAPIClient(
            api_key="derived-sdk-key",
            base_url=base_url,
            cluster=route or "us-west-1",
        )

    monkeypatch.setattr(host_module, "create_workflow_api_client", client_factory)
    monkeypatch.setenv("PYROMIND_API_KEY", "unrelated-host-account")
    context = ToolExecutionContext(
        conversation_id=session.session_id,
        workspace_root=tmp_path,
        request_context=replace(session.context, x_cluster=route),
        model_configuration={},
    )

    environment = await host.sdk_environment(context)

    assert environment == {
        "PYROMIND_API_KEY": "derived-sdk-key",
        "PYROMIND_BASE_URL": base_url,
        "PYROMIND_CLUSTER": route or "us-west-1",
        "PYROMIND_SANDBOX_DEFAULT_IMAGE": default_sandbox_image(route),
        "PYROMIND_TASK_ASSOCIATIONS": "[]",
    }
    assert calls == [
        {
            "env": env,
            "cluster": cluster,
            "auth_token": "request-token",
            "headers": {
                "cookie": "auth_token=request-token",
                "x-cluster": route or "us-west-1",
            },
        }
    ]


async def test_sdk_environment_exports_configured_image_and_current_task_snapshot(
    monkeypatch, sdk_session
):
    adapter, session, host = sdk_session
    session.extra["sandbox_default_image"] = " registry.local/custom-sandbox:1 "
    records = [{"task_id": "123", "run_id": "run-1", "output_dir": "/workspace/run-1"}]
    runs = Mock(spec=PipelineRuns)
    runs.task_associations.return_value = records
    provider = Mock(return_value=runs)
    monkeypatch.setattr(host, "_pipeline_runs", provider)
    monkeypatch.setattr(
        host_module,
        "create_workflow_api_client",
        lambda **params: PyroMindAPIClient(api_key="derived-sdk-key"),
    )

    environment = await host.sdk_environment(adapter._control_tool_context(session))

    assert environment["PYROMIND_SANDBOX_DEFAULT_IMAGE"] == (
        "registry.local/custom-sandbox:1"
    )
    assert json.loads(environment["PYROMIND_TASK_ASSOCIATIONS"]) == records
    provider.assert_called_once_with(session.session_id)
    runs.task_associations.assert_called_once_with()


@pytest.mark.parametrize("configured", [True, False])
async def test_sdk_environment_exports_only_configured_session_model_credentials(
    monkeypatch, sdk_session, configured
):
    adapter, session, host = sdk_session
    for name in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        host_module,
        "create_workflow_api_client",
        lambda **params: PyroMindAPIClient(api_key="derived-sdk-key"),
    )
    model = (
        {
            "base_url": " https://model.example/v1/ ",
            "api_key": "model-access-key",
            "model": "session-model",
        }
        if configured
        else {}
    )
    context = replace(adapter._control_tool_context(session), model_configuration=model)

    environment = await host.sdk_environment(context)

    expected = (
        {
            "LLM_BASE_URL": "https://model.example",
            "LLM_AUTH_TOKEN": "model-access-key",
            "LLM_MODEL": "session-model",
        }
        if configured
        else {}
    )
    actual = {
        name: value for name, value in environment.items() if name.startswith("LLM_")
    }
    assert actual == expected
    assert "request-token" not in repr(environment)


async def test_sdk_snapshot_preserves_legacy_aliases_only_for_current_conversation(
    tmp_path, monkeypatch, sdk_session
):
    adapter, session, host = sdk_session
    session.workspace_root = tmp_path / "workspace"
    store = DataPreparationTaskStore(tmp_path / TASK_ASSOCIATION_DIRNAME)
    for task_id, owner, updated_at in (
        ("legacy-old", session.session_id, "2024-01-01"),
        ("legacy-new", session.session_id, "2025-01-01"),
        ("foreign", "another-conversation", "2026-01-01"),
    ):
        association = DataPreparationTaskAssociation(
            task_id=task_id,
            conversation_id=owner,
            run_id="same-run",
            output_dir="/workspace/same-output",
            input_path="/workspace/input.jsonl",
            script_path="/workspace/pipeline.py",
        )
        association.updated_at = updated_at
        store.save(association)
    (store.root / "broken.json").write_text("{broken")
    (store.root / "incomplete.json").write_text('{"task_id": "incomplete"}')
    product_record = {
        "task_id": "product-task",
        "run_id": "same-run",
        "output_dir": "/workspace/same-output",
    }
    runs = Mock(spec=PipelineRuns)
    runs.task_associations.return_value = [product_record]
    monkeypatch.setattr(host, "_pipeline_runs", lambda scope: runs)
    monkeypatch.setattr(
        host_module,
        "create_workflow_api_client",
        lambda **params: PyroMindAPIClient(api_key="derived-sdk-key"),
    )

    environment = await host.sdk_environment(adapter._control_tool_context(session))
    records = json.loads(environment["PYROMIND_TASK_ASSOCIATIONS"])

    assert records == [
        product_record,
        {**product_record, "task_id": "legacy-new"},
        {**product_record, "task_id": "legacy-old"},
    ]
    assert str(tmp_path) not in environment["PYROMIND_TASK_ASSOCIATIONS"]


async def test_sdk_environment_does_not_return_authentication_errors(
    tmp_path, monkeypatch, sdk_session, caplog
):
    adapter, session, host = sdk_session

    def failed_factory(**params):
        raise RuntimeError("request-token and derived-sdk-key must not escape")

    monkeypatch.setattr(host_module, "create_workflow_api_client", failed_factory)

    with pytest.raises(PermissionError, match="credentials are unavailable") as error:
        await host.sdk_environment(adapter._control_tool_context(session))

    assert "request-token" not in str(error.value)
    assert "derived-sdk-key" not in str(error.value)
    assert "reason=RuntimeError" in caplog.text
    assert "request-token" not in caplog.text
    assert "derived-sdk-key" not in caplog.text


async def test_sdk_rpc_credentials_failure_returns_no_environment(
    monkeypatch, sdk_session
):
    adapter, session, host = sdk_session
    provider = AsyncMock(side_effect=PermissionError("Credentials are unavailable"))
    monkeypatch.setattr(host, "sdk_environment", provider)
    task_context = AsyncMock()
    adapter.set_sdk_context_provider(task_context)

    assert await adapter._runner_request(session, "platform.sdk_environment", {}) == {}
    provider.assert_awaited_once()
    task_context.assert_not_awaited()


async def test_fixed_actor_keeps_local_sdk_when_portal_key_is_unavailable(
    monkeypatch, sdk_session
):
    adapter, session, host = sdk_session
    monkeypatch.setenv("PYROMIND_SANDBOX_ACTOR_URL", "http://router.test")
    monkeypatch.setenv("PYROMIND_SANDBOX_ACTOR_TARGET", "dev/test")
    actor = FixedActorExecutionManager.from_env()
    assert actor is not None
    monkeypatch.setattr(adapter, "_sandbox", actor)
    monkeypatch.setattr(adapter, "_terminal_backend", "sandbox")
    python = "/tmp/actor-sdk/bin/python"
    monkeypatch.setattr(actor, "sdk_python", AsyncMock(return_value=python))
    monkeypatch.setattr(
        actor, "debug_state_path", AsyncMock(return_value="/workspace/state")
    )
    exchange = AsyncMock(
        side_effect=[PermissionError(), {"PYROMIND_API_KEY": "fresh-ak"}]
    )
    monkeypatch.setattr(host, "sdk_environment", exchange)
    signing = AsyncMock(return_value={})
    adapter.set_sdk_context_provider(signing)

    first = await adapter._runner_request(session, "platform.sdk_environment", {})
    assert first == {"PYROMIND_SDK_PYTHON": python}
    signing.assert_not_awaited()
    second = await adapter._runner_request(session, "platform.sdk_environment", {})
    assert second["PYROMIND_API_KEY"] == "fresh-ak"
    assert second["PYROMIND_SDK_PYTHON"] == python
    signing.assert_awaited_once()


async def test_sdk_rpc_refreshes_signed_context_without_persisting_it(
    tmp_path, monkeypatch, sdk_session
):
    adapter, session, host = sdk_session
    environments = [
        {"PYROMIND_API_KEY": "first-account-key"},
        {"PYROMIND_API_KEY": "second-account-key"},
    ]
    monkeypatch.setattr(host, "sdk_environment", AsyncMock(side_effect=environments))
    contexts = [
        {
            "PYROMIND_TASK_CONTEXTS": json.dumps({"workflow_run": f"signed-{index}"}),
            "PYROMIND_PROJECT_EXECUTIONS": "[]",
        }
        for index in (1, 2)
    ]
    provider = AsyncMock(side_effect=contexts)
    adapter.set_sdk_context_provider(provider)
    first_request = session.context

    first = await adapter._runner_request(session, "platform.sdk_environment", {})
    session.context = replace(
        session.context, cookie="auth_token=renewed-token", x_cluster="us-west-1"
    )
    second = await adapter._runner_request(session, "platform.sdk_environment", {})

    for result, expected in ((first, contexts[0]), (second, contexts[1])):
        assert {name: result[name] for name in expected} == expected
        assert result["PYROMIND_SDK_PYTHON"] == "/prepared/sdk/bin/python"
        assert result["PYROMIND_DEBUG_STATE_PATH"] == str(
            tmp_path / "public_data/.pyromind/workflow-debug-state.json"
        )
    assert [call.args for call in provider.await_args_list] == [
        (session.session_id, first_request),
        (session.session_id, session.context),
    ]
    for path in tmp_path.rglob("*.json"):
        assert "signed-1" not in path.read_text()
        assert "signed-2" not in path.read_text()


async def test_sdk_rpc_merges_project_and_legacy_stop_aliases(monkeypatch, sdk_session):
    adapter, session, host = sdk_session
    legacy = {
        "task_id": "legacy-task",
        "run_id": "legacy-run",
        "output_dir": "/workspace/legacy-output",
    }
    project = {"task_id": "sdk-project-task", "run_id": None, "output_dir": None}
    monkeypatch.setattr(
        host,
        "sdk_environment",
        AsyncMock(
            return_value={
                "PYROMIND_API_KEY": "current-account-key",
                "PYROMIND_TASK_ASSOCIATIONS": json.dumps([legacy]),
            }
        ),
    )
    adapter.set_sdk_context_provider(
        AsyncMock(return_value={"PYROMIND_TASK_ASSOCIATIONS": json.dumps([project])})
    )

    result = await adapter._runner_request(session, "platform.sdk_environment", {})

    assert json.loads(result["PYROMIND_TASK_ASSOCIATIONS"]) == [project, legacy]


@pytest.mark.parametrize("stage", ["preparation", "project"])
async def test_echomind_main_uses_current_account_sdk_without_adding_legacy_tools(
    monkeypatch, sdk_session, stage
):
    adapter, session, host = sdk_session
    submissions = {"df_submit_pipeline", "df_submit_evaluation", "submit_workflow"}
    allowed = ["read", "write", "edit", "terminal"]
    if stage == "project":
        allowed += [*sorted(submissions), "df_stop_task"]
    session.config.update(
        profile_id="echomind", tool_allowlist=allowed, delegated_roles=[]
    )
    session.context = replace(session.context, profile_id="echomind")
    adapter.set_workflow_preparer(AsyncMock())
    monkeypatch.setattr(
        host,
        "sdk_environment",
        AsyncMock(return_value={"PYROMIND_API_KEY": "current-account-key"}),
    )
    context_provider = AsyncMock(
        return_value={
            "PYROMIND_TASK_CONTEXTS": "{}",
            "PYROMIND_PROJECT_EXECUTIONS": "[]",
        }
    )
    adapter.set_sdk_context_provider(context_provider)

    result = await adapter._runner_request(session, "platform.sdk_environment", {})
    names = {spec["name"] for spec in adapter._tool_specs(session)}

    assert result["PYROMIND_API_KEY"] == "current-account-key"
    context_provider.assert_awaited_once_with(session.session_id, session.context)
    assert SDK_ATOMIC_TOOL_NAMES.isdisjoint(names)
    assert _LEGACY_SUBMISSION_TOOLS & names == (
        submissions - _MIGRATED_SUBMISSION_TOOLS if stage == "project" else set()
    )


async def test_sdk_context_authority_can_reject_legacy_roles(monkeypatch, sdk_session):
    adapter, session, host = sdk_session
    session.config.update(profile_id="echomind", tool_allowlist=["terminal"])
    session.context = replace(session.context, profile_id="echomind")
    monkeypatch.setattr(
        host,
        "sdk_environment",
        AsyncMock(return_value={"PYROMIND_API_KEY": "current-account-key"}),
    )
    provider = AsyncMock(side_effect=PermissionError("Legacy roles are not enabled"))
    adapter.set_sdk_context_provider(provider)

    with pytest.raises(PermissionError, match="Legacy roles"):
        await adapter._runner_request(session, "platform.sdk_environment", {})
    provider.assert_awaited_once_with(session.session_id, session.context)


@pytest.mark.parametrize("backend", ["os-sandbox", "sandbox"])
async def test_sdk_setup_failure_is_explicit_without_leaking_credentials(
    monkeypatch, sdk_session, caplog, backend
):
    adapter, session, host = sdk_session
    adapter._terminal_backend = backend
    monkeypatch.setattr(
        host,
        "sdk_environment",
        AsyncMock(
            return_value={
                "PYROMIND_API_KEY": "derived-sdk-key",
                "PYROMIND_BASE_URL": "https://api-portal.pyromind.ai/api/v1",
                "PYROMIND_CLUSTER": "us-west-1",
            }
        ),
    )
    prepare = AsyncMock(side_effect=RuntimeError("derived-sdk-key must not escape"))
    monkeypatch.setattr(adapter._sandbox, "sdk_python", prepare)
    prepare_host = Mock(side_effect=RuntimeError("derived-sdk-key must not escape"))
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.adapter.prepare_host_sdk", prepare_host
    )

    with pytest.raises(RuntimeError, match="Platform SDK setup failed") as error:
        await adapter._runner_request(session, "platform.sdk_environment", {})
    if backend == "sandbox":
        prepare.assert_awaited_once()
        prepare_host.assert_not_called()
    else:
        prepare_host.assert_called_once_with()
        prepare.assert_not_awaited()
    migrated = SDK_ATOMIC_TOOL_NAMES - (
        {"sandbox_terminal"} if backend == "os-sandbox" else set()
    )
    assert migrated.isdisjoint(spec["name"] for spec in adapter._tool_specs(session))
    assert "derived-sdk-key" not in str(error.value)
    assert "derived-sdk-key" not in caplog.text


@pytest.mark.parametrize("backend", ["os-sandbox", "sandbox"])
def test_sdk_migration_omits_covered_atomic_schemas(sdk_session, backend):
    adapter, session, host = sdk_session
    adapter._terminal_backend = backend
    retained = {"sandbox_terminal"} if backend == "os-sandbox" else set()

    exposed = {spec["name"] for spec in adapter._tool_specs(session)}

    assert SDK_ATOMIC_TOOL_NAMES & exposed == retained
    assert SDK_ATOMIC_TOOL_NAMES <= {
        spec["name"] for spec in host.specs(sandbox=backend == "sandbox")
    }


@pytest.mark.parametrize("profile", ["pyromind", "echomind"])
@pytest.mark.parametrize("sdk_enabled", [True, False])
def test_inspection_uses_sdk_without_requiring_submission_context(
    sdk_session, profile, sdk_enabled
):
    adapter, session, host = sdk_session
    session.config["profile_id"] = profile
    session.context = replace(session.context, profile_id=profile)
    if profile == "echomind":
        session.config["tool_allowlist"] = ["terminal", "inspect_workflow_task"]
    adapter._platform_sdk_configured = sdk_enabled

    exposed = {spec["name"] for spec in adapter._tool_specs(session)}

    assert ("inspect_workflow_task" in exposed) is not sdk_enabled
    assert "inspect_workflow_task" in {
        spec["name"] for spec in host.specs(sandbox=False)
    }


def test_remote_main_keeps_only_four_platform_tools(sdk_session):
    adapter, session, host = sdk_session
    adapter._terminal_backend = "sandbox"
    session.config["tool_allowlist"] = list(PYROMIND_TOOL_ALLOWLIST)
    adapter.set_sdk_context_provider(AsyncMock())
    adapter.set_workflow_preparer(AsyncMock())
    host.version_call = AsyncMock()

    names = {spec["name"] for spec in adapter._tool_specs(session)}
    platform_names = PLATFORM_SDK_TOOL_NAMES | {
        "training_analysis",
        "label_studio_project",
    }

    assert names & platform_names == {
        "get_storage_url",
        "validate_workflow_dsl",
        "training_analysis",
        "label_studio_project",
    }
    assert {"product_versions", "update_plan"} <= names
    assert SDK_REMOTE_STORAGE_TOOL_NAMES.isdisjoint(names)
    assert SDK_REMOTE_STORAGE_TOOL_NAMES <= {
        spec["name"] for spec in host.specs(sandbox=False)
    }


@pytest.mark.parametrize("backend", ["os-sandbox", "sandbox"])
@pytest.mark.parametrize("sdk_enabled", [True, False])
def test_storage_tool_filter_requires_remote_sdk(sdk_session, backend, sdk_enabled):
    adapter, session, host = sdk_session
    adapter._terminal_backend = backend
    adapter._platform_sdk_configured = sdk_enabled
    exposed = {spec["name"] for spec in adapter._tool_specs(session)}
    original = {spec["name"] for spec in host.specs(sandbox=backend == "sandbox")}
    retained = (
        set()
        if backend == "sandbox" and sdk_enabled
        else SDK_REMOTE_STORAGE_TOOL_NAMES & original
    )

    assert SDK_REMOTE_STORAGE_TOOL_NAMES & exposed == retained


@pytest.mark.parametrize("backend", ["os-sandbox", "sandbox"])
@pytest.mark.parametrize("sdk_enabled", [True, False])
@pytest.mark.parametrize("context_enabled", [True, False])
def test_submission_catalog_hides_only_deployed_sdk_paths(
    sdk_session, backend, sdk_enabled, context_enabled
):
    adapter, session, _ = sdk_session
    adapter._terminal_backend = backend
    adapter._platform_sdk_configured = sdk_enabled
    adapter.set_workflow_preparer(AsyncMock())
    if context_enabled:
        adapter.set_sdk_context_provider(AsyncMock())

    names = {spec["name"] for spec in adapter._tool_specs(session)}

    migrated = (
        _MIGRATED_SUBMISSION_TOOLS if sdk_enabled and context_enabled else frozenset()
    )
    assert _LEGACY_SUBMISSION_TOOLS & names == _LEGACY_SUBMISSION_TOOLS - migrated
    assert "execute_business_plan" not in names
    assert session.config["tool_allowlist"] is None


@pytest.mark.parametrize("sdk_enabled", [True, False])
@pytest.mark.parametrize(
    "permitted",
    [
        {"df_submit_pipeline", "df_submit_evaluation", "submit_workflow"},
        _LEGACY_SUBMISSION_TOOLS,
    ],
)
def test_echomind_keeps_existing_submission_permissions_and_project_binding(
    sdk_session, sdk_enabled, permitted
):
    adapter, session, _ = sdk_session
    session.config.update(
        profile_id="echomind",
        tool_allowlist=["terminal", *sorted(permitted)],
        command_allowlist=["enqueue_task"],
    )
    session.context = replace(session.context, profile_id="echomind")
    adapter.set_workflow_preparer(AsyncMock())
    adapter.set_plan_submission_handler(AsyncMock())
    adapter.set_sdk_context_provider(AsyncMock())
    adapter._platform_sdk_configured = sdk_enabled

    specs = {spec["name"]: spec for spec in adapter._tool_specs(session)}

    assert set(specs) == (
        permitted - _MIGRATED_SUBMISSION_TOOLS if sdk_enabled else permitted
    )
    for name in {"df_submit_pipeline", "df_submit_evaluation", "submit_workflow"} & (
        specs.keys()
    ):
        assert {"plan_task_id", "plan_attempt"} <= (
            specs[name]["input_schema"]["properties"].keys()
        )


@pytest.mark.parametrize("backend", ["os-sandbox", "sandbox"])
def test_submission_catalog_survives_restart_without_changing_legacy_permissions(
    tmp_path, sdk_session, backend
):
    adapter, session, _ = sdk_session
    adapter._terminal_backend = backend
    adapter.set_workflow_preparer(AsyncMock())
    adapter.set_sdk_context_provider(AsyncMock())
    names = {spec["name"] for spec in adapter._tool_specs(session)}
    restored_adapter = PiAdapter(tmp_path, terminal_backend=backend)
    restored_adapter.set_workflow_preparer(AsyncMock())
    restored_adapter.set_sdk_context_provider(AsyncMock())
    restored = _PiSession(
        session.session_id,
        tmp_path,
        session.files,
        session.files.load_session(),
        session.context,
    )

    assert {spec["name"] for spec in restored_adapter._tool_specs(restored)} == names
    assert _MIGRATED_SUBMISSION_TOOLS.isdisjoint(names)
    assert restored.config["tool_allowlist"] is None
    restored_adapter._platform_sdk_configured = False
    assert _LEGACY_SUBMISSION_TOOLS <= {
        spec["name"] for spec in restored_adapter._tool_specs(restored)
    }


@pytest.mark.parametrize("kind", ["data_preparation", "data_cleaning"])
async def test_unsigned_historical_runs_do_not_reintroduce_submission_tools(
    historical_sdk_session, kind
):
    adapter, session = await historical_sdk_session(
        {
            "task_id": "historical-task",
            "kind": kind,
            "status": "running",
            "submitted_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    )

    names = {spec["name"] for spec in adapter._tool_specs(session)}

    assert _MIGRATED_SUBMISSION_TOOLS.isdisjoint(names)
    assert session.active_external_tasks == {"historical-task": kind}
    assert session.files.load_session()["tool_allowlist"] is None


async def test_remote_sdk_budget_environment_uses_execution_workspace_path(
    monkeypatch, sdk_session
):
    adapter, session, host = sdk_session
    adapter._terminal_backend = "sandbox"
    monkeypatch.setattr(
        host, "sdk_environment", AsyncMock(return_value={"PYROMIND_API_KEY": "sdk-key"})
    )
    monkeypatch.setattr(
        adapter._sandbox, "sdk_python", AsyncMock(return_value="/tmp/sdk/bin/python")
    )
    path = "/workspace/conversation/public_data/.pyromind/workflow-debug-state.json"
    prepare = AsyncMock(return_value=path)
    monkeypatch.setattr(adapter._sandbox, "debug_state_path", prepare)

    result = await adapter._runner_request(session, "platform.sdk_environment", {})

    assert result["PYROMIND_DEBUG_STATE_PATH"] == path
    assert str(session.workspace_root) not in result["PYROMIND_DEBUG_STATE_PATH"]
    prepare.assert_awaited_once()


@pytest.mark.parametrize("backend", ["os-sandbox", "sandbox"])
async def test_sdk_rollback_restores_legacy_schemas_and_disables_environment(
    tmp_path, monkeypatch, sdk_session, backend
):
    _, session, _ = sdk_session
    adapter = PiAdapter(tmp_path, terminal_backend=backend, platform_sdk_enabled=False)
    provider = AsyncMock()
    monkeypatch.setattr(adapter._business_tools, "sdk_environment", provider)

    assert SDK_ATOMIC_TOOL_NAMES <= {
        spec["name"] for spec in adapter._tool_specs(session)
    }
    assert await adapter._runner_request(session, "platform.sdk_environment", {}) == {}
    provider.assert_not_awaited()


@pytest.mark.parametrize("backend", ["os-sandbox", "sandbox"])
@pytest.mark.parametrize(
    "scope",
    [
        "config-echomind",
        "request-echomind",
        "restricted",
        "no-auth",
        "anonymous",
        "custom-roles",
        "child",
        "no-read-spec",
    ],
)
async def test_sdk_rpc_does_not_expand_other_scopes(
    monkeypatch, sdk_session, scope, backend
):
    adapter, session, host = sdk_session
    adapter._terminal_backend = backend
    provider = AsyncMock(return_value={"PYROMIND_API_KEY": "must-not-be-issued"})
    monkeypatch.setattr(host, "sdk_environment", provider)
    task_context = AsyncMock()
    adapter.set_sdk_context_provider(task_context)
    if scope == "config-echomind":
        session.config["profile_id"] = "echomind"
    elif scope == "request-echomind":
        session.context = replace(session.context, profile_id="echomind")
    elif scope == "restricted":
        session.config["tool_allowlist"] = [
            "terminal",
            "sandbox_read_file",
            "sandbox_delete_file",
        ]
    elif scope == "no-auth":
        session.context = replace(session.context, cookie=None)
    elif scope == "anonymous":
        session.context = replace(session.context, user_id="anonymous")
    elif scope == "custom-roles":
        session.config["delegated_roles"] = [{"name": "helper", "tools": ["read"]}]
    elif scope == "child":
        session.context = replace(
            session.context,
            task_origin=TaskOrigin(type="subagent", subagent_id="child-1"),
        )
    else:
        monkeypatch.setattr(host, "specs", lambda **kwargs: [])

    assert await adapter._runner_request(session, "platform.sdk_environment", {}) == {}
    provider.assert_not_awaited()
    task_context.assert_not_awaited()
    if scope != "no-read-spec":
        retained = SDK_ATOMIC_TOOL_NAMES
        if scope == "restricted":
            retained = frozenset({"sandbox_read_file", "sandbox_delete_file"})
        assert retained <= {spec["name"] for spec in adapter._tool_specs(session)}


@pytest.mark.parametrize(
    "params", [{"owner": "main"}, {"origin": {"type": "main"}}, {"user_id": "42"}]
)
async def test_sdk_rpc_rejects_caller_supplied_identity(
    monkeypatch, sdk_session, params
):
    adapter, session, host = sdk_session
    provider = AsyncMock()
    monkeypatch.setattr(host, "sdk_environment", provider)

    with pytest.raises(PermissionError, match="not enabled"):
        await adapter._runner_request(session, "platform.sdk_environment", params)

    provider.assert_not_awaited()


async def test_sdk_rpc_refreshes_and_masks_rotated_keys_before_events(
    tmp_path: Path, monkeypatch, sdk_session
):
    adapter, session, host = sdk_session
    session.config["tool_allowlist"] = ["terminal", *PLATFORM_SDK_TOOL_NAMES]
    environments = [
        {
            "PYROMIND_API_KEY": key,
            "PYROMIND_BASE_URL": "https://pre2-api-portal.pyromind.ai/api/v1",
            "PYROMIND_CLUSTER": "cn-east-1#pre2",
        }
        for key in ("first-sdk-key", "second-sdk-key")
    ]
    credentials = []
    model_environments = []
    for index, environment in enumerate(environments):
        environment["LLM_AUTH_TOKEN"] = f"model-key-{index}"
        profile_key = f"dataflow-key-{index}"
        model_environments.append(
            {
                "PYROMIND_DATAFLOW_PROFILES": json.dumps(
                    {"text": {"DF_API_KEY": profile_key}, "vision": {}}
                )
            }
        )
        credentials.extend(
            [
                environment["PYROMIND_API_KEY"],
                environment["LLM_AUTH_TOKEN"],
                profile_key,
            ]
        )
    provider = AsyncMock(side_effect=environments)
    monkeypatch.setattr(host, "sdk_environment", provider)
    monkeypatch.setattr(
        host, "terminal_environment", Mock(side_effect=model_environments)
    )

    for expected in environments:
        result = await adapter._runner_request(session, "platform.sdk_environment", {})
        assert result["PYROMIND_API_KEY"] == expected["PYROMIND_API_KEY"]
        assert result["LLM_AUTH_TOKEN"] == expected["LLM_AUTH_TOKEN"]
    assert provider.await_count == 2

    frame = {
        "protocolVersion": 2,
        "type": "pi.event",
        "sessionId": session.session_id,
        "eventId": "sdk-output",
        "kind": "message.completed",
        "payload": {
            "message_id": "message-1",
            "role": "assistant",
            "content": [{"type": "text", "text": " ".join(credentials)}],
        },
    }
    await adapter._runner_event(session, frame)
    event = session.queue.get_nowait()
    content = event.payload["content"]
    assert content == [
        {"type": "text", "text": " ".join(["<secret-hidden>"] * len(credentials))}
    ]
    assert all(credential not in repr(session) for credential in credentials)
    for path in tmp_path.rglob("*.json"):
        assert all(credential not in path.read_text() for credential in credentials)

    session.context = replace(session.context, cookie=None)
    assert await adapter._runner_request(session, "platform.sdk_environment", {}) == {}
    assert provider.await_count == 2
