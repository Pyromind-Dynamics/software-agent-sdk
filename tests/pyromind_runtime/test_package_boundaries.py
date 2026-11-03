import ast
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
import yaml
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute
from harness_adapter.pi_adapter.business_tool_host import (
    PyromindBusinessToolHost,
    ToolExecutionContext,
    _ToolConversationFacade,
)
from pyromind_agent_server.api.evaluation_runs import (
    create_evaluation_runs_router,
    get_evaluation_storage,
)
from pyromind_agent_server.api.router import create_product_router
from pyromind_runtime.application.event_projection import ProductEventProjector
from pyromind_runtime.application.snapshot_projector import SnapshotProjector
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import HarnessEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot

from openhands.agent_server.dependencies import check_session_api_key
from openhands.tools.data_preparation.evaluation_submit import DfSubmitEvaluationAction
from openhands.tools.data_preparation.inference import InferenceOptions
from openhands.tools.data_preparation.platform_submit import DfSubmitPipelineAction
from openhands.tools.workflow.validate_workflow_dsl import ValidateWorkflowDslAction


ROOT = Path(__file__).resolve().parents[2]


def test_evaluation_auth_boundary_is_separate_from_product_api() -> None:
    def dependency_calls(dependant: Dependant):
        yield dependant.call
        for dependency in dependant.dependencies:
            yield from dependency_calls(dependency)

    for route in create_evaluation_runs_router().routes:
        assert isinstance(route, APIRoute)
        calls = set(dependency_calls(route.dependant))
        assert get_evaluation_storage in calls
        assert check_session_api_key not in calls

    snapshot = next(
        route
        for route in create_product_router().routes
        if isinstance(route, APIRoute) and route.path.endswith("/snapshot")
    )
    assert check_session_api_key in dependency_calls(snapshot.dependant)


@pytest.mark.parametrize("env", ["pre", "pre2", "prod"])
def test_evaluation_validation_shares_submission_route_and_identity(
    tmp_path, monkeypatch, env
):
    monkeypatch.setenv("APP_ENV", "prod" if env.startswith("pre") else "dev")
    context = ToolExecutionContext(
        conversation_id="test-conversation",
        workspace_root=tmp_path,
        request_context=RequestContext(
            user_id="test",
            authorization="Bearer test-token",
            cookie="auth_token=test-cookie",
            x_cluster=f"test-cluster#{env}",
        ),
        model_configuration={},
    )
    host = PyromindBusinessToolHost(
        [], skills_directory=ROOT / ".agents/skills", pipeline_runs=Mock()
    )
    params = host._evaluation_params(context)
    requests = []

    def post(url, **kwargs):
        requests.append((url, kwargs["headers"]))
        return httpx.Response(200, json={"success": True, "data": {"valid": True}})

    monkeypatch.setattr(httpx, "post", post)
    result = params["inference_handler"].validator(
        ValidateWorkflowDslAction(dsl="# workflow: test"),
        _ToolConversationFacade(context),
    )
    assert result.valid is True
    assert params["env"] == env and params["cluster"] == "test-cluster"
    prefix = "" if env == "prod" else f"{env}-"
    assert requests[0][0] == (
        f"https://{prefix}api-portal.pyromind.ai/std2/studio_api/api/workflows/dsl/validate"
    )
    headers = requests[0][1]
    assert headers["x-cluster"] == params["headers"]["x-cluster"]
    assert headers["authorization"] == "Bearer test-token"
    assert headers["cookie"] == "auth_token=test-cookie"


def test_activity_timestamp_flows_through_generic_event_contract() -> None:
    occurred_at = datetime(2026, 9, 15, tzinfo=UTC)
    event = ProductEventProjector().project(
        "conversation-1",
        HarnessEvent(
            session_id="native-session-1",
            type="status.changed",
            payload={"status": "running"},
            occurred_at=occurred_at,
        ),
    )
    assert event is not None
    projector = SnapshotProjector()
    snapshot = projector.reduce(
        ConversationSnapshot(
            conversation_id="conversation-1", capabilities=HarnessCapabilities()
        ),
        event.model_copy(update={"seq": 1}),
    )
    assert snapshot.updated_at == occurred_at
    delayed = event.model_copy(
        update={"seq": 2, "occurred_at": datetime(2026, 9, 14, tzinfo=UTC)}
    )
    assert projector.reduce(snapshot, delayed).updated_at == occurred_at


def _imports(package_root: Path) -> set[str]:
    imported: set[str] = set()
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    return imported


def test_runtime_has_no_server_or_harness_dependencies() -> None:
    imports = _imports(ROOT / "pyromind-runtime" / "pyromind_runtime")
    forbidden = (
        "fastapi",
        "openhands",
        "harness_adapter",
        "pyromind_agent_server",
        "agentgenome",
    )
    assert not any(name.startswith(forbidden) for name in imports)


def test_adapter_does_not_depend_on_pyromind_server() -> None:
    imports = _imports(ROOT / "harness-adapter" / "harness_adapter")
    assert not any(name.startswith("pyromind_agent_server") for name in imports)


def test_evaluation_model_sources_stay_out_of_harness_and_tool_protocol() -> None:
    assert "model_source" not in InferenceOptions.model_fields
    assert "model_source" not in DfSubmitPipelineAction.model_fields
    preparation_nodes = {"CloneAndCacheModel", "DownloadAndCacheModel"}
    for path in (ROOT / "harness-adapter/harness_adapter").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr != "model_source", str(path)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in preparation_nodes, str(path)


def test_evaluation_routes_delegate_to_product_runtime() -> None:
    route = ROOT / "pyromind-agent-server/pyromind_agent_server/api/evaluation_runs.py"
    tree = ast.parse(route.read_text(encoding="utf-8"), filename=str(route))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not imports.intersection(
        {
            "pyromind_runtime.infrastructure.file_product_store",
            "pyromind_runtime.application.pipeline_runs",
            "pyromind_agent_server.external_task_registry",
        }
    )
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert {
        "EvaluationAnnotationService",
        "EvaluationLockRegistry",
        "PyromindEvaluationStorage",
    }.isdisjoint(imported_names)
    assert "build_evaluation_storage" in imported_names
    calls = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "get_product_runtime"
    }
    assert {
        "load_evaluation_annotations",
        "save_evaluation_annotations",
        "read_evaluation_report",
        "export_evaluation_annotations",
    } <= calls


def test_pyromind_start_scripts_use_composed_server_entrypoint() -> None:
    for script_name in ("start.sh", "start_inference.sh"):
        script = (ROOT / script_name).read_text(encoding="utf-8")
        assert (
            "python -m pyromind_agent_server" in script
            or '"${server_python}" -m pyromind_agent_server' in script
        )
        assert "python -m openhands.agent_server" not in script


def test_inference_start_defaults_to_platform_sandbox_for_pi_terminal() -> None:
    script = (ROOT / "start_inference.sh").read_text(encoding="utf-8")

    assert 'export APP_ENV="${APP_ENV:-dev}"' in script
    assert (
        "export PYROMIND_PI_TERMINAL_BACKEND="
        '"${PYROMIND_PI_TERMINAL_BACKEND:-sandbox}"' in script
    )


def test_pre_deployment_uses_platform_sandbox_for_pi_terminal() -> None:
    documents = yaml.safe_load_all(
        (ROOT / "deploy" / "sts.yaml").read_text(encoding="utf-8")
    )
    stateful_set = next(
        document for document in documents if document.get("kind") == "StatefulSet"
    )
    container = stateful_set["spec"]["template"]["spec"]["containers"][0]
    environment = {item["name"]: item.get("value") for item in container["env"]}

    assert environment["PYROMIND_HARNESS_BACKEND"] == "pi"
    assert environment["APP_ENV"] == "pre"
    assert environment["PYROMIND_PI_TERMINAL_BACKEND"] == "sandbox"
    assert environment["PYROMIND_SANDBOX_IDLE_DELETE_SECONDS"] == "1800"


def test_product_image_defaults_to_os_sandbox_for_pi_terminal() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    product_stage = dockerfile.split("FROM base-image-minimal AS product", maxsplit=1)[
        1
    ]

    assert "ENV PYROMIND_PI_TERMINAL_BACKEND=os-sandbox" in product_stage


def test_local_startup_checks_platform_specific_sandbox_dependencies() -> None:
    script = (ROOT / "start_inference.sh").read_text(encoding="utf-8")

    # The checks belong to the os-sandbox backend only; the platform sandbox
    # default must not require host sandbox binaries.
    assert 'if [[ "${PYROMIND_PI_TERMINAL_BACKEND}" == "os-sandbox" ]]; then' in script
    assert "[[ ! -x /usr/bin/sandbox-exec ]]" in script
    assert "for sandbox_dependency in rg bwrap socat" in script


def test_pi_sandbox_initializes_before_using_conversation_temp_path() -> None:
    tools_source = (
        ROOT / "harness-adapter" / "pi-runtime" / "src" / "tools.ts"
    ).read_text(encoding="utf-8")

    terminal_tmp_declaration = tools_source.index(
        "const { terminalTempRoot: terminalOutputTemp } = policy"
    )
    sandbox_initialization = tools_source.index("await createWorkspaceBashOperations(")
    conversation_tmpdir = tools_source.index("process.env.TMPDIR = terminalOutputTemp")
    assert terminal_tmp_declaration < sandbox_initialization < conversation_tmpdir


def test_internal_workflow_lifecycle_events_never_enter_public_protocol() -> None:
    from typing import get_args

    from pyromind_runtime.application.event_projection import ProductEventProjector
    from pyromind_runtime.domain.events import HarnessEvent, ProductEventType

    for kind in ("workflow.modified", "run.finished"):
        assert kind not in get_args(ProductEventType.__value__)
        event = HarnessEvent.model_validate(
            {
                "session_id": "conversation",
                "run_id": "run",
                "type": kind,
                "payload": {},
            }
        )
        assert ProductEventProjector().project("conversation", event) is None


def test_reusable_workflow_runtime_has_no_engine_or_harness_imports() -> None:
    runtime = ROOT / "pyromind-runtime" / "pyromind_runtime"
    for relative in (
        "ports/workflows.py",
        "application/reusable_workflows.py",
        "application/conversation_runtime.py",
    ):
        tree = ast.parse((runtime / relative).read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(
            name.split(".")[0] in {"agentgenome", "core", "harness_adapter", "fastapi"}
            for name in imports
        ), relative


def test_sdk_agentgenome_uses_host_bridge_without_native_execution_fallback() -> None:
    adapter = ROOT / "harness-adapter"
    bridge = (adapter / "harness_adapter/agentgenome_bridge.py").read_text()
    imports = {
        node.module
        for node in ast.walk(ast.parse(bridge))
        if isinstance(node, ast.ImportFrom)
    }
    assert "agentgenome.dispatch" in imports
    assert not imports.intersection(
        {"agentgenome.local_host", "agentgenome.local_service"}
    )
    session = (adapter / "pi-runtime/src/pi-session.ts").read_text()
    assert "additionalExtensionPaths" in session
    assert "experienceHost.assertLoaded(resourceLoader)" in session
    for path in (adapter / "pi-runtime/src").rglob("*.ts"):
        source = path.read_text()
        assert "createGenomeExtension" not in source, path
        assert "createBridgeHost" not in source, path
    host = (adapter / "pi-runtime/src/genome-host.ts").read_text()
    assert "provideGenomeHost" in host
    assert "prepareArguments" not in host
    assert "validateToolArguments" not in host
    assert 'peer.request("workflow.invoke"' not in session
    assert "NativeClient" not in session
    assert "nativeHost" not in session


def test_subagent_lifecycle_has_no_harness_specific_product_branch() -> None:
    source = (
        ROOT / "pyromind-runtime/pyromind_runtime/application/conversation_runtime.py"
    ).read_text()
    tree = ast.parse(source)
    guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_has_active_subagents"
    )
    literals = {
        node.value
        for node in ast.walk(guard)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert literals == {"subagent", "running"}
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "COPY harness-adapter/pi-runtime/agents ./agents" in dockerfile
    assert "COPY harness-adapter/pi-runtime/scripts ./scripts" in dockerfile


def test_native_subagent_report_delivery_stays_in_adapter() -> None:
    for package in ("pyromind-runtime", "pyromind-agent-server"):
        for path in (ROOT / package).rglob("*.py"):
            source = path.read_text()
            assert "subagent-artifacts" not in source, path
            assert "public_data/subagents/" not in source, path
            assert "SubagentReports" not in source, path


def test_subagent_display_rpc_remains_adapter_private() -> None:
    for package in ("pyromind-runtime", "pyromind-agent-server"):
        for path in (ROOT / package).rglob("*.py"):
            source = path.read_text()
            assert "subagents:rpc:v1" not in source, path
            assert "transcriptWindow" not in source, path
            assert "subagent.progress" not in source, path


def test_generated_business_scoring_only_executes_in_cpu_worker():
    runtime = ROOT / "pyromind-runtime/pyromind_runtime"
    for path in runtime.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        assert not {
            "evaluate_predictions",
            "evaluation_products",
            "frozen_business_scoring",
        }.intersection(imports)
        if path.name in {
            "checkpoint_selection.py",
            "training_evaluations.py",
            "review_tasks.py",
            "test_evaluation.py",
        }:
            calls = [
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            ]
            assert not {"eval", "exec", "compile", "__import__"}.intersection(calls)
    from pyromind_runtime.ports.harness import SessionSpec

    assert SessionSpec.model_fields["product_instructions"].default == ""
    assert not {"inference", "evaluation", "products"}.intersection(
        DfSubmitPipelineAction.model_fields
    )
    assert {
        "operation",
        "model_path",
        "scoring_script_path",
        "manifest_path",
    } <= DfSubmitEvaluationAction.model_fields.keys()
