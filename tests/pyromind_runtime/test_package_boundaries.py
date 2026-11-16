import ast
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
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
    monkeypatch.setenv("APP_ENV", "pre2" if env == "prod" else "prod")
    route = "test-cluster" if env == "prod" else f"test-cluster#{env}"
    context = ToolExecutionContext(
        conversation_id="test-conversation",
        workspace_root=tmp_path,
        request_context=RequestContext(
            user_id="test",
            authorization="Bearer test-token",
            cookie="auth_token=test-cookie",
            x_cluster=route,
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
    assert headers["x-cluster"] == params["headers"]["x-cluster"] == route
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

    assert 'export APP_ENV="${APP_ENV:-prod}"' in script
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


def test_stage_projection_cannot_schedule_manager_turns() -> None:
    source = (
        ROOT / "pyromind-runtime/pyromind_runtime/application/snapshot_projector.py"
    )
    tree = ast.parse(source.read_text())
    handler = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "project_stage_activity"
    )
    attributes = {
        node.attr for node in ast.walk(handler) if isinstance(node, ast.Attribute)
    }
    assert not attributes & {"enqueue_manager_message", "notify_external_task", "send"}
    for path in (
        "application/delegated_execution.py",
        "application/work_queue.py",
        "infrastructure/sql_experiment_execution.py",
        "infrastructure/sql_work_queue.py",
    ):
        assert not (ROOT / "pyromind-runtime/pyromind_runtime" / path).exists()


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


def test_return_addresses_stay_in_the_product_contract_and_out_of_model_arguments():
    from pyromind_runtime.domain.task_correlation import TaskCorrelation
    from pyromind_runtime.ports.harness import ExternalTaskNotification

    from openhands.tools.workflow import RunWorkflowAction
    from openhands.tools.workflow_debug.definition import WorkflowDebugAction

    assert "correlation" in ExternalTaskNotification.model_fields
    assert set(TaskCorrelation.model_fields) == {
        "version",
        "source",
        "conversation_id",
        "type",
        "subagent_id",
        "task_kind",
    }
    for action in (RunWorkflowAction, WorkflowDebugAction, DfSubmitPipelineAction):
        assert not {"correlation", "origin", "subagent_id"}.intersection(
            action.model_fields
        )


def test_preparation_policy_is_trusted_profile_configuration():
    from pydantic import ValidationError
    from pyromind_agent_server.api.schemas import CreateConversationRequest
    from pyromind_agent_server.profiles import (
        DEFAULT_PRODUCT_PROFILE,
        ECHOMIND_PRODUCT_PROFILE,
    )

    assert DEFAULT_PRODUCT_PROFILE.preparation_policy is None
    policy = ECHOMIND_PRODUCT_PROFILE.preparation_policy
    assert policy is not None
    roles = ECHOMIND_PRODUCT_PROFILE.delegated_roles
    assert roles is not None and set(policy.allowed_roles) <= {
        role.name for role in roles
    }
    tools = set(ECHOMIND_PRODUCT_PROFILE.tool_allowlist or ())
    assert "set_preparation_plan" in tools
    assert not {"get_preparation", "update_preparation", "inspect_resource"} & tools
    assert {"read", "write", "edit", "terminal"} <= tools
    assert not {
        "grep",
        "find",
        "ls",
        "subagent",
        "sandbox_create",
        "df_submit_evaluation",
        "request_experiment_task",
    }.intersection(tools)
    assert set(ECHOMIND_PRODUCT_PROFILE.development_tool_allowlist or ()) == {
        "read",
        "write",
        "edit",
        "terminal",
        "subagent",
        "ask_user",
        "get_storage_url",
    }
    assert ECHOMIND_PRODUCT_PROFILE.development_skill_allowlist == ()
    with pytest.raises(ValidationError):
        CreateConversationRequest.model_validate(
            {
                "llm": {"model": "test"},
                "preparation_policy": policy.model_dump(mode="json"),
            }
        )


@pytest.mark.parametrize("layout", ["source", "bundle", "configured"])
def test_preparation_checklist_resolves_from_source_bundle_or_configured_root(
    tmp_path, monkeypatch, layout
):
    from pyromind_agent_server import profiles

    package = tmp_path / "bundle" / "pyromind_agent_server" / "profiles.py"
    if layout == "source":
        package = (
            tmp_path
            / "source"
            / "pyromind-agent-server"
            / "pyromind_agent_server"
            / "profiles.py"
        )
        root = package.parents[2] / ".agents/skills"
    elif layout == "configured":
        root = tmp_path / "configured-skills"
    else:
        root = package.parents[1] / ".agents/skills"
    monkeypatch.setattr(profiles, "__file__", str(package))
    monkeypatch.delenv("PYROMIND_SKILLS_PATH", raising=False)
    if layout == "configured":
        monkeypatch.setenv("PYROMIND_SKILLS_PATH", str(root))
    policy = profiles.ECHOMIND_PRODUCT_PROFILE.preparation_policy
    assert policy is not None
    target = root / "model-evaluation-preparation/workflow-nodes.json"
    target.parent.mkdir(parents=True)
    target.write_text(policy.model_dump_json())
    assert profiles._load_preparation_policy() == policy


def test_frozen_bundle_includes_preparation_and_role_skill_assets():
    spec = ROOT / "openhands-agent-server/openhands/agent_server/agent-server.spec"
    function = next(
        node
        for node in ast.parse(spec.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "get_pi_skill_datas"
    )
    namespace: dict[str, Any] = {"project_root": ROOT}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(spec), "exec"),
        namespace,
    )
    targets = {
        (Path(target) / Path(source).name).as_posix()
        for source, target in namespace["get_pi_skill_datas"]()
    }
    assert ".agents/skills/model-evaluation-preparation/workflow-nodes.json" in targets
    assert ".agents/skills/avi-baseline-evaluation/SKILL.md" in targets


def test_execution_capabilities_are_profile_owned_and_cannot_be_model_supplied():
    from pydantic import ValidationError
    from pyromind_agent_server.profiles import (
        DEFAULT_PRODUCT_PROFILE,
        ECHOMIND_PRODUCT_PROFILE,
    )
    from pyromind_runtime.domain.preparation import PreparationPolicy

    roles = {role.name: role for role in ECHOMIND_PRODUCT_PROFILE.delegated_roles or ()}
    policy = ECHOMIND_PRODUCT_PROFILE.preparation_policy
    assert policy is not None
    assert (
        set(roles) == set(policy.allowed_roles) == {"init", "dataset", "train", "eval"}
    )
    assert DEFAULT_PRODUCT_PROFILE.delegated_roles is None
    for role in roles.values():
        assert set(role.model_dump(mode="json")) == {"name", "tools", "skills"}
    assert roles["init"] is not roles["eval"]
    assert roles["init"].tools == roles["eval"].tools
    obsolete = {
        "request_experiment_task",
        "get_experiment_execution",
        "get_experiment_context",
        "update_experiment_context",
        "control_experiment_task",
        "read_experiment_artifact",
        "record_experiment_result",
        "record_execution_checkpoint",
    }
    for role in roles.values():
        assert not obsolete & set(role.tools)
    assert not obsolete & set(ECHOMIND_PRODUCT_PROFILE.development_tool_allowlist or ())
    with pytest.raises(ValidationError):
        PreparationPolicy.model_validate(
            {**policy.model_dump(mode="json"), "execution_policies": {}}
        )


def test_shared_execution_rules_do_not_dispatch_on_agent_or_role_names():
    names = {"echomind", "pyromind", "avi", "pcb", "init", "dataset", "train", "eval"}
    for filename in (
        "application/conversation_runtime.py",
        "application/snapshot_projector.py",
        "infrastructure/sql_preparation.py",
        "infrastructure/file_product_store.py",
    ):
        path = ROOT / "pyromind-runtime/pyromind_runtime" / filename
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Compare):
                continue
            operands = [node.left, *node.comparators]
            discriminants = [
                value
                for value in operands
                if any(
                    key in ast.unparse(value).lower()
                    for key in ("role", "profile", "scenario")
                )
            ]
            literals = {
                value.value.lower()
                for value in operands
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            }
            assert not (discriminants and literals & names), (path, node.lineno)


def test_preparation_exposes_only_review_state_and_catalog():
    from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
    from pyromind_runtime.domain.preparation import PreparationState, PreparationView

    policy = ECHOMIND_PRODUCT_PROFILE.preparation_policy
    assert policy is not None
    assert set(type(policy).model_fields) == {"allowed_roles", "workflow_nodes"}
    for phase in ("collecting", "planning", "ready", "active"):
        state = PreparationState(phase=phase)
        view = PreparationView.from_state(state, policy)
        assert view.phase == phase and view.policy == policy
        assert set(view.model_dump()) == set(state.model_dump()) | {"policy"}
        assert (
            not {"policy", "facts", "missing", "evidence"} & state.model_dump().keys()
        )
