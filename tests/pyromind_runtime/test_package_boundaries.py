import ast
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pyromind_runtime.application.event_projection import ProductEventProjector
from pyromind_runtime.application.snapshot_projector import SnapshotProjector
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.events import HarnessEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot


ROOT = Path(__file__).resolve().parents[2]


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
    forbidden = ("fastapi", "openhands", "harness_adapter", "pyromind_agent_server")
    assert not any(name.startswith(forbidden) for name in imports)


def test_adapter_does_not_depend_on_pyromind_server() -> None:
    imports = _imports(ROOT / "harness-adapter" / "harness_adapter")
    assert not any(name.startswith("pyromind_agent_server") for name in imports)


def test_pyromind_start_scripts_use_composed_server_entrypoint() -> None:
    for script_name in ("start.sh", "start_inference.sh"):
        script = (ROOT / script_name).read_text(encoding="utf-8")
        assert "python -m pyromind_agent_server" in script
        assert "python -m openhands.agent_server" not in script


def test_inference_start_requires_os_sandbox_for_pi_terminal() -> None:
    script = (ROOT / "start_inference.sh").read_text(encoding="utf-8")

    assert 'export APP_ENV="${APP_ENV:-dev}"' in script
    assert 'export PYROMIND_PI_TERMINAL_BACKEND="os-sandbox"' in script


def test_pre_deployment_uses_os_sandbox_for_pi_terminal() -> None:
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
    assert environment["PYROMIND_PI_TERMINAL_BACKEND"] == "os-sandbox"


def test_product_image_defaults_to_os_sandbox_for_pi_terminal() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    product_stage = dockerfile.split("FROM base-image-minimal AS product", maxsplit=1)[
        1
    ]

    assert "ENV PYROMIND_PI_TERMINAL_BACKEND=os-sandbox" in product_stage


def test_local_startup_checks_platform_specific_sandbox_dependencies() -> None:
    script = (ROOT / "start_inference.sh").read_text(encoding="utf-8")

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
