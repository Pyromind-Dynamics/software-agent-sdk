from __future__ import annotations

import ast
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).parents[3]
_RUNTIME_ROOT = _REPOSITORY_ROOT / "openhands-agent-server" / "pyromind_runtime"
_FRONTEND_SOURCE = _REPOSITORY_ROOT.parent / "minimal-chat-frontend" / "src"


def test_openhands_imports_are_confined_to_the_openhands_adapter() -> None:
    violations: list[str] = []
    allowed_root = _RUNTIME_ROOT / "adapters" / "openhands"
    for path in _python_files(_RUNTIME_ROOT):
        if path.is_relative_to(allowed_root):
            continue
        for module in _imported_modules(path):
            if module == "openhands" or module.startswith("openhands."):
                violations.append(f"{path.relative_to(_REPOSITORY_ROOT)}: {module}")

    assert violations == []


def test_pi_adapter_imports_are_confined_to_adapter_composition() -> None:
    violations: list[str] = []
    pi_root = _RUNTIME_ROOT / "adapters" / "pi"
    neutral_roots = (
        _RUNTIME_ROOT / "contracts",
        _RUNTIME_ROOT / "product",
        _RUNTIME_ROOT / "projectors",
        _RUNTIME_ROOT / "tool_host",
    )
    for root in neutral_roots:
        for path in _python_files(root):
            for module in _imported_modules(path):
                if module == "pyromind_runtime.adapters.pi" or module.startswith(
                    "pyromind_runtime.adapters.pi."
                ):
                    violations.append(f"{path.relative_to(_REPOSITORY_ROOT)}: {module}")

    assert pi_root.is_dir()
    assert violations == []


def test_projectors_and_frontend_do_not_consume_provider_metadata() -> None:
    projector_violations = [
        str(path.relative_to(_REPOSITORY_ROOT))
        for path in _python_files(_RUNTIME_ROOT / "projectors")
        if "provider_metadata" in path.read_text(encoding="utf-8")
    ]
    frontend_violations = [
        str(path.relative_to(_REPOSITORY_ROOT.parent))
        for path in _frontend_files()
        if "provider_metadata" in path.read_text(encoding="utf-8")
    ]

    assert projector_violations == []
    assert frontend_violations == []


def test_frontend_has_no_openhands_event_or_websocket_dependency() -> None:
    banned_fragments = (
        "MessageEvent",
        "ActionEvent",
        "ObservationEvent",
        "StreamingDeltaEvent",
        "@openhands",
        "new WebSocket(",
    )
    violations: list[str] = []
    for path in _frontend_files():
        source = path.read_text(encoding="utf-8")
        for fragment in banned_fragments:
            if fragment in source:
                violations.append(
                    f"{path.relative_to(_REPOSITORY_ROOT.parent)}: {fragment}"
                )

    assert violations == []


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.py")))


def _frontend_files() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in _FRONTEND_SOURCE.rglob("*")
            if path.suffix in {".ts", ".tsx"}
        )
    )


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return tuple(modules)
