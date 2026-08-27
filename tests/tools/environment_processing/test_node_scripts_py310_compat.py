"""Platform nodes run Python 3.10; skill scripts must stay compatible.

Regression guard (conversation ebda2d49): render_manifest.py used
``from datetime import UTC`` (3.11+) and the render task died at import time
on the node, wedging the whole edp_render stage until a human fixed the
read-only skill directory.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / ".agents"
    / "skills"
    / "environment-data-processing"
    / "scripts"
)

# Modules/symbols that only exist on Python 3.11+ / 3.12+.
_FORBIDDEN_IMPORTS = {
    "datetime": {"UTC"},
    "typing": {"Self"},
    "tomllib": {None},
    "itertools": {"batched"},
}


def _node_scripts() -> list[Path]:
    scripts = sorted(_SCRIPTS.glob("*.py"))
    assert scripts, f"no skill scripts found under {_SCRIPTS}"
    return scripts


@pytest.mark.parametrize("script", _node_scripts(), ids=lambda p: p.name)
def test_skill_script_is_py310_compatible(script: Path) -> None:
    tree = ast.parse(script.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _FORBIDDEN_IMPORTS:
            names = {alias.name for alias in node.names}
            forbidden = _FORBIDDEN_IMPORTS[node.module] & names
            if forbidden:
                bad = sorted(forbidden)[0]
                pytest.fail(
                    f"{script.name}: `from {node.module} import {bad}` "
                    "requires Python 3.11+; nodes run 3.10 "
                    "(use datetime.timezone.utc etc.)"
                )
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _FORBIDDEN_IMPORTS:
                    pytest.fail(
                        f"{script.name}: `import {alias.name}` requires "
                        "Python 3.11+; nodes run 3.10"
                    )
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "UTC"
            and isinstance(node.value, ast.Name)
            and node.value.id == "datetime"
        ):
            pytest.fail(
                f"{script.name}: `datetime.UTC` requires Python 3.11+; "
                "use `datetime.timezone.utc`"
            )
