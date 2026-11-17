import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def version_guard(tmp_path, monkeypatch):
    script = (
        Path(__file__).resolve().parents[2] / ".github/scripts/check_version_bumps.py"
    )
    spec = importlib.util.spec_from_file_location("release_version_guard", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "PACKAGE_PYPROJECTS",
        {
            "existing": Path("existing/pyproject.toml"),
            "new": Path("new/pyproject.toml"),
        },
    )
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    existing = tmp_path / "existing/pyproject.toml"
    existing.parent.mkdir()
    existing.write_text('[project]\nversion = "1.0.0"\n')
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-qm",
            "base",
        ],
        cwd=tmp_path,
        check=True,
    )
    introduced = tmp_path / "new/pyproject.toml"
    introduced.parent.mkdir()
    introduced.write_text('[project]\nversion = "0.1.0"\n')
    return module, tmp_path


def test_version_guard_accepts_introduced_package(version_guard):
    module, root = version_guard
    assert module.find_version_changes(root, "HEAD") == []


def test_version_guard_still_detects_existing_package_bump(version_guard):
    module, root = version_guard
    (root / "existing/pyproject.toml").write_text('[project]\nversion = "1.1.0"\n')
    changes = module.find_version_changes(root, "HEAD")
    assert [
        (change.package, change.previous_version, change.current_version)
        for change in changes
    ] == [("existing", "1.0.0", "1.1.0")]


def test_version_guard_does_not_ignore_invalid_base(version_guard):
    module, root = version_guard
    with pytest.raises(SystemExit, match="Unable to resolve git ref"):
        module.find_version_changes(root, "missing-base")
