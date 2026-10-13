from __future__ import annotations

from pathlib import Path

import pytest
from pyromind_agent_server.bootstrap import agentgenome_home


@pytest.fixture(autouse=True)
def _clear_home_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTGENOME_HOME", raising=False)
    monkeypatch.delenv("workspace_dir", raising=False)
    monkeypatch.delenv("WORKSPACE_DIR", raising=False)


def test_default_home_lives_under_the_storage_agent_directory(
    tmp_path: Path,
) -> None:
    conversations = tmp_path / "conversations"

    assert agentgenome_home(conversations) == (
        tmp_path / ".pyromind-agent" / "agentgenome"
    )


def test_workspace_root_anchors_the_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("workspace_dir", str(tmp_path / "workspace"))

    assert agentgenome_home(tmp_path / "workspace" / "user" / "conversations") == (
        tmp_path / "workspace" / ".pyromind-agent" / "agentgenome"
    )


def test_configured_home_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTGENOME_HOME", str(tmp_path / "registry"))
    legacy = tmp_path / "agentgenome"
    legacy.mkdir()

    assert agentgenome_home(tmp_path / "conversations") == tmp_path / "registry"
    assert legacy.is_dir()


def test_legacy_registry_moves_under_the_agent_directory(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "agentgenome"
    (legacy / "assets").mkdir(parents=True)
    (legacy / "state.sqlite3").write_bytes(b"state")

    home = agentgenome_home(tmp_path / "conversations")

    assert (home / "state.sqlite3").read_bytes() == b"state"
    assert not legacy.exists()


def test_existing_registry_keeps_a_legacy_copy_untouched(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".pyromind-agent" / "agentgenome"
    home.mkdir(parents=True)
    legacy = tmp_path / "agentgenome"
    legacy.mkdir()

    assert agentgenome_home(tmp_path / "conversations") == home
    assert legacy.is_dir()
