from pathlib import Path

from pyromind_agent_server.__main__ import _reload_excludes
from uvicorn.config import Config
from uvicorn.supervisors.watchfilesreload import FileFilter


def test_reload_excludes_generated_workspace_paths(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    conversations = workspace / "conversations"
    project = workspace / "project"
    for path in (workspace, conversations, project):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("OH_CONVERSATIONS_PATH", str(conversations))
    monkeypatch.setenv("OH_WORKSPACE_PATH", str(project))

    assert _reload_excludes() == sorted(
        {
            str(Path(workspace).resolve()),
            str(Path(conversations).resolve()),
            str(Path(project).resolve()),
        }
    )


def test_uvicorn_reload_filter_ignores_generated_workflow(
    tmp_path, monkeypatch
) -> None:
    repository = tmp_path / "software-agent-sdk"
    workspace = repository / "workspace"
    generated = workspace / "conversations" / "conversation-1" / "workflow.py"
    source = repository / "pyromind-agent-server" / "server.py"
    generated.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    generated.write_text("workflow = None\n", encoding="utf-8")
    source.write_text("app = None\n", encoding="utf-8")
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace))
    config = Config(
        "example:app",
        reload=True,
        reload_dirs=[str(repository)],
        reload_excludes=_reload_excludes(),
    )
    file_filter = FileFilter(config)

    assert not file_filter(generated)
    assert file_filter(source)
