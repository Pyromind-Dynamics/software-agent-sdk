from pyromind_agent_server.__main__ import RELOAD_DIRS
from uvicorn.config import Config
from uvicorn.supervisors.watchfilesreload import FileFilter


def _file_filter(tmp_path):
    repository = tmp_path / "software-agent-sdk"
    repository.mkdir()
    for path in RELOAD_DIRS:
        (repository / path).mkdir(parents=True)
    config = Config(
        "example:app",
        reload=True,
        reload_dirs=[str(repository / path) for path in RELOAD_DIRS],
        reload_includes=["*.py"],
    )
    return repository, config, FileFilter(config)


def test_uvicorn_reload_filter_ignores_conversation_workflow(tmp_path) -> None:
    repository, config, _ = _file_filter(tmp_path)
    generated = (
        repository
        / "workspace"
        / "conversations"
        / "conversation-1"
        / "public_data"
        / "workflow_canvas"
        / "workflow.py"
    )
    generated.parent.mkdir(parents=True)
    generated.write_text("workflow = None\n", encoding="utf-8")

    watched = any(
        directory == generated or directory in generated.parents
        for directory in config.reload_dirs
    )
    assert not watched


def test_uvicorn_reload_filter_includes_all_python_source_dirs(tmp_path) -> None:
    repository, _, file_filter = _file_filter(tmp_path)

    for source_dir in RELOAD_DIRS:
        source = repository / source_dir / "server.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("app = None\n", encoding="utf-8")
        assert file_filter(source), source


def test_uvicorn_reload_filter_only_includes_python_files(tmp_path) -> None:
    repository, _, file_filter = _file_filter(tmp_path)
    source_dir = repository / "pyromind-agent-server"
    source_dir.mkdir(parents=True, exist_ok=True)

    python_source = source_dir / "server.py"
    python_source.write_text("app = None\n", encoding="utf-8")
    text_file = source_dir / "README.md"
    text_file.write_text("docs\n", encoding="utf-8")

    assert file_filter(python_source)
    assert not file_filter(text_file)
