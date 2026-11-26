import os
import ssl
from unittest.mock import Mock

import certifi
from uvicorn.config import Config
from uvicorn.supervisors.watchfilesreload import FileFilter

from pyromind_agent_server import _configure_platform_ca_bundle
from pyromind_agent_server.__main__ import RELOAD_DIRS


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


def test_sdk_tls_preserves_configured_trust_and_recovers_an_empty_store(
    monkeypatch, tmp_path
) -> None:
    for key in (
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ):
        monkeypatch.delenv(key, raising=False)
    defaults = ssl.get_default_verify_paths()._replace(
        cafile=None, capath=str(tmp_path)
    )
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: defaults)
    context = Mock()
    context.cert_store_stats.return_value = {"x509_ca": 0}
    monkeypatch.setattr(ssl, "create_default_context", lambda: context)

    anchor = tmp_path / "12345678.0"
    anchor.touch()
    _configure_platform_ca_bundle()
    assert "SSL_CERT_FILE" not in os.environ
    anchor.unlink()
    _configure_platform_ca_bundle()
    assert os.environ["SSL_CERT_FILE"] == certifi.where()
    monkeypatch.setenv("SSL_CERT_FILE", "/private/company/ca.pem")
    _configure_platform_ca_bundle()
    assert os.environ["SSL_CERT_FILE"] == "/private/company/ca.pem"
