"""Completed reports remain readable after the Sandbox record is reclaimed."""

import pytest
from harness_adapter.pi_adapter import PiAdapter, adapter as adapter_module
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.errors import ProductRuntimeError
from pyromind_runtime.ports.harness import SessionHandle


@pytest.fixture
def archived(tmp_path, monkeypatch):
    root = tmp_path / "conversations"
    files = PiSessionFiles(root / "conversation")
    files.initialize({"session_id": "conversation", "model": {"id": "test"}})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = PiAdapter(root, terminal_backend="sandbox")
    handle = SessionHandle(
        session_id="conversation",
        adapter_session_ref="conversation",
        harness_id="pi",
        capabilities=adapter_module.PI_CAPABILITIES,
    )
    return adapter, handle, files


async def test_archived_report_reads_storage_with_current_request_credentials(
    archived,
    monkeypatch,
):
    adapter, handle, files = archived
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return b"# Verified report\n"

    monkeypatch.setattr(adapter_module, "download_file_from_pyromind", download)
    for cookie in ("auth_token=first", "auth_token=renewed"):
        context = RequestContext(user_id="42", cookie=cookie, x_cluster="cluster#pre")
        assert (
            await adapter.read_workspace_text(
                handle,
                "public_data/run/report.md",
                context,
                max_bytes=2048,
            )
            == "# Verified report\n"
        )
        assert calls[-1]["headers"]["cookie"] == cookie
    assert calls[-1]["storage_path"] == (
        "/.pyromind-agent/conversation/public_data/run/report.md"
    )
    assert calls[-1]["headers"]["x-cluster"] == "cluster#pre"
    assert calls[-1]["max_bytes"] == 2048
    assert not adapter._sessions
    assert files.load_sandbox() is None
    assert "auth_token" not in files.session_path.read_text()


@pytest.mark.parametrize(
    "path",
    [
        "../other/report.md",
        "/other/report.md",
        "public_data/../../other",
        "public_data\\report.md",
    ],
)
async def test_archived_report_rejects_workspace_escape(archived, monkeypatch, path):
    adapter, handle, _ = archived

    def download(**kwargs):
        pytest.fail("invalid paths must not access Storage")

    monkeypatch.setattr(adapter_module, "download_file_from_pyromind", download)
    with pytest.raises(ValueError, match="workspace"):
        await adapter.read_workspace_text(
            handle, path, RequestContext(user_id="42", cookie="secret")
        )


@pytest.mark.parametrize("missing", [True, False])
async def test_storage_failure_is_not_misreported_as_a_missing_report(
    archived,
    monkeypatch,
    missing,
):
    adapter, handle, _ = archived

    def download(**kwargs):
        if missing:
            raise adapter_module.StorageFileNotFoundError("not found")
        raise ValueError("transport detail including sensitive URL")

    monkeypatch.setattr(adapter_module, "download_file_from_pyromind", download)
    context = RequestContext(user_id="42", cookie="secret")
    if missing:
        assert (
            await adapter.read_workspace_text(handle, "public_data/report.md", context)
            is None
        )
    else:
        with pytest.raises(ProductRuntimeError, match="check access") as error:
            await adapter.read_workspace_text(handle, "public_data/report.md", context)
        assert "sensitive" not in str(error.value)
