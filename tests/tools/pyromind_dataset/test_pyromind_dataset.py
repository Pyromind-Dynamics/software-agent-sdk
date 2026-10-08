from pathlib import Path
from typing import Any, cast
from uuid import UUID

import httpx
from pydantic import SecretStr

from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.secret import StaticSecret
from openhands.tools.pyromind_archive.definition import (
    PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET,
)
from openhands.tools.pyromind_dataset.definition import (
    PYROMIND_STORAGE_AUTH_COOKIE_SECRET,
    PYROMIND_STORAGE_HEADERS_STATE_KEY,
    GetStorageUrlAction,
    GetStorageUrlExecutor,
    UploadFileToPyromindAction,
    UploadFileToPyromindExecutor,
    download_file_from_pyromind,
)


class _FakeWorkspace:
    def __init__(self, working_dir: Path) -> None:
        self.working_dir = str(working_dir)


class _Response:
    def __init__(self, status_code: int, payload: Any, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _StreamResponse:
    def __init__(self, content: bytes, *, status_code: int = 200) -> None:
        self._content = content
        self.status_code = status_code
        self.headers = {"content-length": str(len(content))}

    def __enter__(self) -> "_StreamResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def iter_bytes(self):
        yield self._content

    def read(self) -> bytes:
        return self._content


_CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000456")


def _fake_conversation(
    tmp_path: Path,
    *,
    secret_registry: SecretRegistry | None = None,
    agent_state: dict[str, Any] | None = None,
    workspace: object | None = None,
):
    return type(
        "FakeConversation",
        (),
        {
            "id": _CONVERSATION_ID,
            "workspace": (
                workspace if workspace is not None else _FakeWorkspace(tmp_path)
            ),
            "state": type(
                "FakeState",
                (),
                {
                    "secret_registry": secret_registry or SecretRegistry(),
                    "agent_state": agent_state or {},
                },
            )(),
        },
    )()


def _secret_registry() -> SecretRegistry:
    secret_registry = SecretRegistry()
    secret_registry.update_secrets(
        {
            PYROMIND_STORAGE_AUTH_COOKIE_SECRET: StaticSecret(
                value=SecretStr("auth_token=session-token")
            ),
            PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET: StaticSecret(
                value=SecretStr("session-token")
            ),
        }
    )
    return secret_registry


def test_upload_file_to_pyromind_posts_workspace_file(
    monkeypatch,
    tmp_path,
):
    local_file = tmp_path / "metric.py"
    local_file.write_text("def acc():\n    return 1\n", encoding="utf-8")
    calls: dict[str, Any] = {}

    def fake_post(url, *, headers, json, timeout):
        calls.update(
            {
                "url": url,
                "headers": headers,
                "body": json,
                "timeout": timeout,
            }
        )
        return _Response(
            200,
            {
                "success": True,
                "data": {
                    "multipart": False,
                    "upload_url": "https://upload.test/presigned",
                    "method": "PUT",
                    "headers": {},
                },
            },
        )

    def fake_request(method, url, *, content, headers, timeout):
        calls["upload_method"] = method
        calls["upload_url"] = url
        calls["content"] = content.read()
        calls["upload_timeout"] = timeout
        return _Response(200, {})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "request", fake_request)
    conversation = _fake_conversation(
        tmp_path,
        secret_registry=_secret_registry(),
        agent_state={PYROMIND_STORAGE_HEADERS_STATE_KEY: {"x-cluster": "pre"}},
    )

    observation = UploadFileToPyromindExecutor(
        storage_base_url="https://portal.test/storage_api",
        timeout=7.0,
    )(
        UploadFileToPyromindAction(file_path="metric.py"),
        cast(Any, conversation),
    )

    assert not observation.is_error
    assert observation.storage_path == (
        f"/.pyromind-agent/{_CONVERSATION_ID}/metric.py"
    )
    assert calls["url"] == "https://portal.test/storage_api/presigned_upload_url"
    assert calls["headers"]["cookie"] == "auth_token=session-token"
    assert calls["headers"]["x-cluster"] == "pre"
    assert calls["body"] == {
        "filename": "metric.py",
        "path": f"/.pyromind-agent/{_CONVERSATION_ID}",
        "content_type": "application/octet-stream",
        "size": len(b"def acc():\n    return 1\n"),
    }
    assert calls["upload_method"] == "PUT"
    assert calls["upload_url"] == "https://upload.test/presigned"
    assert calls["content"] == b"def acc():\n    return 1\n"
    assert calls["timeout"] == 7.0
    assert calls["upload_timeout"] == 7.0


def test_upload_file_to_pyromind_explicit_target_dir_wins(
    monkeypatch,
    tmp_path,
):
    local_file = tmp_path / "metric.py"
    local_file.write_text("def acc():\n    return 1\n", encoding="utf-8")
    posted_path: dict[str, str] = {}

    def fake_post(url, *, headers, json, timeout):
        posted_path["path"] = json["path"]
        return _Response(
            200,
            {
                "success": True,
                "data": {
                    "multipart": False,
                    "upload_url": "https://upload.test/presigned",
                    "method": "PUT",
                    "headers": {},
                },
            },
        )

    def fake_request(method, url, *, content, headers, timeout):
        content.read()
        return _Response(200, {})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "request", fake_request)
    conversation = _fake_conversation(tmp_path, secret_registry=_secret_registry())

    observation = UploadFileToPyromindExecutor(
        storage_base_url="https://portal.test/storage_api",
    )(
        UploadFileToPyromindAction(
            file_path="metric.py",
            target_dir="/custom/dir",
        ),
        cast(Any, conversation),
    )

    assert not observation.is_error
    assert observation.storage_path == "/custom/dir/metric.py"
    assert posted_path["path"] == "/custom/dir"


def test_upload_file_to_pyromind_warns_before_overwriting(
    monkeypatch,
    tmp_path,
):
    local_file = tmp_path / "metric.py"
    local_file.write_text("def acc():\n    return 1\n", encoding="utf-8")

    def fake_post(url, *, headers, json, timeout):
        if url.endswith("/file_list"):
            return _Response(
                200,
                {
                    "success": True,
                    "data": {
                        "list": [
                            {
                                "name": "metric.py",
                                "path": (
                                    f"/.pyromind-agent/{_CONVERSATION_ID}/metric.py"
                                ),
                            }
                        ]
                    },
                },
            )
        return _Response(
            200,
            {
                "success": True,
                "data": {
                    "multipart": False,
                    "upload_url": "https://upload.test/presigned",
                    "method": "PUT",
                    "headers": {},
                },
            },
        )

    def fake_request(method, url, *, content, headers, timeout):
        content.read()
        return _Response(200, {})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "request", fake_request)
    conversation = _fake_conversation(tmp_path, secret_registry=_secret_registry())

    observation = UploadFileToPyromindExecutor(
        storage_base_url="https://portal.test/storage_api",
    )(
        UploadFileToPyromindAction(file_path="metric.py"),
        cast(Any, conversation),
    )

    assert not observation.is_error, observation.text
    assert (
        "Warning: an existing file at that Storage path was overwritten."
        in observation.text
    )


def test_upload_file_to_pyromind_ignores_a_failing_overwrite_probe(
    monkeypatch,
    tmp_path,
):
    """The advisory listing never blocks the upload it guards."""
    local_file = tmp_path / "metric.py"
    local_file.write_text("def acc():\n    return 1\n", encoding="utf-8")

    def fake_post(url, *, headers, json, timeout):
        if url.endswith("/file_list"):
            raise httpx.ConnectError("listing unavailable")
        return _Response(
            200,
            {
                "success": True,
                "data": {
                    "multipart": False,
                    "upload_url": "https://upload.test/presigned",
                    "method": "PUT",
                    "headers": {},
                },
            },
        )

    def fake_request(method, url, *, content, headers, timeout):
        content.read()
        return _Response(200, {})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "request", fake_request)
    conversation = _fake_conversation(tmp_path, secret_registry=_secret_registry())

    observation = UploadFileToPyromindExecutor(
        storage_base_url="https://portal.test/storage_api",
    )(
        UploadFileToPyromindAction(file_path="metric.py"),
        cast(Any, conversation),
    )

    assert not observation.is_error, observation.text
    assert "Warning:" not in observation.text


def test_upload_file_to_pyromind_multipart_uploads_parts_and_completes(
    monkeypatch,
    tmp_path,
):
    local_file = tmp_path / "big.bin"
    content = b"a" * 50 + b"b" * 10
    local_file.write_bytes(content)
    put_calls: dict[str, bytes] = {}
    complete_calls: list[dict[str, Any]] = []

    def fake_post(url, *, headers, json, timeout):
        if url.endswith("/presigned_upload_url"):
            return _Response(
                200,
                {
                    "success": True,
                    "data": {
                        "multipart": True,
                        "upload_url": None,
                        "method": "PUT",
                        "headers": {},
                        "upload_id": "mp-123",
                        "part_size": 50,
                        "part_count": 2,
                        "part_urls": [
                            "https://upload.test/part1",
                            "https://upload.test/part2",
                        ],
                    },
                },
            )
        if url.endswith("/file_list"):
            return _Response(200, {"success": True, "data": {"list": []}})
        assert url.endswith("/multipart_complete")
        complete_calls.append(json)
        return _Response(200, {"success": True, "data": {"etag": "etag-1"}})

    def fake_request(method, url, *, content, headers, timeout):
        chunk = content if isinstance(content, bytes) else content.read()
        put_calls.setdefault(url, b"")
        put_calls[url] += chunk
        return _Response(200, {})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "request", fake_request)
    conversation = _fake_conversation(tmp_path, secret_registry=_secret_registry())

    observation = UploadFileToPyromindExecutor(
        storage_base_url="https://portal.test/storage_api",
    )(
        UploadFileToPyromindAction(file_path="big.bin"),
        cast(Any, conversation),
    )

    assert not observation.is_error
    assert observation.storage_path == (f"/.pyromind-agent/{_CONVERSATION_ID}/big.bin")
    assert put_calls == {
        "https://upload.test/part1": b"a" * 50,
        "https://upload.test/part2": b"b" * 10,
    }
    assert complete_calls == [
        {
            "path": f"/.pyromind-agent/{_CONVERSATION_ID}",
            "filename": "big.bin",
            "size": len(content),
            "upload_id": "mp-123",
            "part_count": 2,
        }
    ]


def test_download_file_from_pyromind_returns_bounded_script(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        assert json == {"path": "/agentTest/clean.py"}
        return _Response(
            200,
            {"success": True, "data": {"url": "https://download.test/script"}},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(
        httpx,
        "stream",
        lambda *args, **kwargs: _StreamResponse(b"def main():\n    return 0\n"),
    )

    content = download_file_from_pyromind(
        storage_path="/agentTest/clean.py",
        storage_base_url="https://portal.test/storage_api",
        headers={"cookie": "session"},
        timeout=3,
        max_bytes=1024,
    )

    assert content == b"def main():\n    return 0\n"


def test_upload_file_to_pyromind_rejects_workspace_escape(monkeypatch, tmp_path):
    def fake_post(url, *, headers, data, files, timeout):
        raise AssertionError("upload API should not be called")

    monkeypatch.setattr(httpx, "post", fake_post)
    outside = tmp_path.parent / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    conversation = _fake_conversation(tmp_path)

    observation = UploadFileToPyromindExecutor(
        storage_base_url="https://portal.test/storage_api"
    )(
        UploadFileToPyromindAction(file_path=str(outside)),
        cast(Any, conversation),
    )

    assert observation.is_error
    assert "outside the conversation workspace" in observation.text


def test_upload_file_to_pyromind_stages_remote_workspace_file(
    monkeypatch,
    sandbox_workspace,
) -> None:
    content = b"def acc():\n    return 1\n"
    local_file = sandbox_workspace.workspace_dir / "metric.py"
    local_file.write_bytes(content)
    uploaded: list[bytes] = []

    def fake_post(url, *, headers, json, timeout):
        return _Response(
            200,
            {
                "success": True,
                "data": {
                    "multipart": False,
                    "upload_url": "https://upload.test/presigned",
                    "method": "PUT",
                    "headers": {},
                },
            },
        )

    def fake_request(method, url, *, content, headers, timeout):
        uploaded.append(content.read())
        return _Response(200, {})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "request", fake_request)
    conversation = _fake_conversation(
        sandbox_workspace.workspace_dir,
        secret_registry=_secret_registry(),
        workspace=sandbox_workspace,
    )

    observation = UploadFileToPyromindExecutor(
        storage_base_url="https://portal.test/storage_api",
    )(
        UploadFileToPyromindAction(file_path="metric.py"),
        cast(Any, conversation),
    )

    assert not observation.is_error, observation.text
    assert uploaded == [content]


class _StorageWorkspace(_FakeWorkspace):
    """Workspace double that also exposes the sandbox Storage mount."""

    def __init__(
        self, working_dir: Path, storage_path: str = "/target-workspace"
    ) -> None:
        super().__init__(working_dir)
        self.storage_path = storage_path


def test_get_storage_url_maps_workspace_and_storage_paths(
    monkeypatch, tmp_path
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url, *, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _Response(
            200,
            {"success": True, "data": {"url": f"https://download.test{json['path']}"}},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    conversation = _fake_conversation(
        tmp_path,
        secret_registry=_secret_registry(),
        agent_state={PYROMIND_STORAGE_HEADERS_STATE_KEY: {"x-cluster": "pre"}},
        workspace=_StorageWorkspace(tmp_path),
    )

    observation = GetStorageUrlExecutor(
        storage_base_url="https://portal.test/storage_api",
        timeout=7.0,
    )(
        GetStorageUrlAction(
            paths=["public_data/report.html", "storage/datasets/img.png"]
        ),
        cast(Any, conversation),
    )

    workspace_path = f"/.pyromind-agent/{_CONVERSATION_ID}/public_data/report.html"
    assert not observation.is_error
    assert [(entry.path, entry.url) for entry in observation.urls] == [
        (workspace_path, f"https://download.test{workspace_path}"),
        ("/datasets/img.png", "https://download.test/datasets/img.png"),
    ]
    assert [call["json"] for call in calls] == [
        {"path": workspace_path, "force_download": False},
        {"path": "/datasets/img.png", "force_download": False},
    ]
    assert calls[0]["url"] == "https://portal.test/storage_api/get_url"
    assert calls[0]["headers"]["cookie"] == "auth_token=session-token"
    assert calls[0]["headers"]["x-cluster"] == "pre"
    assert calls[0]["timeout"] == 7.0


def test_get_storage_url_rejects_workspace_paths_without_a_mount(
    monkeypatch, tmp_path
) -> None:
    def fake_post(url, *, headers, json, timeout):
        raise AssertionError("no Storage request expected")

    monkeypatch.setattr(httpx, "post", fake_post)
    conversation = _fake_conversation(tmp_path)

    observation = GetStorageUrlExecutor(
        storage_base_url="https://portal.test/storage_api"
    )(
        GetStorageUrlAction(paths=["public_data/report.html"]),
        cast(Any, conversation),
    )

    assert observation.is_error
    assert "has no Storage mount" in observation.text


def test_get_storage_url_reports_per_path_failures(monkeypatch, tmp_path) -> None:
    def fake_post(url, *, headers, json, timeout):
        if json["path"].endswith("missing.png"):
            return _Response(404, {"success": False}, text="not found")
        return _Response(
            200, {"success": True, "data": {"url": "https://download.test/ok"}}
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    conversation = _fake_conversation(
        tmp_path,
        secret_registry=_secret_registry(),
        workspace=_StorageWorkspace(tmp_path),
    )

    observation = GetStorageUrlExecutor(
        storage_base_url="https://portal.test/storage_api"
    )(
        GetStorageUrlAction(paths=["storage/ok.png", "storage/missing.png"]),
        cast(Any, conversation),
    )

    assert not observation.is_error
    assert [entry.path for entry in observation.urls] == ["/ok.png"]
    assert observation.failures == [
        "storage/missing.png: Pyromind storage get_url API returned HTTP 404: not found"
    ]
    assert "--- failed ---" in observation.text
