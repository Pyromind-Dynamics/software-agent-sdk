import json as jsonlib
from pathlib import Path
from typing import Any, cast

import httpx
from pydantic import SecretStr

from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.secret import StaticSecret
from openhands.tools.pyromind_dataset.definition import (
    _PREVIEW_DATASET_DESCRIPTION,
    PYROMIND_STORAGE_AUTH_COOKIE_SECRET,
    PYROMIND_STORAGE_HEADERS_STATE_KEY,
    PreviewDatasetAction,
    PreviewDatasetExecutor,
    UploadFileToPyromindAction,
    UploadFileToPyromindExecutor,
    _with_cleaned_dataset_hint,
)


def test_preview_description_excludes_cleaned_dataset_identifiers() -> None:
    assert "Do not" in _PREVIEW_DATASET_DESCRIPTION
    assert "pyromind/self-cognition" in _PREVIEW_DATASET_DESCRIPTION
    assert "must not be retried" in _PREVIEW_DATASET_DESCRIPTION
    assert "uploaded the data" in _PREVIEW_DATASET_DESCRIPTION
    assert "storage-relative path" in _PREVIEW_DATASET_DESCRIPTION


def test_preview_404_hints_when_path_is_cleaned_dataset_identifier() -> None:
    message = _with_cleaned_dataset_hint(
        "Pyromind storage get_file_metadata API returned HTTP 404",
        "pyromind/self-cognition",
    )
    assert "cleaned dataset identifier" in message
    assert "do not retry" in message


def test_preview_rejects_known_dataset_ids_without_storage_request(monkeypatch):
    def unexpected_post(*args, **kwargs):
        raise AssertionError("storage API must not be called for dataset IDs")

    monkeypatch.setattr(httpx, "post", unexpected_post)
    expected_nodes = {
        "pyromind/self-cognition": "CloneAndCacheDataset",
        "pyromind/geometry-vqa-vlm-demo": "CloneAndCacheDataset",
        "pyromind/easyhard-24k": "DownloadAndCacheDataset",
    }

    for dataset_id, expected_node in expected_nodes.items():
        observation = PreviewDatasetExecutor()(
            PreviewDatasetAction(dataset_path=dataset_id)
        )

        assert observation.is_error
        assert observation.error_code == "NOT_A_STORAGE_PATH"
        assert observation.suggested_node == expected_node
        assert "do not retry" in observation.text


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


def _fake_conversation(
    tmp_path: Path,
    *,
    secret_registry: SecretRegistry | None = None,
    agent_state: dict[str, Any] | None = None,
):
    return type(
        "FakeConversation",
        (),
        {
            "workspace": _FakeWorkspace(tmp_path),
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
            )
        }
    )
    return secret_registry


def test_preview_dataset_reads_jsonl_samples_with_storage_context(
    monkeypatch,
    tmp_path,
):
    calls: list[dict[str, Any]] = []
    jsonl = b'{"prompt":"p1","completion":"c1"}\n{"prompt":"p2","completion":"c2"}\n'

    def fake_post(url, *, headers, json, timeout):
        calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        if url.endswith("/get_file_metadata"):
            return _Response(
                200,
                {
                    "success": True,
                    "data": {
                        "object_name": "datasets/train.jsonl",
                        "bucket_name": "1001",
                        "size": len(jsonl),
                        "content_type": "application/jsonl",
                        "is_dir": False,
                        "metadata": {},
                    },
                },
            )
        if url.endswith("/get_url"):
            return _Response(
                200,
                {"success": True, "data": {"url": "https://download.test/train"}},
            )
        raise AssertionError(f"unexpected URL: {url}")

    def fake_stream(method, url, *, headers, timeout, follow_redirects):
        assert method == "GET"
        assert url == "https://download.test/train"
        assert headers["range"] == "bytes=0-102399"
        assert timeout == 5.0
        assert follow_redirects is True
        return _StreamResponse(jsonl)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "stream", fake_stream)
    conversation = _fake_conversation(
        tmp_path,
        secret_registry=_secret_registry(),
        agent_state={PYROMIND_STORAGE_HEADERS_STATE_KEY: {"x-cluster": "pre"}},
    )

    observation = PreviewDatasetExecutor(
        storage_base_url="https://portal.test/storage_api",
        timeout=5.0,
    )(
        PreviewDatasetAction.model_validate(
            {"dataset_path": "datasets/train.jsonl", "max_samples": 1}
        ),
        cast(Any, conversation),
    )

    assert not observation.is_error
    assert observation.num_rows == 2
    assert observation.previewed_rows == 2
    assert observation.preview_truncated is False
    assert observation.columns == ["prompt", "completion"]
    assert observation.sample_rows == [{"prompt": "p1", "completion": "c1"}]
    assert observation.sample_file_path is not None
    sample_file = Path(observation.sample_file_path)
    assert sample_file.is_file()
    assert sample_file.parent == tmp_path / "preview_dataset"
    assert sample_file.name.startswith("train.jsonl-sample-")
    assert jsonlib.loads(sample_file.read_text(encoding="utf-8")) == [
        {"prompt": "p1", "completion": "c1"}
    ]
    assert observation.sample_size == len(sample_file.read_bytes())
    assert f"sample_file_path={sample_file}" in observation.text
    assert "sample_rows:" not in observation.text
    assert calls[0]["headers"]["cookie"] == "auth_token=session-token"
    assert calls[0]["headers"]["x-cluster"] == "pre"


def test_preview_dataset_formats_text_file_content(
    monkeypatch,
    tmp_path,
):
    lines = [f"echo line-{index}" for index in range(1, 19)]
    content = ("\n".join(lines) + "\n").encode()

    def fake_post(url, *, headers, json, timeout):
        if url.endswith("/get_file_metadata"):
            return _Response(
                200,
                {
                    "success": True,
                    "data": {
                        "object_name": "start-hook.sh",
                        "size": len(content),
                        "content_type": "",
                        "is_dir": False,
                    },
                },
            )
        if url.endswith("/get_url"):
            return _Response(
                200,
                {"success": True, "data": {"url": "https://download.test/hook"}},
            )
        raise AssertionError(f"unexpected URL: {url}")

    def fake_stream(method, url, *, headers, timeout, follow_redirects):
        assert method == "GET"
        assert url == "https://download.test/hook"
        return _StreamResponse(content)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "stream", fake_stream)
    conversation = _fake_conversation(tmp_path)

    observation = PreviewDatasetExecutor(
        storage_base_url="https://portal.test/storage_api",
    )(
        PreviewDatasetAction(dataset_path="/start-hook.sh", n=20),
        cast(Any, conversation),
    )

    assert not observation.is_error
    assert observation.num_rows == 18
    assert observation.previewed_rows == 18
    assert len(observation.sample_rows) == 18
    assert observation.sample_file_path is not None
    sample_file = Path(observation.sample_file_path)
    assert sample_file.parent == tmp_path / "preview_dataset"
    assert sample_file.name.startswith("start-hook.sh-sample-")
    assert sample_file.read_text(encoding="utf-8").splitlines() == lines
    assert f"sample_file_path={sample_file}" in observation.text
    assert "sample_text:" not in observation.text


def test_preview_dataset_defaults_to_ten_sample_rows(
    monkeypatch,
    tmp_path,
):
    lines = [f"line-{index}" for index in range(1, 19)]
    content = ("\n".join(lines) + "\n").encode()

    def fake_post(url, *, headers, json, timeout):
        if url.endswith("/get_file_metadata"):
            return _Response(
                200,
                {
                    "success": True,
                    "data": {
                        "object_name": "small.txt",
                        "size": len(content),
                        "content_type": "text/plain",
                        "is_dir": False,
                    },
                },
            )
        if url.endswith("/get_url"):
            return _Response(
                200,
                {"success": True, "data": {"url": "https://download.test/small"}},
            )
        raise AssertionError(f"unexpected URL: {url}")

    def fake_stream(method, url, *, headers, timeout, follow_redirects):
        return _StreamResponse(content)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "stream", fake_stream)
    conversation = _fake_conversation(tmp_path)
    (tmp_path / "preview_dataset").mkdir()

    observation = PreviewDatasetExecutor(
        storage_base_url="https://portal.test/storage_api",
    )(PreviewDatasetAction(dataset_path="/small.txt"), cast(Any, conversation))

    assert not observation.is_error
    assert observation.num_rows == 18
    assert len(observation.sample_rows) == 10
    assert observation.sample_file_path is not None
    sample_file = Path(observation.sample_file_path)
    assert sample_file.parent == tmp_path / "preview_dataset"
    assert sample_file.read_text(encoding="utf-8").splitlines() == lines[:10]
    assert "rows=18" in observation.text
    assert "sample_rows=10" in observation.text


def test_preview_dataset_tail_strategy_samples_last_rows(
    monkeypatch,
    tmp_path,
):
    lines = [f"line-{index}" for index in range(1, 19)]
    content = ("\n".join(lines) + "\n").encode()

    def fake_post(url, *, headers, json, timeout):
        if url.endswith("/get_file_metadata"):
            return _Response(
                200,
                {
                    "success": True,
                    "data": {
                        "object_name": "small.txt",
                        "size": len(content),
                        "content_type": "text/plain",
                        "is_dir": False,
                    },
                },
            )
        if url.endswith("/get_url"):
            return _Response(
                200,
                {"success": True, "data": {"url": "https://download.test/small"}},
            )
        raise AssertionError(f"unexpected URL: {url}")

    def fake_stream(method, url, *, headers, timeout, follow_redirects):
        return _StreamResponse(content)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "stream", fake_stream)
    conversation = _fake_conversation(tmp_path)

    observation = PreviewDatasetExecutor(
        storage_base_url="https://portal.test/storage_api",
    )(
        PreviewDatasetAction(
            dataset_path="/small.txt",
            n=5,
            sample_strategy="tail",
        ),
        cast(Any, conversation),
    )

    assert not observation.is_error
    assert [row["text"] for row in observation.sample_rows] == lines[-5:]
    assert observation.sample_file_path is not None
    sample_file = Path(observation.sample_file_path)
    assert sample_file.read_text(encoding="utf-8").splitlines() == lines[-5:]
    assert "sample_strategy=tail" in observation.text


def test_preview_dataset_stratified_large_file_uses_multiple_ranges(
    monkeypatch,
    tmp_path,
):
    lines = [f"line-{index:04d} {'x' * 120}" for index in range(1, 500)]
    content = ("\n".join(lines) + "\n").encode()
    ranges: list[tuple[int, int]] = []

    def fake_post(url, *, headers, json, timeout):
        if url.endswith("/get_file_metadata"):
            return _Response(
                200,
                {
                    "success": True,
                    "data": {
                        "object_name": "large.txt",
                        "size": len(content),
                        "content_type": "text/plain",
                        "is_dir": False,
                    },
                },
            )
        if url.endswith("/get_url"):
            return _Response(
                200,
                {"success": True, "data": {"url": "https://download.test/large"}},
            )
        raise AssertionError(f"unexpected URL: {url}")

    def fake_stream(method, url, *, headers, timeout, follow_redirects):
        raw_range = headers["range"].removeprefix("bytes=")
        start, end = [int(part) for part in raw_range.split("-", maxsplit=1)]
        ranges.append((start, end))
        return _StreamResponse(content[start : end + 1])

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "stream", fake_stream)
    conversation = _fake_conversation(tmp_path)

    observation = PreviewDatasetExecutor(
        storage_base_url="https://portal.test/storage_api",
        max_preview_bytes=20 * 1024,
    )(
        PreviewDatasetAction(
            dataset_path="/large.txt",
            n=8,
            sample_strategy="stratified",
        ),
        cast(Any, conversation),
    )

    assert not observation.is_error
    assert observation.preview_truncated is True
    assert len(ranges) > 1
    assert any(start > 0 for start, _ in ranges)
    assert 0 < len(observation.sample_rows) <= 8
    sampled_numbers = [
        int(str(row["text"]).split()[0].removeprefix("line-"))
        for row in observation.sample_rows
    ]
    assert max(sampled_numbers) > 100
    assert observation.sample_file_path is not None
    sample_file = Path(observation.sample_file_path)
    assert len(sample_file.read_bytes()) <= 20 * 1024
    assert "sample_strategy=stratified" in observation.text


def test_preview_dataset_truncates_large_jsonl_and_reduces_samples(
    monkeypatch,
    tmp_path,
):
    rows = [f'{{"i":{i}}}\n'.encode() for i in range(10)]
    content = b"".join(rows)

    def fake_post(url, *, headers, json, timeout):
        if url.endswith("/get_file_metadata"):
            return _Response(
                200,
                {
                    "success": True,
                    "data": {
                        "object_name": "big.jsonl",
                        "size": len(content),
                        "content_type": "application/jsonl",
                        "is_dir": False,
                    },
                },
            )
        return _Response(200, {"success": True, "data": {"url": "https://download"}})

    def fake_stream(method, url, *, headers, timeout, follow_redirects):
        return _StreamResponse(content)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "stream", fake_stream)
    conversation = _fake_conversation(tmp_path)

    observation = PreviewDatasetExecutor(
        storage_base_url="https://portal.test/storage_api",
        max_preview_bytes=len(rows[0]) * 5 + 2,
    )(
        PreviewDatasetAction.model_validate(
            {"dataset_path": "big.jsonl", "max_samples": 10}
        ),
        cast(Any, conversation),
    )

    assert not observation.is_error
    assert observation.preview_truncated is True
    assert observation.num_rows is None
    assert observation.previewed_rows == 5
    assert 0 < len(observation.sample_rows) <= 10
    assert observation.sample_file_path is not None
    sample_file = Path(observation.sample_file_path)
    assert sample_file.is_file()
    assert len(sample_file.read_bytes()) <= len(rows[0]) * 5 + 2
    assert "sample_strategy=head" in observation.text


def test_upload_file_to_pyromind_posts_workspace_file(
    monkeypatch,
    tmp_path,
):
    local_file = tmp_path / "metric.py"
    local_file.write_text("def acc():\n    return 1\n", encoding="utf-8")
    calls: dict[str, Any] = {}

    def fake_post(url, *, headers, data, files, timeout):
        uploaded_file = files["file"]
        calls.update(
            {
                "url": url,
                "headers": headers,
                "data": data,
                "filename": uploaded_file[0],
                "content": uploaded_file[1].read(),
                "timeout": timeout,
            }
        )
        return _Response(
            200,
            {
                "success": True,
                "data": {
                    "uploaded": True,
                    "success_count": 1,
                    "failed_count": 0,
                    "success_files": [{"filename": "metric.py"}],
                    "failed_files": [],
                },
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
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
    assert observation.storage_path == "/agentTest/metric.py"
    assert calls["url"] == "https://portal.test/storage_api/upload_file"
    assert calls["headers"]["cookie"] == "auth_token=session-token"
    assert calls["headers"]["x-cluster"] == "pre"
    assert calls["data"]["path"] == "/agentTest"
    assert calls["filename"] == "metric.py"
    assert calls["content"] == b"def acc():\n    return 1\n"
    assert calls["timeout"] == 7.0


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


def test_preview_dataset_reports_invalid_json(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return _Response(200, jsonlib.JSONDecodeError("bad json", doc="{", pos=0))

    monkeypatch.setattr(httpx, "post", fake_post)

    observation = PreviewDatasetExecutor(
        storage_base_url="https://portal.test/storage_api"
    )(PreviewDatasetAction(dataset_path="data.jsonl"))

    assert observation.is_error
    assert "invalid JSON" in observation.text
