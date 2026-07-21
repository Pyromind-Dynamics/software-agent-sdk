"""Tests for the preview_remote_dataset tool."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openhands.tools.pyromind_remote_dataset.definition import (
    PreviewRemoteDatasetAction,
    PreviewRemoteDatasetExecutor,
    PreviewRemoteDatasetObservation,
    PreviewRemoteDatasetTool,
    _detect_splits,
    _parse_content,
    _RemoteFile,
    _select_file,
)


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------


def test_parse_jsonl_content() -> None:
    lines = [
        json.dumps({"prompt": "hello", "response": "world"}),
        json.dumps({"prompt": "foo", "response": "bar"}),
    ]
    content = ("\n".join(lines) + "\n").encode("utf-8")
    result = _parse_content("data/train.jsonl", content, 10, truncated=False)
    assert result["num_rows"] == 2
    assert "prompt" in result["columns"]
    assert "response" in result["columns"]
    assert len(result["sample_rows"]) == 2


def test_parse_csv_content() -> None:
    content = b"name,age\nAlice,30\nBob,25\n"
    result = _parse_content("data/test.csv", content, 10, truncated=False)
    assert result["num_rows"] == 2
    assert "name" in result["columns"]
    assert "age" in result["columns"]
    assert len(result["sample_rows"]) == 2


def test_parse_text_content() -> None:
    content = b"line one\nline two\nline three\n"
    result = _parse_content("readme.txt", content, 10, truncated=False)
    assert result["num_rows"] == 3
    assert len(result["sample_rows"]) == 3


def test_parse_parquet_content() -> None:
    table = pa.table({"text": ["a", "b", "c"], "label": [1, 2, 3]})
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    content = buf.getvalue().to_pybytes()
    result = _parse_content("data/train.parquet", content, 10, truncated=False)
    assert result["num_rows"] == 3
    assert "text" in result["columns"]
    assert "label" in result["columns"]
    assert len(result["sample_rows"]) == 3


def test_parse_truncated_content() -> None:
    content = b'{"a": 1}\n{"a": 2}\n'
    result = _parse_content("data/train.jsonl", content, 10, truncated=True)
    assert result["num_rows"] is None
    assert result["previewed_rows"] == 2


def test_parse_json_list() -> None:
    data = [{"x": 1}, {"x": 2}]
    content = json.dumps(data).encode("utf-8")
    result = _parse_content("data.json", content, 10, truncated=False)
    assert result["num_rows"] == 2
    assert "x" in result["columns"]


def test_parse_json_object() -> None:
    data = {"key": "value", "num": 42}
    content = json.dumps(data).encode("utf-8")
    result = _parse_content("config.json", content, 10, truncated=False)
    assert result["num_rows"] == 1
    assert "key" in result["columns"]


def test_parse_invalid_json() -> None:
    content = b"{not valid json"
    result = _parse_content("bad.json", content, 10, truncated=False)
    assert result["num_rows"] is None
    assert result["preview_error"] is not None


# ---------------------------------------------------------------------------
# File selection tests
# ---------------------------------------------------------------------------


def test_select_file_no_split() -> None:
    files = [
        _RemoteFile("data/train.jsonl", 100),
        _RemoteFile("data/test.jsonl", 50),
    ]
    selected = _select_file(files, None)
    assert selected is not None
    assert selected.path == "data/train.jsonl"


def test_select_file_with_split() -> None:
    files = [
        _RemoteFile("data/train.jsonl", 100),
        _RemoteFile("data/test.jsonl", 50),
    ]
    selected = _select_file(files, "test")
    assert selected is not None
    assert selected.path == "data/test.jsonl"


def test_select_file_split_not_found() -> None:
    files = [
        _RemoteFile("data/train.jsonl", 100),
        _RemoteFile("data/test.jsonl", 50),
    ]
    selected = _select_file(files, "validation")
    assert selected is None


def test_select_file_prefers_data_over_text() -> None:
    files = [
        _RemoteFile("readme.txt", 10),
        _RemoteFile("data/train.csv", 100),
    ]
    selected = _select_file(files, None)
    assert selected is not None
    assert selected.path == "data/train.csv"


# ---------------------------------------------------------------------------
# Split detection tests
# ---------------------------------------------------------------------------


def test_detect_splits_from_path() -> None:
    files = ["data/train/0000.parquet", "data/test/0000.parquet"]
    splits = _detect_splits(files)
    assert "train" in splits
    assert "test" in splits


def test_detect_splits_from_filename() -> None:
    files = ["train.jsonl", "test.jsonl"]
    splits = _detect_splits(files)
    assert "train" in splits
    assert "test" in splits


def test_detect_splits_empty() -> None:
    splits = _detect_splits(["random/file.jsonl"])
    assert splits == []


# ---------------------------------------------------------------------------
# Action / Observation tests
# ---------------------------------------------------------------------------


def test_action_validation() -> None:
    action = PreviewRemoteDatasetAction(
        dataset_name="openai/gsm8k",
        source="huggingface",
    )
    assert action.n == 10
    assert action.split is None
    assert action.source == "huggingface"


def test_action_source_defaults_none() -> None:
    action = PreviewRemoteDatasetAction(
        dataset_name="openai/gsm8k",
    )
    assert action.source is None


def test_action_invalid_source() -> None:
    with pytest.raises(ValueError):
        PreviewRemoteDatasetAction(
            dataset_name="test",
            source="invalid",
        )


def test_action_invalid_n() -> None:
    with pytest.raises(ValueError):
        PreviewRemoteDatasetAction(
            dataset_name="test",
            source="huggingface",
            n=0,
        )
    with pytest.raises(ValueError):
        PreviewRemoteDatasetAction(
            dataset_name="test",
            source="huggingface",
            n=200,
        )


def test_observation_visualize() -> None:
    obs = PreviewRemoteDatasetObservation.from_text(
        text="test",
        dataset_name="test/dataset",
        source="huggingface",
        num_rows=100,
        columns=["a", "b"],
    )
    text = str(obs.visualize)
    assert "test/dataset" in text
    assert "100" in text


# ---------------------------------------------------------------------------
# Tool creation test
# ---------------------------------------------------------------------------


def test_tool_create() -> None:
    tools = PreviewRemoteDatasetTool.create()
    assert len(tools) == 1
    assert tools[0].name == "preview_remote_dataset"


def test_tool_create_unknown_param() -> None:
    with pytest.raises(ValueError, match="unknown params"):
        PreviewRemoteDatasetTool.create(bad_param=1)


# ---------------------------------------------------------------------------
# Executor integration tests (mocked)
# ---------------------------------------------------------------------------


def test_executor_empty_dataset_name() -> None:
    executor = PreviewRemoteDatasetExecutor()
    action = PreviewRemoteDatasetAction(
        dataset_name="",
        source="huggingface",
    )
    obs = executor(action)
    assert obs.is_error


def test_executor_huggingface_no_files(monkeypatch) -> None:
    class _FakeEntry:
        def __init__(self, path: str, size: int | None = None) -> None:
            self.path = path
            self.size = size

    def fake_list_repo_tree(repo_id, **kwargs):
        return [_FakeEntry("logo.png", 100)]

    monkeypatch.setattr(
        "huggingface_hub.HfApi.list_repo_tree",
        lambda self, *args, **kw: fake_list_repo_tree(*args, **kw),
    )
    executor = PreviewRemoteDatasetExecutor()
    action = PreviewRemoteDatasetAction(
        dataset_name="test/empty",
        source="huggingface",
    )
    obs = executor(action)
    assert not obs.is_error
    assert "no previewable data files" in obs.text.lower()


def test_executor_modelscope_api_error(monkeypatch) -> None:
    class _MockResponse:
        status_code = 500
        text = "server error"

        def json(self):
            return {}

    def fake_get(url, **kwargs):
        return _MockResponse()

    monkeypatch.setattr("httpx.get", fake_get)
    executor = PreviewRemoteDatasetExecutor()
    action = PreviewRemoteDatasetAction(
        dataset_name="test/error",
        source="modelscope",
    )
    obs = executor(action)
    assert obs.is_error
    assert "500" in obs.text


def test_executor_modelscope_success(monkeypatch) -> None:
    jsonl_content = (json.dumps({"prompt": "hi", "response": "hello"}) + "\n").encode(
        "utf-8"
    )

    class _MockResponse:
        status_code = 200
        text = ""
        content = jsonl_content

        def json(self):
            return {
                "Data": {
                    "Files": [
                        {"Path": "data/train.jsonl", "Size": 50},
                    ]
                }
            }

    def fake_get(url, **kwargs):
        return _MockResponse()

    def fake_stream(*args, **kwargs):
        class _StreamCtx:
            def __enter__(self):
                class _Resp:
                    status_code = 200

                    def read(self):
                        return jsonl_content

                return _Resp()

            def __exit__(self, *a):
                return None

        return _StreamCtx()

    monkeypatch.setattr("httpx.get", fake_get)
    monkeypatch.setattr("httpx.stream", fake_stream)

    executor = PreviewRemoteDatasetExecutor()
    action = PreviewRemoteDatasetAction(
        dataset_name="test/dataset",
        source="modelscope",
        n=5,
    )
    obs = executor(action)
    assert not obs.is_error
    assert obs.source == "modelscope"
    assert "prompt" in obs.columns
    assert len(obs.sample_rows) == 1


def test_executor_auto_detect_falls_back_to_modelscope(monkeypatch) -> None:
    """When source=None and HF fails, should fall back to ModelScope."""
    jsonl_content = (json.dumps({"prompt": "hi", "response": "hello"}) + "\n").encode(
        "utf-8"
    )

    class _MockResponse:
        status_code = 200
        text = ""
        content = jsonl_content

        def json(self):
            return {
                "Data": {
                    "Files": [
                        {"Path": "data/train.jsonl", "Size": 50},
                    ]
                }
            }

    def fake_get(url, **kwargs):
        return _MockResponse()

    def fake_stream(*args, **kwargs):
        class _StreamCtx:
            def __enter__(self):
                class _Resp:
                    status_code = 200

                    def read(self):
                        return jsonl_content

                return _Resp()

            def __exit__(self, *a):
                return None

        return _StreamCtx()

    def fake_list_repo_tree(repo_id, **kwargs):
        raise Exception("repo not found")

    monkeypatch.setattr(
        "huggingface_hub.HfApi.list_repo_tree",
        lambda self, *args, **kw: fake_list_repo_tree(*args, **kw),
    )
    monkeypatch.setattr("httpx.get", fake_get)
    monkeypatch.setattr("httpx.stream", fake_stream)

    executor = PreviewRemoteDatasetExecutor()
    action = PreviewRemoteDatasetAction(
        dataset_name="test/dataset",
    )
    obs = executor(action)
    assert not obs.is_error
    assert obs.source == "modelscope"
