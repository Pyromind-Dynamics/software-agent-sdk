"""Tests for AVI Train and Label Studio converters."""

import json
from unittest.mock import MagicMock

import httpx
import pytest

from openhands.tools.label_studio.converter import (
    AVITrainToLabelStudioConverter,
    ConversionError,
    LabelStudioToAVITrainConverter,
)


SAMPLE_META = {
    "sample_id": "sample_001",
    "quality": "defect",
    "score": 0.95,
    "findings": [
        {
            "category": "开路",
            "observation": "线路断开",
            "bbox": {
                "x_min_norm": 100,
                "y_min_norm": 200,
                "x_max_norm": 400,
                "y_max_norm": 500,
            },
        }
    ],
}


def _mock_converter(monkeypatch, sample_dirs, meta=SAMPLE_META):
    converter = AVITrainToLabelStudioConverter(
        dataset_path="/datasets/pcb-001",
        storage_base_url="http://storage.example.com",
        storage_headers={},
    )
    monkeypatch.setattr(converter, "_list_sample_dirs", lambda: sample_dirs)
    monkeypatch.setattr(converter, "_read_json_file", lambda path: meta)
    monkeypatch.setattr(
        converter,
        "_resolve_media_urls",
        lambda paths: {path: f"https://media/{path}?token=x" for path in paths},
    )
    return converter


def test_convert_builds_tasks_with_predictions(monkeypatch):
    converter = _mock_converter(monkeypatch, ["/datasets/pcb-001/sample_001"])
    manifest = converter.convert()
    assert manifest.total_tasks == 1
    path, payload, count = manifest.batch_payloads[0]
    assert path == "tasks-00001.json"
    assert count == 1
    tasks = json.loads(payload)
    task = tasks[0]
    assert task["data"]["sample_id"] == "sample_001"
    predictions = task["predictions"][0]
    results = predictions["result"]
    choice_result = next(r for r in results if r["type"] == "choices")
    assert choice_result["value"]["choices"] == ["defect"]
    bbox_result = next(r for r in results if r["type"] == "rectanglelabels")
    assert bbox_result["id"] == "finding_1"
    assert bbox_result["value"]["x"] == 10.0
    assert bbox_result["value"]["width"] == 30.0
    assert bbox_result["value"]["rectanglelabels"] == ["开路"]
    text_result = next(r for r in results if r["type"] == "textarea")
    assert text_result["id"] == "finding_1"
    assert text_result["value"]["text"] == ["线路断开"]


def test_avi_quality_synonyms_are_normalised(monkeypatch):
    """Upstream spellings such as NG/PASS must land on the config's own choices.

    Label Studio does not render a prediction whose choice is absent from the
    <Choice> list, and it raises no error while doing so -- the pre-annotation
    just disappears, taking the verdict with it.
    """
    cases = (("NG", "defect"), ("PASS", "ok"), ("Good", "ok"), ("bad", "defect"))
    for raw, expected in cases:
        meta = dict(SAMPLE_META, quality=raw)
        converter = _mock_converter(
            monkeypatch, ["/datasets/pcb-001/sample_001"], meta=meta
        )
        manifest = converter.convert()
        tasks = json.loads(manifest.batch_payloads[0][1])
        results = tasks[0]["predictions"][0]["result"]
        choice = next(r for r in results if r["type"] == "choices")
        assert choice["value"]["choices"] == [expected], raw
        assert manifest.unmapped_quality == (), raw


def test_unknown_avi_quality_is_kept_and_recorded(monkeypatch):
    """An unrecognised avi verdict is our own, so keep it -- but make it visible.

    Dropping it would discard what our pipeline produced; keeping it silently
    would hide that the pre-annotation cannot render. So: keep, and record it in
    the manifest and on the observation.
    """
    meta = dict(SAMPLE_META, quality="weird-state")
    converter = _mock_converter(
        monkeypatch, ["/datasets/pcb-001/sample_001"], meta=meta
    )
    manifest = converter.convert()
    tasks = json.loads(manifest.batch_payloads[0][1])
    results = tasks[0]["predictions"][0]["result"]
    choice = next(r for r in results if r["type"] == "choices")
    assert choice["value"]["choices"] == ["weird-state"]
    assert manifest.unmapped_quality == ("weird-state",)
    assert manifest.to_manifest_data().unmapped_quality == ["weird-state"]


def test_unmapped_avi_quality_is_reported_once(monkeypatch):
    """The whole point of the ledger is a short list of distinct spellings."""
    meta = dict(SAMPLE_META, quality="weird-state")
    converter = _mock_converter(
        monkeypatch,
        ["/datasets/pcb-001/s1", "/datasets/pcb-001/s2"],
        meta=meta,
    )
    manifest = converter.convert()
    assert manifest.total_tasks == 2
    assert manifest.unmapped_quality == ("weird-state",)


def test_convert_empty_dataset_raises(monkeypatch):
    converter = _mock_converter(monkeypatch, [])
    with pytest.raises(ConversionError, match="No sample directories"):
        converter.convert()


def test_convert_skips_invalid_meta(monkeypatch):
    converter = AVITrainToLabelStudioConverter(
        dataset_path="/datasets/pcb-001",
        storage_base_url="http://storage.example.com",
        storage_headers={},
    )
    monkeypatch.setattr(
        converter, "_list_sample_dirs", lambda: ["/datasets/pcb-001/sample_001"]
    )
    monkeypatch.setattr(converter, "_read_json_file", lambda path: None)
    monkeypatch.setattr(
        converter,
        "_resolve_media_urls",
        lambda paths: {path: f"https://media/{path}?token=x" for path in paths},
    )
    with pytest.raises(ConversionError, match="skipped"):
        converter.convert()


def test_convert_signs_media_urls(monkeypatch):
    converter = _mock_converter(monkeypatch, ["/datasets/pcb-001/sample_001"])
    requested_paths: list[list[str]] = []

    def fake_resolve(paths: list[str]) -> dict[str, str]:
        requested_paths.append(paths)
        return {
            path: f"https://media.example.com/{path}?media_token=x" for path in paths
        }

    monkeypatch.setattr(converter, "_resolve_media_urls", fake_resolve)
    manifest = converter.convert()
    tasks = json.loads(manifest.batch_payloads[0][1])
    url = tasks[0]["data"]["defect_image"]
    assert "media_token=x" in url
    assert requested_paths == [
        [
            "/datasets/pcb-001/sample_001/defect.jpg",
            "/datasets/pcb-001/sample_001/diff.jpg",
            "/datasets/pcb-001/sample_001/gt.jpg",
        ]
    ]


def test_resolve_media_urls_signs_each_path(monkeypatch):
    converter = AVITrainToLabelStudioConverter(
        dataset_path="/datasets/pcb-001",
        storage_base_url="http://storage.example.com",
        storage_headers={},
    )
    requested: list[tuple[str, dict]] = []

    def fake_storage_post(route: str, body: dict) -> dict:
        requested.append((route, dict(body)))
        return {"data": {"url": f"https://media/{body['path']}?sig=abc"}}

    monkeypatch.setattr(converter, "_storage_post", fake_storage_post)
    paths = ["/datasets/pcb-001/s1/defect.jpg", "/datasets/pcb-001/s1/gt.jpg"]
    urls = converter._resolve_media_urls(paths)
    assert [route for route, _ in requested] == ["get_url", "get_url"]
    assert [body["path"] for _, body in requested] == paths
    assert urls == {p: f"https://media/{p}?sig=abc" for p in paths}


def test_resolve_media_urls_missing_url_raises(monkeypatch):
    converter = AVITrainToLabelStudioConverter(
        dataset_path="/datasets/pcb-001",
        storage_base_url="http://storage.example.com",
        storage_headers={},
    )
    monkeypatch.setattr(converter, "_storage_post", lambda route, body: {"data": {}})
    with pytest.raises(ConversionError, match="returned no url"):
        converter._resolve_media_urls(["/datasets/pcb-001/s1/defect.jpg"])


def test_batch_splitting(monkeypatch):
    converter = _mock_converter(monkeypatch, ["s1", "s2", "s3"])
    converter._batch_task_limit = 2
    manifest = converter.convert()
    assert len(manifest.batch_payloads) == 2
    assert manifest.batch_payloads[0][2] == 2
    assert manifest.batch_payloads[1][2] == 1


AOI_META = {
    "id": "10/B1",
    "label": "skip",
    "note": "对比参考图，差异仅为边缘对齐伪影，无真实物理缺陷。",
    "vlm_verdict": "false_positive",
    "vlm_category": "其他",
    "vlm_confidence": 0.95,
}


def _aoi_converter(monkeypatch, sample_dirs, metas):
    converter = AVITrainToLabelStudioConverter(
        dataset_path="/datasets/aoi-001",
        storage_base_url="http://storage.example.com",
        storage_headers={},
        adapter="aoi_export",
    )
    requested: list[str] = []

    def fake_read(path: str):
        requested.append(path)
        return metas.get(path.rsplit("/", 1)[0])

    monkeypatch.setattr(converter, "_list_sample_dirs", lambda: sample_dirs)
    monkeypatch.setattr(converter, "_read_json_file", fake_read)
    monkeypatch.setattr(
        converter,
        "_resolve_media_urls",
        lambda paths: {path: f"https://media/{path}?token=x" for path in paths},
    )
    return converter, requested


def test_aoi_export_maps_verdict_note_and_score(monkeypatch):
    converter, requested = _aoi_converter(
        monkeypatch,
        ["/datasets/aoi-001/10_B1"],
        {"/datasets/aoi-001/10_B1": dict(AOI_META)},
    )
    manifest = converter.convert()
    assert requested == ["/datasets/aoi-001/10_B1/meta.json"]
    assert manifest.converter_name == "aoi_export"
    task = json.loads(manifest.batch_payloads[0][1])[0]
    prediction = task["predictions"][0]
    results = prediction["result"]
    choice = next(r for r in results if r["type"] == "choices")
    assert choice["from_name"] == "quality_label"
    assert choice["value"]["choices"] == ["ok"]
    note = next(r for r in results if r["type"] == "textarea")
    assert note["from_name"] == "overall_note"
    assert note["value"]["text"] == [AOI_META["note"]]
    assert prediction["score"] == 0.95


def test_aoi_export_defect_verdict_maps_to_defect(monkeypatch):
    meta = dict(AOI_META, vlm_verdict="true", note="")
    converter, _ = _aoi_converter(
        monkeypatch,
        ["/datasets/aoi-001/10_B1"],
        {"/datasets/aoi-001/10_B1": meta},
    )
    task = json.loads(converter.convert().batch_payloads[0][1])[0]
    results = task["predictions"][0]["result"]
    choice = next(r for r in results if r["type"] == "choices")
    assert choice["value"]["choices"] == ["defect"]


def test_aoi_export_unmapped_verdict_has_no_predictions(monkeypatch):
    meta = dict(AOI_META, vlm_verdict="uncertain", note="")
    converter, _ = _aoi_converter(
        monkeypatch,
        ["/datasets/aoi-001/10_B1"],
        {"/datasets/aoi-001/10_B1": meta},
    )
    task = json.loads(converter.convert().batch_payloads[0][1])[0]
    assert "predictions" not in task


def test_aoi_export_skips_dirs_without_meta(monkeypatch):
    converter, requested = _aoi_converter(
        monkeypatch,
        ["/datasets/aoi-001/10_B1", "/datasets/aoi-001/11_B2"],
        {"/datasets/aoi-001/11_B2": dict(AOI_META)},
    )
    manifest = converter.convert()
    assert manifest.total_tasks == 1
    assert sorted(requested) == [
        "/datasets/aoi-001/10_B1/meta.json",
        "/datasets/aoi-001/11_B2/meta.json",
    ]


def test_convert_preserves_sample_order_under_concurrency(monkeypatch):
    expected = [f"sample_{i:03d}" for i in range(24)]
    dirs = [f"/datasets/aoi-001/{name}" for name in expected]
    converter, _ = _aoi_converter(monkeypatch, dirs, {d: dict(AOI_META) for d in dirs})
    manifest = converter.convert()
    tasks = json.loads(manifest.batch_payloads[0][1])
    assert [t["data"]["sample_id"] for t in tasks] == expected


def test_unsupported_adapter_rejected():
    with pytest.raises(ConversionError, match="Unsupported adapter"):
        AVITrainToLabelStudioConverter(
            dataset_path="/datasets/x",
            storage_base_url="http://storage.example.com",
            storage_headers={},
            adapter="bogus",
        )


def test_reverse_converter_roundtrip():
    task = {
        "data": {
            "sample_id": "sample_001",
            "defect_image": "https://media/defect.jpg",
            "diff_image": "https://media/diff.jpg",
            "gt_image": "https://media/gt.jpg",
        },
        "annotations": [
            {
                "result": [
                    {"from_name": "quality_label", "value": {"choices": ["defect"]}},
                    {
                        "id": "finding_1",
                        "from_name": "finding_category",
                        "value": {
                            "x": 10,
                            "y": 20,
                            "width": 30,
                            "height": 30,
                            "rectanglelabels": ["开路"],
                        },
                    },
                    {
                        "id": "finding_1",
                        "from_name": "finding_observation",
                        "value": {"text": ["线路断开"]},
                    },
                ]
            }
        ],
    }
    samples = LabelStudioToAVITrainConverter().convert([task])
    assert len(samples) == 1
    sample = samples[0]
    assert sample["quality"] == "defect"
    assert len(sample["findings"]) == 1
    finding = sample["findings"][0]
    assert finding["category"] == "开路"
    assert finding["bbox"]["x_min_norm"] == 100.0
    assert finding["bbox"]["x_max_norm"] == 400.0
    assert finding["observation"] == "线路断开"


def test_reverse_converter_skips_unannotated():
    samples = LabelStudioToAVITrainConverter().convert(
        [{"data": {"sample_id": "x"}, "annotations": []}]
    )
    assert samples == []


def test_reverse_converter_keeps_the_whole_sample_note():
    """An aoi_export annotation is a verdict plus a note, and no regions at all."""
    task = {
        "data": {
            "sample_id": "sample_001",
            "defect_image": "https://media/defect.jpg",
        },
        "annotations": [
            {
                "result": [
                    {"from_name": "quality_label", "value": {"choices": ["defect"]}},
                    {"from_name": "overall_note", "value": {"text": ["金手指氧化"]}},
                ]
            }
        ],
    }
    samples = LabelStudioToAVITrainConverter().convert([task])
    assert samples[0]["quality"] == "defect"
    assert samples[0]["note"] == "金手指氧化"
    assert samples[0]["findings"] == []


def test_reverse_converter_omits_the_note_an_avi_sample_never_had():
    """avi_train configs carry no note control, so their samples must not grow one."""
    task = {
        "data": {"sample_id": "sample_001"},
        "annotations": [
            {"result": [{"from_name": "quality_label", "value": {"choices": ["ok"]}}]}
        ],
    }
    samples = LabelStudioToAVITrainConverter().convert([task])
    assert samples[0]["quality"] == "ok"
    assert "note" not in samples[0]


class _StreamResponse:
    def __init__(self, payload: bytes, status_code: int = 200) -> None:
        self.status_code = status_code
        self._payload = payload

    def __enter__(self) -> "_StreamResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def iter_bytes(self):
        yield self._payload

    def read(self) -> bytes:
        return self._payload


def _converter(**overrides) -> AVITrainToLabelStudioConverter:
    params = {
        "dataset_path": "/datasets/pcb-001",
        "storage_base_url": "http://storage.example.com",
        "storage_headers": {},
    }
    params.update(overrides)
    return AVITrainToLabelStudioConverter(**params)


def test_task_data_records_object_paths_alongside_urls(monkeypatch):
    """Paths must survive in task data: rendered URLs are short-lived."""
    converter = _mock_converter(monkeypatch, ["/datasets/pcb-001/sample_001"])
    task = json.loads(converter.convert().batch_payloads[0][1])[0]

    assert task["data"]["defect_image"].startswith("https://media/")
    assert (
        task["data"]["defect_image_path"] == "/datasets/pcb-001/sample_001/defect.jpg"
    )
    assert task["data"]["diff_image_path"] == "/datasets/pcb-001/sample_001/diff.jpg"
    assert task["data"]["gt_image_path"] == "/datasets/pcb-001/sample_001/gt.jpg"


def test_resolve_media_urls_delegates_to_the_signer():
    signer = MagicMock()
    signer.sign_many.return_value = {"a": "https://portal/media?path=a"}
    converter = _converter(media_signer=signer)

    assert converter._resolve_media_urls(["a"]) == {"a": "https://portal/media?path=a"}
    signer.sign_many.assert_called_once_with(["a"])


def test_missing_media_url_is_reported(monkeypatch):
    converter = _mock_converter(monkeypatch, ["/datasets/pcb-001/sample_001"])
    monkeypatch.setattr(converter, "_resolve_media_urls", lambda paths: {})
    with pytest.raises(ConversionError, match="No media URL resolved"):
        converter.convert()


def test_download_missing_object_reads_as_absent(monkeypatch):
    converter = _converter()
    monkeypatch.setattr(
        converter, "_storage_post", lambda route, body: {"data": {"url": "https://o/x"}}
    )
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _StreamResponse(b"", 404))

    assert converter._download_file("/datasets/x/meta.json", max_bytes=1024) is None


def test_download_server_error_is_not_mistaken_for_absent(monkeypatch):
    """A swallowed 5xx silently dropped 567 of 686 samples in production."""
    converter = _converter()
    monkeypatch.setattr(
        converter, "_storage_post", lambda route, body: {"data": {"url": "https://o/x"}}
    )
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _StreamResponse(b"boom", 500))

    with pytest.raises(ConversionError, match="HTTP 500"):
        converter._download_file("/datasets/x/meta.json", max_bytes=1024)


def test_download_client_error_raises_immediately(monkeypatch):
    converter = _converter()
    monkeypatch.setattr(
        converter, "_storage_post", lambda route, body: {"data": {"url": "https://o/x"}}
    )
    monkeypatch.setattr(
        httpx, "stream", lambda *a, **k: _StreamResponse(b"denied", 403)
    )

    with pytest.raises(ConversionError, match="HTTP 403"):
        converter._download_file("/datasets/x/meta.json", max_bytes=1024)


def test_download_retries_a_transient_failure(monkeypatch):
    converter = _converter()
    monkeypatch.setattr(
        converter, "_storage_post", lambda route, body: {"data": {"url": "https://o/x"}}
    )
    monkeypatch.setattr(
        "openhands.tools.label_studio.converter.time.sleep", lambda _seconds: None
    )
    attempts: list[int] = []

    def fake_stream(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("connection reset")
        return _StreamResponse(b"{}")

    monkeypatch.setattr(httpx, "stream", fake_stream)

    assert converter._download_file("/datasets/x/meta.json", max_bytes=1024) == b"{}"
    assert len(attempts) == 2


def test_download_gives_up_after_the_retry_budget(monkeypatch):
    converter = _converter()
    monkeypatch.setattr(
        converter, "_storage_post", lambda route, body: {"data": {"url": "https://o/x"}}
    )
    monkeypatch.setattr(
        "openhands.tools.label_studio.converter.time.sleep", lambda _seconds: None
    )
    monkeypatch.setattr(
        httpx,
        "stream",
        lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("down")),
    )

    with pytest.raises(ConversionError, match="after 2 attempts"):
        converter._download_file("/datasets/x/meta.json", max_bytes=1024)


def test_reverse_converter_prefers_the_stored_path():
    samples = LabelStudioToAVITrainConverter().convert(
        [
            {
                "data": {
                    "defect_image": "https://portal/label_studio/media?path=x",
                    "defect_image_path": "/exports/run/10_B1/defect.jpg",
                },
                "annotations": [
                    {
                        "result": [
                            {"from_name": "quality_label", "value": {"choices": ["ok"]}}
                        ]
                    }
                ],
            }
        ]
    )

    assert samples[0]["defect_image_path"] == "/exports/run/10_B1/defect.jpg"
    assert samples[0]["diff_image_path"] == ""
