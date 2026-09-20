"""Tests for AVI Train and Label Studio converters."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from openhands.tools.label_studio.converter import (
    AVITrainToLabelStudioConverter,
    ConversionError,
    LabelStudioToAVITrainConverter,
)
from openhands.tools.label_studio.field_map import parse_field_map


EXAMPLES = Path(__file__).parents[3] / ".agents/skills/label-studio/references/examples"


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
    cases = (
        ("NG", "defect"),
        ("PASS", "ok"),
        ("Good", "ok"),
        ("bad", "defect"),
        ("FAULT", "defect"),
    )
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


JSONL_ROWS = [
    {
        "id": "s1",
        "quality": "defect",
        "defect_image": "/datasets/pcb-001/s1/defect.jpg",
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
    },
    {"quality": "ok", "defect_image": "/datasets/pcb-001/s2/defect.jpg"},
]


def _jsonl_converter(
    monkeypatch,
    rows,
    *,
    field_map=None,
    dataset_path="/datasets/pcb-001/processed.jsonl",
):
    content = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    converter = AVITrainToLabelStudioConverter(
        dataset_path=dataset_path,
        storage_base_url="http://storage.example.com",
        storage_headers={},
        adapter="jsonl",
        field_map=field_map,
        dataset_content=content,
    )
    monkeypatch.setattr(
        converter,
        "_resolve_media_urls",
        lambda paths: {path: f"https://media/{path}?token=x" for path in paths},
    )
    return converter


def test_jsonl_rows_become_tasks(monkeypatch):
    """A processed file imports as written: no sample directory has to exist."""
    manifest = _jsonl_converter(monkeypatch, JSONL_ROWS).convert()
    assert manifest.total_tasks == 2
    tasks = json.loads(manifest.batch_payloads[0][1])

    assert [task["data"]["sample_id"] for task in tasks] == ["s1", "line-2"]
    first = tasks[0]
    assert first["data"]["defect_image"] == (
        "https://media//datasets/pcb-001/s1/defect.jpg?token=x"
    )
    assert first["data"]["defect_image_path"] == "/datasets/pcb-001/s1/defect.jpg"
    # The other two images are optional here, so a row without them renders one
    # view instead of failing the conversion.
    assert "diff_image" not in first["data"]
    assert "gt_image" not in tasks[1]["data"]

    results = first["predictions"][0]["result"]
    choice = next(r for r in results if r["type"] == "choices")
    assert choice["value"]["choices"] == ["defect"]
    region = next(r for r in results if r["type"] == "rectanglelabels")
    assert region["value"]["rectanglelabels"] == ["开路"]
    assert region["value"]["x"] == 10.0


def test_jsonl_image_source_may_be_a_nested_path(monkeypatch):
    """A row that groups its images under one key binds them by dotted path."""
    field_map = parse_field_map(
        {"images": [{"field": "defect_image", "source": "images.defect_image"}]},
        adapter="jsonl",
    )
    converter = _jsonl_converter(
        monkeypatch,
        [{"id": "s1", "images": {"defect_image": "/a/b.jpg"}}],
        field_map=field_map,
    )
    tasks = json.loads(converter.convert().batch_payloads[0][1])
    assert tasks[0]["data"]["defect_image_path"] == "/a/b.jpg"


def test_jsonl_declared_bindings_decide_where_each_value_lands(monkeypatch):
    """A row is read by field name, so a declaration is all a new layout needs.

    This is the shape a preprocessing run tends to write: a bare label, a
    category at sample level, and a list of boxes.
    """
    field_map = parse_field_map(
        {
            "images": [{"field": "image", "source": "image"}],
            "samples": [
                {
                    "field": "label",
                    "control": "quality_label",
                    "type": "choices",
                    "synonyms": {"defect": "defect", "ok": "ok"},
                    "on_unmapped": "keep",
                }
            ],
            "regions": [
                {
                    "source": "boxes",
                    "control": "finding_category",
                    "label": "category",
                }
            ],
        },
        adapter="jsonl",
    )
    converter = _jsonl_converter(
        monkeypatch,
        [
            {
                "id": "s1",
                "label": "defect",
                "category": "开路",
                "image": "/datasets/pcb-001/s1.jpg",
                "boxes": [[100, 200, 400, 500]],
            }
        ],
        field_map=field_map,
    )
    tasks = json.loads(converter.convert().batch_payloads[0][1])
    assert tasks[0]["data"]["image_path"] == "/datasets/pcb-001/s1.jpg"
    results = tasks[0]["predictions"][0]["result"]
    assert {r["from_name"]: r["type"] for r in results} == {
        "quality_label": "choices",
        "finding_category": "rectanglelabels",
    }
    box = next(r for r in results if r["type"] == "rectanglelabels")
    # A bare box list carries no label of its own, so the row's category is
    # copied onto it -- otherwise Label Studio renders no rectangle at all.
    assert box["value"]["rectanglelabels"] == ["开路"]


def test_jsonl_row_without_a_required_image_fails(monkeypatch):
    converter = _jsonl_converter(monkeypatch, [{"id": "s1", "quality": "ok"}])
    with pytest.raises(ConversionError, match="line 1 has no image path"):
        converter.convert()


PIPELINE_ROW = {
    "id": "1-3-0804F/118/B0",
    "source_images": {
        "待检原图": "/datasets/test_100/1-3-0804F/118/B0.bmp",
        "CAM参考图": "/datasets/test_100/1-3-0804F/118/B0_cam.bmp",
    },
    "context": "非铜区可见异物；未提供放行阈值。",
    "label": True,
    "regions": [
        {
            "category": "垃圾/异物",
            "boxes": [[425, 460, 550, 535], [700, 600, 900, 800]],
            "note": "非铜区可见不规则亮色异物颗粒",
        }
    ],
}


def test_a_pipeline_row_imports_through_the_shipped_field_map(monkeypatch):
    """The shipped PCB binding reads a processed row exactly as the pipeline wrote it.

    This is the whole point of the row contract: the example binding declares
    where each field lives, so no reshape step stands between a pre-labeling run
    and the review project.
    """
    declared = json.loads(
        (EXAMPLES / "field-maps" / "pcb_prelabel.json").read_text(encoding="utf-8")
    )
    converter = _jsonl_converter(
        monkeypatch,
        [PIPELINE_ROW],
        field_map=parse_field_map(declared, adapter="jsonl"),
    )
    manifest = converter.convert()
    assert manifest.unmapped_quality == ()
    task = json.loads(manifest.batch_payloads[0][1])[0]
    assert task["data"]["defect_image_path"] == (
        "/datasets/test_100/1-3-0804F/118/B0.bmp"
    )
    assert task["data"]["gt_image_path"] == (
        "/datasets/test_100/1-3-0804F/118/B0_cam.bmp"
    )

    results = task["predictions"][0]["result"]
    choices = next(r for r in results if r["type"] == "choices")
    assert choices["value"]["choices"] == ["defect"]
    whole_note = next(r for r in results if r["from_name"] == "overall_note")
    assert whole_note["value"]["text"] == ["非铜区可见异物；未提供放行阈值。"]

    # One finding naming two boxes renders two rectangles, each carrying the
    # finding's own note, so a defect seen twice is not split in the source data.
    boxes = [r for r in results if r["type"] == "rectanglelabels"]
    assert [r["id"] for r in boxes] == ["finding_1", "finding_1_2"]
    assert [(r["value"]["x"], r["value"]["y"]) for r in boxes] == [
        (42.5, 46.0),
        (70.0, 60.0),
    ]
    assert [r["value"]["rectanglelabels"] for r in boxes] == [
        ["垃圾/异物"],
        ["垃圾/异物"],
    ]
    region_notes = [r for r in results if r["from_name"] == "finding_observation"]
    assert [r["id"] for r in region_notes] == ["finding_1", "finding_1_2"]
    assert all(
        r["value"]["text"] == ["非铜区可见不规则亮色异物颗粒"] for r in region_notes
    )


ROW_WITH_A_ROW_LEVEL_BOX = {
    "id": "1-3-0804F/118/B0",
    "source_images": {"待检原图": "/datasets/test_100/1-3-0804F/118/B0.bmp"},
    "label": True,
    "category": "垃圾/异物",
    "boxes": [[190, 820, 290, 870]],
    "note": "非铜区可见不规则亮色异物颗粒",
}


def test_declared_geometry_falls_back_to_the_wrapped_box(monkeypatch):
    """A binding named after the row's own field still finds the box.

    A row-level ``boxes`` list is imported as bare coordinates, which the
    adapter wraps as ``{"bbox": ..., "category": ...}``. A binding written
    against the row's field name therefore names a key no region carries, and
    dropping the rectangle for it would look like a region without coordinates.
    """
    converter = _jsonl_converter(
        monkeypatch,
        [ROW_WITH_A_ROW_LEVEL_BOX],
        field_map=parse_field_map(
            {
                "images": [
                    {"field": "defect_image", "source": "source_images.待检原图"}
                ],
                "regions": [
                    {
                        "source": "boxes",
                        "control": "finding_category",
                        "label": "category",
                        "geometry": "boxes",
                        "unit": "norm1000",
                        "observation": "note",
                        "observation_control": "finding_observation",
                    }
                ],
            },
            adapter="jsonl",
        ),
    )
    task = json.loads(converter.convert().batch_payloads[0][1])[0]
    results = task["predictions"][0]["result"]

    box = next(r for r in results if r["type"] == "rectanglelabels")
    assert box["value"] == {
        "x": 19.0,
        "y": 82.0,
        "width": 10.0,
        "height": 5.0,
        "rectanglelabels": ["垃圾/异物"],
    }


def test_a_declared_geometry_key_still_wins_over_the_documented_ones(monkeypatch):
    """The declaration decides which field is read when a region carries both."""
    converter = _jsonl_converter(
        monkeypatch,
        [
            {
                "id": "s1",
                "source_images": {"待检原图": "/datasets/x/defect.bmp"},
                "regions": [
                    {
                        "category": "垃圾/异物",
                        "bbox": [0, 0, 100, 100],
                        "value": [500, 500, 600, 600],
                    }
                ],
            }
        ],
        field_map=parse_field_map(
            {
                "images": [
                    {"field": "defect_image", "source": "source_images.待检原图"}
                ],
                "regions": [
                    {
                        "source": "regions",
                        "control": "finding_category",
                        "label": "category",
                        "geometry": "value",
                        "unit": "norm1000",
                    }
                ],
            },
            adapter="jsonl",
        ),
    )
    task = json.loads(converter.convert().batch_payloads[0][1])[0]
    box = next(
        r for r in task["predictions"][0]["result"] if r["type"] == "rectanglelabels"
    )
    assert (box["value"]["x"], box["value"]["y"]) == (50.0, 50.0)


def test_a_region_source_no_row_carries_is_reported(monkeypatch):
    """A binding naming a field the data never has must not fail silently.

    Without this, the only symptom is a project whose editor shows no
    rectangles, which reads the same as a sample an annotator draws by hand.
    """
    converter = _jsonl_converter(
        monkeypatch,
        [{"id": "s1", "image": "/datasets/x/a.jpg", "regions": []}],
        field_map=parse_field_map(
            {
                "images": [{"field": "image", "source": "image"}],
                "regions": [
                    {
                        "source": "boxes",
                        "control": "finding_category",
                        "label": "category",
                    }
                ],
            },
            adapter="jsonl",
        ),
    )
    assert converter.convert().unmatched_regions == ("boxes",)


def test_a_region_source_the_rows_carry_is_not_reported(monkeypatch):
    """The hit is what suppresses the warning, so a matched binding stays quiet."""
    converter = _jsonl_converter(
        monkeypatch,
        [
            {
                "id": "s1",
                "image": "/datasets/x/a.jpg",
                "category": "开路",
                "boxes": [[100, 200, 400, 500]],
            }
        ],
        field_map=parse_field_map(
            {
                "images": [{"field": "image", "source": "image"}],
                "regions": [
                    {
                        "source": "boxes",
                        "control": "finding_category",
                        "label": "category",
                    }
                ],
            },
            adapter="jsonl",
        ),
    )
    assert converter.convert().unmatched_regions == ()


def test_an_aoi_export_never_reports_a_region_source(monkeypatch):
    """Most AOI exports are a whole-sample verdict, so a region less verdict is
    the documented normal case rather than a declaration that missed the data.
    """
    converter, _ = _aoi_converter(
        monkeypatch,
        ["/datasets/aoi-001/10_B1"],
        {"/datasets/aoi-001/10_B1": dict(AOI_META)},
    )
    assert converter.convert().unmatched_regions == ()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b'{"id": "s1"}\n{not json}\n', "line 2 is not valid JSON"),
        (b'["s1"]\n', "line 1 is not a JSON object"),
        (b"", "holds no samples"),
        (b"\n  \n", "holds no samples"),
    ],
)
def test_jsonl_bad_input_is_reported_not_skipped(content, message):
    """Half-importing a file is worse than failing: the annotation work would
    land on a project whose task set silently lost rows."""
    converter = AVITrainToLabelStudioConverter(
        dataset_path="/datasets/pcb-001/processed.jsonl",
        storage_base_url="http://storage.example.com",
        storage_headers={},
        adapter="jsonl",
        dataset_content=content,
    )
    with pytest.raises(ConversionError, match=message):
        converter.convert()


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


@pytest.mark.parametrize(
    ("label", "finding", "expected"),
    [
        (
            "documented_contract",
            {
                "bbox": {
                    "x_min_norm": 400,
                    "y_min_norm": 380,
                    "x_max_norm": 600,
                    "y_max_norm": 620,
                }
            },
            (40.0, 38.0, 20.0, 24.0),
        ),
        (
            "norm_keys_keep_their_own_unit",
            {
                "bbox": {
                    "x_min_norm": 40,
                    "y_min_norm": 30,
                    "x_max_norm": 60,
                    "y_max_norm": 70,
                }
            },
            (4.0, 3.0, 2.0, 4.0),
        ),
        (
            "vlm_pipeline_value_percent",
            {"value": {"x": 40.0, "y": 38.0, "width": 20.0, "height": 24.0}},
            (40.0, 38.0, 20.0, 24.0),
        ),
        (
            "flat_percent",
            {"x": 40.0, "y": 38.0, "width": 20.0, "height": 24.0},
            (40.0, 38.0, 20.0, 24.0),
        ),
        (
            "unit_normalised",
            {"value": {"x": 0.4, "y": 0.38, "width": 0.2, "height": 0.24}},
            (40.0, 38.0, 20.0, 24.0),
        ),
        (
            "norm1000_corner_list",
            {"bbox": [400, 380, 600, 620]},
            (40.0, 38.0, 20.0, 24.0),
        ),
        (
            "norm1000_in_plain_keys",
            {"value": {"x": 400.0, "y": 380.0, "width": 200.0, "height": 240.0}},
            (40.0, 38.0, 20.0, 24.0),
        ),
        (
            "corner_keys",
            {"value": {"x1": 400.0, "y1": 380.0, "x2": 600.0, "y2": 620.0}},
            (40.0, 38.0, 20.0, 24.0),
        ),
    ],
)
def test_region_geometry_reads_every_shape_our_producers_write(
    label: str, finding: dict, expected: tuple[float, float, float, float]
):
    """A shape the converter cannot read is silent: no box, and no error.

    Every shape here was produced by a real producer -- the documented contract,
    our own VLM pipeline, an ad-hoc export script, and unit-normalised model
    output -- which is why all of them have to be read rather than one.
    """
    converter = _converter()
    predictions = converter._build_predictions(
        {"quality": "defect", "findings": [{**finding, "category": "断路"}]}
    )
    assert predictions is not None, label
    box = next(r for r in predictions["result"] if r["type"] == "rectanglelabels")
    value = box["value"]
    got = (value["x"], value["y"], value["width"], value["height"])
    assert got == expected, label


def test_region_without_a_category_is_skipped():
    """Label Studio renders nothing for an empty label list, so it is dropped."""
    converter = _converter()
    predictions = converter._build_predictions(
        {
            "quality": "defect",
            "findings": [
                {
                    "bbox": {
                        "x_min_norm": 100,
                        "y_min_norm": 100,
                        "x_max_norm": 200,
                        "y_max_norm": 200,
                    }
                }
            ],
        }
    )
    assert predictions is not None
    assert [r["type"] for r in predictions["result"]] == ["choices"]


def test_degenerate_region_is_skipped():
    """A zero-area box is not a region; writing it would add a no-op annotation."""
    converter = _converter()
    predictions = converter._build_predictions(
        {
            "quality": "defect",
            "findings": [
                {
                    "category": "断路",
                    "bbox": {
                        "x_min_norm": 100,
                        "y_min_norm": 100,
                        "x_max_norm": 100,
                        "y_max_norm": 200,
                    },
                }
            ],
        }
    )
    assert predictions is not None
    assert [r["type"] for r in predictions["result"]] == ["choices"]


def test_out_of_range_region_is_clipped_rather_than_dropped():
    """A box hanging off the edge still tells the annotator where to look."""
    converter = _converter()
    predictions = converter._build_predictions(
        {
            "quality": "defect",
            "findings": [
                {
                    "category": "断路",
                    "bbox": {
                        "x_min_norm": 900,
                        "y_min_norm": 900,
                        "x_max_norm": 1200,
                        "y_max_norm": 1200,
                    },
                }
            ],
        }
    )
    assert predictions is not None
    box = next(r for r in predictions["result"] if r["type"] == "rectanglelabels")
    assert box["value"]["x"] == 90.0
    assert box["value"]["y"] == 90.0
    assert box["value"]["width"] == 10.0
    assert box["value"]["height"] == 10.0


def test_aoi_export_renders_boxes_when_the_export_carries_them(monkeypatch):
    """Most AOI exports are coordinate-free, but a boxes list is honoured."""
    meta = dict(AOI_META, boxes=[[400, 380, 600, 620]], vlm_category="图电")
    converter, _ = _aoi_converter(
        monkeypatch,
        ["/datasets/aoi-001/10_B1"],
        {"/datasets/aoi-001/10_B1": meta},
    )
    task = json.loads(converter.convert().batch_payloads[0][1])[0]
    results = task["predictions"][0]["result"]
    box = next(r for r in results if r["type"] == "rectanglelabels")
    assert box["from_name"] == "finding_category"
    assert box["to_name"] == "defect_image"
    assert box["value"]["rectanglelabels"] == ["图电"]
    assert box["value"]["x"] == 40.0
    assert box["value"]["width"] == 20.0


def test_aoi_export_prefers_explicit_findings_over_flat_boxes(monkeypatch):
    meta = dict(
        AOI_META,
        findings=[
            {
                "value": {"x": 10.0, "y": 10.0, "width": 5.0, "height": 5.0},
                "category": "阻焊",
            }
        ],
        boxes=[[0, 0, 1000, 1000]],
    )
    converter, _ = _aoi_converter(
        monkeypatch,
        ["/datasets/aoi-001/10_B1"],
        {"/datasets/aoi-001/10_B1": meta},
    )
    task = json.loads(converter.convert().batch_payloads[0][1])[0]
    boxes = [
        r for r in task["predictions"][0]["result"] if r["type"] == "rectanglelabels"
    ]
    assert len(boxes) == 1
    assert boxes[0]["value"]["rectanglelabels"] == ["阻焊"]


def test_aoi_export_without_coordinates_still_has_no_regions(monkeypatch):
    converter, _ = _aoi_converter(
        monkeypatch,
        ["/datasets/aoi-001/10_B1"],
        {"/datasets/aoi-001/10_B1": dict(AOI_META)},
    )
    task = json.loads(converter.convert().batch_payloads[0][1])[0]
    assert [r["type"] for r in task["predictions"][0]["result"]] == [
        "choices",
        "textarea",
    ]


def test_aoi_fault_verdict_maps_to_defect(monkeypatch):
    """Real AOI exports say "fault"; unmapped, the sample kept no verdict at all."""
    meta = dict(AOI_META, vlm_verdict="fault", note="")
    converter, _ = _aoi_converter(
        monkeypatch,
        ["/datasets/aoi-001/10_B1"],
        {"/datasets/aoi-001/10_B1": meta},
    )
    task = json.loads(converter.convert().batch_payloads[0][1])[0]
    choice = next(r for r in task["predictions"][0]["result"] if r["type"] == "choices")
    assert choice["value"]["choices"] == ["defect"]


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
