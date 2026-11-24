import json
import runpy
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".agents/skills/data-processing/scripts/preparation/training_image_paths.py"
)
prepare_images = runpy.run_path(str(SCRIPT))["prepare_images"]


def test_training_derivative_can_open_all_images_without_source_root(tmp_path):
    mount = tmp_path / "workspace"
    source = mount / "source"
    images = source / "images"
    images.mkdir(parents=True)
    (images / "first.jpg").write_bytes(b"first")
    (images / "second.jpg").write_bytes(b"second")
    manifest = mount / "filtered.jsonl"
    row = {"media": ["images/second.jpg", str(images / "first.jpg")], "gt": "label"}
    original = json.dumps(row) + "\n"
    manifest.write_text(original)
    with pytest.raises(ValueError, match=r":1: relative image path"):
        prepare_images(manifest, "media", mount=mount)

    derivative = mount / "another" / "train.jsonl"
    assert prepare_images(
        manifest, "media", source_root=source, output=derivative, mount=mount
    ) == (1, 2)
    assert json.loads(derivative.read_text()) == {
        **row,
        "media": [str(images / "second.jpg"), str(images / "first.jpg")],
    }
    assert prepare_images(derivative, "media", mount=mount) == (1, 2)
    assert manifest.read_text() == original


def test_late_missing_image_does_not_publish_partial_derivative(tmp_path):
    (tmp_path / "ok.jpg").write_bytes(b"data")
    source = tmp_path / "val.jsonl"
    source.write_text('{"image":"ok.jpg"}\n{"image":"missing.jpg"}\n')
    output = tmp_path / "checked.jsonl"
    output.write_text("previous")
    with pytest.raises(ValueError, match=r"val.jsonl:2:.*missing.jpg"):
        prepare_images(
            source, "image", source_root=tmp_path, output=output, mount=tmp_path
        )
    assert output.read_text() == "previous"
