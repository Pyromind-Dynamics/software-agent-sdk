import hashlib
import json
import subprocess
import sys
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest

from pyromind_agent_server import echomind_demo_collection as demo
from pyromind_common.platform.pcb_collection import pcb_selection_command
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.ports.storage import EvaluationStorageMissingError


def test_page_order_and_heldout_exclusion(tmp_path):
    images = tmp_path / "pcb/images"
    for i in range(8):
        sample = images / str(i)
        sample.mkdir(parents=True)
        for file in demo._FILES:
            (sample / file).write_bytes(f"{i}{file}".encode())
    holdout = tmp_path / "val.jsonl"
    holdout.write_text("")

    def select(offset, internal=()):
        return subprocess.run(
            [
                sys.executable,
                "-c",
                demo._SELECT,
                str(tmp_path),
                "/pcb",
                "/pcb/images",
                '["7", "6", "5"]',
                str(offset),
                json.dumps(demo._FILES),
                "/val.jsonl",
                json.dumps(internal),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    assert [r["id"] for r in json.loads(select(0).stdout)] == ["7", "6", "5", "0", "1"]
    assert [r["id"] for r in json.loads(select(3).stdout)] == ["0", "1", "2", "3", "4"]
    assert select(4).returncode
    holdout.write_text(
        json.dumps({"id": "different", "images": ["pcb/images/1/diff.jpg"]})
    )
    assert "overlaps the fixed Eval set" in select(0).stderr
    files = [images / "7" / role for role in demo._FILES]
    fixed = [
        {
            "id": "7",
            "images": [
                "/workspace/" + p.relative_to(tmp_path).as_posix() for p in files
            ],
            "triplet_sha256": hashlib.sha256(
                "".join(
                    hashlib.sha256(p.read_bytes()).hexdigest() for p in files
                ).encode()
            ).hexdigest(),
        }
    ]
    (images / "0" / demo._FILES[0]).write_bytes(files[0].read_bytes())
    assert [r["id"] for r in json.loads(select(0, fixed).stdout)] == [
        "6",
        "5",
        "2",
        "3",
        "4",
    ]
    assert select(1, fixed).returncode


def test_completed_page_reuses_predictions_and_repairs_partial_rosters(monkeypatch):
    project = Mock()
    project.project.conversation_id = uuid4().hex
    image = b"\xff\xd8data"
    digest = hashlib.sha256(image).hexdigest()
    candidates = [
        {
            "id": str(i),
            "images": [f"/workspace/{i}/{n}" for n in demo._FILES],
            "image_sha256": [digest] * 3,
        }
        for i in range(5)
    ]
    prompts = {"system_prompt": "system", "user_prompt": "user"}
    monkeypatch.setattr(demo, "_seed_rows", lambda *args: {"ref": prompts})
    monkeypatch.setattr(demo, "_select", lambda *args: candidates)
    files = {to_storage_api_path(p): image for r in candidates for p in r["images"]}

    def read(path):
        key = to_storage_api_path(path)
        if key not in files:
            raise EvaluationStorageMissingError(key)
        return files[key]

    storage = Mock()
    storage.read.side_effect = read
    storage.write.side_effect = lambda path, data: files.update(
        {to_storage_api_path(path): data}
    )
    infer = Mock(return_value='{"result":"真点","category":"洞孔"}')

    def prepare(
        offset=5, request_id="retry", dataset_labels=False, fixed_validation=False
    ):
        return demo.prepare_demo_source(
            project=project,
            storage=storage,
            seed_source=Mock(),
            raw_directory="/workspace/images",
            offset=offset,
            limit=5,
            request_id=request_id,
            model=None if dataset_labels else "pcb",
            fixed_validation=fixed_validation,
            infer=None if dataset_labels else infer,
        )

    source = prepare()
    rows = [json.loads(row) for row in read(source.train_uri).splitlines()]
    assert len(rows) == 3
    assert rows[0]["label_source"] == "inference_pseudo_label"
    assert rows[0]["label_model"] == "pcb"
    assert len(infer.call_args.args[0][1]["content"]) == 4
    del files[to_storage_api_path(source.val_uri)]
    assert prepare() == source
    assert len(read(source.val_uri).splitlines()) == 2
    assert infer.call_count == 5
    raw_source = prepare(request_id="raw", dataset_labels=True)
    for uri in (raw_source.train_uri, raw_source.val_uri):
        raw_rows = [json.loads(row) for row in read(uri).splitlines()]
        assert all(
            row["label_source"] == "dataset" and "gt" not in row for row in raw_rows
        )
    assert infer.call_count == 5
    fixed = prepare(request_id="all-train", dataset_labels=True, fixed_validation=True)
    assert len(read(fixed.train_uri).splitlines()) == 5
    assert read(fixed.val_uri) == b"\n"
    with pytest.raises(ValueError, match="变更"):
        prepare(offset=6)
    infer.return_value = '{"result":"真点"}'
    with pytest.raises(ValueError, match="category"):
        prepare(request_id="invalid")
    assert not any(demo._digest(b"invalid") in path for path in files)


def test_labeler_uses_serving_openai_route_without_tracing(monkeypatch):
    response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "label"}}]},
        request=httpx.Request("POST", "https://serving.invalid"),
    )
    post = Mock(return_value=response)
    monkeypatch.setattr(demo.httpx, "post", post)
    label = demo.make_inference_labeler("https://serving.invalid/job/v1", "test", "pcb")
    assert label([]) == "label"
    assert post.call_args.args == ("https://serving.invalid/job/v1/chat/completions",)
    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer test"}


def test_report_directory_uses_only_eligible_groups_across_pages(tmp_path):
    images = tmp_path / "datasets/pcb/images"
    validation = tmp_path / "datasets/eval"
    validation.mkdir(parents=True)
    groups = ["irrelevant", *(f"report_{i:02}" for i in range(13))]
    for group in groups:
        directory = images / group
        directory.mkdir(parents=True)
        for filename in demo._FILES:
            (directory / filename).write_bytes(f"{group}/{filename}".encode())
    (images / "report_02/diff.jpg").unlink()
    (validation / "renamed.png").write_bytes((images / "report_01/gt.jpg").read_bytes())
    (validation / "val.jsonl").write_text(
        json.dumps({"id": "report_00", "images": ["renamed.png"]})
    )
    for offset in (0, 5):
        command = pcb_selection_command(
            str(tmp_path), "/datasets/pcb/images", "/datasets/eval/val.jsonl", offset
        )
        output = subprocess.check_output([sys.executable, *command[1:]], text=True)
        rows = json.loads(output)
        expected = [f"report_{i:02}" for i in range(3 + offset, 8 + offset)]
        assert [r["id"] for r in rows] == expected
        for row in rows:
            assert row["image_sha256"] == [
                hashlib.sha256((images / row["id"] / file).read_bytes()).hexdigest()
                for file in demo._FILES
            ]
