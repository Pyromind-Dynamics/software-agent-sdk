"""Review imports preserve image references for Label Studio's link conversion."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from openhands.tools.label_studio.definition import LabelStudioProjectAction
from openhands.tools.label_studio.executor import LabelStudioProjectExecutor


@pytest.mark.parametrize("relative_media_url", [False, True])
def test_three_image_import_preserves_references_without_media_requests(
    tmp_path, monkeypatch, relative_media_url
):
    mount = tmp_path / "mount"
    workspace = SimpleNamespace(
        working_dir=str(mount / ".pyromind-agent/conversation"),
        storage_path=str(mount),
    )
    conversation = MagicMock(id="conversation", workspace=workspace)
    executor = LabelStudioProjectExecutor(
        ls_base_url="https://label.example",
        ls_token_secret="LABEL_STUDIO_TOKEN",
        storage_base_url="https://storage.example",
    )
    slots = ("defect_image", "diff_image", "gt_image")
    sources = {
        "defect_image": "storage/datasets/pcb/defect.jpg",
        "diff_image": str(mount / "datasets/pcb/diff.jpg"),
        "gt_image": "/datasets/pcb/gt.jpg",
    }
    raw = (
        json.dumps({"id": "pcb-1", "nested": sources, "gt": "fail", "answer": "pass"})
        + "\n"
    ).encode()
    field_map = {
        "images": [{"field": field, "source": "nested." + field} for field in slots],
        "samples": [
            {"field": "gt", "control": "ground_truth"},
            {"field": "answer", "control": "model_answer"},
        ],
    }
    xml = (
        "<View>"
        + "".join(f'<Image name="{field}" value="${field}"/>' for field in slots)
        + '<Choices name="ground_truth" toName="defect_image">'
        '<Choice value="fail"/></Choices>'
        '<Choices name="model_answer" toName="defect_image">'
        '<Choice value="pass"/></Choices></View>'
    )
    files = {
        "/reviews/badcase.jsonl": raw,
        "/reviews/label_config.xml": xml.encode(),
        "/reviews/field_map.json": json.dumps(field_map).encode(),
    }
    monkeypatch.setattr(
        executor, "_download_from_storage", lambda path, *a, **k: files[path]
    )
    prefix = "" if relative_media_url else "https://label.example"
    urls = {
        path: f"{prefix}/label_studio/media?image={field}"
        for field, path in sources.items()
    }
    signer = MagicMock(expires_in=3600)
    signer.sign_many.return_value = urls
    api = MagicMock()
    api.find_project_by_title.return_value = None
    api.create_project.return_value = {"id": 71}
    monkeypatch.setattr(executor, "_build_ls_api", lambda _: api)
    monkeypatch.setattr(executor, "_build_media_signer", lambda _: signer)
    monkeypatch.setattr(executor, "_export_ticket", lambda *a: "")
    monkeypatch.setattr(executor, "_upload_to_storage", lambda *a: None)
    states = {}
    monkeypatch.setattr(executor, "_load_state", lambda ref, _: states.get(ref))
    monkeypatch.setattr(
        executor,
        "_save_state",
        lambda _, state, __: states.update({state.project_ref: state}),
    )
    media_get = MagicMock(
        side_effect=AssertionError("Media access belongs to Label Studio")
    )
    monkeypatch.setattr(httpx, "stream", media_get)
    action = LabelStudioProjectAction(
        operation="create",
        adapter="jsonl",
        dataset_path="storage/reviews/badcase.jsonl",
        label_config_path=str(mount / "reviews/label_config.xml"),
        field_map_path="/reviews/field_map.json",
    )

    observation = executor(action, conversation)

    assert not observation.is_error, observation.text
    assert signer.sign_many.call_args.args[0] == list(sources.values())
    task = api.import_tasks.call_args.kwargs["tasks"][0]
    for field, path in sources.items():
        assert task["data"][field + "_path"] == path
        assert task["data"][field] == urls[path]
    assert task["data"]["ground_truth"] == "fail"
    assert task["data"]["model_answer"] == "pass"
    assert files["/reviews/badcase.jsonl"] == raw
    retry = executor(action, conversation)
    assert not retry.is_error
    assert retry.project_id == observation.project_id
    api.create_project.assert_called_once()
    api.import_tasks.assert_called_once()
    media_get.assert_not_called()
