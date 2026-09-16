"""Tests for the Label Studio REST API client."""

from unittest.mock import MagicMock, patch

import pytest

from openhands.tools.label_studio.api_client import (
    LabelStudioAPIClient,
    LabelStudioAPIError,
)


BASE = "http://label-studio.example.com"


def _mock_response(status_code=200, json_data=None, text=""):
    response = MagicMock()
    response.status_code = status_code
    response.text = text or str(json_data)
    response.content = b"{}" if json_data is not None else b""
    response.json.return_value = json_data if json_data is not None else {}
    return response


def test_create_project():
    client = LabelStudioAPIClient(base_url=BASE, token="test-token")
    with patch("httpx.request", return_value=_mock_response(json_data={"id": 42})):
        result = client.create_project(title="test", label_config="<View/>")
    assert result["id"] == 42


def test_import_tasks():
    client = LabelStudioAPIClient(base_url=BASE, token="t")
    with patch(
        "httpx.request",
        return_value=_mock_response(json_data={"task_count": 2}),
    ) as mock_request:
        result = client.import_tasks(project_id=42, tasks=[{"data": {}}, {"data": {}}])
    assert result["task_count"] == 2
    # LS import treats a dict body as a single task; the payload must be a bare list.
    body = mock_request.call_args.kwargs.get("json")
    assert body == [{"data": {}}, {"data": {}}]


def test_http_error_raises():
    client = LabelStudioAPIClient(base_url=BASE, token="t")
    with patch(
        "httpx.request",
        return_value=_mock_response(status_code=400, text="bad request"),
    ):
        with pytest.raises(LabelStudioAPIError, match="HTTP 400"):
            client.create_project(title="x", label_config="y")


def test_get_annotation_from_names():
    client = LabelStudioAPIClient(base_url=BASE, token="t")
    export_data = [
        {
            "annotations": [{"result": [{"from_name": "quality_label"}]}],
            "predictions": [{"result": [{"from_name": "finding_category"}]}],
        }
    ]
    with patch("httpx.request", return_value=_mock_response(json_data=export_data)):
        names = client.get_annotation_from_names(42)
    assert names == {"quality_label", "finding_category"}


def test_find_project_by_title_matches_exactly(monkeypatch):
    client = LabelStudioAPIClient(base_url="http://ls", token="t")
    monkeypatch.setattr(
        client,
        "_request",
        lambda *a, **k: {
            "count": 2,
            "results": [
                {"id": 1, "title": "pyromind_deadbeef-extra"},
                {"id": 2, "title": "pyromind_deadbeef"},
            ],
        },
    )

    assert client.find_project_by_title("pyromind_deadbeef") == {
        "id": 2,
        "title": "pyromind_deadbeef",
    }


def test_find_project_by_title_returns_none_when_absent(monkeypatch):
    client = LabelStudioAPIClient(base_url="http://ls", token="t")
    monkeypatch.setattr(client, "_request", lambda *a, **k: {"count": 0, "results": []})

    assert client.find_project_by_title("pyromind_missing") is None


def test_list_project_tasks_follows_pagination():
    client = LabelStudioAPIClient(base_url=BASE, token="t")
    # Label Studio counts the tasks instead of linking to the next page.
    pages = [
        _mock_response(json_data={"total": 2, "tasks": [{"id": 1, "data": {}}]}),
        _mock_response(json_data={"total": 2, "tasks": [{"id": 2, "data": {}}]}),
    ]

    with patch("httpx.request", side_effect=pages) as mock_request:
        tasks = client.list_project_tasks(7)

    assert [task["id"] for task in tasks] == [1, 2]
    # Tasks are read back per project: the whole org is not the caller's to scan.
    assert mock_request.call_args_list[0].kwargs["params"]["project"] == "7"


def test_update_project_description_patches_the_project():
    client = LabelStudioAPIClient(base_url=BASE, token="t")

    with patch(
        "httpx.request", return_value=_mock_response(json_data={"id": 3})
    ) as mock_request:
        client.update_project_description(project_id=3, description="export target")

    assert mock_request.call_args.args[0] == "PATCH"
    assert mock_request.call_args.args[1].endswith("/api/projects/3")
    assert mock_request.call_args.kwargs["json"] == {"description": "export target"}


def test_update_task_data_sends_the_whole_data_object():
    client = LabelStudioAPIClient(base_url=BASE, token="t")

    with patch(
        "httpx.request", return_value=_mock_response(json_data={"id": 3})
    ) as mock_request:
        client.update_task_data(task_id=3, data={"defect_image": "fresh"})

    assert mock_request.call_args.args[0] == "PATCH"
    assert mock_request.call_args.args[1].endswith("/api/tasks/3")
    assert mock_request.call_args.kwargs["json"] == {"data": {"defect_image": "fresh"}}


def test_validate_config_targets_the_official_route():
    """Label Studio validates at /api/projects/<pk>/validate/.

    No /validate-config route exists, so the previous path answered 404 and took
    every update_config down with it before it could reach Label Studio. Pin the
    route so a rename cannot silently reintroduce that failure.
    """
    client = LabelStudioAPIClient(base_url=BASE, token="t")

    with patch("httpx.request", return_value=_mock_response()) as mock_request:
        client.validate_config(project_id=9, label_config="<View/>")

    assert mock_request.call_args.args[0] == "POST"
    assert mock_request.call_args.args[1].endswith("/api/projects/9/validate/")
    assert mock_request.call_args.kwargs["json"] == {"label_config": "<View/>"}


def test_validate_config_standalone_targets_the_projectless_route():
    """create validates before it converts, and no project exists yet.

    /api/projects/validate/ is Label Studio's project-less validator
    (AllowAny, 204 on success), so the check costs one request and no state.
    """
    client = LabelStudioAPIClient(base_url=BASE, token="t")

    with patch("httpx.request", return_value=_mock_response()) as mock_request:
        client.validate_config_standalone(label_config="<View/>")

    assert mock_request.call_args.args[0] == "POST"
    assert mock_request.call_args.args[1].endswith("/api/projects/validate/")
    assert mock_request.call_args.kwargs["json"] == {"label_config": "<View/>"}
