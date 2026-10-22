"""Preflight checks every populated image binding without forwarding credentials."""

from contextlib import nullcontext

import httpx
import pytest

from openhands.tools.label_studio.converter import ConversionError
from openhands.tools.label_studio.media_validation import validate_task_media


@pytest.mark.parametrize(
    ("status", "content_type", "content", "reason"),
    [
        (404, "application/json", b"{}", "path_not_found"),
        (401, "application/json", b"{}", "authentication_failed"),
        (403, "application/json", b"{}", "authentication_failed"),
        (200, "text/html", b"<html>login", "non_image_response"),
        (200, "image/jpeg", b"", "empty_image_response"),
        (500, "text/plain", b"offline", "unexpected_response"),
    ],
)
def test_preflight_reports_sanitized_slot_and_access_reason(
    monkeypatch, status, content_type, content, reason
):
    monkeypatch.setattr(
        httpx,
        "stream",
        lambda *a, **k: nullcontext(
            httpx.Response(
                status, headers={"content-type": content_type}, content=content
            )
        ),
    )
    with pytest.raises(ConversionError) as exc:
        validate_task_media(
            [{
                "data": {
                    "defect_image": "https://portal/media?media_token=secret",
                    "defect_image_path": "/datasets/a.jpg",
                }
            }],
            10,
        )
    assert reason in str(exc.value)
    assert "defect_image" in str(exc.value)
    assert "/datasets/a.jpg" in str(exc.value)
    assert "secret" not in str(exc.value)


def test_preflight_checks_three_slots_and_late_optional_image(monkeypatch):
    requested = []

    def stream(method, url, **kwargs):
        requested.append(url)
        assert kwargs["headers"] == {"Range": "bytes=0-4095"}
        return nullcontext(
            httpx.Response(206, headers={"content-type": "image/jpeg"}, content=b"jpeg")
        )

    monkeypatch.setattr(httpx, "stream", stream)
    tasks = []
    for fields in [("defect", "diff", "gt"), ("defect", "optional")]:
        data = {}
        for field in fields:
            data[field] = f"https://portal/{field}"
            data[field + "_path"] = f"/datasets/{field}.jpg"
        tasks.append({"data": data})
    validate_task_media(tasks, 10)
    assert requested == [
        "https://portal/defect", "https://portal/diff",
        "https://portal/gt", "https://portal/optional",
    ]
