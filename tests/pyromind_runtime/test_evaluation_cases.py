from __future__ import annotations

import json

import pytest
from pyromind_runtime.application.evaluation_cases import (
    EvaluationCaseError,
    EvaluationCaseRequestError,
    EvaluationCaseService,
    EvaluationMediaNotFoundError,
)


OUTPUT = "/runs/eval"


class FakeStorage:
    def __init__(self, files: dict[str, bytes]):
        self.files = files

    def read(self, path: str) -> bytes:
        return self.files[path]

    def write(self, path: str, payload: bytes) -> None:
        self.files[path] = payload

    def list(self, prefix: str) -> set[str]:
        return set()


def encoded(value: object) -> bytes:
    return json.dumps(value).encode()


@pytest.fixture
def service() -> EvaluationCaseService:
    return EvaluationCaseService(
        FakeStorage(
            {
                f"{OUTPUT}/evaluation_cases_manifest.json": encoded(
                    {
                        "chunks": [
                            {
                                "name": "evaluation_cases/chunk-00000.json",
                                "start": 0,
                                "count": 2,
                            },
                            {
                                "name": "evaluation_cases/chunk-00001.json",
                                "start": 2,
                                "count": 1,
                            },
                        ]
                    }
                ),
                f"{OUTPUT}/evaluation_cases_index.json": encoded(
                    {
                        "cases": [
                            {
                                "id": "alpha",
                                "chunk": 0,
                                "offset": 0,
                                "media": [
                                    {
                                        "media_id": "00000000-000.png",
                                        "mime_type": "image/png",
                                    }
                                ],
                            },
                            {"id": "beta", "chunk": 0, "offset": 1, "media": []},
                            {"id": "gamma", "chunk": 1, "offset": 0, "media": []},
                        ]
                    }
                ),
                f"{OUTPUT}/evaluation_cases/chunk-00000.json": encoded(
                    {"items": [{"id": "alpha"}, {"id": "beta"}]}
                ),
                f"{OUTPUT}/evaluation_cases/chunk-00001.json": encoded(
                    {"items": [{"id": "gamma"}]}
                ),
                f"{OUTPUT}/evaluation_media/00000000-000.png": b"png",
            }
        )
    )


def test_list_page_uses_opaque_cursor_across_chunks(service) -> None:
    first = service.list_page(output_dir=OUTPUT, cursor=None, limit=2, query="")
    assert [item["id"] for item in first.items] == ["alpha", "beta"]
    assert first.total == 3
    assert first.filtered_total == 3
    assert first.next_cursor is not None

    second = service.list_page(
        output_dir=OUTPUT, cursor=first.next_cursor, limit=2, query=""
    )
    assert [item["id"] for item in second.items] == ["gamma"]
    assert second.next_cursor is None


def test_cursor_is_bound_to_normalized_query(service) -> None:
    page = service.list_page(output_dir=OUTPUT, cursor=None, limit=1, query="a")
    assert page.next_cursor is not None
    with pytest.raises(EvaluationCaseRequestError):
        service.list_page(
            output_dir=OUTPUT, cursor=page.next_cursor, limit=1, query="beta"
        )


def test_list_page_filters_ids_and_binds_cursor_to_selection(service) -> None:
    page = service.list_page(
        output_dir=OUTPUT,
        cursor=None,
        limit=1,
        query="",
        allowed_case_ids={"alpha", "gamma"},
        selection_key="unreviewed",
    )
    assert [item["id"] for item in page.items] == ["alpha"]
    assert page.total == 3
    assert page.filtered_total == 2
    assert page.next_cursor is not None

    with pytest.raises(EvaluationCaseRequestError):
        service.list_page(
            output_dir=OUTPUT,
            cursor=page.next_cursor,
            limit=1,
            query="",
            allowed_case_ids={"beta"},
            selection_key="reviewed",
        )


def test_media_must_be_listed_in_index(service) -> None:
    media = service.read_media(output_dir=OUTPUT, media_id="00000000-000.png")
    assert media.payload == b"png"
    assert media.mime_type == "image/png"
    with pytest.raises(EvaluationMediaNotFoundError):
        service.read_media(output_dir=OUTPUT, media_id="../run_manifest.json")


def test_manifest_cannot_select_an_arbitrary_storage_path(service) -> None:
    malicious = EvaluationCaseService(
        FakeStorage(
            {
                f"{OUTPUT}/evaluation_cases_manifest.json": encoded(
                    {"chunks": [{"name": "../run_manifest.json"}]}
                ),
                f"{OUTPUT}/evaluation_cases_index.json": encoded(
                    {"cases": [{"id": "alpha", "chunk": 0, "offset": 0, "media": []}]}
                ),
            }
        )
    )
    with pytest.raises(EvaluationCaseError):
        malicious.list_page(output_dir=OUTPUT, cursor=None, limit=1, query="")
