"""API tests for the evaluation-runs router (Phase 3)."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pyromind_agent_server.api.evaluation_runs as evaluation_routes
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pyromind_agent_server.api.evaluation_runs import (
    create_evaluation_runs_router,
    get_evaluation_storage,
)
from pyromind_agent_server.api.router import create_product_router
from pyromind_agent_server.evaluation_storage import PyromindEvaluationStorage
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.evaluation import EvaluationCommitState
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.storage import (
    EvaluationStorageMissingError,
    EvaluationStorageUnavailableError,
)

from openhands.agent_server.config import Config


CONVERSATION_ID = "conv-eval-1"
# The hex letters matter: the non-canonical-form test needs a UUID whose
# uppercase and hyphen-less spellings differ from the canonical one.
RUN_ID = "2b2b2b2b-2222-4222-8222-2222222222ab"
OUTPUT_DIR = "/tenant/runs/out-1"
CONFIG_SHA = "cfgsha"
TIMESTAMP = "2026-10-09T12:00:00+00:00"
DEV_HEADERS = {
    "x-pyromind-debug-user-id": "42",
    "x-pyromind-debug-user-name": "eval-tester",
}


class FakeStorage:
    """Conforming EvaluationStorage double with injectable failure modes."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.fail_writes: set[str] = set()
        self.fail_reads: set[str] = set()
        self.commits_file: Path | None = None
        self.pipeline_runs_file: Path | None = None

    def read(self, path: str) -> bytes:
        if path in self.fail_reads:
            raise EvaluationStorageUnavailableError(f"injected read failure for {path}")
        if path not in self.files:
            raise EvaluationStorageMissingError(f"missing {path}")
        return self.files[path]

    def write(self, path: str, payload: bytes) -> None:
        if path in self.fail_writes:
            raise EvaluationStorageUnavailableError(
                f"injected write failure for {path}"
            )
        self.files[path] = payload

    def list(self, prefix: str) -> set[str]:
        marker = prefix.rstrip("/") + "/"
        return {p[len(marker) :] for p in self.files if p.startswith(marker)}


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """One app + one conversation with a reserved run and seeded r0 products."""
    monkeypatch.setenv("APP_ENV", "dev")
    conversations = tmp_path / "conversations"
    conversation_dir = conversations / CONVERSATION_ID
    conversation_dir.mkdir(parents=True)
    store = FileProductStore(conversation_dir)
    store.create(
        ConversationSnapshot(
            conversation_id=CONVERSATION_ID, capabilities=HarnessCapabilities()
        ),
        user_id="42",
    )
    run = PipelineRun(
        run_id=uuid.UUID(RUN_ID),
        conversation_id=CONVERSATION_ID,
        definition={"kind": "evaluation"},
        fingerprint="fp-1",
        output_dir=OUTPUT_DIR,
    )
    store.save_pipeline_run(run, expected_revision=None)

    storage = FakeStorage()
    storage.commits_file = conversation_dir / "product" / "evaluation-commits.json"
    storage.pipeline_runs_file = conversation_dir / "product" / "pipeline-runs.json"
    storage.files[f"{OUTPUT_DIR}/run_manifest.json"] = json.dumps(
        {"run_id": RUN_ID, "evaluation_config_sha256": CONFIG_SHA}
    ).encode()
    storage.files[f"{OUTPUT_DIR}/predictions.jsonl"] = (
        json.dumps(
            {
                "id": "b",
                "passed": False,
                "reference": "hello",
                "prediction": "wrong",
                "overall_score": 0.0,
                "rubric_results": [{"name": "correct", "passed": False}],
            }
        )
        + "\n"
    ).encode()
    storage.files[f"{OUTPUT_DIR}/resolved_report_spec.json"] = b"{}"
    storage.files[f"{OUTPUT_DIR}/evaluation_annotations.r0.jsonl"] = (
        json.dumps(
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "evaluation_config_sha256": CONFIG_SHA,
                "case_id": "b",
                "review_status": "unreviewed",
                "human_annotation": None,
            }
        )
        + "\n"
    ).encode()
    storage.files[f"{OUTPUT_DIR}/labeled_evaluation_results.r0.jsonl"] = b"{}\n"

    app = FastAPI()
    app.state.config = Config(
        static_files_path=None, session_api_keys=["session-key"], secret_key=None
    )
    app.include_router(create_evaluation_runs_router())
    app.dependency_overrides[get_evaluation_storage] = lambda: storage
    app.state.product_runtime = ConversationRuntime(conversations, MagicMock())

    class _ConversationServiceStub:
        conversations_dir = conversations

    app.state.conversation_service = _ConversationServiceStub()
    client = TestClient(app, raise_server_exceptions=False)
    return client, storage


def base_url(conversation_id: str = CONVERSATION_ID, run_id: str = RUN_ID) -> str:
    return f"/api/v2/pyromind/conversations/{conversation_id}/evaluation-runs/{run_id}"


def reviewed_record() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "evaluation_config_sha256": CONFIG_SHA,
        "case_id": "b",
        "review_status": "reviewed",
        "human_annotation": {
            "decision": "confirmed_badcase",
            "failure_modes": ["model_miss"],
            "taxonomy_labels": [],
            "disposition": "fix_ground_truth",
            "severity": "P1",
            "comment": "",
            "corrected_answer": None,
            "reviewer_id": None,
            "updated_at": TIMESTAMP,
        },
    }


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-Session-API-Key": "not-a-local-key"},
        {
            "Cookie": "auth_token=opaque-storage-token; other=value",
            "Authorization": "Bearer storage-token",
            "x-cluster": "us-west-2#pre",
        },
    ],
)
def test_evaluation_only_uses_request_storage_credentials(
    environment, monkeypatch: pytest.MonkeyPatch, headers
) -> None:
    client, storage = environment
    monkeypatch.setenv("APP_ENV", "pre")
    client.app.state.config = client.app.state.config.model_copy(
        update={"enable_pyromind_jwt_auth": True}
    )
    received = []

    def build_storage(context: RequestContext):
        received.append(context)
        return storage

    monkeypatch.setattr(evaluation_routes, "build_evaluation_storage", build_storage)
    client.app.dependency_overrides.pop(get_evaluation_storage)
    storage.files[f"{OUTPUT_DIR}/evaluation_report.html"] = (
        b"<html><head></head></html>"
    )
    response = client.get(f"{base_url()}/report", headers=headers)
    assert response.status_code == 200
    assert received == [
        RequestContext(
            user_id="anonymous",
            cookie=headers.get("Cookie"),
            authorization=headers.get("Authorization"),
            x_cluster=headers.get("x-cluster"),
        )
    ]
    assert "opaque-storage-token" not in response.text
    assert "storage-token" not in response.text
    # A later request must not inherit the preceding request's credentials.
    assert client.get(f"{base_url()}/report").status_code == 200
    assert received[-1].user_id == "anonymous"
    assert received[-1].authorization is None
    assert "opaque-storage-token" not in (received[-1].cookie or "")
    assert received[-1].x_cluster is None


def test_bare_report_uses_conversation_route_after_restart(
    environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, storage = environment
    client.app.include_router(create_product_router())
    snapshot = client.get(
        f"/api/v2/pyromind/conversations/{CONVERSATION_ID}/snapshot",
        headers={**DEV_HEADERS, "x-cluster": "us-west-2#pre"},
    )
    assert snapshot.status_code == 200
    assert storage.pipeline_runs_file is not None
    conversations = storage.pipeline_runs_file.parents[2]
    client.app.state.product_runtime = ConversationRuntime(conversations, MagicMock())
    received = []

    def build_storage(context: RequestContext):
        received.append(context)
        return storage

    monkeypatch.setattr(evaluation_routes, "build_evaluation_storage", build_storage)
    client.app.dependency_overrides.pop(get_evaluation_storage)
    client.cookies.set("auth_token", "opaque-storage-cookie")
    storage.files[f"{OUTPUT_DIR}/evaluation_report.html"] = b"<html></html>"
    report = client.get(f"{base_url()}/report")
    assert report.status_code == 200
    assert received[-1].x_cluster == "us-west-2#pre"
    assert "set-cookie" not in report.headers
    assert not report.url.query

    for endpoint in (
        "annotations",
        "annotations/summary",
        "exports/evaluation_annotations.jsonl",
    ):
        response = client.get(f"{base_url()}/{endpoint}")
        assert response.status_code == 200
        assert received[-1].x_cluster == "us-west-2#pre"
        assert "auth_token=opaque-storage-cookie" in received[-1].cookie

    override = client.get(
        f"{base_url()}/annotations", headers={"x-cluster": "us-west-1#pre2"}
    )
    assert override.status_code == 200
    assert received[-1].x_cluster == "us-west-1#pre2"

    other_id = "conv-eval-other"
    (conversations / other_id).mkdir()
    FileProductStore(conversations / other_id).create(
        ConversationSnapshot(
            conversation_id=other_id, capabilities=HarnessCapabilities()
        ),
        user_id="42",
    )
    missing = client.get(f"{base_url(conversation_id=other_id)}/report")
    assert missing.status_code == 404
    assert received[-1].x_cluster is None


def test_evaluation_user_label_does_not_restrict_access(environment) -> None:
    client, _ = environment
    response = client.get(
        f"{base_url()}/annotations",
        headers={**DEV_HEADERS, "x-pyromind-debug-user-id": "43"},
    )
    assert response.status_code == 200


def test_non_evaluation_routes_still_require_auth(environment) -> None:
    client, _ = environment
    client.app.state.config = client.app.state.config.model_copy(
        update={"enable_pyromind_jwt_auth": True}
    )
    client.app.include_router(create_product_router())
    response = client.get(f"/api/v2/pyromind/conversations/{CONVERSATION_ID}/snapshot")
    assert response.status_code == 401


def test_anonymous_annotation_save_and_replay(environment) -> None:
    client, storage = environment
    payload = {
        "save_id": "anonymous-1",
        "base_revision": 0,
        "records": [reviewed_record()],
    }
    first = client.put(f"{base_url()}/annotations", json=payload)
    assert first.status_code == 200
    assert first.json()["revision"] == 1
    replay = client.put(f"{base_url()}/annotations", json=payload)
    assert replay.status_code == 200
    assert replay.json()["revision"] == 1
    saved = json.loads(storage.files[f"{OUTPUT_DIR}/evaluation_annotations.r1.jsonl"])
    assert saved["human_annotation"]["reviewer_id"] == "anonymous"
    exported = client.get(f"{base_url()}/exports/evaluation_annotations.jsonl")
    assert exported.status_code == 200
    assert json.loads(exported.text)["human_annotation"]["reviewer_id"] == "anonymous"


@pytest.mark.parametrize("failure", ["missing", "expired", "forbidden"])
def test_storage_rejection_cannot_be_bypassed_by_local_summary(
    environment, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    client, storage = environment
    saved = client.put(
        f"{base_url()}/annotations",
        json={"save_id": "warmup", "base_revision": 0, "records": [reviewed_record()]},
    )
    assert saved.status_code == 200
    assert client.get(f"{base_url()}/annotations/summary").status_code == 200
    original_read = storage.read

    def rejected_read(path: str) -> bytes:
        if path == f"{OUTPUT_DIR}/run_manifest.json":
            raise EvaluationStorageUnavailableError(failure)
        return original_read(path)

    monkeypatch.setattr(storage, "read", rejected_read)
    assert client.get(f"{base_url()}/annotations/summary").status_code == 503
    rejected = client.patch(
        f"{base_url()}/annotations",
        json={
            "save_id": "blocked",
            "base_revision": 1,
            "changes": [
                {
                    "case_id": "b",
                    "human_annotation": reviewed_record()["human_annotation"],
                }
            ],
        },
    )
    assert rejected.status_code == 503
    assert f"{OUTPUT_DIR}/evaluation_annotations.r2.jsonl" not in storage.files


@pytest.mark.parametrize("manifest", [b"{broken", b"[]", b'{"run_id":"another-run"}'])
def test_report_requires_manifest_bound_to_registered_run(
    environment, manifest
) -> None:
    client, storage = environment
    storage.files[f"{OUTPUT_DIR}/run_manifest.json"] = manifest
    storage.files[f"{OUTPUT_DIR}/evaluation_report.html"] = b"<html></html>"
    response = client.get(f"{base_url()}/report")
    assert response.status_code == 500
    assert "currently unavailable" in response.json()["detail"]


def test_unknown_run_is_not_found(environment) -> None:
    client, _ = environment
    response = client.get(
        f"{base_url(run_id='33333333-3333-4333-8333-333333333333')}/annotations",
        headers=DEV_HEADERS,
    )
    assert response.status_code == 404


def test_unknown_conversation_is_not_found(environment) -> None:
    client, _ = environment
    response = client.get(
        (
            "/api/v2/pyromind/conversations/does-not-exist"
            f"/evaluation-runs/{RUN_ID}/annotations"
        ),
        headers=DEV_HEADERS,
    )
    assert response.status_code == 404


def test_malformed_run_id_is_not_found(environment) -> None:
    client, _ = environment
    response = client.get(
        f"{base_url(run_id='not-a-uuid')}/annotations", headers=DEV_HEADERS
    )
    assert response.status_code == 404


def test_non_canonical_run_id_shares_one_commit_state(environment) -> None:
    """Uppercase/hyphen-less spellings address the same revision history."""
    client, storage = environment
    upper_url = base_url(run_id=RUN_ID.upper())
    dashless_url = base_url(run_id=RUN_ID.replace("-", ""))

    first = client.get(f"{upper_url}/annotations", headers=DEV_HEADERS)
    assert first.status_code == 200
    assert first.json()["revision"] == 0
    assert first.json()["revision"] == 0

    save = client.put(
        f"{upper_url}/annotations",
        headers={**DEV_HEADERS, "content-type": "application/json"},
        json={"save_id": "s1", "base_revision": 0, "records": [reviewed_record()]},
    )
    assert save.status_code == 200
    assert save.json()["revision"] == 1

    canonical = client.get(f"{base_url()}/annotations", headers=DEV_HEADERS)
    assert canonical.status_code == 200
    assert canonical.json()["revision"] == 1

    dashless = client.get(f"{dashless_url}/annotations", headers=DEV_HEADERS)
    assert dashless.status_code == 200
    assert dashless.json()["revision"] == 1

    # Saving on top of revision 1 from the dash-less spelling proves the
    # alternate forms never forked a second commit state (it would sit at
    # revision 0 and answer 409).
    next_save = client.put(
        f"{dashless_url}/annotations",
        headers={**DEV_HEADERS, "content-type": "application/json"},
        json={
            "save_id": "s2",
            "base_revision": 1,
            "records": [
                {
                    **reviewed_record(),
                    "review_status": "unreviewed",
                    "human_annotation": None,
                }
            ],
        },
    )
    assert next_save.status_code == 200
    assert next_save.json()["revision"] == 2

    assert storage.commits_file is not None
    states = json.loads(storage.commits_file.read_text())
    assert list(states) == [RUN_ID]


def test_get_annotations_returns_committed_state(environment) -> None:
    client, _ = environment
    response = client.get(f"{base_url()}/annotations", headers=DEV_HEADERS)
    assert response.status_code == 200
    payload = response.json()
    assert payload["revision"] == 0
    assert [record["case_id"] for record in payload["records"]] == ["b"]
    assert payload["overall_status"] == "in_progress"


def test_put_annotations_saves_and_returns_revision(environment) -> None:
    client, storage = environment
    response = client.put(
        f"{base_url()}/annotations",
        headers={**DEV_HEADERS, "content-type": "application/json"},
        json={"save_id": "s1", "base_revision": 0, "records": [reviewed_record()]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["revision"] == 1
    assert payload["reviewed"] == 1
    assert payload["overall_status"] == "completed"
    # The committed object is on storage and the views mirror it.
    assert f"{OUTPUT_DIR}/evaluation_annotations.r1.jsonl" in storage.files
    snapshot = json.loads(storage.files[f"{OUTPUT_DIR}/evaluation_annotations.json"])
    assert snapshot["revision"] == 1
    # Reviewer attribution: the HTTP route must inject the debug user's id
    # (DEV_HEADERS carries user_id=42) into the persisted annotation.
    committed_rows = [
        json.loads(line)
        for line in storage.files[f"{OUTPUT_DIR}/evaluation_annotations.r1.jsonl"]
        .decode("utf-8")
        .splitlines()
        if line.strip()
    ]
    reviewed_row = next(r for r in committed_rows if r["human_annotation"])
    assert reviewed_row["human_annotation"]["reviewer_id"] == "42"


def test_patch_annotations_merges_dirty_case_and_injects_reviewer(environment) -> None:
    client, storage = environment
    response = client.patch(
        f"{base_url()}/annotations",
        headers={**DEV_HEADERS, "content-type": "application/json"},
        json={
            "save_id": "patch-1",
            "base_revision": 0,
            "changes": [
                {
                    "case_id": "b",
                    "human_annotation": reviewed_record()["human_annotation"],
                }
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["revision"] == 1
    committed = [
        json.loads(line)
        for line in storage.files[f"{OUTPUT_DIR}/evaluation_annotations.r1.jsonl"]
        .decode()
        .splitlines()
    ]
    assert committed[0]["human_annotation"]["reviewer_id"] == "42"


def test_put_annotations_maps_conflict_validation_and_replay(
    environment,
) -> None:
    client, _ = environment
    body = {"save_id": "s1", "base_revision": 0, "records": [reviewed_record()]}
    ok = client.put(
        f"{base_url()}/annotations",
        headers={**DEV_HEADERS, "content-type": "application/json"},
        json=body,
    )
    assert ok.status_code == 200
    conflict = client.put(
        f"{base_url()}/annotations",
        headers={**DEV_HEADERS, "content-type": "application/json"},
        json={"save_id": "s2", "base_revision": 0, "records": [reviewed_record()]},
    )
    assert conflict.status_code == 409
    invalid = client.put(
        f"{base_url()}/annotations",
        headers={**DEV_HEADERS, "content-type": "application/json"},
        json={
            "save_id": "s3",
            "base_revision": 1,
            "records": [
                {
                    "schema_version": 1,
                    "run_id": RUN_ID,
                    "evaluation_config_sha256": CONFIG_SHA,
                    "case_id": "b",
                    "review_status": "reviewed",
                    "human_annotation": {
                        "decision": "confirmed_badcase",
                        "updated_at": TIMESTAMP,
                    },
                }
            ],
        },
    )
    assert invalid.status_code == 422
    assert any("incomplete" in e for e in invalid.json()["detail"]["errors"])
    mismatch = client.put(
        f"{base_url()}/annotations",
        headers={**DEV_HEADERS, "content-type": "application/json"},
        json={"save_id": "s1", "base_revision": 0, "records": [reviewed_record()]},
    )
    # A same-content replay of a committed save_id is idempotent: it returns
    # the stored result instead of re-validating against the new revision.
    assert mismatch.status_code == 200
    replayed = mismatch.json()
    assert replayed["revision"] == 1
    assert replayed["superseded_by"] is None
    followup = client.put(
        f"{base_url()}/annotations",
        headers={**DEV_HEADERS, "content-type": "application/json"},
        json={
            "save_id": "s4",
            "base_revision": 1,
            "records": [
                {
                    **reviewed_record(),
                    "review_status": "unreviewed",
                    "human_annotation": None,
                }
            ],
        },
    )
    assert followup.status_code == 200
    superseded = client.put(
        f"{base_url()}/annotations",
        headers={**DEV_HEADERS, "content-type": "application/json"},
        json={"save_id": "s1", "base_revision": 0, "records": [reviewed_record()]},
    )
    # Once a later revision exists, the replay still lands and reports which
    # revision superseded it.
    assert superseded.status_code == 200
    assert superseded.json()["revision"] == 1
    assert superseded.json()["superseded_by"] == 2
    differing = client.put(
        f"{base_url()}/annotations",
        headers={**DEV_HEADERS, "content-type": "application/json"},
        json={
            "save_id": "s1",
            "base_revision": 0,
            "records": [
                {
                    **reviewed_record(),
                    "review_status": "unreviewed",
                    "human_annotation": None,
                }
            ],
        },
    )
    assert differing.status_code == 422


def test_put_rejects_cross_origin(environment) -> None:
    client, _ = environment
    response = client.put(
        f"{base_url()}/annotations",
        headers={
            **DEV_HEADERS,
            "content-type": "application/json",
            "Origin": "https://evil.example.com",
        },
        json={"save_id": "s1", "base_revision": 0, "records": [reviewed_record()]},
    )
    assert response.status_code == 403


def test_export_downloads_current_revision_and_whitelists(environment) -> None:
    client, storage = environment
    save = client.put(
        f"{base_url()}/annotations",
        headers={**DEV_HEADERS, "content-type": "application/json"},
        json={"save_id": "s1", "base_revision": 0, "records": [reviewed_record()]},
    )
    assert save.status_code == 200
    jsonl = client.get(
        f"{base_url()}/exports/evaluation_annotations.jsonl", headers=DEV_HEADERS
    )
    assert jsonl.status_code == 200
    records = [json.loads(line) for line in jsonl.text.splitlines() if line]
    assert records[0]["review_status"] == "reviewed"
    labeled = client.get(
        f"{base_url()}/exports/labeled_evaluation_results.jsonl", headers=DEV_HEADERS
    )
    assert labeled.status_code == 200
    snapshot = client.get(
        f"{base_url()}/exports/evaluation_annotations.json", headers=DEV_HEADERS
    )
    assert snapshot.status_code == 200
    assert json.loads(snapshot.text)["revision"] == 1
    assert (
        client.get(f"{base_url()}/exports/report.json", headers=DEV_HEADERS).status_code
        == 404
    )
    traversal = client.get(
        f"{base_url()}/exports/..%2Freport.json", headers=DEV_HEADERS
    )
    assert traversal.status_code == 404


def test_cases_are_paginated_and_media_is_whitelisted(environment) -> None:
    client, storage = environment
    storage.files[f"{OUTPUT_DIR}/evaluation_cases_manifest.json"] = json.dumps(
        {
            "chunks": [
                {
                    "name": "evaluation_cases/chunk-00000.json",
                    "start": 0,
                    "count": 2,
                }
            ]
        }
    ).encode()
    storage.files[f"{OUTPUT_DIR}/evaluation_cases_index.json"] = json.dumps(
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
            ]
        }
    ).encode()
    storage.files[f"{OUTPUT_DIR}/evaluation_cases/chunk-00000.json"] = json.dumps(
        {"items": [{"id": "alpha"}, {"id": "beta"}]}
    ).encode()
    storage.files[f"{OUTPUT_DIR}/evaluation_media/00000000-000.png"] = b"png"

    alpha = reviewed_record()
    alpha["case_id"] = "alpha"
    beta = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "evaluation_config_sha256": CONFIG_SHA,
        "case_id": "beta",
        "review_status": "unreviewed",
        "human_annotation": None,
    }
    storage.files[f"{OUTPUT_DIR}/evaluation_annotations.r0.jsonl"] = (
        json.dumps(alpha) + "\n" + json.dumps(beta) + "\n"
    ).encode()

    first = client.get(f"{base_url()}/cases?limit=1")
    assert first.status_code == 200
    assert [item["id"] for item in first.json()["items"]] == ["alpha"]
    assert first.json()["items"][0]["review_status"] == "reviewed"
    assert first.json()["items"][0]["human_annotation"] is not None
    summary = client.get(f"{base_url()}/annotations/summary")
    assert summary.status_code == 200
    assert summary.json()["reviewed"] == 1
    assert summary.json()["unreviewed"] == 1
    assert "records" not in summary.json()
    cursor = first.json()["next_cursor"]
    second = client.get(f"{base_url()}/cases", params={"cursor": cursor})
    assert second.status_code == 200
    assert [item["id"] for item in second.json()["items"]] == ["beta"]

    filtered = client.get(
        f"{base_url()}/cases",
        headers=DEV_HEADERS,
        params={"review_status": "unreviewed"},
    )
    assert filtered.status_code == 200
    assert [item["id"] for item in filtered.json()["items"]] == ["beta"]
    assert filtered.json()["filtered_total"] == 1

    media = client.get(f"{base_url()}/media/00000000-000.png")
    assert media.status_code == 200
    assert media.content == b"png"
    assert media.headers["content-type"] == "image/png"
    assert "private" in media.headers["cache-control"]
    missing = client.get(f"{base_url()}/media/not-listed.png")
    assert missing.status_code == 404


def test_report_is_served_with_marker_and_csp(environment) -> None:
    client, storage = environment
    storage.files[f"{OUTPUT_DIR}/evaluation_report.html"] = (
        b"<!doctype html><html><head><title>t</title></head><body>b</body></html>"
    )
    storage.files[f"{OUTPUT_DIR}/evaluation_cases_manifest.json"] = b"{}"
    response = client.get(f"{base_url()}/report", headers=DEV_HEADERS)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Content-Security-Policy" in response.headers
    assert "img-src 'self'" in response.headers["Content-Security-Policy"]
    assert "window.__EVAL_REPORT_API__" in response.text
    assert '"cases":"cases"' in response.text
    assert '"media":"media/"' in response.text
    assert response.text.index("<head>") < response.text.index(
        "window.__EVAL_REPORT_API__"
    )
    assert response.text.index("window.__EVAL_REPORT_API__") < response.text.index(
        "<title>"
    )


def test_legacy_report_uses_embedded_cases_without_paged_api(environment) -> None:
    client, storage = environment
    storage.files[f"{OUTPUT_DIR}/evaluation_report.html"] = b"<html></html>"
    response = client.get(f"{base_url()}/report", headers=DEV_HEADERS)
    assert response.status_code == 200
    assert '"annotations":"annotations"' in response.text
    assert '"cases":"cases"' not in response.text


def test_report_without_storage_object_is_not_found(environment) -> None:
    client, _ = environment
    response = client.get(f"{base_url()}/report", headers=DEV_HEADERS)
    assert response.status_code == 404


def test_not_found_bodies_do_not_distinguish_existence(environment) -> None:
    """All missing or invalid run references retain the same public body."""
    client, _ = environment
    urls = [
        (
            "/api/v2/pyromind/conversations/does-not-exist"
            f"/evaluation-runs/{RUN_ID}/annotations"
        ),  # conversation missing
        # run unknown in this conversation:
        f"{base_url(run_id='44444444-4444-4444-8444-444444444444')}/annotations",
        f"{base_url(run_id='not-a-uuid')}/annotations",  # malformed run id
    ]
    bodies = []
    for url in urls:
        response = client.get(url, headers=DEV_HEADERS)
        assert response.status_code == 404
        bodies.append(response.text)
    assert len(set(bodies)) == 1
    assert "not found" in bodies[0]


def test_report_storage_failure_is_unavailable_not_missing(environment) -> None:
    """A storage outage must not be reported as "the report does not exist"."""
    client, storage = environment
    storage.files[f"{OUTPUT_DIR}/evaluation_report.html"] = b"<html></html>"
    storage.fail_reads.add(f"{OUTPUT_DIR}/evaluation_report.html")
    response = client.get(f"{base_url()}/report", headers=DEV_HEADERS)
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "temporarily unavailable" in detail
    assert "injected" not in detail


def test_storage_read_failure_during_save_is_server_error(environment) -> None:
    client, storage = environment
    storage.fail_reads.add(f"{OUTPUT_DIR}/predictions.jsonl")
    response = client.put(
        f"{base_url()}/annotations",
        headers={**DEV_HEADERS, "content-type": "application/json"},
        json={"save_id": "s1", "base_revision": 0, "records": [reviewed_record()]},
    )
    assert response.status_code == 500
    assert "nothing was committed" in response.json()["detail"]


def test_annotation_object_read_failure_is_server_error(environment) -> None:
    client, storage = environment
    storage.fail_reads.add(f"{OUTPUT_DIR}/evaluation_annotations.r0.jsonl")
    response = client.get(f"{base_url()}/annotations", headers=DEV_HEADERS)
    assert response.status_code == 500
    assert "loading annotations failed" in response.json()["detail"]


def test_corrupt_commit_state_is_server_error(environment) -> None:
    client, storage = environment
    warmup = client.get(f"{base_url()}/annotations", headers=DEV_HEADERS)
    assert warmup.status_code == 200
    assert storage.commits_file is not None
    storage.commits_file.write_text("{not json")
    response = client.get(f"{base_url()}/annotations", headers=DEV_HEADERS)
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "loading annotations failed" in detail
    assert "{not json" not in detail


def test_corrupt_pipeline_registry_is_server_error(environment) -> None:
    client, storage = environment
    assert storage.pipeline_runs_file is not None
    storage.pipeline_runs_file.write_text("garbage")
    response = client.get(f"{base_url()}/annotations", headers=DEV_HEADERS)
    assert response.status_code == 500
    assert "currently unavailable" in response.json()["detail"]


def test_corrupt_metadata_is_server_error(environment) -> None:
    client, storage = environment
    metadata_file = storage.pipeline_runs_file.parent / "meta.json"
    metadata_file.write_text("{")
    response = client.get(f"{base_url()}/annotations", headers=DEV_HEADERS)
    assert response.status_code == 500
    assert "currently unavailable" in response.json()["detail"]


def test_shared_registry_serializes_two_service_instances() -> None:
    """Phase 3 integration gate: per-request instances must share one lock."""
    from pyromind_runtime.application.evaluation_annotations import (
        EvaluationAnnotationService,
        EvaluationLockRegistry,
    )

    registry = EvaluationLockRegistry()

    class Owner:
        def __init__(self) -> None:
            self.service = EvaluationAnnotationService(
                storage=FakeStorage(), commits=FakeCommitStore(), locks=registry
            )

    first = Owner()
    second = Owner()
    lock_a = first.service._lock_for(RUN_ID)
    lock_b = second.service._lock_for(RUN_ID)
    with lock_a:
        done = threading.Event()

        def contender() -> None:
            with lock_b:
                done.set()

        thread = threading.Thread(target=contender)
        thread.start()
        assert not done.wait(0.2)
    thread.join(timeout=5)
    assert done.is_set()


class FakeCommitStore:
    def __init__(self) -> None:
        self.states: dict[str, EvaluationCommitState] = {}

    def load_evaluation_commit(self, run_id: str) -> EvaluationCommitState | None:
        return self.states.get(run_id)

    def save_evaluation_commit(self, state: EvaluationCommitState) -> None:
        self.states[state.run_id] = state


def test_export_serves_latest_even_when_view_writes_fail(environment) -> None:
    """Phase 3 review fix: downloads resolve through the commit state.

    View refreshes fail for every save (the views never exist at all), yet
    the exports still serve the latest committed revision assembled from
    the authoritative objects.
    """
    client, storage = environment
    storage.fail_writes = {
        f"{OUTPUT_DIR}/evaluation_annotations.jsonl",
        f"{OUTPUT_DIR}/labeled_evaluation_results.jsonl",
        f"{OUTPUT_DIR}/evaluation_annotations.json",
    }
    unreviewed = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "evaluation_config_sha256": CONFIG_SHA,
        "case_id": "b",
        "review_status": "unreviewed",
        "human_annotation": None,
    }
    for index, (save_id, records) in enumerate(
        [("s1", [unreviewed]), ("s2", [reviewed_record()])]
    ):
        response = client.put(
            f"{base_url()}/annotations",
            headers={**DEV_HEADERS, "content-type": "application/json"},
            json={"save_id": save_id, "base_revision": index, "records": records},
        )
        assert response.status_code == 200
    jsonl = client.get(
        f"{base_url()}/exports/evaluation_annotations.jsonl", headers=DEV_HEADERS
    )
    assert jsonl.status_code == 200
    records = [json.loads(line) for line in jsonl.text.splitlines() if line]
    assert records[0]["review_status"] == "reviewed"
    snapshot = client.get(
        f"{base_url()}/exports/evaluation_annotations.json", headers=DEV_HEADERS
    )
    assert snapshot.status_code == 200
    assert json.loads(snapshot.text)["revision"] == 2


def test_missing_runtime_is_a_503(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    app = FastAPI()
    app.state.config = Config(
        static_files_path=None, session_api_keys=[], secret_key=None
    )
    app.include_router(create_evaluation_runs_router())

    app.state.conversation_service = None
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get(f"{base_url()}/annotations", headers=DEV_HEADERS)
    assert response.status_code == 503
    assert "runtime" in response.json()["detail"].lower()


def test_origin_check_trust_boundary(environment, monkeypatch) -> None:
    client, _ = environment
    body = {"save_id": "s1", "base_revision": 0, "records": [reviewed_record()]}

    def put(origin: str | None, extra: dict[str, str] | None = None):
        headers = {**DEV_HEADERS, "content-type": "application/json"}
        if origin is not None:
            headers["Origin"] = origin
        if extra:
            headers.update(extra)
        return client.put(f"{base_url()}/annotations", headers=headers, json=body)

    forwarded = {
        "X-Forwarded-Proto": "https",
        "X-Forwarded-Host": "console.example.com",
    }
    # Direct requests: same origin passes, scheme mismatch is cross-origin.
    assert put("http://testserver").status_code != 403
    assert put("https://testserver").status_code == 403
    # Default: client-supplied forwarded headers are NOT trusted.
    assert put("https://console.example.com", forwarded).status_code == 403
    # Opting into trusted proxy headers lets the proxy values define the
    # origin — but only for their own origin, never for any other.
    monkeypatch.setenv("PYROMIND_TRUST_PROXY_HEADERS", "1")
    assert put("https://console.example.com", forwarded).status_code != 403
    assert put("https://evil.example.com", forwarded).status_code == 403
    monkeypatch.delenv("PYROMIND_TRUST_PROXY_HEADERS")
    # An explicit allowlist wins over everything, forwarded headers included.
    monkeypatch.setenv("PYROMIND_REPORT_ALLOWED_ORIGINS", "https://portal.example.com")
    assert put("https://portal.example.com").status_code != 403
    assert put("http://testserver").status_code == 403


def test_storage_list_rejects_business_failures(monkeypatch) -> None:
    """HTTP 200 with success=false must raise, not yield a fake listing."""
    import pyromind_agent_server.evaluation_storage as evaluation_storage_module

    adapter = PyromindEvaluationStorage(base_url="https://storage.test", headers={})

    class FakeResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    def business_failure(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            {
                "success": False,
                "message": "login expired",
                "data": {
                    "isLoggedIn": False,
                    "list": ["evaluation_annotations.r0.jsonl"],
                },
            }
        )

    monkeypatch.setattr(evaluation_storage_module.httpx, "post", business_failure)
    with pytest.raises(RuntimeError, match="login expired"):
        adapter.list("/runs/out")

    def missing_list(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse({"success": True, "data": {}})

    monkeypatch.setattr(evaluation_storage_module.httpx, "post", missing_list)
    with pytest.raises(RuntimeError, match="unexpected payload"):
        adapter.list("/runs/out")
