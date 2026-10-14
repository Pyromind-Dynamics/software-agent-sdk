"""Tests for the DataFlow pipeline progress check tool."""

import json
from typing import Any, ClassVar, cast

from openhands.tools.data_preparation.progress import (
    DfCheckProgressAction,
    DfCheckProgressExecutor,
)


def _make_executor(files: dict[str, bytes]) -> DfCheckProgressExecutor:
    executor = DfCheckProgressExecutor(storage_base_url="https://storage.test/api")

    def fake_get_size_and_url(path: str, headers: dict[str, str]):
        if path not in files:
            return f"not found: {path}"
        return len(files[path]), f"https://cdn.test{path}"

    def fake_download_range(url: str, start: int | None, end: int | None):
        path = url.removeprefix("https://cdn.test")
        data = files[path]
        if start is None or end is None:
            return data
        return data[start : end + 1]

    executor._get_size_and_url = fake_get_size_and_url
    executor._download_range = fake_download_range
    return executor


def test_check_progress_full_snapshot() -> None:
    progress = {
        "total": 100,
        "processed": 40,
        "succeeded": 38,
        "failed": 2,
        "elapsed_ms": 20000,
        "records_per_second": 2.0,
        "eta_ms": 30000,
        "updated_at": "2026-07-28T10:00:00Z",
    }
    processed = "".join(json.dumps({"id": i}) + "\n" for i in range(5))
    report = {"status": "running", "source_integrity": {"unchanged": True}}
    files = {
        "/run/progress.json": json.dumps(progress).encode(),
        "/run/processed.jsonl": processed.encode(),
        "/run/report.json": json.dumps(report).encode(),
    }
    executor = _make_executor(files)

    obs = executor(DfCheckProgressAction(output_dir="/run/"), conversation=None)

    assert obs.output_dir == "/run"
    assert obs.progress_found is True
    assert obs.total == 100
    assert obs.processed == 40
    assert obs.succeeded == 38
    assert obs.failed == 2
    assert obs.percent == 40.0
    assert obs.eta_ms == 30000
    assert len(obs.latest_records) == 5
    assert obs.latest_records[-1] == {"id": 4}
    assert obs.report_found is True
    assert obs.report == report
    assert "report.status=running" in obs.text


def test_check_progress_missing_progress_file() -> None:
    executor = _make_executor({})

    obs = executor(DfCheckProgressAction(output_dir="/run"), conversation=None)

    assert obs.progress_found is False
    assert obs.percent is None
    assert obs.total is None
    assert obs.latest_records == []
    assert obs.report_found is False
    assert obs.report is None


def test_check_progress_tail_drops_partial_first_line() -> None:
    records = [json.dumps({"id": i, "pad": "x" * 20}) for i in range(10)]
    processed = "".join(record + "\n" for record in records)
    progress = {"total": 10, "processed": 10, "succeeded": 10, "failed": 0}
    files = {
        "/run/progress.json": json.dumps(progress).encode(),
        "/run/processed.jsonl": processed.encode(),
    }
    executor = _make_executor(files)
    executor._tail_bytes = 100  # force a mid-line start

    obs = executor(
        DfCheckProgressAction(output_dir="/run", tail_lines=3),
        conversation=None,
    )

    assert obs.percent == 100.0
    assert len(obs.latest_records) <= 3
    assert all(isinstance(record, dict) for record in obs.latest_records)
    assert obs.latest_records[-1]["id"] == 9


def test_check_progress_tail_lines_zero_returns_no_records() -> None:
    """tail_lines=0 is valid and means no records are shown (# tail -0 pitfall)."""
    processed = "".join(json.dumps({"id": i}) + "\n" for i in range(5))
    progress = {"total": 5, "processed": 5, "succeeded": 5, "failed": 0}
    files = {
        "/run/progress.json": json.dumps(progress).encode(),
        "/run/processed.jsonl": processed.encode(),
    }
    executor = _make_executor(files)

    obs = executor(
        DfCheckProgressAction(output_dir="/run", tail_lines=0),
        conversation=None,
    )

    assert obs.percent == 100.0
    assert obs.latest_records == []


def test_report_urls_only_resolve_artifacts_in_the_run_directory() -> None:
    report = {
        "status": "succeeded",
        "artifacts": {
            "evaluation_report.html": "/run/evaluation_report.html",
            "outside.html": "/run/../other/report.html",
        },
    }
    executor = _make_executor(
        {
            "/run/report.json": json.dumps(report).encode(),
            "/run/evaluation_report.html": b"<!doctype html><html>report</html>",
            "/run/../other/report.html": b"private",
        }
    )
    observation = executor(DfCheckProgressAction(output_dir="/run"))
    assert observation.artifact_urls == {
        "evaluation_report.html": "https://cdn.test/run/evaluation_report.html",
    }


class _FakeSecretRegistry:
    def get_secret_value(self, name: str) -> str:
        return ""


class _FakeState:
    agent_state: ClassVar[dict] = {}
    secret_registry = _FakeSecretRegistry()


class _FakeConversation:
    id = "conv-stable-1"
    state = _FakeState()


RUN_ID = "44444444-4444-4444-8444-444444444444"
OWNER_DIR = f"/.pyromind-agent/conv-owner-9/data_preparation/{RUN_ID}"


def _eval_report() -> dict:
    return {
        "status": "succeeded",
        "evaluation_status": "validated",
        "report_spec": {"template_id": "evaluation-basic-v1"},
        "badcase_count": 1,
        "artifacts": {},
    }


def _eval_files(output_dir: str, run_id: str) -> dict[str, bytes]:
    report = _eval_report()
    report["artifacts"]["evaluation_report.html"] = (
        f"{output_dir}/evaluation_report.html"
    )
    return {
        f"{output_dir}/report.json": json.dumps(report).encode(),
        f"{output_dir}/run_manifest.json": json.dumps({"run_id": run_id}).encode(),
    }


def test_stable_report_link_uses_the_owning_conversation() -> None:
    """Cross-conversation check: the link must carry the run's owner.

    The run belongs to conv-owner-9; the tool runs inside conv-stable-1.
    Building the URL from the current chat would 404 at the report API,
    which resolves the run under its owning conversation.
    """
    executor = _make_executor(_eval_files(OWNER_DIR, RUN_ID))
    executor._public_api_base = "https://agent.test"

    obs = executor(
        DfCheckProgressAction(output_dir=OWNER_DIR + "/"),
        conversation=cast(Any, _FakeConversation()),
    )

    assert obs.evaluation_report_url == (
        "https://agent.test/api/v2/pyromind/conversations/conv-owner-9"
        f"/evaluation-runs/{RUN_ID}/report"
    )
    assert "Stable report link" in obs.text


def test_stable_report_link_works_for_the_current_chat_too() -> None:
    output_dir = f"/.pyromind-agent/conv-stable-1/data_preparation/{RUN_ID}"
    executor = _make_executor(_eval_files(output_dir, RUN_ID))
    executor._public_api_base = "https://agent.test"

    obs = executor(
        DfCheckProgressAction(output_dir=output_dir),
        conversation=cast(Any, _FakeConversation()),
    )

    assert obs.evaluation_report_url == (
        "https://agent.test/api/v2/pyromind/conversations/conv-stable-1"
        f"/evaluation-runs/{RUN_ID}/report"
    )


def test_stable_report_link_absent_when_output_dir_layout_unknown() -> None:
    # Without the platform layout the owning conversation cannot be
    # inferred, and a link built from the current chat would 404 — emit none.
    files = _eval_files("/run", RUN_ID)
    executor = _make_executor(files)

    obs = executor(
        DfCheckProgressAction(output_dir="/run/"),
        conversation=cast(Any, _FakeConversation()),
    )

    assert obs.evaluation_report_url is None


def test_stable_report_link_absent_when_path_run_id_disagrees() -> None:
    # A path whose run segment differs from the manifest would 404 at
    # PipelineRuns.resolve (output_dir mismatch), so the link is invalid.
    other_run = "55555555-5555-4555-8555-555555555555"
    output_dir = f"/.pyromind-agent/conv-owner-9/data_preparation/{other_run}"
    executor = _make_executor(_eval_files(output_dir, RUN_ID))

    obs = executor(
        DfCheckProgressAction(output_dir=output_dir),
        conversation=cast(Any, _FakeConversation()),
    )

    assert obs.evaluation_report_url is None


def test_stable_report_link_absent_for_non_uuid_run_id() -> None:
    # A non-UUID run_id is rejected by PipelineRuns.resolve — the report
    # API would 404 on this link, so the tool must not emit it.
    files = _eval_files(OWNER_DIR, "run-123")
    executor = _make_executor(files)

    obs = executor(
        DfCheckProgressAction(output_dir=OWNER_DIR),
        conversation=cast(Any, _FakeConversation()),
    )

    assert obs.evaluation_report_url is None


def test_stable_report_link_absent_without_html_artifact() -> None:
    # Without the report HTML artifact the report page cannot render.
    report = _eval_report()
    executor = _make_executor(
        {
            f"{OWNER_DIR}/report.json": json.dumps(report).encode(),
            f"{OWNER_DIR}/run_manifest.json": json.dumps({"run_id": RUN_ID}).encode(),
        }
    )

    obs = executor(
        DfCheckProgressAction(output_dir=OWNER_DIR),
        conversation=cast(Any, _FakeConversation()),
    )

    assert obs.evaluation_report_url is None


def test_stable_report_link_absent_for_non_evaluation_reports() -> None:
    report = {"status": "running", "source_integrity": {"unchanged": True}}
    executor = _make_executor(
        {
            f"{OWNER_DIR}/report.json": json.dumps(report).encode(),
            f"{OWNER_DIR}/run_manifest.json": json.dumps(
                {"run_id": "run-123"}
            ).encode(),
        }
    )

    obs = executor(
        DfCheckProgressAction(output_dir=OWNER_DIR),
        conversation=cast(Any, _FakeConversation()),
    )

    assert obs.evaluation_report_url is None


def test_stable_report_link_absent_without_conversation() -> None:
    executor = _make_executor(_eval_files(OWNER_DIR, RUN_ID))

    obs = executor(DfCheckProgressAction(output_dir=OWNER_DIR), conversation=None)

    assert obs.evaluation_report_url is None


def test_stable_report_link_absent_when_manifest_missing() -> None:
    report = _eval_report()
    report["artifacts"]["evaluation_report.html"] = (
        f"{OWNER_DIR}/evaluation_report.html"
    )
    executor = _make_executor({f"{OWNER_DIR}/report.json": json.dumps(report).encode()})

    obs = executor(
        DfCheckProgressAction(output_dir=OWNER_DIR),
        conversation=cast(Any, _FakeConversation()),
    )

    assert obs.evaluation_report_url is None
