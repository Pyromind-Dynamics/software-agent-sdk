"""Read progress from the mounted run without a Storage HTTP Tool."""

import json

import pytest

from pyromind_common.platform.progress import read_progress


@pytest.fixture
def mounted_run(tmp_path):
    mount = tmp_path / "storage"
    run = mount / "jobs/one"
    run.mkdir(parents=True)
    return mount, run


def test_reads_progress_counters_report_and_only_real_run_artifacts(mounted_run):
    mount, run = mounted_run
    (run / "progress.json").write_text(
        json.dumps(
            {
                "total": 20,
                "processed": 10,
                "succeeded": 9,
                "failed": 1,
                "elapsed_ms": 2000,
                "records_per_second": 5.0,
                "eta_ms": 2000,
                "updated_at": "2026-11-24T10:00:00Z",
            }
        )
    )
    (run / "report.html").write_text("<!doctype html><title>Report</title>")
    (run / "report.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "artifacts": {
                    "report.html": "/jobs/one/report.html",
                    "missing.html": "missing.html",
                    "outside.html": "/other/report.html",
                },
            }
        )
    )
    (run / "processed.jsonl").write_text('{"id": 1}\n{"id": 2}\n')

    result = read_progress("/workspace/jobs/one", storage_root=mount, tail_lines=1)

    assert result.output_dir == "/workspace/jobs/one"
    assert result.progress_found and result.report_found
    assert (result.total, result.processed, result.succeeded, result.failed) == (
        20,
        10,
        9,
        1,
    )
    assert result.percent == 50
    assert result.eta_ms == 2000 and result.records_per_second == 5
    assert result.latest_records == [{"id": 2}]
    assert result.tail_truncated
    assert result.artifact_paths == {"report.html": "/workspace/jobs/one/report.html"}
    assert "artifacts" in result.errors
    assert result.evaluation_report_url is None


def test_missing_progress_does_not_infer_completion_from_a_report(mounted_run):
    mount, run = mounted_run
    (run / "report.json").write_text('{"status": "succeeded"}')

    result = read_progress("/jobs/one", storage_root=mount)

    assert not result.progress_found
    assert result.percent is None and result.processed is None and result.total is None
    assert result.latest_records == []
    assert result.report == {"status": "succeeded"}
    assert result.errors == {}


def test_bounded_tail_retains_utf8_records_and_ignores_a_partial_last_row(mounted_run):
    mount, run = mounted_run
    contents = json.dumps({"padding": "字" * 1000}, ensure_ascii=False) + "\n"
    contents += "".join(
        json.dumps({"id": number, "text": "中文"}, ensure_ascii=False) + "\n"
        for number in (1, 2, 3)
    )
    contents += '{"id": 4, "text": "未完'
    (run / "verdicts.jsonl").write_text(contents)

    result = read_progress(
        "/jobs/one",
        storage_root=mount,
        tail_filename="verdicts.jsonl",
        tail_lines=2,
        tail_bytes=140,
    )

    assert result.latest_records == [
        {"id": 2, "text": "中文"},
        {"id": 3, "text": "中文"},
    ]
    assert result.tail_truncated
    assert (
        result.errors["verdicts.jsonl"] == "Skipped incomplete or invalid JSONL records"
    )


def test_complete_last_record_without_a_newline_is_read(mounted_run):
    mount, run = mounted_run
    (run / "processed.jsonl").write_text('{"id": 1}')

    result = read_progress("/jobs/one", storage_root=mount)

    assert result.latest_records == [{"id": 1}]
    assert not result.tail_truncated
    assert not result.errors


@pytest.mark.parametrize("payload", [b"x" * 33, b'{"processed": "oops"}', b"\xff"])
def test_invalid_or_oversize_progress_is_reported_without_raw_file_contents(
    mounted_run, payload
):
    mount, run = mounted_run
    (run / "progress.json").write_bytes(payload)

    result = read_progress("/jobs/one", storage_root=mount, max_json_bytes=32)

    assert not result.progress_found
    assert result.processed is None and result.percent is None
    assert result.errors["progress.json"]
    assert "oops" not in result.model_dump_json()


def test_symlinked_run_files_cannot_read_outside_the_selected_run(mounted_run):
    mount, run = mounted_run
    source = mount / "private.json"
    source.write_text('{"processed": 100}')
    (run / "progress.json").symlink_to(source)

    result = read_progress("/jobs/one", storage_root=mount)

    assert not result.progress_found
    assert "progress.json" in result.errors
    assert result.processed is None


@pytest.mark.parametrize("filename", ["../secret", "/secret", "run/verdicts.jsonl"])
def test_tail_filename_cannot_escape_the_run(mounted_run, filename):
    mount, _ = mounted_run
    with pytest.raises(ValueError, match="tail_filename"):
        read_progress("/jobs/one", storage_root=mount, tail_filename=filename)


def test_storage_root_is_explicit_and_a_symlink_cannot_escape_it(mounted_run, tmp_path):
    mount, _ = mounted_run
    (mount / "elsewhere").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        read_progress("/elsewhere", storage_root=mount)


def test_stable_report_url_is_only_an_explicit_confirmed_input(mounted_run):
    mount, run = mounted_run
    (run / "run_manifest.json").write_text(
        '{"run_id": "11111111-1111-1111-1111-111111111111"}'
    )
    assert read_progress("/jobs/one", storage_root=mount).evaluation_report_url is None
    confirmed = (
        "https://agent.test/api/v2/pyromind/conversations/c/evaluation-runs/r/report"
    )

    result = read_progress("/jobs/one", storage_root=mount, stable_report_url=confirmed)

    assert result.evaluation_report_url == confirmed
    with pytest.raises(ValueError, match="confirmed HTTP report URL"):
        read_progress(
            "/jobs/one",
            storage_root=mount,
            stable_report_url="https://agent.test/report?token=secret",
        )
