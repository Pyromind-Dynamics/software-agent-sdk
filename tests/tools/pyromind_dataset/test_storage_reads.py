import httpx
import pytest

from openhands.tools.pyromind_dataset import definition as storage


@pytest.fixture
def download(monkeypatch):
    waits = []
    monkeypatch.setattr(storage.time, "sleep", waits.append)

    def run(handler, *, max_bytes=1024):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            monkeypatch.setattr(httpx, "post", client.post)
            monkeypatch.setattr(httpx, "stream", client.stream)
            return storage.download_file_from_pyromind(
                storage_path="/.pyromind-agent/conversation/public_data/report.md",
                storage_base_url="https://portal.test/storage_api",
                headers={"cookie": "test-request-credential"},
                timeout=1,
                max_bytes=max_bytes,
            )

    return run, waits


@pytest.mark.parametrize("phase", ["get_url", "download"])
@pytest.mark.parametrize("failure", [502, 503, 504, "timeout"])
def test_temporary_report_read_failures_retry_without_changing_path(
    download, phase, failure
):
    run, waits = download
    attempts = []
    paths = []

    def handler(request):
        current = "get_url" if request.url.host == "portal.test" else "download"
        if current == "get_url":
            paths.append(request.content)
            assert request.headers["cookie"] == "test-request-credential"
        else:
            assert "cookie" not in request.headers
            assert "authorization" not in request.headers
        if current == phase:
            attempts.append(current)
            if len(attempts) < 3:
                if failure == "timeout":
                    raise httpx.ReadTimeout(
                        "sensitive transport detail", request=request
                    )
                return httpx.Response(failure, text="sensitive server detail")
        if current == "get_url":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {"url": "https://signed.test/report?secret=value"},
                },
            )
        return httpx.Response(200, content=b"# Report")

    assert run(handler) == b"# Report"
    assert len(attempts) == 3 and waits == [0.5, 1.0]
    assert len(set(paths)) == 1


@pytest.mark.parametrize("status", [401, 403, 404, 503])
def test_report_read_failure_classification_and_retry_limit(download, status):
    run, waits = download
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="private URL and credential")

    expected = (
        storage.StorageFileNotFoundError if status == 404 else storage.StorageReadError
    )
    with pytest.raises(expected) as error:
        run(handler)
    assert len(calls) == (3 if status == 503 else 1)
    assert waits == ([0.5, 1.0] if status == 503 else [])
    assert "private" not in str(error.value)
    assert "credential" not in str(error.value)


def test_report_size_failure_is_not_retried(download):
    run, waits = download
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.host == "portal.test":
            return httpx.Response(
                200,
                json={"success": True, "data": {"url": "https://signed.test/report"}},
            )
        return httpx.Response(200, content=b"too large")

    with pytest.raises(ValueError, match="exceeds"):
        run(handler, max_bytes=2)
    assert len(calls) == 2 and waits == []
