import pytest

from openhands.tools.terminal.terminal.subprocess_terminal import (
    _sandbox_nproc_limit,
    _sandbox_vmem_kb,
)


def test_sandbox_vmem_kb_defaults_to_500m(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OH_SANDBOX_VMEM_LIMIT", raising=False)
    assert _sandbox_vmem_kb() == 500 * 1024


def test_sandbox_vmem_kb_reads_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OH_SANDBOX_VMEM_LIMIT", "512M")
    assert _sandbox_vmem_kb() == 512 * 1024


def test_sandbox_vmem_kb_rejects_invalid(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OH_SANDBOX_VMEM_LIMIT", "banana")
    with pytest.raises(RuntimeError, match="OH_SANDBOX_VMEM_LIMIT"):
        _sandbox_vmem_kb()


def test_sandbox_nproc_limit_defaults_to_2(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OH_SANDBOX_NPROC_LIMIT", raising=False)
    assert _sandbox_nproc_limit() == 2


def test_sandbox_nproc_limit_reads_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OH_SANDBOX_NPROC_LIMIT", "3")
    assert _sandbox_nproc_limit() == 3


def test_sandbox_nproc_limit_rejects_invalid(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OH_SANDBOX_NPROC_LIMIT", "banana")
    with pytest.raises(RuntimeError, match="OH_SANDBOX_NPROC_LIMIT"):
        _sandbox_nproc_limit()


def test_sandbox_nproc_limit_requires_shell_plus_one(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OH_SANDBOX_NPROC_LIMIT", "1")
    with pytest.raises(RuntimeError, match="at least 2"):
        _sandbox_nproc_limit()
