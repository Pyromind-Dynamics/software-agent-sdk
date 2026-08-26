import shutil
import subprocess

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


def _run_nproc_guard() -> str:
    """Run the nproc-guard expression emitted into the shell init command.

    Mirrors the guard logic in ``SubprocessTerminal.initialize`` so we can test
    the macOS branch (no usable /proc/self/uid_map) against a real shell.
    """
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    script = (
        "map=4294967295; "
        "if [ -r /proc/self/uid_map ]; then "
        "map=$(awk 'NR==1{print $3}' /proc/self/uid_map); fi; "
        '[ "$map" != "4294967295" ] '
        "&& ulimit -u 2 2>/dev/null; "
        'printf "%s" "$(ulimit -u)"'
    )
    return subprocess.run(
        [bash, "-c", script], capture_output=True, text=True, check=True
    ).stdout


def test_nproc_guard_skips_ulimit_without_usable_uid_map():
    """Without a private user namespace, ulimit -u must not be applied.

    ``/proc/self/uid_map`` is absent on macOS (and on Linux outside a user
    namespace), where applying ``ulimit -u 2`` would cap the whole shell and
    break pipelines that fork several children (``du | sort | head``).
    """
    nproc = _run_nproc_guard()
    assert int(nproc) > 2
