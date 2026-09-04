#!/usr/bin/env python3
"""Frozen runtime for ProcessingProfile-driven sandbox batch validation.

Interprets a declarative ``ProcessingProfile`` (steps + verdict + output)
against a manifest of records. All control flow lives here — image-dedup
caching, resume, per-record sandbox cleanup — so a mis-authored profile can
only fail a run, never silently corrupt it.

Usage:
    python sandbox_runner.py \\
        --profile profiles/tmax-validation.json \\
        --manifest /path/to/manifest.jsonl \\
        --output-dir /path/to/run-dir \\
        --env pre --cluster us-west-1 \\
        [--auth-token <token>] \
        [--set LLM_BASE_URL=...] [--set LLM_AUTH_TOKEN=...] \
        [--set LLM_MODEL=...] \
        [--limit N]

``--auth-token`` falls back to the ``PYROMIND_AUTH_TOKEN`` environment
variable. Step params may reference CLI-provided secrets via
``{secret:KEY}`` placeholders; substituted values never reach the verdicts
file. The verdicts file under ``--output-dir`` doubles as the resume
checkpoint: re-running the same command skips already-judged records and
reuses verdicts cached per image.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import sys
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyromind_sdk.client.models import ResourceConfig, SandboxRequest, SandboxType

from openhands.sdk.profiles.processing_profile import (
    ProcessingProfile,
    ProcessingStep,
    VerdictRule,
)
from openhands.tools.sandbox import create_sandbox_api_client


if TYPE_CHECKING:
    from pyromind_sdk.client.sandbox import SandboxClient


logger = logging.getLogger("sandbox_runner")

DEFAULT_CPU = 4
DEFAULT_WAIT_TIMEOUT = 300
DEFAULT_EXEC_TIMEOUT = 300

# Sandbox lifecycle failure categories surfaced in verdicts.jsonl.
CATEGORY_CREATE_FAILED = "create_failed"
CATEGORY_PROBE_FAILED = "probe_failed"
CATEGORY_EXEC_FAILED = "exec_failed"
CATEGORY_VERIFIER_FAILED = "verifier_failed"
CATEGORY_PI_INSTALL_FAILED = "pi_install_failed"
CATEGORY_PI_RUN_FAILED = "pi_run_failed"
# Distinct from verifier_failed: the verifier script itself could not run
# because the image lacks an artifact/command it references. Those records
# are usually recoverable by fixing the image, not by discarding the data.
CATEGORY_VERIFIER_ENV_MISSING = "verifier_env_missing"

DEFAULT_AGENT_TIMEOUT = 1800
DEFAULT_POLL_INTERVAL = 10.0
PROBE_TIMEOUT = 30
RUNNING_MARK = "__RUNNING__"
# pi requires node >= 22.19 (its engines field) and tmax images ship node 12
# (or none), so install_pi always provisions this exact Node version.
PI_NODE_VERSION = "22.19.0"
PI_TASK_FILE = "/workspace/__tmax_pi_task__.md"
PI_RUN_SCRIPT = "/workspace/.pi_run.sh"
PI_INSTALL_LOG = "/workspace/.pi_install.log"
PI_EXIT_FILE = "/workspace/.pi_exit.txt"
PI_MODELS_REL = ".pi/agent/models.json"  # under $HOME
PI_INSTALL_EXIT_FILE = "/workspace/.pi_install_exit.txt"
PI_INSTALL_DONE = "/workspace/.pi_install.done"
PI_INSTALL_LOCK = "/opt/.pi_install.lock"
PI_RUN_LOCK = "/workspace/.pi_run.lock"
HEARTBEAT_SECONDS = 60.0

_SECRET_PLACEHOLDER = re.compile(r"\{secret:([A-Za-z0-9_]+)\}")


class StepError(Exception):
    """A profile step failed; ``category`` maps onto the verdict entry."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


class _RecordTemplate(dict):
    """format_map target that rejects missing manifest fields loudly."""

    def __missing__(self, key: str) -> str:
        raise StepError(
            CATEGORY_EXEC_FAILED, f"manifest record is missing field {key!r}"
        )


def _substitute(
    value: Any, record: dict[str, Any], secrets: dict[str, str] | None = None
) -> Any:
    """Replace ``{field}`` and ``{secret:KEY}`` placeholders in step params.

    ``{secret:KEY}`` slots are stashed before ``format_map`` runs — str.format
    would otherwise parse ``secret:KEY`` as field name ``secret`` plus a
    format spec — and resolved from the CLI-provided secrets afterwards.
    """
    if isinstance(value, str):
        secret_slots: dict[str, str] = {}

        def _stash(match: re.Match[str]) -> str:
            token = f"\x00secret{len(secret_slots)}\x00"
            secret_slots[token] = match.group(1)
            return token

        stashed = _SECRET_PLACEHOLDER.sub(_stash, value)
        substituted = stashed.format_map(_RecordTemplate(record))
        for token, key in secret_slots.items():
            secret_value = secrets.get(key) if secrets else None
            if not secret_value:
                raise StepError(
                    CATEGORY_EXEC_FAILED,
                    f"step references secret {key!r} but it was not provided "
                    f"(pass --set {key}=...)",
                )
            substituted = substituted.replace(token, str(secret_value))
        return substituted
    if isinstance(value, list):
        return [_substitute(item, record, secrets) for item in value]
    if isinstance(value, dict):
        return {k: _substitute(v, record, secrets) for k, v in value.items()}
    return value


@dataclass
class VerdictEntry:
    """One line of the verdicts artifact."""

    task_id: str
    image: str
    verdict: str  # "usable" | "error"
    exit_code: int | None = None
    error_category: str | None = None
    cached: bool = False
    reward: float | None = None
    note: str | None = None

    @classmethod
    def from_json(cls, line: str) -> VerdictEntry | None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or "task_id" not in data:
            return None
        reward = data.get("reward")
        return cls(
            task_id=str(data["task_id"]),
            image=str(data.get("image", "")),
            verdict=str(data.get("verdict", "error")),
            exit_code=data.get("exit_code"),
            error_category=data.get("error_category"),
            cached=bool(data.get("cached", False)),
            reward=float(reward) if reward is not None else None,
            note=data.get("note"),
        )


@dataclass
class RunSummary:
    total: int = 0
    usable: int = 0
    error: int = 0
    resumed: int = 0
    cached: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step primitives
# ---------------------------------------------------------------------------


def _create_sandbox(
    step: ProcessingStep, record: dict[str, Any], client: SandboxClient
) -> str:
    params = _substitute(step.params, record)
    image = params.get("image")
    if not image:
        raise StepError(CATEGORY_CREATE_FAILED, "create_sandbox requires 'image'")
    cpu = int(params.get("cpu", DEFAULT_CPU))
    memory = params.get("memory") or f"{cpu * 2}Gi"
    wait_timeout = int(params.get("wait_timeout", DEFAULT_WAIT_TIMEOUT))
    request = SandboxRequest(
        # A short unique suffix avoids INSTANCE_EXIST when a same-task sandbox
        # is still alive (concurrent runs of the same manifest, or a leaked
        # sandbox from an interrupted run): the name is a label, not an id.
        name=f"profile-{record.get('task_id', 'record')}-{uuid.uuid4().hex[:6]}",
        sandbox_type=SandboxType.CUSTOM,
        resources=ResourceConfig(cpu=str(cpu), memory=str(memory)),
        image=str(image),
    )
    try:
        created = client.create(request)
        if wait_timeout <= 0:
            return created.id
        reached = client.wait_for_sandbox_status(
            created.id, target_status="running", timeout=wait_timeout
        )
    except StepError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise StepError(CATEGORY_CREATE_FAILED, f"create sandbox failed: {exc}")
    if not reached:
        raise StepError(
            CATEGORY_CREATE_FAILED,
            f"sandbox {created.id} did not reach 'running' within {wait_timeout}s "
            "(image pull may have failed)",
        )
    return created.id


def _probe(
    step: ProcessingStep,
    record: dict[str, Any],
    client: SandboxClient,
    sandbox_id: str,
) -> None:
    params = _substitute(step.params, record)
    command = params.get("command")
    if not command:
        raise StepError(CATEGORY_PROBE_FAILED, "probe requires 'command'")
    timeout = int(params.get("timeout", 60))
    try:
        result = client.exec_command(sandbox_id, str(command), timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        raise StepError(CATEGORY_PROBE_FAILED, f"probe failed: {exc}")
    if result.returncode != 0:
        raise StepError(
            CATEGORY_PROBE_FAILED,
            f"probe exited {result.returncode} (missing workdir or bash): "
            f"{(result.output or '').strip()[:200]}",
        )


def _write_file(
    step: ProcessingStep,
    record: dict[str, Any],
    client: SandboxClient,
    sandbox_id: str,
) -> None:
    params = _substitute(step.params, record)
    path = params.get("path")
    content_field = params.get("content_field")
    if not path or not content_field:
        raise StepError(
            CATEGORY_EXEC_FAILED, "write_file requires 'path' and 'content_field'"
        )
    content = record.get(str(content_field))
    if content is None:
        raise StepError(
            CATEGORY_EXEC_FAILED,
            f"manifest record is missing field {content_field!r} for write_file",
        )
    try:
        client.write_file(sandbox_id, str(path), str(content).encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise StepError(CATEGORY_EXEC_FAILED, f"write_file failed: {exc}")


def _exec(
    step: ProcessingStep,
    record: dict[str, Any],
    client: SandboxClient,
    sandbox_id: str,
    secrets: dict[str, str] | None = None,
) -> tuple[int, str]:
    params = _substitute(step.params, record, secrets)
    command = params.get("command")
    if not command:
        raise StepError(CATEGORY_EXEC_FAILED, "exec requires 'command'")
    timeout = int(params.get("timeout", DEFAULT_EXEC_TIMEOUT))
    marker = str(params.get("marker") or "")
    if marker:
        # Step-level idempotency: a retried run skips a step whose marker
        # already exists instead of re-executing the command inside the
        # sandbox (markers live and die with the sandbox).
        probe = f"test -f {shlex.quote(marker)} && echo yes || echo no"
        marked = client.exec_command(sandbox_id, probe, timeout=PROBE_TIMEOUT)
        if (marked.output or "").strip() == "yes":
            logger.info("exec marker %s present; skipping step", marker)
            return 0, ""
    try:
        result = client.exec_command(sandbox_id, str(command), timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        raise StepError(CATEGORY_EXEC_FAILED, f"exec failed: {exc}")
    returncode = int(result.returncode)
    if marker and returncode == 0:
        client.exec_command(
            sandbox_id, f"touch {shlex.quote(marker)}", timeout=PROBE_TIMEOUT
        )
    return returncode, result.output or ""


def _install_pi(
    step: ProcessingStep,
    record: dict[str, Any],
    client: SandboxClient,
    sandbox_id: str,
    secrets: dict[str, str] | None = None,
) -> None:
    """Install Node + the pi coding agent and register the gateway in models.json.

    pi reads provider definitions from ~/.pi/agent/models.json; we write one
    entry ("mygw") pointing at the LLM gateway configured in the step env so
    pi talks to it directly (OpenAI chat-completions protocol).
    """
    params = _substitute(step.params, record, secrets)
    env = params.get("env") or {}
    base = str(env.get("LLM_BASE_URL") or "").strip().rstrip("/")
    token = str(env.get("LLM_AUTH_TOKEN") or "").strip()
    model = str(env.get("LLM_MODEL") or "").strip()
    if not (base and token and model):
        raise StepError(
            CATEGORY_PI_INSTALL_FAILED,
            "install_pi requires LLM_BASE_URL / LLM_AUTH_TOKEN / LLM_MODEL in step env",
        )
    base_v1 = base if base.endswith("/v1") else base + "/v1"
    timeout = int(params.get("timeout", DEFAULT_AGENT_TIMEOUT))
    poll_interval = float(params.get("poll_interval", DEFAULT_POLL_INTERVAL))
    # models.json must land where run_pi reads it: run_pi executes the agent
    # as the workspace user with HOME={workdir}, so the install shell targets
    # the same HOME instead of the root exec shell's (/root). The chmod keeps
    # it readable/writable after the su drop (root-owned files default 644).
    pi_home = str(record.get("workdir") or "/home/user")
    inner = rf"""set -e
export HOME={shlex.quote(pi_home)}
case "$(uname -m)" in aarch64 | arm64) narch=arm64 ;; *) narch=x64 ;; esac
dir="/opt/node-v{PI_NODE_VERSION}-linux-$narch"
if [ ! -x "$dir/bin/node" ]; then
    url="https://nodejs.org/dist/v{PI_NODE_VERSION}/node-v{PI_NODE_VERSION}-linux-$narch.tar.gz"
    python3 -c "import sys,urllib.request as u;u.urlretrieve(*sys.argv[1:])" \
        "$url" /tmp/node-dist.tar.gz
    mkdir -p /opt
    tar -xzf /tmp/node-dist.tar.gz -C /opt
    rm -f /tmp/node-dist.tar.gz
fi
export PATH="$dir/bin:$PATH"
if [ ! -x /opt/pi/node_modules/.bin/pi ]; then
    mkdir -p /opt/pi
    npm install --prefix /opt/pi --ignore-scripts --no-fund --no-audit \
        @earendil-works/pi-coding-agent >/tmp/pi_install.log 2>&1
fi
mkdir -p "$HOME/.pi/agent"
cat > "$HOME/{PI_MODELS_REL}" <<EOF2
{{
  "providers": {{
    "mygw": {{
      "name": "My Gateway",
      "baseUrl": "{shlex.quote(base_v1)}",
      "api": "openai-completions",
      "apiKey": "{shlex.quote(token)}",
      "models": [
        {{ "id": "{shlex.quote(model)}", "contextWindow": 1048576, "maxTokens": 8192 }}
      ]
    }}
  }}
}}
EOF2
chmod -R a+rwX "$HOME/.pi" 2>/dev/null || true
for b in node npm npx pi; do
    p=$(command -v "$b" 2>/dev/null || true)
    if [ -n "$p" ] && [ ! -e "/usr/local/bin/$b" ]; then
        ln -s "$p" "/usr/local/bin/$b" 2>/dev/null || true
    fi
done
ls /opt/pi/node_modules/.bin/pi >/dev/null
cat "$HOME/{PI_MODELS_REL}" >/dev/null
node --version
touch {PI_INSTALL_DONE}
"""
    script = (
        f"{{ {inner}}} >{shlex.quote(PI_INSTALL_LOG)} 2>&1 "
        f"|| {{ tail -30 {shlex.quote(PI_INSTALL_LOG)}; exit 1; }}"
    )
    # Detached install: the exec channel caps a single command well below a
    # cold npm install, so run the script with an exit-code file and poll it
    # with heartbeats instead of one blocking exec. The mkdir lock makes a
    # duplicated launch (HTTP-level retry) exit 9 instead of double-running
    # npm install in the same sandbox.
    cleanup = (
        f"rc=$?; rmdir {shlex.quote(PI_INSTALL_LOCK)} 2>/dev/null; "
        f"echo $rc > {shlex.quote(PI_INSTALL_EXIT_FILE)}"
    )
    launch = (
        f"rm -f {shlex.quote(PI_INSTALL_EXIT_FILE)} {shlex.quote(PI_INSTALL_DONE)}; "
        f"if mkdir {shlex.quote(PI_INSTALL_LOCK)} 2>/dev/null; then "
        f"nohup bash -c {shlex.quote(script + chr(10) + cleanup)} "
        ">/dev/null 2>&1 & echo launched; else "
        f"echo 9 > {shlex.quote(PI_INSTALL_EXIT_FILE)}; fi"
    )
    try:
        marked = client.exec_command(
            sandbox_id,
            f"test -f {shlex.quote(PI_INSTALL_DONE)} && echo yes || echo no",
            timeout=PROBE_TIMEOUT,
        )
        if (marked.output or "").strip() == "yes":
            logger.info("pi already installed in %s; skipping install", sandbox_id)
            return
        client.exec_command(sandbox_id, launch, timeout=60)
        exit_code = _poll_exit_file(
            client,
            sandbox_id,
            exit_file=PI_INSTALL_EXIT_FILE,
            timeout=timeout,
            poll_interval=poll_interval,
            label="pi install",
            progress_file=PI_INSTALL_LOG,
            kill_pattern="npm install",
        )
    except StepError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise StepError(CATEGORY_PI_INSTALL_FAILED, f"pi install failed: {exc}")
    if exit_code is None:
        raise StepError(
            CATEGORY_PI_INSTALL_FAILED,
            f"pi install timed out after {timeout}s: "
            f"{_remote_tail(client, sandbox_id, PI_INSTALL_LOG, limit=600)}",
        )
    if exit_code == 9:
        raise StepError(
            CATEGORY_PI_INSTALL_FAILED,
            "pi install is already running in this sandbox (install lock held)",
        )
    if exit_code != 0:
        raise StepError(
            CATEGORY_PI_INSTALL_FAILED,
            f"pi install exited {exit_code}: "
            f"{_remote_tail(client, sandbox_id, PI_INSTALL_LOG, limit=600)}",
        )


def _run_pi(
    step: ProcessingStep,
    record: dict[str, Any],
    client: SandboxClient,
    sandbox_id: str,
    secrets: dict[str, str] | None = None,
    trace_dir: Path | None = None,
) -> None:
    """Solve the task with the pi coding agent (OpenAI-completions gateway).

    pi reads the registered "mygw" provider/model from models.json (written by
    install_pi under the run user's HOME) and streams JSON events to the trace
    file. Like any long agent run it executes detached with an exit-code file
    poll: the platform exec channel caps a single command well below real
    solving time. Credentials live only in models.json, so the launch script
    needs nothing but the model name from the step env.
    """
    params = _substitute(step.params, record, secrets)
    env = params.get("env") or {}
    if not isinstance(env, dict):
        raise StepError(CATEGORY_PI_RUN_FAILED, "run_pi 'env' must be an object")
    workdir = params.get("workdir")
    if not workdir:
        raise StepError(CATEGORY_PI_RUN_FAILED, "run_pi requires 'workdir'")
    prompt = record.get(params.get("prompt_field", "prompt"))
    if prompt is None:
        raise StepError(
            CATEGORY_PI_RUN_FAILED,
            f"manifest record is missing field "
            f"{params.get('prompt_field', 'prompt')!r}",
        )
    model = str(env.get("LLM_MODEL") or "").strip()
    if not model:
        raise StepError(CATEGORY_PI_RUN_FAILED, "run_pi requires LLM_MODEL in env")
    run_as_user = str(params.get("run_as_user", "")).strip()
    export_trace = bool(params.get("export_trace", False))
    timeout = int(params.get("timeout", DEFAULT_AGENT_TIMEOUT))
    poll_interval = float(params.get("poll_interval", DEFAULT_POLL_INTERVAL))

    workdir = str(workdir)
    trace_path = f"{workdir}/.pi_trace.jsonl"
    err_path = f"{workdir}/.pi_run.err"

    try:
        client.write_file(sandbox_id, PI_TASK_FILE, str(prompt).encode("utf-8"))
        script = (
            f"cd {shlex.quote(workdir)} && "
            f"export HOME={shlex.quote(workdir)}; "
            f'/opt/pi/node_modules/.bin/pi -p "$(cat {PI_TASK_FILE})" '
            f"--mode json --provider mygw --model {shlex.quote(model)} "
            "--no-session --no-context-files "
            "--tools bash,write,read,edit,grep,find,ls "
            f" > {shlex.quote(trace_path)} 2> {shlex.quote(err_path)}\n"
        )
        client.write_file(sandbox_id, PI_RUN_SCRIPT, script.encode("utf-8"))
        if run_as_user:
            pi_home = f"HOME={shlex.quote(workdir)} bash "
            wrap = (
                f"su -s /bin/bash {shlex.quote(run_as_user)} "
                f"-c {shlex.quote(pi_home + PI_RUN_SCRIPT)}"
            )
        else:
            wrap = f"bash {PI_RUN_SCRIPT}"
        agent_cmd = (
            f"{wrap}; rc=$?; rmdir {shlex.quote(PI_RUN_LOCK)} 2>/dev/null; "
            f"echo $rc > {shlex.quote(PI_EXIT_FILE)}"
        )
        launch = (
            f"chmod 644 {PI_TASK_FILE} {PI_RUN_SCRIPT}; "
            f"rm -f {shlex.quote(PI_EXIT_FILE)}; "
            f"if mkdir {shlex.quote(PI_RUN_LOCK)} 2>/dev/null; then "
            f"nohup bash -c {shlex.quote(agent_cmd)} >/dev/null 2>&1 & "
            "echo launched; else "
            f"echo 9 > {shlex.quote(PI_EXIT_FILE)}; fi"
        )
        client.exec_command(sandbox_id, launch, timeout=60)
        exit_code = _poll_exit_file(
            client,
            sandbox_id,
            exit_file=PI_EXIT_FILE,
            timeout=timeout,
            poll_interval=poll_interval,
            label="pi agent",
            progress_file=trace_path,
            kill_pattern="node_modules/.bin/pi",
        )
    except StepError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise StepError(CATEGORY_PI_RUN_FAILED, f"pi run failed: {exc}")

    if trace_dir is not None and export_trace:
        _export_agent_trace(client, sandbox_id, trace_path, trace_dir, record)

    if exit_code is None:
        raise StepError(
            CATEGORY_PI_RUN_FAILED,
            f"pi did not finish within {timeout}s and was killed; "
            f"partial trace kept in {trace_path}",
        )
    if exit_code != 0:
        raise StepError(
            CATEGORY_PI_RUN_FAILED,
            f"pi exited {exit_code}: {_stderr_tail(client, sandbox_id, err_path)}",
        )


def _remote_tail(
    client: SandboxClient, sandbox_id: str, path: str, limit: int = 300
) -> str:
    """Best-effort tail of a file inside the sandbox, flattened to one line."""
    try:
        result = client.exec_command(
            sandbox_id, f"tail -c {limit} {shlex.quote(path)}", timeout=30
        )
    except Exception:  # noqa: BLE001
        return "unavailable"
    text = " ".join((result.output or "").split())
    return text[:200] or "empty"


def _remote_kill(client: SandboxClient, sandbox_id: str, pattern: str) -> None:
    try:
        client.exec_command(
            sandbox_id,
            f"pkill -9 -f {shlex.quote(pattern)} 2>/dev/null; true",
            timeout=30,
        )
    except Exception:  # noqa: BLE001
        logger.warning("failed to kill %r in %s", pattern, sandbox_id)


def _poll_exit_file(
    client: SandboxClient,
    sandbox_id: str,
    *,
    exit_file: str,
    timeout: int,
    poll_interval: float,
    label: str,
    progress_file: str | None = None,
    kill_pattern: str | None = None,
) -> int | None:
    """Poll a detached command's exit-code file; None on timeout.

    The exec channel streams no output while a detached stage runs, so a
    heartbeat (elapsed + best-effort progress tail) is logged every
    HEARTBEAT_SECONDS to keep long installs/solves visible in the platform
    log instead of a multi-minute silent gap.
    """
    probe = (
        f"test -f {shlex.quote(exit_file)} && cat {shlex.quote(exit_file)} "
        f"|| echo {RUNNING_MARK}"
    )
    deadline = time.monotonic() + timeout
    next_beat = time.monotonic() + HEARTBEAT_SECONDS
    while True:
        result = client.exec_command(sandbox_id, probe, timeout=PROBE_TIMEOUT)
        out = (result.output or "").strip()
        if out.isdigit():
            return int(out)
        now = time.monotonic()
        if progress_file is not None and now >= next_beat and now < deadline:
            logger.info(
                "%s still running in %s: elapsed=%ds tail=%s",
                label,
                sandbox_id,
                int(timeout - (deadline - now)),
                _remote_tail(client, sandbox_id, progress_file),
            )
            next_beat = now + HEARTBEAT_SECONDS
        if now >= deadline:
            break
        time.sleep(poll_interval)
    if progress_file is not None:
        logger.warning(
            "%s timed out in %s after %ds; last output: %s",
            label,
            sandbox_id,
            timeout,
            _remote_tail(client, sandbox_id, progress_file, limit=600),
        )
    if kill_pattern is not None:
        _remote_kill(client, sandbox_id, kill_pattern)
    return None


def _export_agent_trace(
    client: SandboxClient,
    sandbox_id: str,
    trace_path: str,
    trace_dir: Path,
    record: dict[str, Any],
) -> None:
    """Pull the agent trace back so training-data conversion has a local input."""
    task_id = str(record.get("task_id", "")) or "record"
    try:
        raw = client.read_file(sandbox_id, trace_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("no agent trace to export for %s: %s", task_id, exc)
        return
    target = Path(trace_dir) / f"{task_id}.pi_trace.jsonl"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    except OSError as exc:
        logger.warning("failed to write trace for %s: %s", task_id, exc)


def _stderr_tail(client: SandboxClient, sandbox_id: str, err_path: str) -> str:
    try:
        result = client.exec_command(
            sandbox_id, f"tail -c 600 {shlex.quote(err_path)}", timeout=30
        )
    except Exception:  # noqa: BLE001
        return "no stderr captured"
    return (result.output or "").strip()[:200] or "no stderr captured"


def _delete_sandbox(client: SandboxClient, sandbox_id: str) -> None:
    """Tear the record's sandbox down: kill leftovers, pause, then delete.

    The platform keeps instances Running after their workloads finish and
    rejects DELETE on Running instances with a 400, so pausing first is the
    normal path, not error recovery; direct-delete only covers instances
    that already stopped on their own.
    """
    # Background processes (agent, npm) keep the instance Running and would
    # also wedge the pause, so kill them best-effort first.
    _remote_kill(client, sandbox_id, "node_modules/.bin/pi")
    _remote_kill(client, sandbox_id, "npm install")
    _remote_kill(client, sandbox_id, ".pi_run.sh")
    try:
        try:
            client.pause(sandbox_id)
        except Exception as exc:  # noqa: BLE001
            # Already stopped or pause unsupported — let the delete decide.
            logger.info("pause before delete skipped for %s: %s", sandbox_id, exc)
        try:
            client.delete(sandbox_id)
        except Exception as exc:  # noqa: BLE001
            if "can not delete" not in str(exc).lower():
                raise
            # The pause raced the state transition; one beat, then retry.
            time.sleep(2)
            client.pause(sandbox_id)
            client.delete(sandbox_id)
    except Exception as exc:  # noqa: BLE001
        # Cleanup is best-effort: a leaked sandbox must not mask the verdict,
        # but it must be loud in the logs.
        logger.warning("failed to delete sandbox %s: %s", sandbox_id, exc)


# ---------------------------------------------------------------------------
# Per-record validation
# ---------------------------------------------------------------------------


def decide_verdict(
    rule: VerdictRule,
    exit_code: int | None,
    error_category: str | None,
    reward: float | None = None,
) -> tuple[str, str | None]:
    """Map a run outcome onto ``(verdict, error_category)``.

    With ``kind='reward_file'`` a parsed reward marks the record usable
    (any 0-1 value is a valid training signal; the value itself is recorded
    on the entry); a missing/unparsable reward falls back to exit-code
    matching.
    """
    if error_category is not None:
        return "error", error_category
    if rule.kind == "reward_file" and reward is not None:
        return "usable", None
    if exit_code is not None and exit_code in rule.success_codes:
        return "usable", None
    return "error", CATEGORY_VERIFIER_FAILED


def _read_reward(
    client: SandboxClient, sandbox_id: str, reward_path: str
) -> float | None:
    """Read the verifier's reward file; return None when unusable."""
    try:
        raw = client.read_file(sandbox_id, reward_path)
        return float(raw.decode("utf-8").strip())
    except Exception:  # noqa: BLE001
        return None


def validate_record(
    profile: ProcessingProfile,
    record: dict[str, Any],
    client: SandboxClient,
    secrets: dict[str, str] | None = None,
    trace_dir: Path | None = None,
) -> VerdictEntry:
    """Run one manifest record through the profile steps inside a sandbox."""
    sandbox_id: str | None = None
    exit_code: int | None = None
    error_category: str | None = None
    reward: float | None = None
    exec_output = ""
    failure_note: str | None = None
    try:
        for step in profile.steps:
            if step.name == "create_sandbox":
                if sandbox_id is not None:
                    raise StepError(
                        CATEGORY_CREATE_FAILED, "duplicate create_sandbox step"
                    )
                sandbox_id = _create_sandbox(step, record, client)
            elif step.name == "probe":
                _probe(step, record, client, _require_sandbox(sandbox_id, step))
            elif step.name == "write_file":
                _write_file(step, record, client, _require_sandbox(sandbox_id, step))
            elif step.name == "exec":
                exit_code, exec_output = _exec(
                    step, record, client, _require_sandbox(sandbox_id, step), secrets
                )
            elif step.name == "install_pi":
                _install_pi(
                    step, record, client, _require_sandbox(sandbox_id, step), secrets
                )
            elif step.name == "run_pi":
                _run_pi(
                    step,
                    record,
                    client,
                    _require_sandbox(sandbox_id, step),
                    secrets,
                    trace_dir=trace_dir,
                )
            elif step.name == "delete_sandbox":
                pass  # cleanup is frozen in the finally block below
        if (
            error_category is None
            and sandbox_id is not None
            and profile.verdict.kind == "reward_file"
        ):
            reward = _read_reward(client, sandbox_id, profile.verdict.reward_path)
    except StepError as exc:
        error_category = exc.category
        error_detail = str(exc)
        logger.info(
            "record %s failed at profile step: %s", record.get("task_id"), error_detail
        )
        # Surface the failure reason on the verdict itself: the sandbox is
        # deleted in the finally block, so step-level detail (e.g. the agent
        # stderr tail) would otherwise be lost for offline diagnosis.
        failure_note = error_detail
    finally:
        if sandbox_id is not None:
            _delete_sandbox(client, sandbox_id)

    verdict, category = decide_verdict(
        profile.verdict, exit_code, error_category, reward
    )
    note = failure_note
    if category == CATEGORY_VERIFIER_FAILED and note is None:
        # A verifier that cannot find its own artifacts means the image is
        # missing pieces, not that the record is unusable; keep it visible
        # as a separate bucket for later image repair.
        for marker in ("No such file or directory", "command not found"):
            if marker in exec_output:
                category = CATEGORY_VERIFIER_ENV_MISSING
                note = f"verifier output contains {marker!r} (image may lack artifacts)"
                break
    return VerdictEntry(
        task_id=str(record.get("task_id", "")),
        image=str(record.get("image", "")),
        verdict=verdict,
        exit_code=exit_code,
        error_category=category,
        reward=reward,
        note=note,
    )


def _require_sandbox(sandbox_id: str | None, step: ProcessingStep) -> str:
    if sandbox_id is None:
        raise StepError(
            CATEGORY_EXEC_FAILED,
            f"step {step.name!r} runs before any create_sandbox step",
        )
    return sandbox_id


# ---------------------------------------------------------------------------
# Batch run (resume + image-dedup cache)
# ---------------------------------------------------------------------------


def load_existing_verdicts(
    verdicts_path: Path,
) -> tuple[set[str], dict[str, VerdictEntry]]:
    """Rebuild the resume checkpoint and per-image cache from prior output."""
    completed: set[str] = set()
    image_cache: dict[str, VerdictEntry] = {}
    if not verdicts_path.exists():
        return completed, image_cache
    for line in verdicts_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = VerdictEntry.from_json(line)
        if entry is None:
            continue
        completed.add(entry.task_id)
        # Only real (non-cached) verdicts seed the image cache.
        if not entry.cached and entry.image:
            image_cache[entry.image] = entry
    return completed, image_cache


def run_batch(
    profile: ProcessingProfile,
    records: Iterable[dict[str, Any]],
    output_dir: Path,
    client: SandboxClient,
    *,
    limit: int | None = None,
    secrets: dict[str, str] | None = None,
    dedup_by_image: bool = False,
) -> RunSummary:
    """Validate every record, appending verdicts (the resume checkpoint).

    ``dedup_by_image`` reuses a prior verdict for records sharing an image.
    It is safe only for environment-only probes; with per-record harness
    runs (install_pi/run_pi) each record must be judged on its own prompt.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    verdicts_path = output_dir / profile.output.filename
    completed, image_cache = load_existing_verdicts(verdicts_path)
    summary = RunSummary()
    records = list(records)
    total = len(records)

    with open(verdicts_path, "a", encoding="utf-8") as out:
        for record in records:
            if limit is not None and summary.total >= limit:
                break
            summary.total += 1
            task_id = str(record.get("task_id", ""))
            if task_id in completed:
                summary.resumed += 1
                continue

            cached = (
                image_cache.get(str(record.get("image", "")))
                if dedup_by_image
                else None
            )
            if cached is not None:
                entry = VerdictEntry(
                    task_id=task_id,
                    image=cached.image,
                    verdict=cached.verdict,
                    exit_code=cached.exit_code,
                    error_category=cached.error_category,
                    cached=True,
                    reward=cached.reward,
                )
                summary.cached += 1
            else:
                entry = validate_record(
                    profile, record, client, secrets, trace_dir=output_dir / "traces"
                )
                if entry.image:
                    image_cache[entry.image] = entry

            out.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
            out.flush()
            logger.info(
                "[%d/%d] task_id=%s verdict=%s reward=%s%s",
                summary.total,
                total,
                task_id,
                entry.verdict,
                entry.reward,
                " (cached)" if entry.cached else "",
            )
            completed.add(task_id)
            if entry.verdict == "usable":
                summary.usable += 1
            else:
                summary.error += 1
                if entry.error_category:
                    summary.errors.append(f"{task_id}: {entry.error_category}")

    return summary


# ---------------------------------------------------------------------------
# IO + CLI
# ---------------------------------------------------------------------------


def load_profile(path: Path) -> ProcessingProfile:
    return ProcessingProfile.model_validate_json(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest line {lineno} is not valid JSON: {exc}")
        if not isinstance(data, dict):
            raise ValueError(f"manifest line {lineno} is not a JSON object")
        records.append(data)
    return records


def _default_client_factory(
    *, env: str, cluster: str, auth_token: str
) -> SandboxClient:
    client = create_sandbox_api_client(
        env=env, cluster=cluster, auth_token=auth_token, headers={}
    )
    return client.sandboxes


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[..., SandboxClient] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--env", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Secrets referenced by step params as {secret:KEY} "
            "(e.g. --set LLM_AUTH_TOKEN=sk-...); repeatable"
        ),
    )
    parser.add_argument(
        "--dedup-by-image",
        action="store_true",
        help=(
            "Reuse verdicts across records sharing an image "
            "(environment-only probes; never safe with run_pi)"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    auth_token = args.auth_token or os.environ.get("PYROMIND_AUTH_TOKEN")
    if not auth_token:
        logger.error("auth token required via --auth-token or PYROMIND_AUTH_TOKEN")
        return 2

    secrets: dict[str, str] = {}
    for item in args.set:
        if "=" not in item:
            logger.error("--set expects KEY=VALUE, got %r", item)
            return 2
        key, _, value = item.partition("=")
        secrets[key] = value

    profile = load_profile(args.profile)
    records = load_manifest(args.manifest)
    factory = client_factory or _default_client_factory
    client = factory(env=args.env, cluster=args.cluster, auth_token=auth_token)

    summary = run_batch(
        profile,
        records,
        args.output_dir,
        client,
        limit=args.limit,
        secrets=secrets,
        dedup_by_image=args.dedup_by_image,
    )
    logger.info(
        "run complete: total=%d usable=%d error=%d resumed=%d cached=%d",
        summary.total,
        summary.usable,
        summary.error,
        summary.resumed,
        summary.cached,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
