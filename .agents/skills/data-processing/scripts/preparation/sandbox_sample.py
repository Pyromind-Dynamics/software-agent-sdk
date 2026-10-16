#!/usr/bin/env python3
"""Run one DataFlow sample inside the sandbox that owns the workspace.

Invoke this script through terminal either with the single-value flags
(``--pipeline`` plus ``--input``/``--output``) or, when a run needs fields they
do not carry, with ``--config`` and a non-secret JSON config. Input, output,
model profiles and validation remain inside the execution sandbox. The legacy
``--spec`` transport remains available for older recorded executions.

The interpreter is resolved in this order:

1. ``python`` from the run spec, set by ``PYROMIND_SANDBOX_DATAFLOW_PYTHON``: it
   wins when it exists in this sandbox and matches the pin, and is an error when
   it exists with another version. A path that only exists on the agent host is
   ignored, because the sandbox is a different filesystem.
2. An interpreter baked into the sandbox image at one of the well-known paths,
   when its ``open-dataflow`` matches the pin.
3. The sandbox's own ``python3`` when ``open-dataflow`` matches the pin.
4. A venv at ``venv`` in the run spec, created on first use and reused.

``--check`` reports which interpreter the run would use without changing the
sandbox; ``--prepare`` resolves the same interpreter and creates the cached venv
when it is missing. Both exist so a caller never has to probe for the runtime
with ``import dataflow`` or a filesystem search.

The first-use install takes the lock next to the venv, so a warm-up and a real
run sharing one sandbox cannot corrupt each other's venv.

Only the envelope, the run's log tails, and the artifacts the pipeline writes
cross the sandbox boundary. Nothing is copied to the agent-server host.
"""

from __future__ import annotations

import argparse
import ast
import base64
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


HEARTBEAT_SECONDS = 20
LOG_TAIL_CHARS = 6000
ENVELOPE_LINE_WIDTH = 120
_PREFLIGHT_TIMEOUT_SECONDS = 30
# Suffixes DataFlow's file storages can pick a reader for. A produced artifact that
# carries any other name is aliased to the type its content actually is.
_DATAFLOW_SUFFIXES = frozenset(
    {".json", ".jsonl", ".csv", ".parquet", ".pickle", ".xlsx"}
)
_SNIFF_BYTES = 65536

DATAFLOW_VERSION = "1.0.11"
DATAFLOW_RUNTIME_PACKAGES = (
    f"open-dataflow=={DATAFLOW_VERSION}",
    "numpy==1.26.4",
    "Pillow==12.1.1",
    "opencv-python-headless==4.10.0.84",
    "matplotlib==3.9.4",
)
DEFAULT_DATAFLOW_VENV = f"/tmp/pyromind-dataflow-venv-{DATAFLOW_VERSION}"
# Interpreters a sandbox image may bake in at build time. They are probed in
# order after the run spec's explicit ``python`` and win only when their
# ``open-dataflow`` matches the pin.
IMAGE_DATAFLOW_PYTHONS = (
    "/opt/pyromind/dataflow-venv/bin/python",
    "/home/openhands/dataflow-venv/bin/python",
)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _interpreter_version(python: str) -> str | None:
    """Return the installed ``open-dataflow`` version, or None."""
    probe = (
        "import importlib.metadata, sys\n"
        "try:\n"
        "    print(importlib.metadata.version('open-dataflow'))\n"
        "except importlib.metadata.PackageNotFoundError:\n"
        "    sys.exit(3)\n"
    )
    try:
        result = subprocess.run(
            [python, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip() or None


def _system_python(spec: dict[str, Any]) -> str:
    configured = spec.get("system_python")
    if isinstance(configured, str) and configured:
        return configured
    return shutil.which("python3") or sys.executable


def _interpreter_spec() -> dict[str, Any]:
    """Interpreter settings shared by the CLI config and the check entry."""
    return {
        "python": os.environ.get("PYROMIND_SANDBOX_DATAFLOW_PYTHON"),
        "venv": os.environ.get("PYROMIND_SANDBOX_DATAFLOW_VENV", DEFAULT_DATAFLOW_VENV),
        "dataflow_version": DATAFLOW_VERSION,
        "packages": list(DATAFLOW_RUNTIME_PACKAGES),
        "system_python": os.environ.get("PYROMIND_SANDBOX_SYSTEM_PYTHON"),
    }


def _image_interpreters() -> tuple[str, ...]:
    """Candidate interpreters a prepared sandbox image may already provide."""
    home = Path.home() / "dataflow-venv" / "bin" / "python"
    return tuple(dict.fromkeys([str(home), *IMAGE_DATAFLOW_PYTHONS]))


def _cached_venv_python(spec: dict[str, Any]) -> str | None:
    """The cached venv interpreter when its stamp matches the pin."""
    venv = Path(str(spec["venv"]))
    python = venv / "bin" / "python"
    stamp = venv / "open-dataflow-version"
    if not python.is_file() or not stamp.is_file():
        return None
    try:
        if stamp.read_text(encoding="utf-8").strip() != str(spec["dataflow_version"]):
            return None
    except OSError:
        return None
    return str(python)


def _configured_python(configured: str) -> str | None:
    """The configured interpreter when this sandbox actually has it.

    The value can come from the agent host, where the same variable names a
    different path, so a missing interpreter falls back to the rest of the
    resolution order instead of failing the run.
    """
    if os.sep in configured:
        return configured if Path(configured).is_file() else None
    return shutil.which(configured)


@contextmanager
def _install_lock(venv: Path) -> Iterator[None]:
    """Serialize first-use installs for one cached venv."""
    lock_path = Path(f"{venv}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _bootstrap_venv(spec: dict[str, Any]) -> tuple[str | None, str]:
    """Create the cached DataFlow venv when it is missing or stale."""
    venv = Path(str(spec["venv"]))
    expected = str(spec["dataflow_version"])
    python = venv / "bin" / "python"
    with _install_lock(venv):
        cached = _cached_venv_python(spec)
        if cached is not None:
            return cached, "reused"
        system_python = _system_python(spec)
        _log(f"installing open-dataflow=={expected} into {venv} (first use)")
        venv.parent.mkdir(parents=True, exist_ok=True)
        if venv.exists():
            shutil.rmtree(venv, ignore_errors=True)
        try:
            created = subprocess.run(
                [system_python, "-m", "venv", str(venv)],
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"could not create {venv}: {exc}"
        if created.returncode != 0:
            return None, (
                f"could not create {venv}: "
                f"{(created.stderr or created.stdout or '').strip()[-2000:]}"
            )
        packages = [str(name) for name in spec["packages"]]
        installed = subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--use-deprecated=legacy-resolver",
                *packages,
            ],
            capture_output=True,
            text=True,
            timeout=float(spec.get("install_timeout", 3600)),
        )
        if installed.returncode != 0:
            shutil.rmtree(venv, ignore_errors=True)
            detail = (installed.stderr or installed.stdout or "").strip()[-2000:]
            return None, f"could not install {', '.join(packages)}: {detail}"
        (venv / "open-dataflow-version").write_text(expected, encoding="utf-8")
        return str(python), "installed"


def _resolve_interpreter(
    spec: dict[str, Any], *, bootstrap: bool = True
) -> tuple[str | None, str]:
    expected = str(spec["dataflow_version"])
    configured = spec.get("python")
    if isinstance(configured, str) and configured:
        resolved = _configured_python(configured)
        if resolved is None:
            _log(
                f"configured interpreter {configured} is not present in this "
                "sandbox; resolving from the image instead"
            )
        else:
            found = _interpreter_version(resolved)
            if found != expected:
                return None, (
                    f"configured interpreter {configured} has open-dataflow "
                    f"{found or 'missing'}, expected {expected}"
                )
            return resolved, "configured"
    for candidate in _image_interpreters():
        if _interpreter_version(candidate) == expected:
            return candidate, "image"
    system_python = _system_python(spec)
    if _interpreter_version(system_python) == expected:
        return system_python, "image"
    cached = _cached_venv_python(spec)
    if cached is not None:
        return cached, "reused"
    if not bootstrap:
        return None, (
            f"no interpreter with open-dataflow=={expected} is available; "
            f"run this script with --prepare to install {spec['venv']}"
        )
    return _bootstrap_venv(spec)


def _interpreter_report(spec: dict[str, Any], *, prepare: bool) -> dict[str, Any]:
    """Describe the interpreter a run would use, optionally provisioning it."""
    python, detail = _resolve_interpreter(spec, bootstrap=prepare)
    report: dict[str, Any] = {
        "marker": "prepare" if prepare else "check",
        "ready": python is not None,
        "expected": str(spec["dataflow_version"]),
        "venv": str(spec["venv"]),
    }
    if python is None:
        report["detail"] = detail
        return report
    report["interpreter"] = python
    report["source"] = detail
    report["version"] = _interpreter_version(python)
    return report


def _preflight(env: dict[str, str], profile: str) -> str | None:
    """Verify the sandbox can reach the model endpoint before a long run."""
    if profile == "none" or os.environ.get("DF_SKIP_PREFLIGHT"):
        return None
    url = env.get("DF_API_URL")
    model = env.get("DF_MODEL_NAME")
    if not url or not model:
        return None
    headers = {"content-type": "application/json"}
    api_key = env.get("DF_API_KEY")
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
    ).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(
            request, timeout=_PREFLIGHT_TIMEOUT_SECONDS
        ) as reply:
            body = reply.read(300).decode("utf-8", "replace")
            status = reply.status
    except urllib.error.HTTPError as exc:
        detail = exc.read(300).decode("utf-8", "replace")
        if exc.code in (401, 403):
            return f"auth failed (HTTP {exc.code}); check DF_API_KEY"
        if exc.code in (400, 404):
            return (
                f"rejected model {model!r} (HTTP {exc.code}: {detail}); "
                "check DF_MODEL_NAME"
            )
        return f"failed (HTTP {exc.code}): {detail}"
    except (urllib.error.URLError, OSError) as exc:
        return (
            f"the sandbox cannot reach {url!r}: {type(exc).__name__}: {exc}. "
            "Model calls happen inside the sandbox, so the endpoint must be "
            "reachable from there."
        )
    if status == 200:
        return None
    if not body.lstrip().startswith(("{", "[")):
        return (
            f"got a non-JSON response (HTTP {status}); DF_API_URL likely points "
            "at a web page rather than a chat-completions endpoint"
        )
    return f"failed (HTTP {status}): {body}"


class _Keepalive:
    """Print a heartbeat while a silent step runs.

    The sandbox terminal bridge drops an idle connection, so a step that prints
    nothing for minutes needs periodic output to stay attached.
    """

    def __init__(self, seconds: int = HEARTBEAT_SECONDS) -> None:
        self._seconds = seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self._seconds):
            print("...", flush=True)

    def __enter__(self) -> _Keepalive:
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()


def _run_step(
    python: str,
    args: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout: float,
) -> tuple[int, str, str]:
    """Run one step, streaming its output and keeping the tail for the envelope."""
    process = subprocess.Popen(
        [python, *args],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def pump(stream: Any, sink: list[str]) -> None:
        for line in iter(stream.readline, ""):
            sink.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        stream.close()

    readers = [
        threading.Thread(
            target=pump, args=(process.stdout, stdout_chunks), daemon=True
        ),
        threading.Thread(
            target=pump, args=(process.stderr, stderr_chunks), daemon=True
        ),
    ]
    for reader in readers:
        reader.start()
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        for reader in readers:
            reader.join(timeout=5)
        tail = f"\nTimed out after {int(timeout)}s"
        return (
            124,
            "".join(stdout_chunks)[-LOG_TAIL_CHARS:],
            "".join(stderr_chunks)[-LOG_TAIL_CHARS:] + tail,
        )
    for reader in readers:
        reader.join(timeout=5)
    return (
        return_code,
        "".join(stdout_chunks)[-LOG_TAIL_CHARS:],
        "".join(stderr_chunks)[-LOG_TAIL_CHARS:],
    )


def _source_digest(target: str) -> str:
    path = Path(target)
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = (
        [path]
        if path.is_file()
        else sorted(item for item in path.rglob("*") if item.is_file())
    )
    for item in files:
        if path.is_dir():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(python: str, script: Path, target: str) -> tuple[str | None, str]:
    """Fingerprint the source so the run can prove it stayed read-only.

    A frozen package can carry this CLI without its sibling helper, so a missing
    helper falls back to hashing inline instead of failing the run.
    """
    if script.is_file():
        result = subprocess.run(
            [python, str(script), target],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            return None, (result.stderr or "").strip()[-2000:]
        return (result.stdout or "").strip(), ""
    try:
        return _source_digest(target), ""
    except OSError as exc:
        return None, str(exc)


def _read_output_records(path: str | None, limit: int = 3) -> tuple[list[Any], int]:
    if not path:
        return [], 0
    output = Path(path)
    records: list[Any] = []
    total = 0
    try:
        with output.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                if len(records) < limit:
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        records.append(value)
    except OSError:
        return [], 0
    return records, total


def _read_report_failure(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    report = Path(path)
    if not report.is_file():
        return {}
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    failure = payload.get("failure")
    if not isinstance(failure, dict):
        return {}
    nested = failure.get("failure")
    return nested if isinstance(nested, dict) else failure


def _classify(
    *,
    pipeline_rc: int,
    validation_rc: int | None,
    report_rc: int | None,
    source_integrity_rc: int,
    report_failure: dict[str, Any],
    timeout: int,
) -> tuple[int, str | None, str | None, str | None]:
    """Map the step results onto the tool's failure vocabulary."""
    if source_integrity_rc != 0:
        rc = pipeline_rc if pipeline_rc != 0 else 90
        return (
            rc,
            "source_integrity",
            "source_data_modified",
            "Pipeline modified its source data.",
        )
    if pipeline_rc == 124 or validation_rc == 124 or report_rc == 124:
        step = (
            "pipeline"
            if pipeline_rc == 124
            else "validation"
            if validation_rc == 124
            else "report"
        )
        return (
            124,
            "timeout",
            f"dataflow_{step}_timeout",
            f"DataFlow {step} timed out after {timeout}s.",
        )
    if pipeline_rc != 0:
        stage = report_failure.get("stage")
        error = report_failure.get("error")
        return (
            pipeline_rc,
            stage if isinstance(stage, str) and stage.strip() else "pipeline_execution",
            "dataflow_pipeline_failed",
            error
            if isinstance(error, str) and error.strip()
            else f"DataFlow pipeline exited with code {pipeline_rc}.",
        )
    if validation_rc not in (None, 0):
        return (
            validation_rc,
            "schema_validation",
            "dataflow_schema_validation_failed",
            f"DataFlow output schema validation exited with code {validation_rc}.",
        )
    if report_rc not in (None, 0):
        return (
            report_rc,
            "report_generation",
            "dataflow_report_generation_failed",
            f"DataFlow report generation exited with code {report_rc}.",
        )
    return 0, None, None, None


def _pipeline_args(spec: dict[str, Any]) -> list[str]:
    """Positional arguments for the pipeline, freezing a support file if set."""
    args = [str(arg) for arg in spec["args"]]
    support = spec.get("support_file")
    if not support:
        return args
    support_dir = Path(str(spec["support_dir"]))
    support_dir.mkdir(parents=True, exist_ok=True)
    frozen = support_dir / Path(str(support)).name
    shutil.copy2(str(support), str(frozen))
    return [*args, str(frozen)]


def _run(spec: dict[str, Any]) -> dict[str, Any]:
    marker = str(spec["marker"])
    profile = str(spec.get("model_profile", "text"))
    inherited = {
        k: v
        for k, v in os.environ.items()
        if k != "PYROMIND_DATAFLOW_PROFILES" and not k.startswith("DF_")
    }
    env = {**inherited, **{str(k): str(v) for k, v in spec["env"].items()}}
    timeout = int(spec["timeout"])
    runtime_dir = Path(str(spec["runtime_dir"]))
    python_path = os.pathsep.join(
        part for part in (str(runtime_dir), env.get("PYTHONPATH", "")) if part
    )
    env["PYTHONPATH"] = python_path
    output_schema = spec.get("output_schema")
    output_path = spec.get("output_path")
    input_path = spec.get("input_path")
    log_dir = spec.get("log_dir")
    report_path = f"{log_dir}/report.json" if log_dir else None
    for directory in (log_dir, spec.get("state_dir")):
        if directory:
            Path(str(directory)).mkdir(parents=True, exist_ok=True)

    def envelope(**fields: Any) -> dict[str, Any]:
        return {"marker": marker, **fields}

    with _Keepalive():
        python, detail = _resolve_interpreter(spec)
    if python is None:
        return envelope(
            rc=1,
            pipeline_rc=1,
            validation_rc=None,
            report_rc=None,
            source_integrity_rc=0,
            stdout_tail="",
            stderr_tail=detail,
            record_count=0,
            sample_records=[],
            report_failure={},
            failure_stage="runtime_dependency",
            error_code="dataflow_not_installed",
            error_message=(
                "The sandbox has no usable DataFlow runtime. "
                f"{detail}\nInstall it into the sandbox image (set "
                "PYROMIND_SANDBOX_DATAFLOW_PYTHON), or run this script with "
                "--prepare once the sandbox can reach PyPI so it can build the "
                "cached venv."
            ),
            interpreter=None,
            interpreter_source="unavailable",
        )
    failure = _preflight(env, profile)
    if failure is not None:
        return envelope(
            rc=1,
            pipeline_rc=1,
            validation_rc=None,
            report_rc=None,
            source_integrity_rc=0,
            stdout_tail="",
            stderr_tail=failure,
            record_count=0,
            sample_records=[],
            report_failure={},
            failure_stage="model_configuration",
            error_code="dataflow_llm_preflight_failed",
            error_message=f"DataFlow LLM preflight failed: {failure}",
            interpreter=python,
            interpreter_source=detail,
        )

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    fingerprint_before: str | None = None
    fingerprint_after: str | None = None
    fingerprint_error: str | None = None
    source_integrity_rc = 0
    if input_path:
        fingerprint_before, fingerprint_error = _fingerprint(
            python, runtime_dir / "source_fingerprint.py", str(input_path)
        )
        if fingerprint_before is None:
            return envelope(
                rc=1,
                pipeline_rc=1,
                validation_rc=None,
                report_rc=None,
                source_integrity_rc=0,
                stdout_tail="",
                stderr_tail=fingerprint_error,
                record_count=0,
                sample_records=[],
                report_failure={},
                failure_stage="input_resolution",
                error_code="workspace_input_not_found",
                error_message=f"Invalid standard pipeline input: {fingerprint_error}",
                interpreter=python,
                interpreter_source=detail,
            )

    pipeline_rc, out, err = _run_step(
        python,
        [str(spec["pipeline"]), *_pipeline_args(spec)],
        cwd=str(spec["cwd"]),
        env=env,
        timeout=timeout,
    )
    stdout_parts.append(out)
    stderr_parts.append(err)

    if input_path:
        fingerprint_after, fingerprint_error = _fingerprint(
            python, runtime_dir / "source_fingerprint.py", str(input_path)
        )
        if fingerprint_before != fingerprint_after:
            source_integrity_rc = 90
            stderr_parts.append(
                "\nSource data changed during pipeline execution; "
                "the source must remain read-only."
            )
    if log_dir and input_path:
        Path(log_dir, "source_integrity.json").write_text(
            json.dumps(
                {
                    "before": fingerprint_before,
                    "after": fingerprint_after,
                    "unchanged": source_integrity_rc == 0,
                    "error": fingerprint_error,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    validation_rc: int | None = None
    if pipeline_rc == 0 and output_schema not in (None, "structured"):
        validation_args = [
            str(runtime_dir / "validate_prepared_data.py"),
            str(output_path),
            "--schema",
            str(output_schema),
            "--report",
            str(Path(log_dir or spec["state_dir"], "validation.json")),
        ]
        if output_schema == "vision" and spec.get("image_root"):
            validation_args.extend(["--image-root", str(spec["image_root"])])
        validation_rc, out, err = _run_step(
            python,
            validation_args,
            cwd=str(spec["cwd"]),
            env=env,
            timeout=min(timeout, 600),
        )
        stdout_parts.append(f"--- validation ---\n{out}")
        stderr_parts.append(f"--- validation ---\n{err}")

    report_rc: int | None = None
    if output_path and log_dir:
        report_args = [
            str(runtime_dir / "generate_report.py"),
            "--log-dir",
            str(log_dir),
            "--pipeline-exit-code",
            str(pipeline_rc),
            "--execution-revision",
            str(spec.get("execution_revision", 1)),
            "--resumed",
            "false",
            "--output-file",
            str(output_path),
        ]
        fingerprint = spec.get("runtime_fingerprint")
        if fingerprint:
            report_args.extend(["--runtime-fingerprint", str(fingerprint)])
        report_rc, out, err = _run_step(
            python,
            report_args,
            cwd=str(spec["cwd"]),
            env=env,
            timeout=min(timeout, 600),
        )
        stdout_parts.append(f"--- report ---\n{out}")
        stderr_parts.append(f"--- report ---\n{err}")

    record_count: int | None = None
    sample_records: list[Any] = []
    if pipeline_rc == 0 and output_path and Path(str(output_path)).is_file():
        sample_records, record_count = _read_output_records(str(output_path))
    report_failure = _read_report_failure(report_path)
    rc, failure_stage, error_code, error_message = _classify(
        pipeline_rc=pipeline_rc,
        validation_rc=validation_rc,
        report_rc=report_rc,
        source_integrity_rc=source_integrity_rc,
        report_failure=report_failure,
        timeout=timeout,
    )
    if rc != 0 and source_integrity_rc != 0 and pipeline_rc == 0:
        stderr_parts.append("\nPipeline modified its source data.")
    return envelope(
        rc=rc,
        pipeline_rc=pipeline_rc,
        validation_rc=validation_rc,
        report_rc=report_rc,
        source_integrity_rc=source_integrity_rc,
        stdout_tail="".join(stdout_parts)[-LOG_TAIL_CHARS:],
        stderr_tail="".join(stderr_parts)[-LOG_TAIL_CHARS:],
        record_count=record_count,
        sample_records=sample_records,
        report_failure=report_failure,
        failure_stage=failure_stage,
        error_code=error_code,
        error_message=error_message,
        interpreter=python,
        interpreter_source=detail,
        output_exists=bool(output_path) and Path(str(output_path)).is_file(),
        input_alias=spec.get("input_alias"),
    )


def _emit_envelope(envelope: dict[str, Any], marker: str) -> None:
    """Print the result as wrapped base64 between two marker lines.

    The tool reads this back through the sandbox terminal bridge, which echoes
    the command and may re-wrap a long line. Base64 keeps the payload free of
    whitespace, so re-wrapping is harmless once the tool joins the lines.
    """
    encoded = base64.b64encode(
        json.dumps(envelope, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    print(f"__PM_DF_BEGIN__{marker}")
    for start in range(0, len(encoded), ENVELOPE_LINE_WIDTH):
        print(encoded[start : start + ENVELOPE_LINE_WIDTH])
    print(f"__PM_DF_END__{marker}")


def validate_cli_image_pipeline(pipeline: Path) -> None:
    """Keep managed vision pipelines declarative when called without a Tool."""
    source = pipeline.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(pipeline))
    compile(tree, str(pipeline), "exec")
    helper = ast.parse((Path(__file__).parent / "image_utils.py").read_text())
    public = next(
        ast.literal_eval(n.value)
        for n in helper.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
    )
    forbidden = {
        "base64",
        "dataflow",
        "httpx",
        "openai",
        "preparation_runtime",
        "requests",
        "urllib",
    }
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.comprehension)):
            raise ValueError(
                "Managed image pipelines must be configuration-only; no loops"
            )
        if isinstance(node, ast.Import):
            if any(
                n.name.split(".")[0] in forbidden | {"image_utils"} for n in node.names
            ):
                raise ValueError("Use explicit supported image_utils imports")
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in forbidden:
                raise ValueError(
                    "Managed image pipelines cannot implement model transport"
                )
            if node.module == "image_utils":
                names = {n.name for n in node.names}
                if names - set(public):
                    raise ValueError("Unsupported image_utils API")
                imported.update(names)
    if not {"ImagePipelineConfig", "run_image_pipeline_from_cli"} <= imported:
        raise ValueError(
            "Managed image pipelines require ImagePipelineConfig "
            "and run_image_pipeline_from_cli"
        )


def _layout_root(cwd: Path) -> Path:
    """The directory that owns ``public_data/``, for the storage whitelist.

    A run started by a caller that owns the workspace runs with the workspace root
    as its cwd. A caller replaying a frozen graph runs a package directory nested
    inside it (``public_data/workflow-runs/<run>/package``), so the whitelist is
    anchored at the nearest ancestor carrying ``public_data/`` instead of cwd.
    """
    for candidate in (cwd, *cwd.parents):
        if (candidate / "public_data").is_dir():
            return candidate
    return cwd


def _is_json_document(line: str) -> bool:
    try:
        json.loads(line)
    except (json.JSONDecodeError, RecursionError):
        return False
    return True


def _sniff_suffix(path: Path) -> str | None:
    """Guess the DataFlow file type of a file whose name does not carry one."""
    try:
        with path.open("rb") as stream:
            head = stream.read(_SNIFF_BYTES)
    except OSError:
        return None
    text = head.decode("utf-8", "replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    first = lines[0]
    if first in {"{", "["} or first.startswith("["):
        return ".json"
    if len(lines) > 1 and all(_is_json_document(line) for line in lines[:-1]):
        return ".jsonl"
    if _is_json_document(first):
        return ".jsonl"
    if _is_json_document(text):
        return ".json"
    return ".csv" if "," in first or "\t" in first else ".txt"


def _alias_input(source: Path, directory: Path) -> dict[str, Any] | None:
    """Expose an unknown-extension input under the type its content is.

    DataFlow picks its reader from the file extension, so an artifact produced by
    an earlier graph node (``artifacts/<node>.out``) cannot be read as-is. The
    original file stays untouched; the pipeline is handed a link that carries the
    sniffed suffix.
    """
    if not source.is_file() or source.suffix.lower() in _DATAFLOW_SUFFIXES:
        return None
    suffix = _sniff_suffix(source)
    if suffix is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    alias = directory / f"{source.stem or 'input'}{suffix}"
    for link in (os.symlink, os.link):
        try:
            link(source, alias)
        except OSError:
            continue
        return {
            "source": str(source),
            "path": str(alias),
            "type": suffix.lstrip("."),
            "link": link.__name__,
        }
    shutil.copyfile(source, alias)
    return {
        "source": str(source),
        "path": str(alias),
        "type": suffix.lstrip("."),
        "link": "copy",
    }


def cli_spec(config: dict[str, Any], workspace: Path) -> dict[str, Any]:
    """Resolve the public CLI contract; credentials only come from process env."""
    allowed = {
        "pipeline_path",
        "args",
        "support_file_path",
        "timeout",
        "python",
        "output_schema",
        "model_profile",
        "gateway",
    }
    if set(config) - allowed:
        raise ValueError(
            "unknown config fields; credentials and env must not be saved in config"
        )
    cwd = workspace.resolve()
    root = _layout_root(cwd)
    public = root / "public_data"

    def path(value: str, output: bool = False) -> Path:
        p = (cwd / value).resolve()
        storage = (root / "storage").resolve()
        if not p.is_relative_to(public) and (output or not p.is_relative_to(storage)):
            raise ValueError("paths must be in public_data/ or read-only storage/")
        return p

    pipeline = path(config["pipeline_path"], True)
    if not pipeline.is_file():
        raise ValueError("pipeline file does not exist")
    args = config.get("args", [])
    if not isinstance(args, list) or not all(isinstance(v, str) for v in args):
        raise ValueError("args must be strings")
    schema = config.get("output_schema")
    supported = {
        "text",
        "dpo",
        "vision",
        "structured",
        "multiturn",
        "function_call",
        "quality_evaluation",
        "text2sql",
        "artifacts",
    }
    if schema is not None and schema not in supported:
        raise ValueError("unsupported output_schema")
    if schema and len(args) < 2:
        raise ValueError(
            "schema validation requires input and output as first two args"
        )
    source = path(args[0]) if args else None
    output = path(args[1], True) if len(args) > 1 else None
    if source and not source.exists():
        raise ValueError("input does not exist")
    if (
        source
        and output
        and (output == source or (source.is_dir() and output.is_relative_to(source)))
    ):
        raise ValueError("output must not overwrite input")
    state = public / ".dataflow-runs" / __import__("uuid").uuid4().hex
    alias = _alias_input(source, state / "input") if source else None
    if alias is not None:
        _log(
            f"input alias: {alias['source']} -> {alias['path']} "
            f"(type {alias['type']}, {alias['link']})"
        )
    normalized = list(args)
    if source:
        normalized[0] = str(alias["path"]) if alias else str(source)
    if output:
        normalized[1] = str(output)
    support = (
        path(config["support_file_path"]) if config.get("support_file_path") else None
    )
    if support:
        if len(args) != 2:
            raise ValueError("support_file_path requires exactly input and output args")
        json.loads(support.read_text())
    profile = config.get("model_profile", "none")
    if profile not in {"none", "text", "vision"}:
        raise ValueError("unsupported model_profile")
    if profile == "vision":
        validate_cli_image_pipeline(pipeline)
    profiles = json.loads(os.environ.get("PYROMIND_DATAFLOW_PROFILES", "{}"))
    environment = profiles.get(profile, {}) if profile != "none" else {}
    gateway = config.get("gateway")
    if gateway:
        if profile != "vision" or set(gateway) != {"api_url", "model", "api_key_env"}:
            raise ValueError(
                "gateway requires vision profile, api_url, model and api_key_env"
            )
        key = os.environ.get(gateway["api_key_env"])
        if not key:
            raise ValueError("gateway key environment variable is not configured")
        environment = {
            "DF_API_URL": gateway["api_url"],
            "DF_API_BASE_URL": gateway["api_url"].removesuffix("/chat/completions"),
            "DF_MODEL_NAME": gateway["model"],
            "DF_API_KEY": key,
        }
    if profile != "none" and not environment:
        raise ValueError(
            "model profile is not configured in the host execution environment"
        )
    timeout = config.get("timeout", 3600)
    if not isinstance(timeout, int) or not 1 <= timeout <= 7200:
        raise ValueError("timeout must be between 1 and 7200 seconds")
    log = output.parent if output else state
    environment = {
        **environment,
        "DF_STATE_DIR": str(log),
        "DF_LOG_DIR": str(log),
        "DF_RESUME": "0",
    }
    if schema:
        environment["DF_OUTPUT_SCHEMA"] = schema
    interpreter = _interpreter_spec()
    return {
        "marker": "cli",
        "runtime_dir": str(Path(__file__).resolve().parent),
        "python": config.get("python") or interpreter["python"],
        "venv": interpreter["venv"],
        "dataflow_version": interpreter["dataflow_version"],
        "packages": interpreter["packages"],
        "pipeline": str(pipeline),
        "cwd": str(root),
        "args": normalized,
        "model_profile": profile,
        "env": environment,
        "timeout": timeout,
        "support_file": str(support) if support else None,
        "support_dir": str(state / "support"),
        "input_path": str(alias["path"])
        if alias
        else (str(source) if source else None),
        "output_path": str(output) if output else None,
        "input_alias": alias,
        "output_schema": schema,
        "image_root": str(source if source and source.is_dir() else source.parent)
        if source
        else None,
        "state_dir": str(state),
        "log_dir": str(log),
    }


def direct_config(args: argparse.Namespace) -> dict[str, Any]:
    """The public config ``--config`` would carry, assembled from flags."""
    arguments = [args.input] if args.input else []
    if args.output:
        arguments.append(args.output)
    config: dict[str, Any] = {
        "pipeline_path": args.pipeline,
        "args": [*arguments, *args.arg],
    }
    if args.model_profile:
        config["model_profile"] = args.model_profile
    if args.output_schema:
        config["output_schema"] = args.output_schema
    if args.timeout is not None:
        config["timeout"] = args.timeout
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--spec", type=Path)
    modes.add_argument("--config", type=Path)
    modes.add_argument(
        "--pipeline",
        help="workspace path of the pipeline script; runs it without a config file",
    )
    modes.add_argument(
        "--check",
        action="store_true",
        help="report the DataFlow interpreter a run would use",
    )
    modes.add_argument(
        "--prepare",
        action="store_true",
        help="resolve the DataFlow interpreter, installing the cached venv if needed",
    )
    parser.add_argument(
        "--input",
        help="pipeline's first argument, read from storage/ or public_data/",
    )
    parser.add_argument(
        "--output",
        help="pipeline's second argument, written under public_data/",
    )
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        help=(
            "further pipeline argument after input and output; repeatable, and a "
            "value starting with '-' must be written as --arg=<value>"
        ),
    )
    parser.add_argument(
        "--model-profile",
        choices=["none", "text", "vision"],
        help="model configuration injected from the execution environment",
    )
    parser.add_argument(
        "--output-schema",
        help="validated output shape; see references/terminal-cli.md",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="seconds the pipeline may run, 1-7200",
    )
    args = parser.parse_args()
    if args.check or args.prepare:
        report = _interpreter_report(_interpreter_spec(), prepare=bool(args.prepare))
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["ready"] else 1
    direct = (
        args.input,
        args.output,
        args.arg,
        args.model_profile,
        args.output_schema,
        args.timeout,
    )
    if args.pipeline is None and any(direct):
        parser.error(
            "--input/--output/--arg/--model-profile/--output-schema/--timeout "
            "require --pipeline"
        )
    if args.output and not args.input:
        parser.error("--output requires --input")
    if args.pipeline is not None:
        spec = cli_spec(direct_config(args), Path.cwd())
    else:
        config = args.config or args.spec
        payload = json.loads(config.read_text(encoding="utf-8"))
        spec = cli_spec(payload, Path.cwd()) if args.config else payload
    marker = str(spec["marker"])
    try:
        envelope = _run(spec)
    except Exception as exc:  # noqa: BLE001 - the tool turns this into a failure
        envelope = {
            "marker": marker,
            "rc": 1,
            "pipeline_rc": 1,
            "validation_rc": None,
            "report_rc": None,
            "source_integrity_rc": 0,
            "stdout_tail": "",
            "stderr_tail": f"{type(exc).__name__}: {exc}",
            "record_count": 0,
            "sample_records": [],
            "report_failure": {},
            "failure_stage": "pipeline_execution",
            "error_code": "dataflow_pipeline_failed",
            "error_message": (
                f"Sandbox sample runner failed: {type(exc).__name__}: {exc}"
            ),
        }
    if args.spec is None:
        # A pipeline may leave its last line unterminated; keep the envelope alone.
        print()
        print(json.dumps(envelope, ensure_ascii=False))
        return int(envelope["rc"])
    _emit_envelope(envelope, marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
