#!/usr/bin/env python3
"""Run one DataFlow sample inside the sandbox that owns the workspace.

``df_run_pipeline`` executes on the agent-server host, which has no filesystem
view of a sandbox workspace. When the conversation workspace is served by a
platform sandbox, the tool uploads this script and one run spec instead of
staging the input onto the host, and the whole sample sequence happens where the
data already lives: resolve the DataFlow interpreter, run the pipeline, validate
the output schema, generate the report, and print one JSON envelope the tool
reads back.

The interpreter is resolved in this order:

1. ``python`` from the run spec, set by ``PYROMIND_SANDBOX_DATAFLOW_PYTHON``.
2. The sandbox's own ``python3`` when ``open-dataflow`` matches the pin.
3. A venv at ``venv`` in the run spec, created on first use and reused.

Only the envelope, the run's log tails, and the artifacts the pipeline writes
cross the sandbox boundary. Nothing is copied to the agent-server host.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


HEARTBEAT_SECONDS = 20
LOG_TAIL_CHARS = 6000
ENVELOPE_LINE_WIDTH = 120
_PREFLIGHT_TIMEOUT_SECONDS = 30


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


def _bootstrap_venv(spec: dict[str, Any]) -> tuple[str | None, str]:
    """Create the cached DataFlow venv when it is missing or stale."""
    venv = Path(str(spec["venv"]))
    expected = str(spec["dataflow_version"])
    python = venv / "bin" / "python"
    stamp = venv / "open-dataflow-version"
    if python.is_file() and stamp.is_file():
        try:
            if stamp.read_text(encoding="utf-8").strip() == expected:
                return str(python), "reused"
        except OSError:
            pass
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
    stamp.write_text(expected, encoding="utf-8")
    return str(python), "installed"


def _resolve_interpreter(spec: dict[str, Any]) -> tuple[str | None, str]:
    expected = str(spec["dataflow_version"])
    configured = spec.get("python")
    if isinstance(configured, str) and configured:
        found = _interpreter_version(configured)
        if found != expected:
            return None, (
                f"configured interpreter {configured} has open-dataflow "
                f"{found or 'missing'}, expected {expected}"
            )
        return configured, "configured"
    system_python = _system_python(spec)
    if _interpreter_version(system_python) == expected:
        return system_python, "image"
    return _bootstrap_venv(spec)


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


def _fingerprint(python: str, script: Path, target: str) -> tuple[str | None, str]:
    if not script.is_file():
        return None, f"missing runtime helper {script}"
    result = subprocess.run(
        [python, str(script), target],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        return None, (result.stderr or "").strip()[-2000:]
    return (result.stdout or "").strip(), ""


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
    env = {**os.environ, **{str(k): str(v) for k, v in spec["env"].items()}}
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
                "PYROMIND_SANDBOX_DATAFLOW_PYTHON) or allow the sandbox to "
                "reach PyPI so the tool can create its cached venv."
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
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
    _emit_envelope(envelope, marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
