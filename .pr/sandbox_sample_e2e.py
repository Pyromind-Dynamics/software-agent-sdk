"""Exercise the sandbox DataFlow runner against a fake sandbox root.

``df_run_pipeline`` now executes inside the conversation sandbox. The unit
suite covers the tool side of that path with doubles; this script runs the real
runner script end to end on the agent host, using a temporary directory as the
sandbox root and the local DataFlow venv as the interpreter:

    uv run python .pr/sandbox_sample_e2e.py
    uv run python .pr/sandbox_sample_e2e.py --bootstrap

It checks the success envelope (interpreter resolution, source fingerprint,
pipeline run, report) and the failure envelope (non-zero pipeline exit), then
decodes both with the tool's own parser.

``--bootstrap`` instead covers the first run in a sandbox whose image has no
DataFlow: the runner has to create its cached venv, and the second run has to
reuse it. That path installs from PyPI, so it is opt-in.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = REPO_ROOT / ".agents/skills/data-processing/scripts/preparation"
DATAFLOW_PYTHON = REPO_ROOT / "workspace/runtime/dataflow-venv/bin/python"
RUNTIME_FILENAMES = (
    "avi_pcb_runtime.py",
    "df_logging.py",
    "generate_report.py",
    "image_utils.py",
    "preparation_runtime.py",
    "source_fingerprint.py",
    "validate_prepared_data.py",
)
RUNNER_FILENAME = "sandbox_sample.py"
DATAFLOW_VERSION = "1.0.10"

sys.path.insert(0, str(REPO_ROOT / "openhands-tools"))

from openhands.tools.data_preparation.sandbox_execution import (  # noqa: E402
    _parse_envelope,
)


SUCCESS_PIPELINE = '''\
"""Minimal standard pipeline: copy the JSONL input, adding one field."""

import json
import sys
from pathlib import Path

source, destination = Path(sys.argv[1]), Path(sys.argv[2])
with open(destination, "w", encoding="utf-8") as out:
    for path in sorted(source.rglob("*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["prepared"] = True
                out.write(json.dumps(record) + "\\n")
print(f"prepared {destination}")
'''

FAILING_PIPELINE = """\
import sys

print("pipeline exploded", file=sys.stderr)
sys.exit(3)
"""


def _build_root(root: Path) -> Path:
    runtime_dir = root / "runtime"
    runtime_dir.mkdir(parents=True)
    for name in (*RUNTIME_FILENAMES, RUNNER_FILENAME):
        shutil.copy2(SKILL_SCRIPTS / name, runtime_dir / name)

    inputs = root / "storage/inputs"
    inputs.mkdir(parents=True)
    with (inputs / "records.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(3):
            handle.write(json.dumps({"id": index, "text": f"row {index}"}) + "\n")
    return runtime_dir


def _spec(
    root: Path, runtime_dir: Path, *, marker: str, failing: bool
) -> dict[str, Any]:
    sample_dir = root / "ws/public_data/sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    pipeline = sample_dir / "pipeline.py"
    pipeline.write_text(
        FAILING_PIPELINE if failing else SUCCESS_PIPELINE, encoding="utf-8"
    )
    output = sample_dir / ("failed.jsonl" if failing else "out.jsonl")
    state_dir = sample_dir / f".{output.stem}.state"
    return {
        "python": str(DATAFLOW_PYTHON),
        "venv": str(root / "venv-cache"),
        "dataflow_version": DATAFLOW_VERSION,
        "packages": [],
        "model_profile": "none",
        "cwd": str(sample_dir),
        "pipeline": str(pipeline),
        "args": [str(root / "storage/inputs"), str(output)],
        "support_file": None,
        "support_dir": str(state_dir / "support"),
        "output_schema": "structured",
        "output_path": str(output),
        "input_path": str(root / "storage/inputs"),
        "image_root": None,
        "log_dir": str(sample_dir),
        "state_dir": str(state_dir),
        "env": {},
        "timeout": 300,
        "execution_revision": 1,
        "runtime_fingerprint": "e2e-fingerprint",
        "runtime_dir": str(runtime_dir),
        "marker": marker,
        "system_python": sys.executable,
    }


def _bootstrap_spec(root: Path, runtime_dir: Path, *, marker: str) -> dict[str, Any]:
    """A spec whose sandbox image has no DataFlow, forcing the cached venv."""
    spec = _spec(root, runtime_dir, marker=marker, failing=False)
    spec["python"] = None
    spec["packages"] = [f"open-dataflow=={DATAFLOW_VERSION}"]
    spec["venv"] = str(root / "bootstrap-venv")
    spec["system_python"] = _dataflow_free_python()
    return spec


def _dataflow_free_python() -> str:
    """A host interpreter without DataFlow, standing in for a bare image."""
    for candidate in ("/usr/local/bin/python3", "/opt/homebrew/bin/python3"):
        if Path(candidate).is_file():
            return candidate
    raise SystemExit("no DataFlow-free python3 found for the bootstrap check")


def _run_runner(root: Path, spec: dict[str, Any]) -> str:
    spec_path = root / f"run-{spec['marker']}.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    completed = subprocess.run(
        [
            str(DATAFLOW_PYTHON),
            str(root / "runtime" / RUNNER_FILENAME),
            "--spec",
            str(spec_path),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if "__PM_DF_BEGIN__" not in completed.stdout:
        raise SystemExit(
            f"runner printed no envelope\n--- stdout ---\n{completed.stdout[-4000:]}"
            f"\n--- stderr ---\n{completed.stderr[-4000:]}"
        )
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="exercise the cached-venv path instead of the pinned interpreter",
    )
    options = parser.parse_args()
    if not DATAFLOW_PYTHON.is_file():
        raise SystemExit(f"missing local DataFlow interpreter: {DATAFLOW_PYTHON}")
    root = Path(tempfile.mkdtemp(prefix="pm-df-sandbox-e2e-"))
    try:
        runtime_dir = _build_root(root)
        checks: list[str] = []

        if options.bootstrap:
            first = _parse_envelope(
                _run_runner(
                    root, _bootstrap_spec(root, runtime_dir, marker="bootstrap-1")
                ),
                "bootstrap-1",
            )
            assert first["rc"] == 0, first
            assert first["interpreter_source"] == "installed", first
            second = _parse_envelope(
                _run_runner(
                    root, _bootstrap_spec(root, runtime_dir, marker="bootstrap-2")
                ),
                "bootstrap-2",
            )
            assert second["rc"] == 0, second
            assert second["interpreter_source"] == "reused", second
            assert second["interpreter"] == first["interpreter"], second
            checks.append(
                f"bootstrap venv installed then reused: {first['interpreter']}"
            )
            for line in checks:
                print(f"[ok] {line}")
            return 0

        success_marker = "e2e-success"
        success = _parse_envelope(
            _run_runner(
                root, _spec(root, runtime_dir, marker=success_marker, failing=False)
            ),
            success_marker,
        )
        assert success["rc"] == 0, success
        assert success["interpreter_source"] == "configured", success
        assert success["record_count"] == 3, success
        assert success["sample_records"][0]["prepared"] is True, success
        assert success["output_exists"] is True, success
        checks.append(
            "success envelope: rc=0, 3 records, interpreter="
            f"{success['interpreter']} ({success['interpreter_source']})"
        )
        assert (root / "ws/public_data/sample/out.jsonl").is_file()
        checks.append("pipeline artifacts stayed inside the fake sandbox root")

        failure_marker = "e2e-failure"
        failure = _parse_envelope(
            _run_runner(
                root, _spec(root, runtime_dir, marker=failure_marker, failing=True)
            ),
            failure_marker,
        )
        assert failure["rc"] == 3, failure
        assert failure["error_code"] == "dataflow_pipeline_failed", failure
        assert "pipeline exploded" in failure["stderr_tail"], failure
        checks.append(
            f"failure envelope: rc=3, error_code={failure['error_code']}, "
            f"stage={failure['failure_stage']}"
        )

        for line in checks:
            print(f"[ok] {line}")
        print(f"[ok] sandbox root={root}")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
