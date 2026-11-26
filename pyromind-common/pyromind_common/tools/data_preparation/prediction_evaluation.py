"""Platform packaging for CPU-only evaluation of existing predictions."""

from __future__ import annotations

import json
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from pyromind_common.platform.evaluation_core import (
    _PREFLIGHT as _PREFLIGHT,
    PredictionEvaluationOptions as PredictionEvaluationOptions,
    build_prediction_evaluation_dsl as build_prediction_evaluation_dsl,
    prepare_definition as prepare_definition,
)
from pyromind_common.tools.data_preparation.sandbox_execution import (
    DEFAULT_SANDBOX_SYSTEM_PYTHON,
    ENV_SANDBOX_SYSTEM_PYTHON,
    require_sandbox_success,
    sandbox_execute,
    sandbox_upload,
    supports_sandbox_execution,
)


def preflight_scoring(workspace: Any, runtime_dir: Path, definition: dict) -> None:
    # User-authored Python only runs in a sandbox or the platform CPU worker.
    if not supports_sandbox_execution(workspace):
        return
    remote = f"/tmp/scoring-preflight-{uuid4().hex}.zip"
    python = os.environ.get(ENV_SANDBOX_SYSTEM_PYTHON) or DEFAULT_SANDBOX_SYSTEM_PYTHON
    with tempfile.TemporaryDirectory(prefix="scoring-preflight-") as directory:
        archive = Path(directory) / "preflight.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as files:
            for name in ("evaluate_predictions.py", "evaluate_inference.py"):
                files.write(runtime_dir / name, name)
            files.writestr("__main__.py", _PREFLIGHT)
            files.writestr("score_eval.py", definition["scoring_script"])
            files.writestr("spec.json", json.dumps(definition["scoring_spec"]))
        try:
            sandbox_upload(workspace, archive, remote)
            command = f"trap 'rm -f {remote}' EXIT; {shlex.quote(python)} {remote}"
            require_sandbox_success(
                sandbox_execute(workspace, f"sh -c {shlex.quote(command)}", timeout=30),
                "scoring validation examples",
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
