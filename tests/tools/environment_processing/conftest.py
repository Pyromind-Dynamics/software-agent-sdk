from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from pyromind_common.platform.preparation import StagingDirectory


@pytest.fixture
def edp_runtime(tmp_path: Path) -> Path:
    root = tmp_path / "runtime"
    (root / "profiles").mkdir(parents=True)
    (root / "sandbox_runner.py").write_text("print('frozen runner')\n")
    (root / "profiles/tmax-validation.json").write_text(
        json.dumps(
            {
                "name": "tmax-validation",
                "steps": [
                    {"name": "create_sandbox"},
                    {
                        "name": "run_pi",
                        "params": {"env": {"LLM_MODEL": "{secret:LLM_MODEL}"}},
                    },
                ],
            }
        )
    )
    (root / "profiles/environment-only.json").write_text(
        json.dumps({"steps": [{"name": "create_sandbox"}, {"name": "probe"}]})
    )
    pod = root / "pod_runtime/pyromind_common"
    pod.mkdir(parents=True)
    (pod / "__init__.py").write_text("")
    (root / "pod_runtime/pod_requirements.txt").write_text("pyromind-sdk==0.1.17\n")
    return root


@pytest.fixture
def edp_run_id() -> UUID:
    return UUID("11111111-1111-1111-1111-111111111111")


@pytest.fixture
def edp_staging(tmp_path: Path, edp_run_id: UUID) -> StagingDirectory:
    return StagingDirectory(
        local=tmp_path / "frozen", storage=f"/workspace/edp/batch-001/{edp_run_id}"
    )
