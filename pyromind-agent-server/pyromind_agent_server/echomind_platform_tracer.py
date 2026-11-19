from __future__ import annotations

import re
from hashlib import sha256

from pyromind_agent_server.echomind_runtime_slots import RuntimeSlotSetupRequest
from pyromind_agent_server.echomind_storage import canonical_uuid
from pyromind_runtime.ports.storage import (
    EvaluationStorage,
    EvaluationStorageMissingError,
    EvaluationStorageUnavailableError,
)


_HOOK = b"""#!/bin/sh
set -eu
exec python3 - <<'PYROMIND_ECHOMIND_HOOK'
import json
import os
import tempfile
from pathlib import Path

data = Path('/app/data')
spans = data / 'spans'
markers = data / 'project-batches'


def batch_is_sealed(batch, index):
    shards = [
        path for path in batch.glob('spans_*.parquet')
        if path.is_file() and not path.is_symlink()
    ]
    if not shards:
        return False
    metadata = batch / 'batch.json'
    try:
        if metadata.exists():
            state = json.loads(metadata.read_bytes())
            return (
                isinstance(state, dict)
                and state.get('index') == index
                and state.get('state') in ('ready', 'in_training')
            )
        # Older images trigger immediately after flushing a full batch,
        # before moving to the next directory, and omit batch.json.
        import pyarrow.parquet as parquet

        limit = int(os.environ.get('BUFFER_SIZE', '0'))
        return (
            limit > 0
            and sum(parquet.read_metadata(path).num_rows for path in shards) >= limit
        )
    except (OSError, ValueError):
        return False


if spans.is_dir():
    for batch in spans.iterdir():
        if not (
            batch.name.isascii() and batch.name.isdecimal()
            and batch.is_dir() and not batch.is_symlink()
        ):
            continue
        index = int(batch.name)
        marker = markers / f'{index}.json'
        if marker.exists() or not batch_is_sealed(batch, index):
            continue
        markers.mkdir(exist_ok=True)
        staged = None
        try:
            with tempfile.NamedTemporaryFile(dir=markers, delete=False) as file:
                staged = Path(file.name)
                os.fchmod(file.fileno(), 0o644)
                file.write(json.dumps({'batch_index': index}).encode())
                file.flush()
                os.fsync(file.fileno())
            os.replace(staged, marker)
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
PYROMIND_ECHOMIND_HOOK
"""


def _hook_stem(project_id: str) -> str:
    return f"echomind-project-{canonical_uuid(project_id, kind='project')}"


def is_platform_tracer_script(project_id: str, script_path: str | None) -> bool:
    if script_path is None:
        return False
    return (
        re.fullmatch(
            rf"/app/trigger/{_hook_stem(project_id)}-[0-9a-f]{{64}}\.sh", script_path
        )
        is not None
    )


def prepare_platform_tracer_setup(
    setup: RuntimeSlotSetupRequest,
    *,
    project_id: str,
    storage: EvaluationStorage,
) -> RuntimeSlotSetupRequest:
    tracer = setup.tracer
    if (
        setup.attach_existing_tracer_id is not None
        or tracer.execution_mode != "manual"
        or tracer.custom_runtime_script_path
    ):
        return setup

    filename = f"{_hook_stem(project_id)}-{sha256(_HOOK).hexdigest()}.sh"
    storage_path = f"/EchoMind/trigger/{filename}"
    try:
        saved = storage.read(storage_path)
    except EvaluationStorageMissingError:
        storage.write(storage_path, _HOOK)
        saved = storage.read(storage_path)
    if saved != _HOOK:
        raise EvaluationStorageUnavailableError(
            "EchoMind trigger script contents do not match the expected version."
        )

    return setup.model_copy(
        update={
            "tracer": tracer.model_copy(
                update={"custom_runtime_script_path": f"/app/trigger/{filename}"}
            )
        }
    )
