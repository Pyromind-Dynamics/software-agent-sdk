from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import PurePosixPath

from pyromind_agent_server.echomind_sandbox_storage import (
    SandboxEvaluationStorage,
    _require_sandbox,
)
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.ports.project_storage import StorageBindingBlocked
from pyromind_runtime.ports.storage import EvaluationStorageUnavailableError


_ARCHIVE_SCRIPT = """
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from datetime import datetime
from pathlib import Path


def safe_path(path):
    if path != path.resolve() or root not in path.parents:
        raise ValueError
    return path


def file_digest(path, destination=None):
    safe_path(path)
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags), "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_dev != root_device:
            raise ValueError
        digest = hashlib.sha256()
        size = 0
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
            if destination is not None:
                destination.write(chunk)
        if size != info.st_size:
            raise ValueError
        return {"size": size, "sha256": digest.hexdigest()}


def raise_error(error):
    raise error


def snapshot(directory, destination=None):
    safe_path(directory)
    if not directory.is_dir():
        raise ValueError
    files = {}
    entries = os.walk(directory, onerror=raise_error, followlinks=False)
    for dirpath, dirs, names in entries:
        current = safe_path(Path(dirpath))
        if current.stat().st_dev != root_device:
            raise ValueError
        if destination is not None:
            target = destination / current.relative_to(directory)
            target.mkdir()
        for name in dirs:
            safe_path(current / name)
        for name in names:
            source = current / name
            relative = str(source.relative_to(directory))
            if destination is None:
                files[relative] = file_digest(source)
            else:
                with (target / name).open("xb") as file:
                    files[relative] = file_digest(source, file)
                    file.flush()
                    os.fsync(file.fileno())
    return files


def check_batch(directory, files):
    metadata_path = safe_path(directory / "batch.json")
    with metadata_path.open("rb") as file:
        metadata = json.loads(file.read(65536))
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("index")) is not int
        or metadata["index"] != batch_index
        or metadata.get("state") not in ("ready", "in_training")
        or type(metadata.get("collected")) is not int
        or metadata["collected"] <= 0
        or not any(
            re.fullmatch(r"spans_[A-Za-z0-9_-]+[.]parquet", name) and info["size"]
            for name, info in files.items()
        )
    ):
        raise ValueError
    if sealed_before is not None:
        ready_at = metadata.get("ready_at")
        if not isinstance(ready_at, str):
            raise ValueError
        if ready_at.endswith("Z"):
            ready_at = ready_at[:-1] + "+00:00"
        sealed = datetime.fromisoformat(ready_at)
        if sealed.utcoffset() is None or sealed > sealed_before:
            raise ValueError


def verify_existing():
    archived = archive / "batch"
    files = snapshot(archived)
    check_batch(archived, files)
    expected = {**identity, "files": files}
    manifest = safe_path(archive / "manifest.json")
    if json.loads(manifest.read_bytes()) != expected:
        raise ValueError
    if source.exists() and snapshot(source) != files:
        raise ValueError


def publish():
    if archive.exists():
        verify_existing()
        return
    if not source.is_dir():
        raise ValueError
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.parent.stat().st_dev != root_device:
        raise ValueError
    with tempfile.TemporaryDirectory(
        prefix=".batch-archive-", dir=archive.parent
    ) as temp:
        staged = Path(temp) / "archive"
        staged.mkdir()
        files = snapshot(source, staged / "batch")
        check_batch(staged / "batch", files)
        if snapshot(staged / "batch") != files or snapshot(source) != files:
            raise ValueError
        with (staged / "manifest.json").open("x", encoding="utf-8") as file:
            json.dump({**identity, "files": files}, file, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.rename(staged, archive)
        except OSError:
            if not archive.is_dir():
                raise
            verify_existing()


try:
    mount, tracer_resource_id, index, relative_archive, cutoff = sys.argv[1:]
    root = Path(mount).resolve(strict=True)
    root_device = root.stat().st_dev
    batch_index = int(index)
    sealed_before = datetime.fromisoformat(cutoff) if cutoff else None
    identity = {
        "version": 1,
        "tracer_resource_id": tracer_resource_id,
        "batch_index": batch_index,
    }
    spans = root / "EchoMind" / tracer_resource_id / "data" / "spans"
    source = safe_path(spans / index)
    archive = safe_path(root / relative_archive)
    publish()
except (OSError, ValueError):
    print("unverified")
else:
    print("verified")
"""


def archive_tracer_batch(
    storage: SandboxEvaluationStorage,
    *,
    tracer_resource_id: str,
    batch_index: int,
    archive_directory: str,
    sealed_before: datetime | None = None,
) -> None:
    """Preserve and verify a sealed Tracer batch inside the Project's Storage."""
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", tracer_resource_id)
        or type(batch_index) is not int
        or batch_index < 0
    ):
        raise ValueError("Invalid Tracer batch identity")
    if sealed_before is not None and sealed_before.utcoffset() is None:
        raise ValueError("Tracer batch verification needs a timezone-aware cutoff")
    destination = PurePosixPath(to_storage_api_path(archive_directory))
    _, *parts = destination.parts
    if (
        len(parts) < 4
        or parts[0] != ".pyromind-agent"
        or not re.fullmatch(r"[0-9a-f]{32}", parts[1])
        or parts[2] != "project-batches"
        or not any(
            destination != prefix and destination.is_relative_to(prefix)
            for prefix in storage.write_prefixes
        )
    ):
        raise ValueError("Tracer archive must stay within the Project's batch Storage")
    config = storage.binding.config
    try:
        _require_sandbox(storage.client, config)
        response = storage.client.exec_command(
            config.sandbox_id,
            [
                "python3",
                "-c",
                _ARCHIVE_SCRIPT,
                config.mount_root,
                tracer_resource_id,
                str(batch_index),
                destination.relative_to("/").as_posix(),
                sealed_before.astimezone(UTC).isoformat() if sealed_before else "",
            ],
            timeout=180,
        )
    except StorageBindingBlocked as exc:
        raise EvaluationStorageUnavailableError(str(exc)) from None
    except Exception:
        raise EvaluationStorageUnavailableError(
            "Tracer batch archive could not be verified in the bound Storage Sandbox"
        ) from None
    if response.returncode != 0 or response.output.strip() != "verified":
        raise EvaluationStorageUnavailableError(
            "Tracer batch archive verification failed"
        )
