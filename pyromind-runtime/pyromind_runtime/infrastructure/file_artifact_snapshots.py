"""Snapshots for a mounted, per-user Storage tree (also usable in local tests)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.versions import (
    SnapshotResult,
    SnapshotSource,
    canonical,
    fingerprint,
)


def storage_path(value: str) -> str:
    if value.startswith("storage/"):
        value = value[len("storage") :]
    path = PurePosixPath(value)
    if not value.startswith("/") or ".." in path.parts or "\\" in value:
        raise ValueError("expected an absolute Storage path")
    return str(path)


def tree_manifest(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("snapshot trees cannot contain symbolic links")
        if path.is_file():
            with path.open("rb") as stream:
                result[path.relative_to(root).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        elif not path.is_dir():
            raise ValueError("snapshot trees must contain ordinary files")
    return result


class FileArtifactSnapshots:
    """Each user gets a separate root, named by SHA-256 of their identity."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def user_root(self, context: RequestContext) -> Path:
        return self.root / hashlib.sha256(context.user_id.encode()).hexdigest()

    def path(self, reference: str, context: RequestContext) -> Path:
        root = self.user_root(context)
        candidate = root / storage_path(reference).lstrip("/")
        if not candidate.resolve().is_relative_to(root):
            raise ValueError("Storage path escapes user namespace")
        if any(
            item.is_symlink()
            for item in (candidate, *candidate.parents)
            if item.is_relative_to(root)
        ):
            raise ValueError("Storage path cannot use symbolic links")
        return candidate

    def save(
        self, source: SnapshotSource, target: str, context: RequestContext
    ) -> SnapshotResult:
        destination = self.path(target, context)
        if source.reference_only:
            if source.storage_ref is None:
                raise ValueError("reference-only sources require a Storage path")
            reference = storage_path(source.storage_ref)
            if not self.path(reference, context).exists():
                raise FileNotFoundError("source artifact does not exist")
            return SnapshotResult(storage_ref=reference, reference_only=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (destination.parent / f".{destination.name}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            receipt = destination / "manifest.json"
            if receipt.exists():
                result = SnapshotResult.model_validate_json(receipt.read_text())
                self.verify(result, context)
                return result
            with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
                staging = Path(directory) / "snapshot"
                content = staging / "content"
                content.mkdir(parents=True)
                if source.text is not None:
                    (content / source.filename).write_text(
                        source.text, encoding="utf-8"
                    )
                    content_ref = target + "/content/" + source.filename
                else:
                    assert source.storage_ref is not None
                    original = self.path(source.storage_ref, context)
                    if destination.is_relative_to(original):
                        raise ValueError("snapshot destination cannot be inside source")
                    if original.is_dir():
                        before = tree_manifest(original)
                        shutil.copytree(
                            original, content, dirs_exist_ok=True, symlinks=True
                        )
                        if before != tree_manifest(original) or before != tree_manifest(
                            content
                        ):
                            raise ValueError("source changed during snapshot")
                        content_ref = target + "/content"
                    else:
                        with original.open("rb") as stream:
                            before_hash = hashlib.file_digest(
                                stream, "sha256"
                            ).hexdigest()
                        shutil.copyfile(original, content / original.name)
                        copied = tree_manifest(content)
                        if copied[original.name] != before_hash:
                            raise ValueError("source changed during snapshot")
                        content_ref = target + "/content/" + original.name
                files = tree_manifest(content)
                result = SnapshotResult(
                    storage_ref=content_ref,
                    manifest_ref=target + "/manifest.json",
                    digest=fingerprint(files),
                )
                (staging / "files.json").write_text(canonical(files))
                (staging / "manifest.json").write_text(result.model_dump_json())
                for path in staging.rglob("*"):
                    if path.is_file():
                        with path.open("rb") as stream:
                            os.fsync(stream.fileno())
                os.rename(staging, destination)
                return result

    def verify(self, result: SnapshotResult, context: RequestContext) -> None:
        if result.reference_only or not result.manifest_ref:
            raise ValueError("reference-only artifact cannot be restored")
        receipt = self.path(result.manifest_ref, context)
        files = tree_manifest(receipt.parent / "content")
        if not files or fingerprint(files) != result.digest:
            raise ValueError("snapshot missing or content digest changed")
        if files != json.loads((receipt.parent / "files.json").read_text()):
            raise ValueError("snapshot manifest changed")

    def read_text(self, result: SnapshotResult, context: RequestContext) -> str:
        self.verify(result, context)
        path = self.path(result.storage_ref, context)
        if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("text comparison requires a file under 2 MiB")
        return path.read_text(encoding="utf-8")
