"""Platform Storage transport for the product snapshot port."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from uuid import uuid4

from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.versions import SnapshotResult, SnapshotSource
from pyromind_runtime.infrastructure.file_artifact_snapshots import (
    FileArtifactSnapshots,
    storage_path,
)

from openhands.tools.pyromind_dataset.definition import (
    _default_storage_base_url,
    download_file_from_pyromind,
    upload_local_file_to_pyromind,
)


class PlatformArtifactSnapshots:
    """Retain a durable local staging receipt so retries upload identical bytes."""

    def __init__(self, staging_root: Path) -> None:
        self.staging = FileArtifactSnapshots(staging_root)

    @staticmethod
    def _headers(context: RequestContext) -> dict[str, str]:
        return {
            key: value
            for key, value in (
                ("cookie", context.cookie),
                ("authorization", context.authorization),
                ("x-cluster", context.x_cluster),
            )
            if value
        }

    def _download(self, reference: str, context: RequestContext) -> bytes:
        return download_file_from_pyromind(
            storage_path=storage_path(reference),
            storage_base_url=_default_storage_base_url(),
            headers=self._headers(context),
            timeout=60,
            max_bytes=32 * 1024 * 1024,
        )

    def save(
        self, source: SnapshotSource, target: str, context: RequestContext
    ) -> SnapshotResult:
        if source.reference_only:
            if source.storage_ref is None:
                raise ValueError("reference-only input requires a Storage path")
            return SnapshotResult(
                storage_ref=storage_path(source.storage_ref), reference_only=True
            )
        receipt = self.staging.path(target + "/manifest.json", context)
        if receipt.exists():
            result = SnapshotResult.model_validate_json(receipt.read_text())
            self.staging.verify(result, context)
        elif source.text is not None:
            result = self.staging.save(source, target, context)
        else:
            assert source.storage_ref is not None
            filename = PurePosixPath(storage_path(source.storage_ref)).name
            temporary_ref = f"/.version-downloads/{uuid4()}/{filename}"
            temporary = self.staging.path(temporary_ref, context)
            temporary.parent.mkdir(parents=True)
            try:
                temporary.write_bytes(self._download(source.storage_ref, context))
                result = self.staging.save(
                    SnapshotSource(storage_ref=temporary_ref), target, context
                )
            finally:
                temporary.unlink(missing_ok=True)
                temporary.parent.rmdir()
        root = receipt.parent
        # Content precedes the receipt. The persistent staging tree is immutable,
        # so retries (including after a process crash) never upload new content.
        files = [*sorted((root / "content").rglob("*")), root / "files.json", receipt]
        for path in files:
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            upload_local_file_to_pyromind(
                local_path=path,
                target_dir=target + "/" + relative.parent.as_posix(),
                storage_base_url=_default_storage_base_url(),
                headers=self._headers(context),
                timeout=60,
            )
        self.verify(result, context)
        return result

    def verify(self, result: SnapshotResult, context: RequestContext) -> None:
        if result.reference_only or not result.manifest_ref:
            raise ValueError("reference-only artifact cannot be restored")
        root = str(PurePosixPath(result.manifest_ref).parent)
        files = json.loads(self._download(root + "/files.json", context))
        from pyromind_runtime.domain.versions import fingerprint

        if (
            not isinstance(files, dict)
            or not files
            or fingerprint(files) != result.digest
        ):
            raise ValueError("snapshot manifest changed")
        for name, digest in files.items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("invalid snapshot manifest path")
            actual = hashlib.sha256(
                self._download(root + "/content/" + name, context)
            ).hexdigest()
            if actual != digest:
                raise ValueError("snapshot missing or content digest changed")

    def read_text(self, result: SnapshotResult, context: RequestContext) -> str:
        self.verify(result, context)
        content = self._download(result.storage_ref, context)
        if len(content) > 2 * 1024 * 1024:
            raise ValueError("text comparison requires a file under 2 MiB")
        return content.decode("utf-8")
