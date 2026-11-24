"""Platform Storage transport for the product snapshot port."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from uuid import uuid4

import httpx

from pyromind_common.tools.pyromind_dataset.definition import (
    _default_storage_base_url,
    _extract_api_data,
    download_file_from_pyromind,
    storage_file_names,
    upload_local_file_to_pyromind,
)
from pyromind_common.tools.utils.platform_env import (
    platform_headers,
    platform_portal_url,
)
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.versions import SnapshotResult, SnapshotSource, fingerprint
from pyromind_runtime.infrastructure.file_artifact_snapshots import (
    FileArtifactSnapshots,
    storage_path,
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
        elif source.text is not None or source.files is not None:
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

    @staticmethod
    def _snapshot_root(result: SnapshotResult) -> str:
        if result.reference_only or not result.manifest_ref:
            raise ValueError("reference-only artifact cannot be restored")
        return str(PurePosixPath(result.manifest_ref).parent)

    def _manifest(
        self, root: str, result: SnapshotResult, context: RequestContext
    ) -> dict[str, str]:
        files = json.loads(self._download(root + "/files.json", context))
        if (
            not isinstance(files, dict)
            or not files
            or fingerprint(files) != result.digest
        ):
            raise ValueError("snapshot manifest changed")
        for name in files:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("invalid snapshot manifest path")
        return files

    def _verify_files(
        self, root: str, files: dict[str, str], context: RequestContext
    ) -> None:
        for name, digest in files.items():
            actual = hashlib.sha256(
                self._download(root + "/content/" + name, context)
            ).hexdigest()
            if actual != digest:
                raise ValueError("snapshot missing or content digest changed")

    def verify(self, result: SnapshotResult, context: RequestContext) -> None:
        root = self._snapshot_root(result)
        self._verify_files(root, self._manifest(root, result, context), context)

    def verify_reference(self, storage_ref: str, context: RequestContext) -> None:
        path = PurePosixPath(storage_path(storage_ref))
        if path.name not in storage_file_names(
            directory=str(path.parent),
            search=path.name,
            headers=self._headers(context),
            storage_base_url=_default_storage_base_url(),
            timeout=8,
        ):
            raise ValueError("Registered Storage reference is unavailable")

    def read_text(self, result: SnapshotResult, context: RequestContext) -> str:
        root = self._snapshot_root(result)
        reference = storage_path(result.storage_ref)
        relative = PurePosixPath(reference).relative_to(root + "/content").as_posix()
        with ThreadPoolExecutor(max_workers=2) as pool:
            manifest = pool.submit(self._manifest, root, result, context)
            content = self._download(reference, context)
            files = manifest.result()
        if hashlib.sha256(content).hexdigest() != files.pop(relative, None):
            raise ValueError("snapshot missing or content digest changed")
        self._verify_files(root, files, context)
        if len(content) > 2 * 1024 * 1024:
            raise ValueError("text comparison requires a file under 2 MiB")
        return content.decode("utf-8")

    def file_sizes(
        self, references: tuple[str, ...], context: RequestContext
    ) -> dict[str, int]:
        headers = platform_headers(self._headers(context))
        base = platform_portal_url(_default_storage_base_url(), headers).rstrip("/")
        by_parent: dict[str, dict[str, str]] = {}
        for reference in references:
            path = PurePosixPath(storage_path(reference))
            by_parent.setdefault(str(path.parent), {})[path.name] = reference

        def metadata(route: str, body: dict[str, str]) -> dict:
            try:
                response = httpx.post(
                    f"{base}/{route}",
                    headers=headers,
                    json=body,
                    timeout=2,
                )
                response.raise_for_status()
                payload = response.json()
                data = (
                    _extract_api_data(route, payload)
                    if isinstance(payload, dict)
                    else None
                )
                return data if isinstance(data, dict) else {}
            except (httpx.HTTPError, ValueError):
                return {}

        def file_size(item: dict) -> int | None:
            value = item.get("size")
            if isinstance(value, str) and value.isdecimal() and len(value) <= 16:
                value = int(value)
            if (
                type(value) is int
                and 0 <= value < 2**53
                and item.get("is_dir") is not True
            ):
                return value
            return None

        def sizes_in_directory(directory: str, names: dict[str, str]) -> dict[str, int]:
            entries = metadata("file_list", {"path": directory, "search": ""}).get(
                "list"
            )
            if not isinstance(entries, list):
                return {}
            sizes: dict[str, int] = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if not isinstance(name, str) or name not in names:
                    continue
                if (size := file_size(entry)) is not None:
                    sizes[names[name]] = size
            return sizes

        with ThreadPoolExecutor(max_workers=4) as pool:
            sizes = {
                ref: size
                for sizes in pool.map(sizes_in_directory, by_parent, by_parent.values())
                for ref, size in sizes.items()
            }
            missing = [ref for ref in references if ref not in sizes]
            fallback = pool.map(
                lambda ref: file_size(metadata("get_file_metadata", {"path": ref})),
                missing,
            )
            sizes.update(
                (ref, size)
                for ref, size in zip(missing, fallback, strict=True)
                if size is not None
            )
            return sizes
