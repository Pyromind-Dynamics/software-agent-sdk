from __future__ import annotations

import asyncio
import base64
import posixpath
from collections.abc import Mapping

from pydantic import JsonValue

from pyromind_runtime.contracts.content import JsonObject
from pyromind_runtime.contracts.sandbox import (
    SandboxExecutionBackend,
    SandboxFileInfo,
)


_MAX_FILE_BYTES = 4 * 1024 * 1024
_MAX_OUTPUT_BYTES = 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_TIMEOUT_SECONDS = 600.0


class SandboxGatewayError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.path = path


class BoundedSandboxGateway:
    def __init__(
        self,
        backend: SandboxExecutionBackend,
        *,
        sandbox_id: str,
        workspace_root: str,
        environment: Mapping[str, str] | None = None,
        max_file_bytes: int = _MAX_FILE_BYTES,
        max_output_bytes: int = _MAX_OUTPUT_BYTES,
    ) -> None:
        root = posixpath.normpath(workspace_root)
        if not root.startswith("/"):
            raise ValueError("sandbox workspace_root must be absolute")
        self._backend = backend
        self._sandbox_id = sandbox_id
        self._workspace_root = root
        self._environment = dict(environment or {})
        self._max_file_bytes = max_file_bytes
        self._max_output_bytes = max_output_bytes

    @property
    def cwd(self) -> str:
        return self._workspace_root

    async def handle(self, method: str, params: JsonObject) -> JsonValue:
        if not method.startswith("env."):
            raise SandboxGatewayError(
                "invalid", f"Unsupported sandbox method: {method}"
            )
        try:
            value = await self._dispatch(method[4:], params)
            return {"ok": True, "value": value}
        except SandboxGatewayError as exc:
            error: JsonObject = {"code": exc.code, "message": str(exc)}
            if exc.path is not None:
                error["path"] = exc.path
            return {"ok": False, "error": error}
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return {
                "ok": False,
                "error": {"code": "timeout", "message": "Sandbox operation timed out"},
            }
        except Exception:
            return {
                "ok": False,
                "error": {"code": "unknown", "message": "Sandbox operation failed"},
            }

    async def close(self) -> None:
        await self._backend.close()

    async def _dispatch(self, operation: str, params: JsonObject) -> JsonValue:
        if operation == "absolutePath":
            return await self._checked_path(
                self._string(params, "path"), allow_missing=True
            )
        if operation == "joinPath":
            parts = params.get("parts")
            if not isinstance(parts, list) or not all(
                isinstance(part, str) for part in parts
            ):
                raise SandboxGatewayError(
                    "invalid", "parts must be an array of strings"
                )
            string_parts = [part for part in parts if isinstance(part, str)]
            return await self._checked_path(
                posixpath.join(*string_parts), allow_missing=True
            )
        if operation == "readTextFile":
            content = await self._read(params)
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SandboxGatewayError("invalid", "File is not valid UTF-8") from exc
        if operation == "readTextLines":
            content = await self._read(params)
            try:
                lines = content.decode("utf-8").splitlines()
            except UnicodeDecodeError as exc:
                raise SandboxGatewayError("invalid", "File is not valid UTF-8") from exc
            max_lines = self._optional_int(params, "maxLines")
            selected = lines if max_lines is None else lines[:max_lines]
            return [line for line in selected]
        if operation == "readBinaryFile":
            return base64.b64encode(await self._read(params)).decode("ascii")
        if operation in {"writeFile", "appendFile"}:
            path = await self._checked_path(
                self._string(params, "path"), allow_missing=True
            )
            content = self._bytes(params)
            if len(content) > self._max_file_bytes:
                raise SandboxGatewayError(
                    "invalid", "File content exceeds size limit", path=path
                )
            if operation == "writeFile":
                await self._backend.write_file(self._sandbox_id, path, content)
            else:
                await self._backend.append_file(self._sandbox_id, path, content)
            return None
        if operation == "renameFile":
            source = await self._checked_path(self._string(params, "sourcePath"))
            destination = await self._checked_path(
                self._string(params, "destinationPath"),
                allow_missing=True,
            )
            await self._backend.rename_file(self._sandbox_id, source, destination)
            return None
        if operation == "fileInfo":
            path = await self._checked_path(self._string(params, "path"))
            return self._file_info(
                await self._backend.file_info(self._sandbox_id, path)
            )
        if operation == "listDir":
            path = await self._checked_path(self._string(params, "path"))
            entries = await self._backend.list_dir(self._sandbox_id, path)
            for entry in entries:
                self._ensure_within_workspace(entry.path)
            return [self._file_info(entry) for entry in entries]
        if operation == "canonicalPath":
            path = await self._checked_path(self._string(params, "path"))
            canonical = await self._backend.canonical_path(self._sandbox_id, path)
            self._ensure_within_workspace(canonical)
            return canonical
        if operation == "exists":
            path = self._lexical_path(self._string(params, "path"))
            if not await self._backend.exists(self._sandbox_id, path):
                await self._check_existing_parent(path)
                return False
            await self._checked_path(path)
            return True
        if operation == "createDir":
            path = await self._checked_path(
                self._string(params, "path"), allow_missing=True
            )
            await self._backend.create_dir(
                self._sandbox_id,
                path,
                recursive=self._optional_bool(params, "recursive", True),
            )
            return None
        if operation == "remove":
            path = await self._checked_path(
                self._string(params, "path"),
                allow_missing=self._optional_bool(params, "force", False),
            )
            if path == self._workspace_root:
                raise SandboxGatewayError(
                    "permission_denied", "Cannot remove workspace root", path=path
                )
            await self._backend.remove(
                self._sandbox_id,
                path,
                recursive=self._optional_bool(params, "recursive", False),
                force=self._optional_bool(params, "force", False),
            )
            return None
        if operation == "createTempDir":
            prefix = self._optional_string(params, "prefix") or "tmp-"
            path = await self._backend.create_temp_dir(
                self._sandbox_id,
                self._workspace_root,
                prefix,
            )
            return await self._checked_path(path)
        if operation == "createTempFile":
            prefix = self._optional_string(params, "prefix") or ""
            suffix = self._optional_string(params, "suffix") or ""
            path = await self._backend.create_temp_file(
                self._sandbox_id,
                self._workspace_root,
                prefix,
                suffix,
            )
            return await self._checked_path(path)
        if operation == "exec":
            return await self._exec(params)
        if operation == "cleanup":
            return None
        raise SandboxGatewayError(
            "not_supported", f"Unsupported sandbox operation: {operation}"
        )

    async def _read(self, params: JsonObject) -> bytes:
        path = await self._checked_path(self._string(params, "path"))
        content = await self._backend.read_file(self._sandbox_id, path)
        if len(content) > self._max_file_bytes:
            raise SandboxGatewayError(
                "invalid", "File exceeds read size limit", path=path
            )
        return content

    async def _exec(self, params: JsonObject) -> JsonObject:
        command = self._string(params, "command")
        cwd_value = self._optional_string(params, "cwd") or self._workspace_root
        cwd = await self._checked_path(cwd_value)
        timeout = self._optional_number(params, "timeout") or _DEFAULT_TIMEOUT_SECONDS
        if timeout <= 0 or timeout > _MAX_TIMEOUT_SECONDS:
            raise SandboxGatewayError(
                "invalid", "Command timeout is outside the allowed range"
            )
        requested_env = params.get("env", {})
        if not isinstance(requested_env, dict) or not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in requested_env.items()
        ):
            raise SandboxGatewayError("invalid", "env must contain string values")
        unknown_names = set(requested_env) - set(self._environment)
        if unknown_names:
            raise SandboxGatewayError(
                "permission_denied", "Command environment variable is not allowed"
            )
        inherit = self._optional_bool(params, "inheritEnv", True)
        environment = dict(self._environment) if inherit else {}
        environment.update(
            {
                name: value
                for name, value in requested_env.items()
                if isinstance(name, str) and isinstance(value, str)
            }
        )
        async with asyncio.timeout(timeout):
            result = await self._backend.exec(
                self._sandbox_id,
                command,
                cwd=cwd,
                environment=environment,
                timeout_seconds=timeout,
            )
        stdout = self._bounded_output(result.stdout)
        stderr = self._bounded_output(result.stderr)
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exitCode": result.exit_code,
        }

    async def _checked_path(self, path: str, *, allow_missing: bool = False) -> str:
        addressed = self._lexical_path(path)
        if await self._backend.exists(self._sandbox_id, addressed):
            canonical = await self._backend.canonical_path(self._sandbox_id, addressed)
            self._ensure_within_workspace(canonical)
        elif allow_missing:
            await self._check_existing_parent(addressed)
        else:
            await self._check_existing_parent(addressed)
            raise SandboxGatewayError(
                "not_found", "Path does not exist", path=addressed
            )
        return addressed

    async def _check_existing_parent(self, path: str) -> None:
        parent = posixpath.dirname(path)
        while parent != self._workspace_root:
            if await self._backend.exists(self._sandbox_id, parent):
                canonical = await self._backend.canonical_path(self._sandbox_id, parent)
                self._ensure_within_workspace(canonical)
                return
            next_parent = posixpath.dirname(parent)
            if next_parent == parent:
                break
            parent = next_parent
        canonical_root = await self._backend.canonical_path(
            self._sandbox_id,
            self._workspace_root,
        )
        self._ensure_within_workspace(canonical_root)

    def _lexical_path(self, path: str) -> str:
        candidate = (
            path if path.startswith("/") else posixpath.join(self._workspace_root, path)
        )
        normalized = posixpath.normpath(candidate)
        self._ensure_within_workspace(normalized)
        return normalized

    def _ensure_within_workspace(self, path: str) -> None:
        normalized = posixpath.normpath(path)
        try:
            within = posixpath.commonpath((self._workspace_root, normalized))
        except ValueError as exc:
            raise SandboxGatewayError(
                "permission_denied", "Path escapes workspace", path=path
            ) from exc
        if within != self._workspace_root:
            raise SandboxGatewayError(
                "permission_denied", "Path escapes workspace", path=path
            )

    def _bounded_output(self, value: str) -> str:
        encoded = value.encode("utf-8", errors="replace")
        if len(encoded) <= self._max_output_bytes:
            return value
        return encoded[: self._max_output_bytes].decode("utf-8", errors="ignore")

    @staticmethod
    def _file_info(info: SandboxFileInfo) -> JsonObject:
        return {
            "name": info.name,
            "path": info.path,
            "kind": info.kind,
            "size": info.size,
            "mtimeMs": info.mtime_ms,
        }

    @staticmethod
    def _string(params: JsonObject, name: str) -> str:
        value = params.get(name)
        if not isinstance(value, str) or not value:
            raise SandboxGatewayError("invalid", f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _optional_string(params: JsonObject, name: str) -> str | None:
        value = params.get(name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise SandboxGatewayError("invalid", f"{name} must be a string")
        return value

    @staticmethod
    def _optional_int(params: JsonObject, name: str) -> int | None:
        value = params.get(name)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SandboxGatewayError(
                "invalid", f"{name} must be a non-negative integer"
            )
        return value

    @staticmethod
    def _optional_number(params: JsonObject, name: str) -> float | None:
        value = params.get(name)
        if value is None:
            return None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SandboxGatewayError("invalid", f"{name} must be a number")
        return float(value)

    @staticmethod
    def _optional_bool(params: JsonObject, name: str, default: bool) -> bool:
        value = params.get(name, default)
        if not isinstance(value, bool):
            raise SandboxGatewayError("invalid", f"{name} must be a boolean")
        return value

    @staticmethod
    def _bytes(params: JsonObject) -> bytes:
        value = params.get("content")
        encoding = params.get("encoding", "utf8")
        if not isinstance(value, str):
            raise SandboxGatewayError("invalid", "content must be a string")
        if encoding == "utf8":
            return value.encode()
        if encoding == "base64":
            try:
                return base64.b64decode(value, validate=True)
            except ValueError as exc:
                raise SandboxGatewayError(
                    "invalid", "content is not valid base64"
                ) from exc
        raise SandboxGatewayError("invalid", "encoding must be utf8 or base64")
