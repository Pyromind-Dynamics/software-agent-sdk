from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import shlex
from collections.abc import Mapping
from typing import Final, cast
from urllib.parse import urlsplit, urlunsplit

import httpx

from pyromind_runtime.adapters.pi.sandbox import SandboxGatewayError
from pyromind_runtime.contracts.content import JsonObject
from pyromind_runtime.contracts.sandbox import (
    SandboxExecResult,
    SandboxExecutionBackend,
    SandboxFileInfo,
    SandboxFileKind,
)


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_AUTH_COOKIE_NAME: Final = "auth_token"
_HELPER_SCRIPT: Final = r"""
import base64,json,os,shutil,stat,sys,tempfile
p=json.loads(base64.urlsafe_b64decode(sys.argv[1]).decode())
op=p['op']
def info(path):
 s=os.lstat(path)
 if stat.S_ISLNK(s.st_mode): kind='symlink'
 elif stat.S_ISDIR(s.st_mode): kind='directory'
 else: kind='file'
 return {
  'name':os.path.basename(path),'path':path,'kind':kind,
  'size':s.st_size,'mtime_ms':s.st_mtime*1000,
 }
if op=='exists':
 out=os.path.lexists(p['path'])
elif op=='canonical':
 if not os.path.lexists(p['path']): raise FileNotFoundError(p['path'])
 out=os.path.realpath(p['path'])
elif op=='read':
 with open(p['path'],'rb') as f:
  out=base64.b64encode(f.read()).decode()
elif op=='write':
 os.makedirs(os.path.dirname(p['path']),exist_ok=True)
 with open(p['path'],'wb') as f:
  f.write(base64.b64decode(p['content']))
 out=None
elif op=='append':
 os.makedirs(os.path.dirname(p['path']),exist_ok=True)
 with open(p['path'],'ab') as f:
  f.write(base64.b64decode(p['content']))
 out=None
elif op=='rename':
 os.replace(p['source'],p['destination'])
 out=None
elif op=='info':
 out=info(p['path'])
elif op=='list':
 names=sorted(os.listdir(p['path']))
 out=[info(os.path.join(p['path'],name)) for name in names]
elif op=='mkdir':
 if p['recursive']: os.makedirs(p['path'],exist_ok=True)
 else: os.mkdir(p['path'])
 out=None
elif op=='remove':
 path=p['path']
 if not os.path.lexists(path) and p['force']: out=None
 elif os.path.isdir(path) and not os.path.islink(path):
  if p['recursive']: shutil.rmtree(path)
  else: os.rmdir(path)
  out=None
 else:
  os.unlink(path)
  out=None
elif op=='temp_dir':
 out=tempfile.mkdtemp(prefix=p['prefix'],dir=p['parent'])
elif op=='temp_file':
 fd,out=tempfile.mkstemp(
  prefix=p['prefix'],suffix=p['suffix'],dir=p['parent'],
 )
 os.close(fd)
else:
 raise ValueError('unsupported helper operation')
print(json.dumps({'value':out},separators=(',',':')))
""".strip()


class PyromindHttpSandboxBackend(SandboxExecutionBackend):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        cluster: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        request_timeout_seconds: float = 30.0,
    ) -> None:
        if not base_url:
            raise ValueError("Pyromind sandbox base_url is required")
        self._base_url = base_url.rstrip("/")
        self._access_key_url = _access_key_url(self._base_url)
        self._headers = {
            "accept": "application/json",
        }
        if cluster:
            self._headers["x-cluster"] = cluster
        for name, value in (headers or {}).items():
            if name.lower() in {"authorization", "cookie", "x-cluster"}:
                self._headers[name.lower()] = value
        # A server-side access key is authoritative. In particular, do not let
        # an unrelated inbound Authorization header replace it.
        if api_key and api_key.strip():
            self._headers["authorization"] = f"Bearer {api_key.strip()}"
        self._credential_lock = asyncio.Lock()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=request_timeout_seconds,
        )

    async def create_sandbox(self, request: JsonObject) -> str:
        payload = await self._json_request(
            "POST",
            "/sandboxes",
            json_body=request,
        )
        sandbox_id = payload.get("sandbox_id") or payload.get("id")
        if not isinstance(sandbox_id, str) or not sandbox_id:
            raise SandboxGatewayError(
                "unknown", "Sandbox create response is missing id"
            )
        return sandbox_id

    async def wait_until_running(
        self,
        sandbox_id: str,
        *,
        timeout_seconds: float = 300.0,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            payload = await self._json_request("GET", f"/sandboxes/{sandbox_id}")
            status = payload.get("status")
            if isinstance(status, str) and status.lower() in {"running", "ready"}:
                return
            if isinstance(status, str) and status.lower() in {
                "failed",
                "error",
                "deleted",
            }:
                raise SandboxGatewayError("unknown", "Sandbox failed to become ready")
            remaining = deadline - asyncio.get_running_loop().time()
            await asyncio.sleep(min(2.0, max(0.0, remaining)))
        raise TimeoutError("Sandbox creation timed out")

    async def delete_sandbox(self, sandbox_id: str) -> None:
        await self._request("DELETE", f"/sandboxes/{sandbox_id}")

    async def exists(self, sandbox_id: str, path: str) -> bool:
        value = await self._helper(sandbox_id, {"op": "exists", "path": path})
        if not isinstance(value, bool):
            raise SandboxGatewayError("unknown", "Invalid sandbox exists response")
        return value

    async def canonical_path(self, sandbox_id: str, path: str) -> str:
        value = await self._helper(sandbox_id, {"op": "canonical", "path": path})
        if not isinstance(value, str):
            raise SandboxGatewayError("unknown", "Invalid sandbox canonical path")
        return value

    async def read_file(self, sandbox_id: str, path: str) -> bytes:
        value = await self._helper(sandbox_id, {"op": "read", "path": path})
        if not isinstance(value, str):
            raise SandboxGatewayError("unknown", "Invalid sandbox file content")
        try:
            return base64.b64decode(value, validate=True)
        except ValueError as exc:
            raise SandboxGatewayError(
                "unknown", "Invalid sandbox file encoding"
            ) from exc

    async def write_file(
        self,
        sandbox_id: str,
        path: str,
        content: bytes,
    ) -> None:
        await self._helper(
            sandbox_id,
            {
                "op": "write",
                "path": path,
                "content": base64.b64encode(content).decode(),
            },
        )

    async def append_file(
        self,
        sandbox_id: str,
        path: str,
        content: bytes,
    ) -> None:
        await self._helper(
            sandbox_id,
            {
                "op": "append",
                "path": path,
                "content": base64.b64encode(content).decode(),
            },
        )

    async def rename_file(
        self,
        sandbox_id: str,
        source_path: str,
        destination_path: str,
    ) -> None:
        await self._helper(
            sandbox_id,
            {"op": "rename", "source": source_path, "destination": destination_path},
        )

    async def file_info(self, sandbox_id: str, path: str) -> SandboxFileInfo:
        value = await self._helper(sandbox_id, {"op": "info", "path": path})
        if not isinstance(value, dict):
            raise SandboxGatewayError("unknown", "Invalid sandbox file info")
        return _file_info(value)

    async def list_dir(
        self,
        sandbox_id: str,
        path: str,
    ) -> tuple[SandboxFileInfo, ...]:
        value = await self._helper(sandbox_id, {"op": "list", "path": path})
        if not isinstance(value, list):
            raise SandboxGatewayError("unknown", "Invalid sandbox directory listing")
        return tuple(_file_info(item) for item in value if isinstance(item, dict))

    async def create_dir(
        self,
        sandbox_id: str,
        path: str,
        *,
        recursive: bool,
    ) -> None:
        await self._helper(
            sandbox_id,
            {"op": "mkdir", "path": path, "recursive": recursive},
        )

    async def remove(
        self,
        sandbox_id: str,
        path: str,
        *,
        recursive: bool,
        force: bool,
    ) -> None:
        await self._helper(
            sandbox_id,
            {
                "op": "remove",
                "path": path,
                "recursive": recursive,
                "force": force,
            },
        )

    async def create_temp_dir(
        self,
        sandbox_id: str,
        parent: str,
        prefix: str,
    ) -> str:
        value = await self._helper(
            sandbox_id,
            {"op": "temp_dir", "parent": parent, "prefix": prefix},
        )
        if not isinstance(value, str):
            raise SandboxGatewayError("unknown", "Invalid sandbox temp directory")
        return value

    async def create_temp_file(
        self,
        sandbox_id: str,
        parent: str,
        prefix: str,
        suffix: str,
    ) -> str:
        value = await self._helper(
            sandbox_id,
            {
                "op": "temp_file",
                "parent": parent,
                "prefix": prefix,
                "suffix": suffix,
            },
        )
        if not isinstance(value, str):
            raise SandboxGatewayError("unknown", "Invalid sandbox temp file")
        return value

    async def exec(
        self,
        sandbox_id: str,
        command: str,
        *,
        cwd: str,
        environment: Mapping[str, str],
        timeout_seconds: float,
    ) -> SandboxExecResult:
        for name in environment:
            if not _ENV_NAME.fullmatch(name):
                raise SandboxGatewayError(
                    "invalid", "Invalid environment variable name"
                )
        assignments = " ".join(
            f"{name}={shlex.quote(value)}" for name, value in environment.items()
        )
        isolated = f"env -i {assignments} /bin/sh -lc {shlex.quote(command)}"
        payload = await self._exec_request(
            sandbox_id,
            isolated,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )
        output = payload.get("output", "")
        exception = payload.get("exception_info", "")
        return_code = payload.get("returncode", -1)
        return SandboxExecResult(
            stdout=output if isinstance(output, str) else str(output),
            stderr=exception if isinstance(exception, str) else str(exception),
            exit_code=return_code if isinstance(return_code, int) else -1,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _helper(self, sandbox_id: str, params: JsonObject) -> object:
        encoded = base64.urlsafe_b64encode(
            json.dumps(params, separators=(",", ":")).encode()
        ).decode()
        command = f"python3 -c {shlex.quote(_HELPER_SCRIPT)} {shlex.quote(encoded)}"
        payload = await self._exec_request(
            sandbox_id,
            command,
            cwd="/",
            timeout_seconds=30,
        )
        output = payload.get("output")
        return_code = payload.get("returncode")
        if return_code != 0 or not isinstance(output, str):
            raise SandboxGatewayError("unknown", "Sandbox filesystem helper failed")
        try:
            result = json.loads(output.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise SandboxGatewayError(
                "unknown", "Invalid sandbox helper response"
            ) from exc
        if not isinstance(result, dict) or "value" not in result:
            raise SandboxGatewayError("unknown", "Invalid sandbox helper response")
        return result["value"]

    async def _exec_request(
        self,
        sandbox_id: str,
        command: str,
        *,
        cwd: str,
        timeout_seconds: float,
    ) -> JsonObject:
        return await self._json_request(
            "POST",
            f"/sandboxes/{sandbox_id}/exec",
            json_body={
                "command": command,
                "cwd": cwd,
                "timeout": min(600, max(1, math.ceil(timeout_seconds))),
            },
        )

    async def _json_request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        json_body: JsonObject | None = None,
    ) -> JsonObject:
        response = await self._request(
            method,
            path,
            params=params,
            json_body=json_body,
        )
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise SandboxGatewayError(
                "unknown", "Sandbox API returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise SandboxGatewayError("unknown", "Sandbox API returned invalid payload")
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            raise SandboxGatewayError("unknown", "Sandbox API response is missing data")
        return data

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        json_body: JsonObject | None = None,
    ) -> httpx.Response:
        await self._ensure_authorization()
        try:
            response = await self._client.request(
                method,
                f"{self._base_url}{path}",
                params=params,
                json=json_body,
                headers=self._headers,
            )
        except httpx.TimeoutException as exc:
            raise TimeoutError("Sandbox API request timed out") from exc
        except httpx.HTTPError as exc:
            raise SandboxGatewayError("unknown", "Sandbox API request failed") from exc
        if response.is_success:
            return response
        code = {
            400: "invalid",
            422: "invalid",
            401: "permission_denied",
            403: "permission_denied",
            404: "not_found",
        }.get(response.status_code, "unknown")
        validation_detail = (
            _safe_validation_detail(response)
            if response.status_code == 422
            else None
        )
        detail_suffix = (
            f": {validation_detail}" if validation_detail is not None else ""
        )
        raise SandboxGatewayError(
            code,
            f"Sandbox API {method} {path} returned HTTP "
            f"{response.status_code}{detail_suffix}",
        )

    async def _ensure_authorization(self) -> None:
        """Exchange the portal login cookie for an API access key when needed."""
        if self._headers.get("authorization"):
            return
        cookie_header = self._headers.get("cookie")
        auth_token = _cookie_value(cookie_header, _AUTH_COOKIE_NAME)
        if auth_token is None:
            raise SandboxGatewayError(
                "permission_denied",
                "Sandbox authentication is unavailable; sign in to Pyromind "
                "or configure PYROMIND_API_KEY",
            )

        async with self._credential_lock:
            if self._headers.get("authorization"):
                return
            exchange_headers = {
                "accept": "application/json",
                "auth_token": auth_token,
                "cookie": cookie_header or "",
            }
            if cluster := self._headers.get("x-cluster"):
                exchange_headers["x-cluster"] = cluster
            try:
                response = await self._client.request(
                    "POST",
                    self._access_key_url,
                    json={},
                    headers=exchange_headers,
                )
            except httpx.TimeoutException as exc:
                raise TimeoutError("Sandbox credential exchange timed out") from exc
            except httpx.HTTPError as exc:
                raise SandboxGatewayError(
                    "unknown", "Sandbox credential exchange failed"
                ) from exc
            if not response.is_success:
                code = (
                    "permission_denied"
                    if response.status_code in {401, 403}
                    else "unknown"
                )
                raise SandboxGatewayError(
                    code,
                    f"Sandbox credential exchange returned HTTP {response.status_code}",
                )
            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                raise SandboxGatewayError(
                    "unknown", "Sandbox credential exchange returned invalid JSON"
                ) from exc
            data = payload.get("data") if isinstance(payload, dict) else None
            access_key = (
                data.get("accessKey") or data.get("access_key")
                if isinstance(data, dict)
                else None
            )
            if not isinstance(access_key, str) or not access_key.strip():
                raise SandboxGatewayError(
                    "unknown",
                    "Sandbox credential exchange response is missing accessKey",
                )
            self._headers["authorization"] = f"Bearer {access_key.strip()}"
            # The API access key is sufficient from here on; avoid forwarding
            # the browser's full cookie header to sandbox API endpoints.
            self._headers.pop("cookie", None)


def _safe_validation_detail(response: httpx.Response) -> str | None:
    """Return only non-sensitive FastAPI validation metadata."""
    try:
        payload = response.json()
    except json.JSONDecodeError:
        content_type = response.headers.get("content-type", "").lower()
        text = response.text.strip()
        if content_type.startswith("text/plain") and text and len(text) <= 500:
            return text
        return None
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail")
    message = payload.get("message")
    text_parts = [
        value.strip()
        for value in (message, detail)
        if isinstance(value, str) and value.strip()
    ]
    if text_parts:
        return "; ".join(text_parts)[:500]
    nested_candidates = [payload.get("error")]
    if isinstance(detail, dict):
        nested_candidates.extend((detail, detail.get("error")))
    for candidate in nested_candidates:
        if not isinstance(candidate, dict):
            continue
        nested_code = candidate.get("code") or candidate.get("error_code")
        nested_message = candidate.get("message")
        if isinstance(nested_message, str) and nested_message.strip():
            prefix = f"{nested_code}: " if isinstance(nested_code, str) else ""
            return f"{prefix}{nested_message.strip()}"[:500]
    if not isinstance(detail, list):
        return None

    errors: list[str] = []
    for item in detail[:5]:
        if not isinstance(item, dict):
            return None
        location = item.get("loc")
        message = item.get("msg")
        error_type = item.get("type")
        if (
            not isinstance(location, list)
            or not all(isinstance(part, str | int) for part in location)
            or not isinstance(message, str)
            or not isinstance(error_type, str)
        ):
            return None
        rendered_location = ".".join(str(part) for part in location)
        errors.append(f"{rendered_location}: {message} ({error_type})")
    return "; ".join(errors) or None


def _file_info(value: Mapping[object, object]) -> SandboxFileInfo:
    kind = value.get("kind")
    if kind not in {"file", "directory", "symlink"}:
        raise SandboxGatewayError("unknown", "Invalid sandbox file kind")
    name = value.get("name")
    path = value.get("path")
    size = value.get("size")
    mtime_ms = value.get("mtime_ms")
    if (
        not isinstance(name, str)
        or not isinstance(path, str)
        or not isinstance(size, int)
        or not isinstance(mtime_ms, (int, float))
    ):
        raise SandboxGatewayError("unknown", "Invalid sandbox file info")
    return SandboxFileInfo(
        name=name,
        path=path,
        kind=cast(SandboxFileKind, kind),
        size=size,
        mtime_ms=float(mtime_ms),
    )


def _cookie_value(cookie_header: str | None, name: str) -> str | None:
    if not cookie_header:
        return None
    for part in cookie_header.split(";"):
        candidate = part.strip()
        if not candidate or "=" not in candidate:
            continue
        cookie_name, value = candidate.split("=", 1)
        if cookie_name.strip() == name:
            stripped = value.strip()
            return stripped or None
    return None


def _access_key_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/api/v1"):
        path = path[: -len("/api/v1")]
    return urlunsplit(
        (parsed.scheme, parsed.netloc, f"{path}/account/find_access_key", "", "")
    )
