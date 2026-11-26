"""Attach development sessions to an operations-managed sandbox Actor."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any, Self
from urllib.parse import urlsplit

from filelock import FileLock

from harness_adapter.pi_adapter.actor_workspace import ActorTransport, ActorWorkspace
from harness_adapter.pi_adapter.business_tool_host import ToolExecutionContext
from harness_adapter.pi_adapter.fixed_actor_resources import (
    prepare_actor_resources,
    prepare_actor_sdk,
)
from harness_adapter.pi_adapter.persistence import (
    PiSessionFiles,
    _atomic_json,
    _load_object,
)
from harness_adapter.pi_adapter.protocol import FixedActorUnavailable
from harness_adapter.pi_adapter.sandbox_runtime import (
    DEFAULT_MOUNT_PATH,
    PYROMIND_AGENT_STORAGE_ROOT,
    resource_files,
    sync_debug_budget,
)


_FORK_SCRIPT = """\
set -eu
source_dir=$1
target=$2
receipt=$3
marker=.pyromind-actor-fork
matches_receipt() {
    test -d "$target" && test ! -L "$target" && test -f "$target/$marker" &&
        test ! -L "$target/$marker" && test "$(cat -- "$target/$marker")" = "$receipt"
}
if test -e "$target" || test -L "$target"; then
    matches_receipt || exit 43
    exit 0
fi
test -d "$source_dir" && test ! -L "$source_dir" || exit 42
temporary=$(mktemp -d -- "${target%/*}/.fork-XXXXXXXX")
trap 'rm -rf -- "$temporary"' EXIT
cp -a -- "$source_dir"/. "$temporary"/
rm -f -- "$temporary/$marker"
printf '%s\n' "$receipt" > "$temporary/$marker"
mv -nT -- "$temporary" "$target"
matches_receipt || exit 43
"""


class FixedActorExecutionManager:
    def __init__(
        self,
        transport: ActorTransport,
        *,
        resource_roots: Sequence[tuple[str, os.PathLike[str]]] = (),
    ) -> None:
        self._transport = transport
        self._resource_roots = tuple(resource_roots)
        self._locks: dict[str, asyncio.Lock] = {}
        self._prepared: dict[str, ActorWorkspace] = {}
        self._interpreters: dict[str, str] = {}

    @classmethod
    def from_env(
        cls, *, resource_roots: Sequence[tuple[str, os.PathLike[str]]] = ()
    ) -> Self | None:
        base = os.environ.get("PYROMIND_SANDBOX_ACTOR_URL", "").strip()
        target = os.environ.get("PYROMIND_SANDBOX_ACTOR_TARGET", "").strip()
        if not base and not target:
            return None
        if not base or not target:
            raise ValueError(
                "Set PYROMIND_SANDBOX_ACTOR_URL and PYROMIND_SANDBOX_ACTOR_TARGET "
                "together"
            )
        url = urlsplit(base)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or any(ord(ch) < 33 for ch in base)
        ):
            raise ValueError("PYROMIND_SANDBOX_ACTOR_URL must be a router HTTP(S) URL")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", target):
            raise ValueError("PYROMIND_SANDBOX_ACTOR_TARGET requires account/actor")
        return cls(
            ActorTransport(base.rstrip("/"), target), resource_roots=resource_roots
        )

    def _workspace(self, conversation_id: str) -> ActorWorkspace:
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", conversation_id):
            raise ValueError("Invalid Actor workspace conversation id")
        path = f"{DEFAULT_MOUNT_PATH}{PYROMIND_AGENT_STORAGE_ROOT}/{conversation_id}"
        return ActorWorkspace(working_dir=path, transport=self._transport)

    def _endpoint(self, workspace: ActorWorkspace) -> dict[str, Any]:
        return {
            "base_url": self._transport.base_url,
            "actor_target": self._transport.actor_target,
            "sandbox_id": self._transport.actor_target,
            "workspace_path": workspace.working_dir,
            "storage_path": workspace.storage_path,
        }

    @staticmethod
    def _record(files: PiSessionFiles) -> dict[str, Any] | None:
        try:
            return _load_object(files.directory / "actor-workspace.json")
        except FileNotFoundError:
            return None

    def recorded_workspace(
        self, conversation_id: str, files: PiSessionFiles
    ) -> ActorWorkspace | None:
        workspace = self._workspace(conversation_id)
        return workspace if self._record(files) == self._endpoint(workspace) else None

    def execution_workspace_path(self, conversation_id: str) -> str | None:
        workspace = self._prepared.get(conversation_id)
        return workspace.working_dir if workspace else None

    def _prepare(self, workspace: ActorWorkspace, pending: dict[str, Any]) -> None:
        workspace.transport.ready()
        source = pending.get("source_conversation_id")
        if source is not None:
            if not isinstance(source, str):
                raise ValueError("Invalid Actor fork source conversation id")
            source_path = self._workspace(source).working_dir
            if source_path == workspace.working_dir:
                raise ValueError("An Actor conversation cannot fork itself")
            command = shlex.join(
                [
                    "/bin/bash",
                    "-c",
                    _FORK_SCRIPT,
                    "actor-fork",
                    source_path,
                    workspace.working_dir,
                    f"{source_path} -> {workspace.working_dir}",
                ]
            )
            result = workspace.execute_command(
                command, cwd=workspace.storage_path, timeout=600
            )
            if result.exit_code:
                reason = {
                    42: "source workspace is unavailable",
                    43: "target workspace already exists",
                }.get(result.exit_code, f"copy failed (exit code {result.exit_code})")
                raise RuntimeError(f"PI_SANDBOX_FORK_FAILED: {reason}")
        canvas = PurePosixPath(workspace.working_dir) / "public_data/workflow_canvas"
        workspace.checked_command(
            f"mkdir -p -- {shlex.quote(str(canvas))}", cwd=workspace.storage_path
        )
        workflow = pending.get("workflow_dsl")
        workflow_path = str(canvas / "workflow.py")
        if isinstance(workflow, str):
            if workflow.strip():
                workspace.transport.write_file(workflow_path, workflow.encode())
            else:
                workspace.checked_command(f"rm -f -- {shlex.quote(workflow_path)}")
        prepare_actor_resources(
            workspace, resource_files(self._resource_roots, workspace.working_dir)
        )

    def _attach(self, workspace: ActorWorkspace, files: PiSessionFiles) -> None:
        endpoint = self._endpoint(workspace)
        files.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        with FileLock(files.directory / "actor-workspace.lock"):
            record = self._record(files)
            if record is not None and record != endpoint:
                raise RuntimeError(
                    "Conversation is already bound to another Actor or router URL. "
                    "Restore its original route or start a new conversation."
                )
            pending = files.load_pending_sandbox_staging()
            self._prepare(workspace, pending or {})
            if record is None:
                _atomic_json(files.directory / "actor-workspace.json", endpoint)
            if pending is not None and files.load_pending_sandbox_staging() == pending:
                files.clear_pending_sandbox_staging()

    async def workspace(
        self,
        context: ToolExecutionContext,
        files: PiSessionFiles,
        *,
        refresh: bool = False,
    ) -> ActorWorkspace:
        key = context.conversation_id
        async with self._locks.setdefault(key, asyncio.Lock()):
            pending = files.load_pending_sandbox_staging()
            if key not in self._prepared or pending is not None or refresh:
                self._prepared.pop(key, None)
                self._interpreters.pop(key, None)
                workspace = self._workspace(key)
                await asyncio.to_thread(self._attach, workspace, files)
                self._prepared[key] = workspace
            return self._prepared[key]

    async def ensure(
        self,
        context: ToolExecutionContext,
        files: PiSessionFiles,
        *,
        refresh: bool = False,
        replace: bool = False,
    ) -> dict[str, Any]:
        try:
            if replace:
                raise RuntimeError("A pre-provisioned Actor cannot be replaced")
            workspace = await self.workspace(context, files, refresh=refresh)
            return self._endpoint(workspace)
        except Exception as exc:
            raise FixedActorUnavailable(
                f"Cannot attach to Actor {self._transport.actor_target}: {exc}"
            ) from exc

    async def sdk_python(
        self, context: ToolExecutionContext, files: PiSessionFiles
    ) -> str:
        workspace = await self.workspace(context, files)
        key = context.conversation_id
        async with self._locks[key]:
            cached = self._interpreters.get(key)
            if cached:
                result = await asyncio.to_thread(
                    workspace.execute_command, f"test -x {shlex.quote(cached)}"
                )
                if result.exit_code == 0:
                    return cached
            python = await asyncio.to_thread(prepare_actor_sdk, workspace)
            self._interpreters[key] = python
            return python

    async def _debug_budget(
        self, context: ToolExecutionContext, files: PiSessionFiles, *, push: bool
    ) -> str:
        workspace = await self.workspace(context, files)

        async def run(script: str) -> str:
            return await asyncio.to_thread(
                workspace.checked_command, script, timeout=60
            )

        return await sync_debug_budget(
            context, files, workspace.working_dir, run, push=push
        )

    async def debug_state_path(
        self, context: ToolExecutionContext, files: PiSessionFiles
    ) -> str:
        return await self._debug_budget(context, files, push=False)

    async def push_debug_budget(
        self, context: ToolExecutionContext, files: PiSessionFiles
    ) -> None:
        await self._debug_budget(context, files, push=True)

    def prewarm(self, context: ToolExecutionContext, files: PiSessionFiles) -> None:
        """A fixed Actor is already running. Attach on the first workspace request."""

    async def pause(self, context: ToolExecutionContext, files: PiSessionFiles) -> None:
        del files
        key = context.conversation_id
        async with self._locks.setdefault(key, asyncio.Lock()):
            self._prepared.pop(key, None)
            self._interpreters.pop(key, None)

    async def delete(
        self, context: ToolExecutionContext, files: PiSessionFiles
    ) -> None:
        await self.pause(context, files)
