"""Single-process scheduler lifecycle; business decisions remain in Runtime."""

from __future__ import annotations

import asyncio
import fcntl
import logging
from pathlib import Path
from typing import IO

from pyromind_runtime.application.conversation_runtime import ConversationRuntime


logger = logging.getLogger(__name__)


class CheckpointScheduler:
    def __init__(self, runtime: ConversationRuntime, lock_path: Path):
        self.runtime, self.lock_path = runtime, lock_path
        self._lock: IO[str] | None = None
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None:
            return
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.lock_path.open("a")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError(
                "Checkpoint polling requires one server worker per conversation store"
            ) from None
        self._lock = lock
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            self.runtime.checkpoint_wakeup.clear()
            try:
                await self.runtime.poll_checkpoint_evaluations()
            except Exception:
                logger.exception("Checkpoint polling pass failed")
            try:
                await asyncio.wait_for(
                    self.runtime.checkpoint_wakeup.wait(), timeout=60
                )
            except TimeoutError:
                pass

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._lock is not None:
            self._lock.close()
            self._lock = None
