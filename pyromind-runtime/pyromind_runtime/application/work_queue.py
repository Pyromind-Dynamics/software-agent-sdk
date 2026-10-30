"""Bounded workers with explicit reconciliation; no Harness-specific policies."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping

from pyromind_runtime.ports.work_queue import WorkQueue


logger = logging.getLogger(__name__)
Handler = Callable[[dict], Awaitable[dict]]
Reconciler = Callable[[dict], Awaitable[bool]]


class DurableWorker:
    def __init__(
        self,
        queue: WorkQueue,
        handlers: Mapping[str, Handler],
        reconcile: Reconciler,
        *,
        lease_seconds: int = 90,
        heartbeat_seconds: float = 20,
        poll_seconds: float = 1,
    ):
        if heartbeat_seconds >= lease_seconds or heartbeat_seconds <= 0:
            raise ValueError("heartbeat must be shorter than the lease")
        self.queue = queue
        self.handlers = dict(handlers)
        self.reconcile = reconcile
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.poll_seconds = poll_seconds
        self._tasks: list[asyncio.Task] = []
        self._closing = asyncio.Event()

    async def recover(self) -> None:
        for item in await asyncio.to_thread(self.queue.expired):
            if item["kind"] not in self.handlers:
                continue
            safe = await self.reconcile(item)
            await asyncio.to_thread(self.queue.reconcile_expired, item["id"], safe)

    async def run_once(self) -> bool:
        item = await asyncio.to_thread(
            self.queue.claim, self.lease_seconds, tuple(self.handlers)
        )
        if item is None:
            return False
        handler = self.handlers.get(item["kind"])
        if handler is None:
            await asyncio.to_thread(
                self.queue.finish,
                item["id"],
                item["generation"],
                {"error": "unsupported work kind"},
                "failed",
            )
            return True
        task = asyncio.ensure_future(handler(item))
        heartbeat = asyncio.create_task(self._heartbeat(item))
        try:
            done, _ = await asyncio.wait(
                {task, heartbeat}, return_when=asyncio.FIRST_COMPLETED
            )
            if heartbeat in done:
                # Lost ownership cancels execution before any result can commit.
                await heartbeat
                raise RuntimeError("heartbeat stopped unexpectedly")
            result = await task
            await asyncio.to_thread(
                self.queue.finish, item["id"], item["generation"], result
            )
        except asyncio.CancelledError:
            # Keep running evidence for reconciliation instead of requeuing on shutdown.
            raise
        except PermissionError:
            await asyncio.to_thread(
                self.queue.finish,
                item["id"],
                item["generation"],
                {"reason": "reauthorization required"},
                "waiting_authorization",
            )
        except Exception:
            logger.exception("Work item failed: %s", item["id"])
            # Handler failures may follow external acceptance. Never replay blindly.
            try:
                await asyncio.to_thread(
                    self.queue.finish,
                    item["id"],
                    item["generation"],
                    {"reason": "execution requires reconciliation"},
                    "reconciliation_required",
                )
            except ValueError:
                logger.info(
                    "Work ownership expired before failure receipt: %s", item["id"]
                )
        finally:
            task.cancel()
            heartbeat.cancel()
            await asyncio.gather(task, heartbeat, return_exceptions=True)
        return True

    async def _heartbeat(self, item: dict) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_seconds)
            await asyncio.to_thread(
                self.queue.heartbeat, item["id"], item["generation"], self.lease_seconds
            )

    async def _run(self) -> None:
        while not self._closing.is_set():
            try:
                if await self.run_once():
                    continue
                await self.recover()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Queue poll failed")
            try:
                await asyncio.wait_for(self._closing.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass

    async def start(self, concurrency: int = 2) -> None:
        if self._tasks:
            return
        if concurrency < 1:
            raise ValueError("worker concurrency must be positive")
        self._closing.clear()
        self._tasks = [asyncio.create_task(self._run()) for _ in range(concurrency)]

    async def close(self) -> None:
        self._closing.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
