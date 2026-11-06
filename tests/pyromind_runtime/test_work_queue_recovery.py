import asyncio
from datetime import timedelta

import pytest
from pyromind_runtime.application.work_queue import DurableWorker
from pyromind_runtime.infrastructure.sql_work_queue import (
    SQLWorkQueue,
    StaleLeaseError,
    now,
    roles,
    work,
)

from .test_execution_queue import ledger as ledger_fixture


ledger = ledger_fixture


def expire(queue, item):
    with queue.engine.begin() as conn:
        conn.execute(
            work.update()
            .where(work.c.id == item["id"])
            .values(lease_until=now() - timedelta(seconds=1))
        )


@pytest.mark.parametrize("safe_to_retry", [True, False])
async def test_concurrent_recovery_updates_one_generation_and_never_replays_twice(
    ledger, safe_to_retry
):
    queue = ledger.queue
    identity = queue.enqueue("owner", "lane", "task", "prepare", {})
    original = queue.claim()
    assert original is not None
    expire(queue, original)
    ready = asyncio.Event()
    inspected = []
    executed = []

    async def reconcile(item):
        inspected.append(item["generation"])
        if len(inspected) == 2:
            ready.set()
        await ready.wait()
        return safe_to_retry

    async def handle(item):
        executed.append(item["id"])
        return {"completed": True}

    workers = [
        DurableWorker(queue, {"prepare": handle}, reconcile),
        DurableWorker(SQLWorkQueue(ledger.engine), {"prepare": handle}, reconcile),
    ]
    await asyncio.wait_for(asyncio.gather(*(worker.recover() for worker in workers)), 5)
    saved = queue.list("owner")[0]
    assert inspected == [original["generation"], original["generation"]]
    assert saved["state"] == ("queued" if safe_to_retry else "reconciliation_required")
    assert saved["generation"] == original["generation"] + 1
    with pytest.raises(StaleLeaseError):
        queue.heartbeat(identity, original["generation"])
    await asyncio.gather(*(worker.run_once() for worker in workers))
    assert executed == ([identity] if safe_to_retry else [])


@pytest.mark.parametrize("change", ["renewed", "completed"])
def test_stale_expiry_snapshot_cannot_overwrite_a_renewed_or_finished_task(
    ledger, change
):
    queue = ledger.queue
    identity = queue.enqueue("owner", "lane", "task", "prepare", {})
    original = queue.claim()
    assert original is not None
    expire(queue, original)
    observed = queue.expired()[0]
    lease = now() + timedelta(seconds=90)
    with ledger.engine.begin() as conn:
        conn.execute(
            work.update().where(work.c.id == identity).values(lease_until=lease)
        )
        conn.execute(
            roles.update()
            .where(roles.c.lane == original["lane"])
            .values(lease_until=lease)
        )
    if change == "completed":
        queue.finish(identity, original["generation"], {"answer": "preserved"})
    before = queue.list("owner")[0]
    assert (
        queue.reconcile_expired(identity, True, generation=observed["generation"])
        is False
    )
    assert queue.list("owner")[0] == before


def test_old_recovery_verdict_cannot_recover_a_new_expired_generation(ledger):
    queue = ledger.queue
    identity = queue.enqueue("owner", "lane", "task", "prepare", {})
    original = queue.claim()
    assert original is not None
    expire(queue, original)
    observed = queue.expired()[0]
    assert queue.reconcile_expired(identity, True, generation=observed["generation"])
    replacement = queue.claim()
    assert replacement is not None
    assert replacement["generation"] > observed["generation"]
    expire(queue, replacement)
    before = queue.list("owner")[0]
    assert (
        queue.reconcile_expired(identity, False, generation=observed["generation"])
        is False
    )
    assert queue.list("owner")[0] == before
    assert queue.reconcile_expired(
        identity, False, generation=replacement["generation"]
    )
    assert queue.list("owner")[0]["state"] == "reconciliation_required"


@pytest.mark.parametrize("safe_to_retry", [True, False])
async def test_worker_recovers_safe_terminal_reconciliation(ledger, safe_to_retry):
    queue = ledger.queue
    identity = queue.enqueue("owner", "lane", "notice", "manager_turn", {})
    original = queue.claim()
    assert original is not None
    queue.finish(
        identity,
        original["generation"],
        {"reason": "execution requires reconciliation"},
        "reconciliation_required",
    )
    executed = []

    async def reconcile(item):
        return safe_to_retry

    async def handle(item):
        executed.append(item["id"])
        return {"delivered": True}

    worker = DurableWorker(queue, {"manager_turn": handle}, reconcile)
    await worker.recover()
    saved = queue.list("owner")[0]
    assert saved["state"] == ("queued" if safe_to_retry else "reconciliation_required")
    assert await worker.run_once() is safe_to_retry
    assert executed == ([identity] if safe_to_retry else [])
