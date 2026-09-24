"""Measure Pi conversation memory, release, and re-attach cost.

Run with the repo venv::

    uv run python .pr/pi_capacity_probe.py memory --conversations 8
    uv run python .pr/pi_capacity_probe.py grace --grace 5

``memory`` materialises N real Pi runners (one Node process per conversation),
samples RSS after each step, releases them through the runtime eviction path,
and times re-attach. ``grace`` drives one real LLM turn and prints a timeline of
status, active conversation count, and live runner processes so the grace-period
release can be observed end to end.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import psutil
from harness_adapter.pi_adapter import PiAdapter
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.harness import SessionSpec


CONTEXT = RequestContext(user_id="probe")


def _runner_processes(pid: int) -> list[psutil.Process]:
    try:
        root = psutil.Process(pid)
    except psutil.Error:
        return []
    alive: list[psutil.Process] = []
    for process in [root, *root.children(recursive=True)]:
        try:
            if process.is_running() and "index.js" in " ".join(process.cmdline()):
                alive.append(process)
        except psutil.Error:
            continue
    return alive


def _rss_mb(processes: list[psutil.Process]) -> float:
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.Error:
            continue
    return total / (1024 * 1024)


def sample(label: str) -> tuple[float, float]:
    pid = os.getpid()
    try:
        self_process = psutil.Process(pid)
        python_rss = self_process.memory_info().rss / (1024 * 1024)
    except psutil.Error:
        python_rss = 0.0
    runners = _runner_processes(pid)
    per_runner: list[float] = []
    for runner in runners:
        try:
            per_runner.append(runner.memory_info().rss / (1024 * 1024))
        except psutil.Error:
            continue
    runner_rss = sum(per_runner)
    print(
        f"[sample] {label:<28} runners={len(runners):<3} "
        f"runner_rss={runner_rss:8.1f}MB python_rss={python_rss:7.1f}MB "
        f"total={runner_rss + python_rss:8.1f}MB",
        flush=True,
    )
    if per_runner:
        print(
            f"[runner] {label:<28} median={statistics.median(per_runner):7.1f}MB "
            f"min={min(per_runner):7.1f}MB max={max(per_runner):7.1f}MB",
            flush=True,
        )
    return runner_rss, python_rss


def _runtime_kwargs(args: argparse.Namespace) -> dict[str, int]:
    params = inspect.signature(ConversationRuntime.__init__).parameters
    kwargs: dict[str, int] = {}
    if "release_grace_seconds" in params:
        kwargs["release_grace_seconds"] = args.grace
    if "max_active_conversations" in params:
        kwargs["max_active_conversations"] = args.cap
    return kwargs


def _spec(root: Path, conversation_id: str, prompt: str | None = None) -> SessionSpec:
    return SessionSpec(
        conversation_id=conversation_id,
        user_id="probe",
        workspace_root=str(root / conversation_id),
        initial_message=((TextContent(text=prompt),) if prompt else ()),
        model_configuration={
            "model": os.getenv("LLM_MODEL", "openai/gpt-5"),
            "base_url": os.getenv("LLM_BASE_URL", ""),
            "api_key": os.getenv("OPENAI_API_KEY", "probe-key"),
        },
    )


def _build(
    root: Path, args: argparse.Namespace, **overrides: int
) -> ConversationRuntime:
    adapter = PiAdapter(root, terminal_backend=args.backend)
    kwargs = _runtime_kwargs(args)
    kwargs.update(overrides)
    return ConversationRuntime(
        root,
        adapter,
        default_harness_id="pi",
        idle_eviction_seconds=args.idle,
        **kwargs,
    )


async def run_memory(args: argparse.Namespace) -> None:
    root = Path(tempfile.mkdtemp(prefix="pi-probe-")).resolve()
    print(f"[setup] conversations_dir={root} backend={args.backend}", flush=True)
    runtime = _build(root, args)
    try:
        baseline_runner, baseline_python = sample("baseline")
        creation_ms: list[float] = []
        for index in range(args.conversations):
            started = time.perf_counter()
            snapshot = await runtime.create_conversation(
                _spec(root, f"probe-{index:03d}"), CONTEXT
            )
            creation_ms.append((time.perf_counter() - started) * 1000)
            if index == 0:
                print(
                    f"[setup] first conversation ready id={snapshot.conversation_id} "
                    f"status={snapshot.status}",
                    flush=True,
                )
        runner_rss, python_rss = sample(f"{args.conversations} conversations live")
        print(
            f"[result] create_ms p50={statistics.median(creation_ms):.0f} "
            f"min={min(creation_ms):.0f} max={max(creation_ms):.0f}",
            flush=True,
        )
        per_conversation = (runner_rss - baseline_runner) / args.conversations
        print(
            f"[result] runner_rss_per_conversation={per_conversation:.1f}MB "
            f"python_rss={python_rss:.1f}MB (baseline {baseline_python:.1f}MB)",
            flush=True,
        )

        released_at = time.perf_counter()
        await runtime._evict_idle()
        release_ms = (time.perf_counter() - released_at) * 1000
        sample(f"released all ({release_ms:.0f}ms)")
        print(
            f"[result] release_all_ms={release_ms:.0f} "
            f"active_after={len(runtime._active)}",
            flush=True,
        )

        reattach_ms: list[float] = []
        for index in range(min(args.reattach, args.conversations)):
            conversation_id = f"probe-{index:03d}"
            started = time.perf_counter()
            await runtime._ensure_active(conversation_id, CONTEXT)
            reattach_ms.append((time.perf_counter() - started) * 1000)
        sample(f"re-attached {len(reattach_ms)}")
        if reattach_ms:
            print(
                f"[result] reattach_ms p50={statistics.median(reattach_ms):.0f} "
                f"min={min(reattach_ms):.0f} max={max(reattach_ms):.0f}",
                flush=True,
            )
    finally:
        await runtime.close()
        sample("after runtime close")
        shutil.rmtree(root, ignore_errors=True)


async def run_grace(args: argparse.Namespace) -> None:
    root = Path(tempfile.mkdtemp(prefix="pi-probe-")).resolve()
    print(
        f"[setup] conversations_dir={root} backend={args.backend} "
        f"grace={args.grace}s model={os.getenv('LLM_MODEL', '-')}",
        flush=True,
    )
    runtime = _build(root, args)
    try:
        snapshot = await runtime.create_conversation(
            _spec(root, "probe-grace", "Reply with exactly: ok"), CONTEXT
        )
        conversation_id = snapshot.conversation_id
        print(f"[setup] conversation={conversation_id}", flush=True)
        deadline = time.monotonic() + args.timeout
        seen: list[str] = []
        while time.monotonic() < deadline:
            status = runtime._store(conversation_id).load_snapshot().status
            runners = len(_runner_processes(os.getpid()))
            row = (
                f"t={time.monotonic() - deadline + args.timeout:6.1f}s "
                f"status={status:<12} active={len(runtime._active)} "
                f"runners={runners}"
            )
            if not seen or seen[-1].split("status=")[1] != row.split("status=")[1]:
                print(f"[timeline] {row}", flush=True)
            seen.append(row)
            if status in {"idle", "finished", "error"} and runners == 0:
                break
            await asyncio.sleep(0.5)
        else:
            print("[timeline] timed out waiting for release", flush=True)

        started = time.perf_counter()
        await runtime._ensure_active(conversation_id, CONTEXT)
        print(
            f"[result] reactivate_ms={(time.perf_counter() - started) * 1000:.0f} "
            f"active={len(runtime._active)}",
            flush=True,
        )
    finally:
        await runtime.close()
        shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("memory", "grace"))
    parser.add_argument("--backend", default="os-sandbox")
    parser.add_argument("--conversations", type=int, default=8)
    parser.add_argument("--reattach", type=int, default=3)
    parser.add_argument("--grace", type=int, default=5)
    parser.add_argument("--idle", type=int, default=0)
    parser.add_argument("--cap", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    runner = run_memory if args.mode == "memory" else run_grace
    asyncio.run(runner(args))


if __name__ == "__main__":
    main()
