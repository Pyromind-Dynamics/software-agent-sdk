#!/usr/bin/env python3
"""Convert coding-agent traces into SFT training format.

Consumes the ``traces/<task_id>.*_trace.jsonl`` files exported by
``sandbox_runner.py`` plus the run's verdicts, and emits one
``{"messages": [...]}`` sample per line. Two trace formats are supported and
detected per file:

- pi coding agent (``--mode json`` events): ``message_end`` events with
  roles user / assistant / toolResult. The launch prompt is the first user
  message, so ``--manifest`` is NOT prepended again for these traces.
- Claude Code (``--output-format stream-json``): assistant/user events; the
  task prompt never appears in the stream, so pass ``--manifest`` to prepend
  each task's ``prompt`` as the leading ``user`` message.

``--system-prompt`` prepends a coding-agent system message for both formats.
Without ``--manifest`` (CC traces) or a leading user message, the output lacks
the problem statement and must not be used for SFT.

Mapping onto the OpenAI-style message schema:
- assistant ``text`` blocks   -> message ``content``
- assistant tool calls        -> ``tool_calls`` (function call, JSON string args)
- tool results                -> ``role=tool`` messages keyed by tool call id
- ``thinking`` blocks are dropped (not an SFT target)

Consecutive assistant stream events are merged into one message. ``<system-reminder>``
injections are kept verbatim: they are part of the real rollout distribution the
trained actor will see. Default ``--min-reward`` is 1.0 — SFT imitates, so only
solved trajectories belong in the cold-start set; lower it explicitly to include
partial-credit tasks.

Usage:
    python convert_to_sft.py \\
        --traces-dir run-dir/traces \\
        --verdicts run-dir/verdicts.jsonl \\
        --out sft.jsonl \\
        [--manifest run-dir/manifest.jsonl] \\
        [--system-prompt "You are a helpful coding assistant."] \\
        [--min-reward 1.0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(parts)
    return str(content)


def _assistant_event(
    event: dict,
) -> tuple[str, list[dict]]:
    """Split one CC assistant stream event into (text, tool_calls)."""
    content = event.get("message", {}).get("content", [])
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    tool_calls: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = str(block.get("text", ""))
            if text:
                texts.append(text)
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name", "")),
                        "arguments": json.dumps(
                            block.get("input", {}), ensure_ascii=False
                        ),
                    },
                }
            )
    return "\n\n".join(texts), tool_calls


def _user_events(event: dict) -> list[dict]:
    """Turn one CC user stream event into user and/or tool messages."""
    content = event.get("message", {}).get("content")
    if isinstance(content, str):
        return [{"role": "user", "content": content}] if content.strip() else []
    messages: list[dict] = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id", "")),
                    "content": _result_text(block.get("content")),
                }
            )
        elif block.get("type") == "text":
            text = str(block.get("text", ""))
            if text:
                messages.append({"role": "user", "content": text})
    return messages


def convert_trace(events: list[dict]) -> list[dict]:
    """Map CC stream-json events onto OpenAI-style messages."""
    messages: list[dict] = []
    for event in events:
        event_type = event.get("type")
        if event_type == "assistant":
            text, tool_calls = _assistant_event(event)
            if not text and not tool_calls:
                continue
            if messages and messages[-1]["role"] == "assistant":
                previous = messages[-1]
                if text:
                    previous["content"] = (
                        f"{previous.get('content') or ''}\n\n{text}"
                        if previous.get("content")
                        else text
                    )
                if tool_calls:
                    previous.setdefault("tool_calls", []).extend(tool_calls)
            else:
                message: dict[str, Any] = {"role": "assistant", "content": text}
                if tool_calls:
                    message["tool_calls"] = tool_calls
                messages.append(message)
        elif event_type == "user":
            messages.extend(_user_events(event))
    return messages


def is_pi_trace(events: list[dict]) -> bool:
    """pi ``--mode json`` streams use message_end events; CC does not."""
    return any(event.get("type") == "message_end" for event in events)


def convert_pi_trace(events: list[dict]) -> list[dict]:
    """Map pi ``--mode json`` events onto OpenAI-style messages.

    Every event of interest is a complete ``message_end``: the launch prompt
    is the single ``user`` message, assistant turns carry ``text``/``toolCall``
    blocks, and tool outputs arrive as ``toolResult`` messages keyed by
    ``toolCallId``.
    """
    messages: list[dict] = []
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message", {})
        role = message.get("role")
        if role == "user":
            text = _result_text(message.get("content"))
            if text:
                messages.append({"role": "user", "content": text})
        elif role == "assistant":
            texts: list[str] = []
            tool_calls: list[dict] = []
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = str(block.get("text", ""))
                    if text:
                        texts.append(text)
                elif block.get("type") == "toolCall":
                    tool_calls.append(
                        {
                            "id": str(block.get("id", "")),
                            "type": "function",
                            "function": {
                                "name": str(block.get("name", "")),
                                "arguments": json.dumps(
                                    block.get("arguments", {}), ensure_ascii=False
                                ),
                            },
                        }
                    )
            if not texts and not tool_calls:
                continue
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": "\n\n".join(texts),
            }
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
        elif role == "toolResult":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(message.get("toolCallId", "")),
                    "content": _result_text(message.get("content")),
                }
            )
    return messages


def find_trace_path(traces_dir: Path, task_id: str) -> Path | None:
    """Locate the trace file for one task (pi first, then CC)."""
    for suffix in (".pi_trace.jsonl", ".cc_trace.jsonl"):
        candidate = traces_dir / f"{task_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_trace_events(trace_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_sft_messages(
    events: list[dict],
    prompt: str | None,
    system_prompt: str | None,
) -> list[dict] | None:
    """Turn one trace into an OpenAI-style messages sample.

    Returns None when the trace yields no valid sample: either no messages
    at all, or the trajectory does not end with an assistant turn.
    """
    if is_pi_trace(events):
        messages = convert_pi_trace(events)
        # The launch prompt is already the first user message in a pi
        # trace; prepending the manifest prompt would duplicate it.
        trace_has_prompt = bool(messages) and messages[0].get("role") == "user"
    else:
        messages = convert_trace(events)
        trace_has_prompt = False
    if not messages or messages[-1]["role"] != "assistant":
        return None
    if not trace_has_prompt and not prompt:
        # A CC trace without the manifest prompt lacks the problem
        # statement: not a usable SFT sample (see module docstring).
        return None
    leading: list[dict[str, str]] = []
    if system_prompt:
        leading.append({"role": "system", "content": system_prompt})
    if prompt and not trace_has_prompt:
        leading.append({"role": "user", "content": prompt})
    return leading + messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-dir", required=True, type=Path)
    parser.add_argument("--verdicts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-reward", type=float, default=1.0)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--system-prompt", default=None)
    args = parser.parse_args(argv)

    rewards: dict[str, float | None] = {}
    for line in args.verdicts.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("verdict") != "usable":
            continue
        task_id = str(entry.get("task_id", ""))
        reward = entry.get("reward")
        rewards[task_id] = float(reward) if reward is not None else None

    prompts: dict[str, str] = {}
    if args.manifest is not None:
        for line in args.manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            prompts[str(entry.get("task_id", ""))] = str(entry.get("prompt", ""))

    converted = 0
    skipped_reward = 0
    skipped_no_trace = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for task_id, reward in rewards.items():
            if reward is None or reward < args.min_reward:
                skipped_reward += 1
                continue
            trace_path = find_trace_path(args.traces_dir, task_id)
            if trace_path is None:
                skipped_no_trace += 1
                continue
            messages = build_sft_messages(
                load_trace_events(trace_path),
                prompts.get(task_id),
                args.system_prompt,
            )
            if messages is None:
                print(
                    f"warning: {task_id} trace yields no valid SFT sample",
                    file=sys.stderr,
                )
                continue
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            converted += 1

    print(
        f"converted={converted} skipped-low-reward={skipped_reward} "
        f"skipped-no-trace={skipped_no_trace} out={args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
