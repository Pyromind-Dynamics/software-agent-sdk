# Harness-neutral product runtime

`pyromind_runtime` is the stable product boundary around agent harnesses. Its
dependency direction is:

```text
HTTP commands / SSE / ProductEvent
                 ↓
Product runtime, event store, snapshots, projectors
                 ↓
HarnessProtocol, ToolSpec, WorkspaceRef, SandboxRef
                 ↓
OpenHandsAdapter | PiAdapter | future adapters
```

Only `adapters/openhands` may import OpenHands events and services. Pi process,
JSONL, model, and tool types stay in `adapters/pi` and `pi-runtime`. Projectors
and clients must not consume `provider_metadata`.

Each conversation persists `metadata.json`, `events.jsonl`, and `snapshot.json`.
Product event sequence numbers are assigned at append time. SSE replays events
after the requested sequence before switching to live delivery.

The default harness only selects new conversations. Existing conversations use
their persisted `harness_id`. Inactive sessions are recreated only when their
effective capability set declares `resume=true`; Pi never fakes resume.

Forking is capability-gated. The OpenHands adapter wraps its existing full
conversation fork. The first version only forks an idle conversation at its
current `through_seq`; historical workspace rollback is deliberately rejected.

Run the focused suite with:

```bash
uv run pytest tests/agent_server/pyromind_runtime
```
