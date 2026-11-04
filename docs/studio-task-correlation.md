# Studio task return addresses

Studio's `out_id` remains a string. Agent submissions serialize this v1 object:

```json
{"version":1,"source":"agent","conversation_id":"conversation-id","type":"main","task_kind":"workflow_run"}
```

Children use `"type":"subagent","subagent_id":"opaque-instance-id"` in the same
object. The server fills the address from the caller, without model-visible
routing parameters. UUID conversation IDs use the Runtime's canonical hex form.
Debug submissions use `workflow_debug`; data processing tools retain their task
category. Studio stores this text unchanged and returns it in task and node
Kafka events. No platform JSON parsing or database migration is required.

## Layer and persistence boundaries

- Runtime domain owns `TaskOrigin`, `TaskCorrelation`, and the JSON codec.
- Tools and the submission adapter depend on these neutral models. A scoped
  context carries the caller across threaded tools, without shared mutable
  conversation state. Both tool/server packages declare the Runtime dependency.
- Submission requests, task snapshots, and the External Task Registry persist
  correlation. An approval uses the origin saved before confirmation. Credentials
  remain in the ephemeral request context and are never included in `out_id`.
- Kafka consumers decode the common contract. Runtime checks the saved task owner
  and return address; the SQL observation ledger checks them before applying
  platform status, including observations received before registration.
- Only the adapter maps opaque child IDs to plugin session files and native run
  IDs/indexes. It reuses `pi-subagents` status artifacts and `steer`/`resume` RPCs.
  A resumed session retains its recipient ID while its native run ID may change.

An early callback is recorded as `external_task.observed` and replayed after the
matching task registers. Terminal results are deduplicated and pending delivery
is durable. An active conversation retries unavailable recipients every two
seconds; reattachment also drains the backlog. Suspended execution and explicitly
stopped children are not automatically revived. Results never fall back to the
main Agent or a replacement child. Graph and managed-task continuations retain
their existing orchestration owner instead of starting a second Agent turn.

## Compatibility and validation

The Kafka format switches directly to v1 JSON strings. `agent1#...` callbacks are
discarded; finish old-format in-flight tasks before rollout. Optional correlation
fields keep existing persisted sessions readable. The new observation event is
additive. HTTP endpoints and frontend interaction remain unchanged.

Run `uv run pytest tests/pyromind_runtime`, the workflow tool/Kafka handler tests,
and `npm test` in `harness-adapter/pi-runtime`. The native callback integration
test starts a local mock model service and real child processes. The platform
test `tests/core/workflow/test_out_id_passthrough.py` in `k8s_middleware` verifies
feature persistence and both Kafka event types, including Chinese, quotes,
backslashes, and additional opaque metadata; it requires that project's platform
dependencies.
