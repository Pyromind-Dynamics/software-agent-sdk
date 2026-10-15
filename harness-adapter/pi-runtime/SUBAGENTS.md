# Pi subagents integration

The adapter loads the unmodified, published `pi-subagents@0.78.0` package through
Pi's resource loader. Pi packages are pinned to `0.87.0`. The plugin owns child
sessions, concurrency, background processes, durable artifacts, completion
notifications and resume. There is no adapter-owned agent database or scheduler.

## Roles and calls

`agents/search.md` and `agents/general_purpose.md` use the plugin's native agent
format. The build copies them to `dist/agents`. On session creation, the adapter
copies them into the conversation's private Pi agent directory and substitutes
the compiled host-extension path. The plugin parses the profiles. New launches
read the configured profiles; resume retains the plugin's saved launch contract.

```ts
subagent({ agent: "search", task: "Find the dataset input contract", async: true })
subagent({ agent: "general_purpose", task: "Update the assigned public_data file", async: false })
subagent({ action: "status", id: "<run-id>" })
subagent({ action: "resume", id: "<run-id>", message: "Continue with this evidence" })
subagent({ action: "stop", id: "<run-id>" })
```

Resume returns a new run ID. Use that ID for the next continuation. The native
limits are four top-level async runs and four children within one run, not four
children globally. Capacity rejection, leases and recovery eligibility follow
the plugin. Headless Pi automatically drains outstanding child work at the end
of a turn; an async tool receipt does not promise the parent turn ends immediately.

## Host boundary

A required child extension fails launch when the conversation host is unavailable
or a selected tool uses a default host backend. Role extensions register only
the configured tools and a model provider connected to the private Unix socket.
The socket has mode 0600; its stable path is derived from the conversation root.
It contains no credentials. Each model request uses the current host model
runtime. Request credentials are not sent to child processes or saved in plugin
artifacts. Session and leaf IDs distinguish tool calls across resumes.

All tools execute through the adapter's existing path policies and sandbox.
Search tools use read-only backend operations; mirrored knowledge/skills can be
read on the host after canonical path checks. Terminal permission checks and
mutations share one execution lane with the parent. Disconnection aborts active
IPC work. Reconnection uses the current host and credentials.

Only repository-owned roles can be launched. Native capability ceilings, disabled
plugin features and the host's `tool_call` policy prevent tool, model, cwd,
session-directory, workflow-script and extension overrides. Children have no
platform tools, AgentGenome, nested delegation or supervisor tool. A missing
decision is returned in their handoff. Main-session business tools are unaffected.

## Lifecycle and product integration

The plugin delivers its own completion messages and wake prompts. The runner
normalizes extension-initiated turns as well as user turns. Platform callbacks
keep their existing durable queue; both paths enter the same Pi session loop.
No duplicate adapter completion notifier is installed.

The adapter projects native lifecycle events, status RPC and version-3 lifecycle
artifacts into generic `operation.*` events with category `subagent`. Status
observation does not start, replay or reschedule work. Product Runtime uses those
operations to protect eviction, fork and rollback without knowing the harness.
Cancellation calls native stop and waits for terminal status. Unexpected host
exit interrupts outstanding product operations; reattachment reconciles plugin
artifacts and leaves resume eligibility to the plugin.

Child mutations use `workflow.modified` and the existing completion/sync path,
without ending a concurrent parent run. Native cost accounting supplies cumulative
parent-plus-child usage. Full child traces remain in plugin sessions; the product
receives operation status and bounded handoffs.

Long final reports are copied to `public_data/subagents/<run-id>/report.md`
through the existing policy-bound write tool and the shared mutation lane. Native
artifact metadata must match the current parent session and role; only final
output files are read, with canonical path checks and a 16 MiB bound. Search
agents retain their read-only tool set. Short inline results do not cause copies.
The `tool_result` hook waits for foreground copies. The `context` hook projects
background notifications and retained results before a model request, without
changing native notifications or their delivery bookkeeping. Native source
artifacts remain authoritative; successful copies are cached only in memory.
Failed copies preserve the summary and native references for a later-turn retry,
never rerun the child, and never publish an unwritten workspace reference.
After a runner restart, native metadata reconstructs the mapping for reports
already referenced by workspace path. Report writes may initialize the sandbox;
parent `read` and the public path policy are unchanged.

All plugin directories live under the conversation's `pi/` directory. No public
API, HTTP route or snapshot schema is added. Dependency direction remains:
server -> adapter -> runtime; only the adapter imports Pi and pi-subagents.
Old product snapshots need no migration. Existing Pi sessions are covered by
Pi's own session reader and the regression tests.

## AgentGenome compatibility artifact

The deployed `0.6.3` artifact is newer than this workspace's reference repository.
`vendor/agentgenome-pi-extension-0.6.3-pi087.tgz` changes only its peer declaration
to `0.84.2 || 0.87.0`; runtime code and embedded Python wheels are byte-for-byte
unchanged. Reproduce it with `python3 scripts/repack-agentgenome.py`, then refresh
the npm lockfile. Tests compare every archived runtime file and exercise the real
AgentGenome stage/judge workflow against Pi 0.87.0. No forced peer resolution is used.

References: [plugin](https://github.com/nicobailon/pi-subagents),
[roles](https://github.com/nicobailon/pi-subagents/blob/main/docs/agents.md),
[host APIs](https://github.com/nicobailon/pi-subagents/blob/main/docs/extension-api.md).

## Web display (optional operation details v1)

The Adapter queries native `status` RPC (`view: transcript`, 200 lines), at most
once per two seconds per active run with one request in flight. It publishes
changed windows using existing `operation.progress`, not model tool calls.
`details.subagent` carries `version: 1`, run/role/task, originating tool call,
optional previous run, timestamps, native status, and a `trace` object (text,
availability, truncation, update time, line/byte limits). Windows are capped at
32 KiB; native metadata headers are removed. Unknown formats are unavailable.
Completed foreground output never overwrites the last live trace. On restart,
terminal reconciliation omits absent display details so saved windows survive.

The frontend groups only proven tool-call/run relationships into one collapsed
card, with bounded live activity and the final result. It never interprets native
terminal text as structured tool events. The optional display convention uses
existing JSON details: no HTTP endpoints, commands, ports, capabilities or
snapshot migrations. Dependencies still point from Adapter to Product Runtime;
plugin RPC and native format handling stay below the Adapter boundary. Polling
continues while cards are collapsed; change detection bounds event volume.
