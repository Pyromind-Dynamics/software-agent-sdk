# Pyromind Pi runner

This package is the TypeScript process behind `PiAdapter`. It owns Pi-specific
messages, model configuration, tools, and the JSONL subprocess protocol. The
Python product runtime only receives neutral `HarnessEvent` values.

Requirements:

- Node.js 22.19 or newer.
- `npm ci` in this directory.
- A writable local workspace root configured by `OH_WORKSPACE_PATH`.

Pi's native `read/write/edit/bash` tools use `NodeExecutionEnv` and therefore
run directly as the agent-server operating-system user. The session working
directory is created below `OH_WORKSPACE_PATH`, but this is not a security
sandbox: shell commands and absolute paths can still access anything permitted
to that user. Run the server as a dedicated low-privilege user or inside a
container when isolation is required.

Useful checks:

```bash
npm run check
npm test
```

Set `PYROMIND_DEFAULT_HARNESS=pi` to route new product conversations to Pi. For
backward-compatible startup scripts, `PYROMIND_ENABLE_PI=true` also selects Pi
when no explicit default harness is configured. An explicit
`PYROMIND_DEFAULT_HARNESS` always wins. The server then requires
`PYROMIND_PI_MODEL_API_KEY` (or `OPENAI_API_KEY`). No Pyromind sandbox API key,
image, cluster, or sandbox service is used.

Optional settings include `PYROMIND_PI_MODEL_PROVIDER`, `PYROMIND_PI_MODEL_ID`,
`PYROMIND_PI_MODEL_BASE_URL`, `PYROMIND_PI_THINKING_LEVEL`, and
`PYROMIND_PI_NODE_BINARY`. `PYROMIND_SKILLS_PATH` selects the trusted host skill
root. `PYROMIND_PI_SKILLS` is a comma-separated allow-list and defaults to
`generate-workflow-dsl`. Configured local skills are advertised to the model
with Pi's AgentSkills prompt format.

Pi sessions support cancellation, partial messages, custom Python business
tools, and native `read/write/edit/bash`. They do not claim cross-server resume,
steering, permission replies, or conversation forks.
