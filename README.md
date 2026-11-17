# Pyromind Agent Server

Pyromind and EchoMind use the Pi harness. The former OpenHands execution engine,
SDK conversation endpoints and WebSocket chat transport have been retired.

## Packages

- `pyromind-agent-server`: FastAPI application, authentication, projects, callbacks and service lifecycle.
- `pyromind-runtime`: product commands, snapshots, events, workflows and persistent state.
- `harness-adapter`: Pi sessions, tools, sandbox integration and the TypeScript runner under `pi-runtime`.
- `pyromind-common`: shared business tools, content models, permissions, secrets and workspace contracts. It does not contain an agent execution loop.
- `pyromind-embodied-runtime`: deterministic Python 3.10 processing used by frozen EDP sandbox jobs.

## Development

Requires Python 3.13, uv, Node.js 22 and npm.

```bash
uv sync --group dev
npm ci --prefix harness-adapter/pi-runtime
npm run build --prefix harness-adapter/pi-runtime
export OPENAI_API_KEY=your-key
export APP_ENV=dev
export PYROMIND_PI_TERMINAL_BACKEND=os-sandbox
./start.sh
```

`start.sh` defaults to portal authentication; local standalone use can set
`OH_ENABLE_PYROMIND_JWT_AUTH=false`. Development debug headers retain their
existing environment restriction. Linux `os-sandbox` execution requires the
platform sandbox dependencies checked by the Pi runner.

```bash
uv run pytest tests
npm test --prefix harness-adapter/pi-runtime
uv run pre-commit run --all-files
make build-server
```

Some tests require a separately prepared DataFlow environment or platform
credentials; their real results and environment limitations must be reported.

## Compatibility

The product APIs remain `/api/v2/pyromind/conversations` and
`/api/v2/echomind/conversations`; snapshot and SSE event contracts remain in use.
Project APIs, workflow callbacks, debug requests, progress polling, workspace
artifact URLs, authentication cookies and session headers retain their routes.
Warm pools still initialize through `/api/init`.

Existing `OH_*`, `OPENHANDS_CONFIG_DIR`, `OPENHANDS_AGENT_SERVER_CONFIG_PATH` and
build metadata environment keys are deployment compatibility names. The packaged
binary retains the `openhands-agent-server` filename and `agent-server` CLI alias;
both start `pyromind_agent_server` and use Pi.

Historical Product Store records marked `openhands`, including records missing
an engine ID, remain OpenHands records. They are read-only and never attach to
Pi. New records explicitly use `pi`. A legacy command returns HTTP 409 with
`legacy_harness_retired`; pending task callbacks can record terminal status
without restarting the retired engine. This does not convert an old SDK
checkpoint or automatically migrate a running remote task.
