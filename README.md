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

### Run each developer's Agent on a dedicated Actor

Place `start_actor_dev.sh` beside your personal `start_inference.sh` and export
the three sandbox connection settings.

```bash
export KUBECONFIG="$(cat <<'YAML'
# Paste your complete kubeconfig YAML here.
YAML
)"
export PYROMIND_SANDBOX_ACTOR_URL=http://127.0.0.1:8081
export PYROMIND_SANDBOX_ACTOR_TARGET=your-atespace/your-actor
./start_actor_dev.sh
```

The wrapper connects to the Actor, then runs your `start_inference.sh` with
the sandbox terminal backend. Run `./start_actor_dev.sh` for the complete
startup. Your existing script manages the model, credentials, database, Kafka,
dependencies and other service settings.

For the localhost 8081 URL the wrapper manages a router forward using
`KUBECONFIG`. Multiline YAML is
written to a temporary file readable only by the current user, passed to
`kubectl`, and removed on exit. File paths are also accepted, with
`$HOME/.kube/config` as the default. An existing connection is reused.
A directly reachable router URL also works.
Use embedded certificate data or absolute certificate paths in pasted YAML.

Operations should provision one Actor and the corresponding JuiceFS user disk
per developer, with an egress policy for the required services. Pin the same
image digest in every template. The tested `amd64-v3` digest is
`sha256:ad1e8070460cec98cf0ca4eb8f7945a39e64ffff5bcd202a5246111c45ee025f`.

### Run checks

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
