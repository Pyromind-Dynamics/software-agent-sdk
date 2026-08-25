# OpenHands Embodied Runtime

This package contains the deterministic embodied-data cleaning code executed by
Pyromind sandbox jobs. It supports Python 3.10 independently of the main
OpenHands packages, which continue to require Python 3.12 or newer.

Build the sandbox wheel from the repository root:

```bash
uv build --package openhands-embodied-runtime --wheel
```

The sandbox entrypoint is:

```bash
python -m openhands_embodied_runtime.sandbox_runner
```
