# Pyromind build and release workflows

The workspace contains five distributions: pyromind-common, pyromind-embodied-runtime, pyromind-runtime, harness-adapter and pyromind-agent-server. The product service uses the Pi runner.

## Current checks and artifacts

- `tests.yml` installs the workspace, builds the Node runner, runs Python tests and runs Node tests.
- `server.yml` builds the Pyromind executable, validates the product OpenAPI, builds server images and publishes image manifests. Its event triggers and manual inputs are defined in that workflow.
- `release-binaries.yml` builds and smoke-tests executables on main pushes and published releases. Release/manual runs attach binaries and checksums; the workflow also checks the corresponding Docker images.
- `make build-server` builds `dist/pyromind-agent-server` (`.exe` on Windows) using the product PyInstaller spec.
- `uv build --all-packages` builds local wheel/sdist artifacts. There is currently no automatic PyPI package publication or release-preparation workflow in this repository.

## Configuration boundary

Image repository defaults, deployment target aliases, binary compatibility filenames and external configuration keys remain as currently configured in the workflow and Dockerfile. Renaming documentation does not change those settings or publish artifacts. Coordinate any later configuration migration with deployed consumers and the actual image repository.

## Validation before publishing

Run focused Python tests, the Runtime boundary suite, Node checks/tests and the intended artifact/image smoke tests. Record existing failures and unavailable environments accurately. Package builds alone do not prove that a Linux image starts or that CN platform business requests succeed.
