# OpenHands Embodied Runtime

This package contains the deterministic embodied-data cleaning code executed by
Pyromind sandbox jobs. It supports Python 3.10 independently of the main
OpenHands packages, which continue to require Python 3.12 or newer.

Supported mounted sources are self-collected S2 episode directories and complete
Hugging Face LeRobot v2.1 datasets. A LeRobot source may include `labels.json`,
`annotations.json`, or `labels.jsonl` beside `meta/`; labels are joined by source
episode ID. Label documents without Parquet state/action data and MP4 media can be
inspected but cannot be materialized.

Build the sandbox wheel from the repository root:

```bash
uv build --package openhands-embodied-runtime --wheel --out-dir dist
```

Embodied jobs use the standard Pyromind Python 3.10 Sandbox image:

```bash
pyrominddynamics/jupyter-lab-with-ssh:v0.9
```

Do not replace this image with an example or local registry name. Provision the
runtime after the Sandbox starts, either from the deployment package index or a
wheel path mounted by the deployment:

```bash
python3.10 -m pip install 'openhands-embodied-runtime==1.29.5'
# Or, for a deployment-mounted developer wheel:
python3.10 -m pip install /mounted/runtime/openhands_embodied_runtime-1.29.5-py3-none-any.whl
```

The sandbox entrypoint is:

```bash
python -m openhands_embodied_runtime.sandbox_runner
```

The platform runs `plan` and `full` through a generic Python 3.10 sandbox.
Quality-rejected episodes remain isolated and are described in the structured
report; accepted episodes are still merged and published. `resume` retries only
unexpected runtime failures and never reprocesses accepted or rejected episodes.

State/camera alignment gaps above 100 ms are reported as warnings. An episode is
rejected only when a camera lead, camera lag, or internal state gap exceeds
500 ms. A gap exactly equal to 500 ms is accepted.
