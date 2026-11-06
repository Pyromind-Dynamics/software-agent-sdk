# Pyromind sandbox all-in-one image

This image contains the Pyromind sandbox runtime assets needed after startup:

```text
/opt/pyromind/
  dataflow-venv/
    bin/python
    open-dataflow-version
  agent-resources/
    skills/
    knowledge/
    manifest.json
```

## Build

The default base is a public Jupyter image so the build can be tested without
private registry access:

```bash
deploy/sandbox-image/build.sh
```

For the production sandbox base, provide the platform image explicitly:

```bash
BASE_IMAGE=pyrominddynamics/jupyter-lab-with-ssh:v0.9 \
  deploy/sandbox-image/build.sh
```

The script detects the base image user, builds `linux/amd64` by default,
validates the result, and writes:

```text
dist/pyromind-sandbox-all-in-one-<revision>-linux-amd64.tar.gz
```

Useful overrides:

```text
BASE_IMAGE
IMAGE_REPOSITORY
PLATFORM
RUNTIME_USER
DATAFLOW_VERSION
PIP_INDEX_URL
TORCH_VERSION
TORCHVISION_VERSION
TORCHAUDIO_VERSION
TORCH_CPU_INDEX_URL
OUTPUT_DIR
SKIP_VERIFY
KEEP_BUILD_CONTEXT
```

## Verify

```bash
deploy/sandbox-image/verify.sh pyromind-sandbox-all-in-one:local
```

The check confirms the DataFlow version, CPU-only PyTorch stack, Python
dependencies, resource manifest, skill and knowledge paths, and that
`sandbox_sample.py --check` resolves the image-baked interpreter.
