ARG BASE_IMAGE=quay.io/jupyter/base-notebook:python-3.12
FROM ${BASE_IMAGE}

ARG RUNTIME_USER=root
ARG RESOURCE_REVISION=unknown
ARG RESOURCE_DIGEST=unknown
ARG RESOURCE_FILE_COUNT=0
ARG DATAFLOW_VERSION=1.0.13
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

USER root

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_CACHE_DIR=/root/.cache/pip \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=20 \
    PIP_PROGRESS_BAR=off

RUN apt-get -o Acquire::Retries=5 update \
 && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      jq \
      libglib2.0-0 \
      procps \
      python3-dev \
      python3-venv \
      ripgrep \
      tmux \
 && rm -rf /var/lib/apt/lists/*

ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG TORCHAUDIO_VERSION=2.6.0
ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
 && python3 -m venv /opt/pyromind/dataflow-venv \
 && /opt/pyromind/dataflow-venv/bin/python -m pip install \
      --index-url "${TORCH_CPU_INDEX_URL}" \
      --no-deps \
      --retries 20 \
      --timeout 120 \
      "torch==${TORCH_VERSION}+cpu" \
      "torchvision==${TORCHVISION_VERSION}+cpu" \
      "torchaudio==${TORCHAUDIO_VERSION}+cpu" \
 && /opt/pyromind/dataflow-venv/bin/python -m pip install \
      --index-url "${PIP_INDEX_URL}" \
      --retries 20 \
      --timeout 120 \
      "open-dataflow==${DATAFLOW_VERSION}" \
      "numpy==1.26.4" \
      "Pillow==12.1.1" \
      "opencv-python-headless==4.10.0.84" \
      "matplotlib==3.9.4" \
 && /opt/pyromind/dataflow-venv/bin/python -m pip check \
 && /opt/pyromind/dataflow-venv/bin/python -c \
      "import importlib.metadata as m; import cv2, matplotlib, numpy, PIL, torch, torchaudio, torchvision; assert m.version('open-dataflow') == '${DATAFLOW_VERSION}'; assert torch.version.cuda is None" \
 && /opt/pyromind/dataflow-venv/bin/python -m pip freeze \
      > /opt/pyromind/dataflow-venv/pip-freeze.txt \
 && printf '%s\n' \
      "torch==${TORCH_VERSION}+cpu" \
      "torchvision==${TORCHVISION_VERSION}+cpu" \
      "torchaudio==${TORCHAUDIO_VERSION}+cpu" \
      > /opt/pyromind/dataflow-venv/runtime-stack.txt \
 && printf '%s\n' "${DATAFLOW_VERSION}" \
      > /opt/pyromind/dataflow-venv/open-dataflow-version

COPY agent-resources /opt/pyromind/agent-resources

RUN test -f /opt/pyromind/agent-resources/manifest.json \
 && test -d /opt/pyromind/agent-resources/skills \
 && test -d /opt/pyromind/agent-resources/knowledge \
 && chmod -R a+rX /opt/pyromind \
 && find /opt/pyromind/agent-resources -type f \
      \( -name '*.py' -o -name '*.sh' \) \
      -exec chmod a+rx {} +

ENV PYROMIND_AGENT_RESOURCES=/opt/pyromind/agent-resources \
    PYROMIND_SKILLS_PATH=/opt/pyromind/agent-resources/skills \
    PYROMIND_KNOWLEDGE_BASE_PATH=/opt/pyromind/agent-resources/knowledge \
    PYROMIND_PUBLIC_READ_PATHS=/opt/pyromind/agent-resources/skills \
    PYROMIND_SANDBOX_DATAFLOW_PYTHON=/opt/pyromind/dataflow-venv/bin/python \
    DATAFLOW_PYTHON=/opt/pyromind/dataflow-venv/bin/python

LABEL org.opencontainers.image.title="Pyromind sandbox all-in-one" \
      org.opencontainers.image.description="Pyromind sandbox image with DataFlow and agent resources" \
      ai.pyromind.resources.revision="${RESOURCE_REVISION}" \
      ai.pyromind.resources.digest="${RESOURCE_DIGEST}" \
      ai.pyromind.resources.file-count="${RESOURCE_FILE_COUNT}" \
      ai.pyromind.dataflow.version="${DATAFLOW_VERSION}"

USER ${RUNTIME_USER}
