# syntax=docker/dockerfile:1.7

# NOTE: LC_ALL/LANG must be set to C.UTF-8 for libtmux to work correctly.
# Without proper locale, tmux converts UTF-8 separator characters to underscores,
# breaking libtmux's format parsing.
ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.11.6
ARG USERNAME=openhands
ARG UID=10001
ARG GID=10001
ARG PORT=8000

################################################################################
# Builder
# Copy source + build a self-contained venv.
#
# SELF-CONTAINED /agent-server CONTRACT:
# uv installs python-build-standalone into /agent-server/uv-managed-python and
# creates .venv against it. Both live under /agent-server, so downstream
# consumers can COPY /agent-server onto any base image and the venv works.
################################################################################
FROM python:${PYTHON_VERSION}-bookworm AS builder
ARG PYTHON_VERSION UV_VERSION USERNAME UID GID

ENV UV_PROJECT_ENVIRONMENT=/agent-server/.venv
ENV UV_PYTHON_INSTALL_DIR=/agent-server/uv-managed-python

COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /uvx /bin/

RUN groupadd -g ${GID} ${USERNAME} \
 && useradd -m -u ${UID} -g ${GID} -s /usr/sbin/nologin ${USERNAME} \
 && mkdir -p /agent-server/uv-managed-python \
 && chown -R ${USERNAME}:${USERNAME} /agent-server

USER ${USERNAME}
WORKDIR /agent-server

# Copy workspace source
# Before docker build, run:
#   git submodule update --init --recursive docs-mintlify
COPY --chown=${USERNAME}:${USERNAME} pyproject.toml uv.lock ./
COPY --chown=${USERNAME}:${USERNAME} openhands-sdk ./openhands-sdk
COPY --chown=${USERNAME}:${USERNAME} openhands-tools ./openhands-tools
COPY --chown=${USERNAME}:${USERNAME} openhands-workspace ./openhands-workspace
COPY --chown=${USERNAME}:${USERNAME} openhands-agent-server ./openhands-agent-server
COPY --chown=${USERNAME}:${USERNAME} knowledge ./knowledge
COPY --chown=${USERNAME}:${USERNAME} docs-mintlify ./docs-mintlify

# Install dependencies and workspace packages
RUN --mount=type=cache,target=/home/${USERNAME}/.cache,uid=${UID},gid=${GID} \
    uv python install ${PYTHON_VERSION} && \
    uv venv --python-preference only-managed --python ${PYTHON_VERSION} .venv && \
    uv sync --frozen --no-editable --managed-python && \
    readlink -f .venv/bin/python | grep -q '^/agent-server/uv-managed-python/'

################################################################################
# Runtime
# Lightweight image with only the self-contained /agent-server directory.
################################################################################
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
ARG USERNAME UID GID PORT

# Install base packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash ca-certificates curl git jq tmux tar tini \
        build-essential coreutils procps findutils grep sed && \
    rm -rf /var/lib/apt/lists/*; \
    grep -Eq "^[^:]*:[^:]*:${GID}:" /etc/group || groupadd -g "${GID}" "${USERNAME}"; \
    grep -Eq "^${USERNAME}:" /etc/passwd || \
        useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}"; \
    mkdir -p /workspace/project; \
    chown -R "${USERNAME}:${USERNAME}" /workspace

# Copy self-contained agent-server (venv + managed python + source)
COPY --chown=${USERNAME}:${USERNAME} --from=builder /agent-server /agent-server

USER ${USERNAME}
WORKDIR /agent-server

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

EXPOSE ${PORT}

ENTRYPOINT ["tini", "--", "/agent-server/.venv/bin/python", "-m", "openhands.agent_server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
