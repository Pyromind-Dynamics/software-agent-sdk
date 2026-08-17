#!/usr/bin/env bash
# ============================================================
# Pyromind Agent Server - Startup Script
# ============================================================
# Usage:
#   chmod +x start.sh
#   ./start.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SOFTWARE_AGENT_SDK_DIR="${SOFTWARE_AGENT_SDK_DIR:-${SCRIPT_DIR}}"

# ----------------------------------------------------------
# LLM Configuration
# ----------------------------------------------------------
# Single-provider defaults (used when llm_config.json is absent).
# For automatic failover, write ${OPENHANDS_CONFIG_DIR}/llm_config.json:
#   {"llms": [
#     {"name": "openrouter", "model": "openai/deepseek-v4-flash-0731",
#      "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
#     {"name": "deepseek", "model": "openai/deepseek-chat",
#      "base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY"}
#   ]}
# Keys are never stored in the config file: staging/production deployment
# manifests (e.g. Kubernetes Secret + secretKeyRef) inject the env vars above.
export LLM_BASE_URL="${LLM_BASE_URL:-http://208.64.254.187:8000/v1}"
if [[ ! -f "${LLM_CONFIG_PATH:-${OPENHANDS_CONFIG_DIR:-${WORKSPACE_DIR:-${SOFTWARE_AGENT_SDK_DIR}/workspace}}/llm_config.json}" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" && -n "${LLM_API_KEY:-}" ]]; then
    export OPENAI_API_KEY="${LLM_API_KEY}"
  fi
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is required (no llm_config.json found)." >&2
    exit 1
  fi
  export OPENAI_API_KEY
fi
# LiteLLM requires a provider prefix (e.g. openai/) for custom OpenAI-compatible endpoints.
export LLM_MODEL="${LLM_MODEL:-openai/deepseek-v4-flash-0731}"

export DF_API_BASE_URL=https://openrouter.ai/api/v1
export DF_API_URL=https://openrouter.ai/api/v1/chat/completions
export DF_MODEL_NAME=google/gemma-4-31b-it
export DF_API_KEY="${OPENROUTER_API_KEY}"

# ----------------------------------------------------------
# Agent Server Configuration
# ----------------------------------------------------------
# Session API key for authenticating requests (leave empty for unsecured dev mode)
export SESSION_API_KEY="${SESSION_API_KEY:-}"

# Secret key for encrypting LLM API keys in stored conversations
# Generate with: openssl rand -hex 32
if [[ -n "${OH_SECRET_KEY:-}" ]]; then
  export OH_SECRET_KEY
elif [[ -n "${SESSION_API_KEY}" ]]; then
  export OH_SECRET_KEY="${SESSION_API_KEY}"
fi

# Local start.sh runs without Pyromind portal cookies by default. Deployments
# that require portal login should set OH_ENABLE_PYROMIND_JWT_AUTH=true.
export OH_ENABLE_PYROMIND_JWT_AUTH="${OH_ENABLE_PYROMIND_JWT_AUTH:-true}"

# Allow all CORS origins in development
export OH_ALLOW_CORS_ORIGIN_REGEX="${OH_ALLOW_CORS_ORIGIN_REGEX:-https?://.+}"

# ----------------------------------------------------------
# Workspace
# ----------------------------------------------------------
# Deployment may set lowercase workspace_dir. Keep uppercase WORKSPACE_DIR as a
# convenience alias, then derive the Config fields the server actually consumes.
export workspace_dir="${workspace_dir:-${WORKSPACE_DIR:-${SOFTWARE_AGENT_SDK_DIR}/workspace}}"
export WORKSPACE_DIR="${workspace_dir}"
export OPENHANDS_CONFIG_DIR="${OPENHANDS_CONFIG_DIR:-${WORKSPACE_DIR}}"
export OPENHANDS_AGENT_SERVER_CONFIG_PATH="${OPENHANDS_AGENT_SERVER_CONFIG_PATH:-${OPENHANDS_CONFIG_DIR}/openhands_agent_server_config.json}"
export LLM_CONFIG_PATH="${LLM_CONFIG_PATH:-${OPENHANDS_CONFIG_DIR}/llm_config.json}"
export LLM_FAILOVER_COOLDOWN_SECONDS="${LLM_FAILOVER_COOLDOWN_SECONDS:-300}"
export OH_CONVERSATIONS_PATH="${OH_CONVERSATIONS_PATH:-${WORKSPACE_DIR}/conversations}"
export OH_CONVERSATION_STORAGE_QUOTA="${OH_CONVERSATION_STORAGE_QUOTA:-500M}"
export OH_SANDBOX_MEMORY_LIMIT="${OH_SANDBOX_MEMORY_LIMIT:-500M}"
export OH_WORKSPACE_PATH="${OH_WORKSPACE_PATH:-${WORKSPACE_DIR}/project}"
export OH_BASH_EVENTS_DIR="${OH_BASH_EVENTS_DIR:-${WORKSPACE_DIR}/bash_events}"
mkdir -p \
  "${OPENHANDS_CONFIG_DIR}" \
  "${OH_CONVERSATIONS_PATH}" \
  "${OH_WORKSPACE_PATH}" \
  "${OH_BASH_EVENTS_DIR}"

# ----------------------------------------------------------
# Pyromind Knowledge Base
# ----------------------------------------------------------
# Points to the knowledge/ folder in this repository by default.
export PYROMIND_KNOWLEDGE_BASE_PATH="${PYROMIND_KNOWLEDGE_BASE_PATH:-${SOFTWARE_AGENT_SDK_DIR}/knowledge}"
export PYROMIND_SKILLS_PATH="${PYROMIND_SKILLS_PATH:-${SOFTWARE_AGENT_SDK_DIR}/.agents/skills}"
export PYROMIND_PUBLIC_READ_PATHS="${PYROMIND_PUBLIC_READ_PATHS:-${PYROMIND_SKILLS_PATH}}"

for required_dir in basic jupyterlab nodes sdk studio; do
  if [[ ! -d "${PYROMIND_KNOWLEDGE_BASE_PATH}/${required_dir}" ]]; then
    echo "ERROR: knowledge directory missing: ${PYROMIND_KNOWLEDGE_BASE_PATH}/${required_dir}" >&2
    exit 1
  fi
done

if [[ ! -f "${PYROMIND_KNOWLEDGE_BASE_PATH}/dataset_processing_workflow.py" ]]; then
  echo "ERROR: knowledge workflow example missing: ${PYROMIND_KNOWLEDGE_BASE_PATH}/dataset_processing_workflow.py" >&2
  exit 1
fi

if [[ ! -d "${PYROMIND_SKILLS_PATH}" ]]; then
  echo "ERROR: skills directory missing: ${PYROMIND_SKILLS_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PYROMIND_SKILLS_PATH}/generate-workflow-dsl/SKILL.md" ]]; then
  echo "ERROR: workflow DSL skill missing: ${PYROMIND_SKILLS_PATH}/generate-workflow-dsl/SKILL.md" >&2
  exit 1
fi

# Validate the multi-provider LLM config when present (single env-var mode
# otherwise). Failing validation aborts startup with a clear message.
if [[ -f "${LLM_CONFIG_PATH}" ]]; then
  LLM_CONFIG_SUMMARY="$(python3 - "${LLM_CONFIG_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
if isinstance(data, dict):
    sections = [("llms", data.get("llms"))]
    if "multimodal_llms" in data:
        sections.append(("multimodal_llms", data["multimodal_llms"]))
else:
    sections = [("llms", data)]
summary = []
for name, entries in sections:
    if entries is None:
        continue
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"'{name}' must be a non-empty list")
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("model"):
            raise SystemExit(f"{name} entry {i} must declare a 'model'")
    summary.append(f"{len(entries)} {name}")
print(", ".join(summary))
PY
)"
fi

# ----------------------------------------------------------
# Start Agent Server
# ----------------------------------------------------------
cd "${SOFTWARE_AGENT_SDK_DIR}"

echo "============================================"
echo " Pyromind Agent Server"
echo "============================================"
if [[ -n "${LLM_CONFIG_SUMMARY:-}" ]]; then
  echo " LLM providers:     ${LLM_CONFIG_SUMMARY} from ${LLM_CONFIG_PATH} (auto-failover on)"
else
  echo " LLM base URL:      ${LLM_BASE_URL} (single provider)"
  echo " Failover tip:      create ${LLM_CONFIG_PATH} with ordered 'llms'/'multimodal_llms' lists"
fi
echo " Server root:       ${SOFTWARE_AGENT_SDK_DIR}"
echo " Knowledge Base:    ${PYROMIND_KNOWLEDGE_BASE_PATH}"
echo " Public read paths: ${PYROMIND_PUBLIC_READ_PATHS}"
echo " Skills:            ${PYROMIND_SKILLS_PATH}"
echo " Workspace root:    ${WORKSPACE_DIR}"
echo " Conversations:     ${OH_CONVERSATIONS_PATH}"
echo " Project workspace: ${OH_WORKSPACE_PATH}"
echo " Bash events:       ${OH_BASH_EVENTS_DIR}"
echo " Config path:       ${OPENHANDS_AGENT_SERVER_CONFIG_PATH}"
echo " Pyromind JWT auth: ${OH_ENABLE_PYROMIND_JWT_AUTH}"
echo " Session API key:   $([[ -n "${SESSION_API_KEY}" ]] && echo configured || echo disabled)"
echo " Host:              127.0.0.1"
echo " Port:              8000"
echo " Auto-reload:       enabled"
echo "============================================"
echo ""

uv run python -m openhands.agent_server \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
