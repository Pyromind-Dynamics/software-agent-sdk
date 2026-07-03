#!/usr/bin/env bash
# ============================================================
# Pyromind Agent Server - Startup Script
# ============================================================
# Usage:
#   chmod +x start.sh
#   ./start.sh
# ============================================================

set -euo pipefail

# ----------------------------------------------------------
# LLM Configuration
# ----------------------------------------------------------
export LLM_BASE_URL="https://openrouter.ai/api/v1/"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is required. Export it before running start.sh." >&2
  exit 1
fi
export OPENAI_API_KEY
export LLM_MODEL="openai/gpt-5.5"

# ----------------------------------------------------------
# Agent Server Configuration
# ----------------------------------------------------------
# Session API key for authenticating requests (leave empty for unsecured dev mode)
export SESSION_API_KEY=""

# Secret key for encrypting LLM API keys in stored conversations
# Generate with: openssl rand -hex 32
export OH_SECRET_KEY=""

# Allow all CORS origins in development
export OH_ALLOW_CORS_ORIGIN_REGEX="https?://.+"

# ----------------------------------------------------------
# Pyromind Knowledge Base Path
# ----------------------------------------------------------
# Points to the knowledge/ folder in this repository by default
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYROMIND_KNOWLEDGE_BASE_PATH="${SCRIPT_DIR}/knowledge"

PYROMIND_SUBMODULES=(
  "knowledge/docs-mintlify/zh/docs"
  "knowledge/pyromind-sdk-example/docs"
)
missing=()
for path in "${PYROMIND_SUBMODULES[@]}"; do
  if [[ ! -d "${SCRIPT_DIR}/${path}" ]]; then
    missing+=("${path}")
  fi
done
if ((${#missing[@]} > 0)); then
  echo "WARNING: knowledge submodule content is missing:"
  for path in "${missing[@]}"; do
    echo "  - ${SCRIPT_DIR}/${path}"
  done
  echo "Initialize submodules before starting:"
  echo "  git submodule update --init --recursive knowledge/docs-mintlify knowledge/pyromind-sdk-example"
  echo ""
fi

# ----------------------------------------------------------
# Start Agent Server
# ----------------------------------------------------------
echo "============================================"
echo " Pyromind Agent Server"
echo "============================================"
echo " LLM Base URL:      ${LLM_BASE_URL}"
echo " Knowledge Base:    ${PYROMIND_KNOWLEDGE_BASE_PATH}"
echo " Mintlify docs:     ${PYROMIND_KNOWLEDGE_BASE_PATH}/docs-mintlify/zh/docs"
echo " Node docs:         ${PYROMIND_KNOWLEDGE_BASE_PATH}/pyromind-sdk-example/docs"
echo " Host:              127.0.0.1"
echo " Port:              8000"
echo " Auto-reload:       enabled"
echo "============================================"
echo ""

uv run python -m openhands.agent_server \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
