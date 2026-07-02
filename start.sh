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
export OPENAI_API_KEY=""
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
# Pyromind Knowledge Base (sync docs-mintlify → knowledge/)
# ----------------------------------------------------------
# Points to the knowledge/ folder in this repository by default
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYROMIND_KNOWLEDGE_BASE_PATH="${SCRIPT_DIR}/knowledge"

echo "Initializing docs-mintlify submodule..."
git -C "${SCRIPT_DIR}" submodule update --init --recursive docs-mintlify

DOCS_SRC="${SCRIPT_DIR}/docs-mintlify/zh/docs"
KNOWLEDGE_DIR="${SCRIPT_DIR}/knowledge"
if [[ ! -d "${DOCS_SRC}" ]]; then
  echo "ERROR: docs source directory not found: ${DOCS_SRC}" >&2
  exit 1
fi

echo "Syncing ${DOCS_SRC} → ${KNOWLEDGE_DIR}/"
cp -a "${DOCS_SRC}/." "${KNOWLEDGE_DIR}/"

# ----------------------------------------------------------
# Start Agent Server
# ----------------------------------------------------------
echo "============================================"
echo " Pyromind Agent Server"
echo "============================================"
echo " LLM Base URL:      ${LLM_BASE_URL}"
echo " Knowledge Base:    ${PYROMIND_KNOWLEDGE_BASE_PATH}"
echo " Docs source:       ${DOCS_SRC}"
echo " Host:              127.0.0.1"
echo " Port:              8000"
echo " Auto-reload:       enabled"
echo "============================================"
echo ""

uv run python -m openhands.agent_server \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
