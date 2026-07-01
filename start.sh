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
export OPENAI_API_KEY="your-secret-key-here"

# ----------------------------------------------------------
# Agent Server Configuration
# ----------------------------------------------------------
# Session API key for authenticating requests (leave empty for unsecured dev mode)
export SESSION_API_KEY="pyromind_agent"

# Secret key for encrypting LLM API keys in stored conversations
# Generate with: openssl rand -hex 32
export OH_SECRET_KEY="pyromind_agent"

# Allow all CORS origins in development
export OH_ALLOW_CORS_ORIGIN_REGEX="https?://.+"

# ----------------------------------------------------------
# Pyromind Knowledge Base Path
# ----------------------------------------------------------
# Points to the knowledge/ folder in this repository by default
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYROMIND_KNOWLEDGE_BASE_PATH="${SCRIPT_DIR}/knowledge"

# ----------------------------------------------------------
# Start Agent Server
# ----------------------------------------------------------
echo "============================================"
echo " Pyromind Agent Server"
echo "============================================"
echo " LLM Base URL:      ${LLM_BASE_URL}"
echo " Knowledge Base:    ${PYROMIND_KNOWLEDGE_BASE_PATH}"
echo " Host:              127.0.0.1"
echo " Port:              8000"
echo " Auto-reload:       enabled"
echo "============================================"
echo ""

uv run python -m openhands.agent_server \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
