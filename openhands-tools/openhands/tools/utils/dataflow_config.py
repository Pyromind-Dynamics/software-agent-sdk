"""Central definitions for DataFlow LLM env var names and defaults.

Kept dependency-free so both ``data_preparation`` and ``pyromind_dataset``
(which are mutually importing packages) can consume these constants.
"""

# DataFlow LLM env var names
ENV_DF_API_BASE_URL = "DF_API_BASE_URL"
ENV_DF_API_URL = "DF_API_URL"
ENV_DF_MODEL_NAME = "DF_MODEL_NAME"
ENV_DF_API_KEY = "DF_API_KEY"

# Agent LLM env var names used as fallbacks for the DataFlow vision profile
ENV_LLM_BASE_URL = "LLM_BASE_URL"
ENV_LLM_MODEL = "LLM_MODEL"

# Fallback order: DF_* env vars, then these defaults, then the conversation
# LLM config
DEFAULT_DATAFLOW_API_BASE_URL = "http://208.64.254.187:8000/v1"
DEFAULT_DATAFLOW_API_URL = f"{DEFAULT_DATAFLOW_API_BASE_URL}/chat/completions"
DEFAULT_DATAFLOW_MODEL_NAME = "openai/deepseek-v4-flash-0731"