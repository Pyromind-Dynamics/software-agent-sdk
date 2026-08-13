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

# Last-resort defaults; the conversation LLM config takes precedence
DEFAULT_DATAFLOW_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_DATAFLOW_API_URL = f"{DEFAULT_DATAFLOW_API_BASE_URL}/chat/completions"
DEFAULT_DATAFLOW_MODEL_NAME = "google/gemma-4-31b-it"
