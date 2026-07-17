"""Shared constants for Pyromind conversations."""

PYROMIND_APP_TAG_KEY = "app"
PYROMIND_APP_TAG_VALUE = "pyromind"
PYROMIND_WORKFLOW_EVENT_KEY = "pyromind_workflow"

PYROMIND_TERMINAL_PARAMS: dict[str, object] = {
    "command_working_subdir": "public_data",
    "restrict_workspace_discovery": True,
}

PYROMIND_RUNTIME_CONTRACT = """\
This agent authors and validates workflow DSL; Pyromind platform nodes perform actual
Storage data loading and processing, Benchmark, training, inference, and other workload
execution. Use dedicated platform tools for preview/upload when a skill requires them,
and never use the local terminal as a substitute for platform operations.

Shell and file tools are confined to this conversation's private workspace. Terminal
commands run from `public_data/`, and workspace-discovery commands are rejected. Use
`terminal` only to execute an exact, already-known conversation-local script; do not
use it to inspect files or data, discover paths, or probe the environment. Use canvas
context, tool results, and `file_editor` with exact relative paths instead. Do not
access host-absolute paths, `/workspace`, or paths outside the conversation workspace;
`/workspace` remains valid inside workflow DSL node parameters.
"""
