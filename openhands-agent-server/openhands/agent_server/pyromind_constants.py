"""Shared constants for Pyromind conversations."""

PYROMIND_APP_TAG_KEY = "app"
PYROMIND_APP_TAG_VALUE = "pyromind"
PYROMIND_WORKFLOW_EVENT_KEY = "pyromind_workflow"

PYROMIND_TERMINAL_PARAMS: dict[str, object] = {}

PYROMIND_LEGACY_TERMINAL_PARAM_KEYS = frozenset(
    {"command_working_subdir", "restrict_workspace_discovery"}
)

PYROMIND_LEGACY_RUNTIME_CONTRACT = """\
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

PYROMIND_RUNTIME_CONTRACT = """\
This agent authors and validates workflow DSL; Pyromind platform nodes perform actual
Storage data loading and processing, Benchmark, training, inference, and other workload
execution. Use the platform capabilities selected by the relevant skill, and never use
the local workspace runtime as a substitute for platform operations.

`public_data/` is the writable area for all agent-created conversation-local files.
All workspace paths are relative to the conversation root, and every created file must
use a `public_data/...` path. Local execution is limited to conversation-local auxiliary
files needed to author or validate the workflow.

In user messages and tool arguments, `workspace/` or `/workspace/` prefixes mean
platform user storage (a legacy alias), never the local working directory.
Local paths always start with `public_data/`.
"""
