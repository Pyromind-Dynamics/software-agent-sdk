from openhands.agent_server.agent_profiles.spec import AgentProfileSpec
from openhands.agent_server.pyromind_constants import PYROMIND_RUNTIME_CONTRACT


PYROMIND_TOOL_ALLOWLIST = (
    "read",
    "write",
    "edit",
    "terminal",
    "subagent",
    "update_plan",
    "validate_workflow_dsl",
    "product_versions",
    "submit_workflow",
    "get_storage_url",
    "upload_file_to_pyromind",
    "run_dataset_cleaning",
    "df_submit_pipeline",
    "df_submit_evaluation",
    "df_check_progress",
    "df_stop_task",
    "edp_render",
    "edp_submit",
    "edp_aggregate",
    "sandbox_create",
    "sandbox_delete",
    "sandbox_read_file",
    "sandbox_write_file",
    "sandbox_delete_file",
    "sandbox_terminal",
    "sandbox_upload",
    "sandbox_download",
    "workflow_debug",
    "analyze_task_failure",
    "training_analysis",
    "label_studio_project",
)

# Knowledge-base retrieval guidance layered on top of the codex base prompt via
# get_codex_agent(custom_instructions=...).
PYROMIND_KB_INSTRUCTIONS = """\
The Pyromind platform knowledge base is available through the read-only logical
path `{knowledge_alias}/`. Do not use or request its host filesystem path.

Knowledge base layout:
- basic/: platform basics
- jupyterlab/: JupyterLab and script-based training
- sdk/: Python SDK and script-based workflow APIs
- studio/: Studio workflow documentation
- nodes/<NodeType>/<NodeType>.md: node parameters, I/O, and ports
- dataset_processing_workflow.py: workflow DSL example

The shared skill documents are available through the read-only logical path
`{skills_alias}/`. Do not use or request their host filesystem path.

User paths are platform storage paths (e.g. `datasets/...`,
`/.pyromind-agent/...`, or a `workspace/` prefix) — never this conversation's
local files. In a sandbox session they are mounted read-only at `storage/`;
read them there with the file tools. Only `public_data/...` paths are local.

Skill usage rules:
- Every user turn is preceded by a `<system_reminder>` stating whether
  `public_data/workflow_canvas/workflow.py` exists. It is authoritative; never
  run a command or read another file to confirm it. When the workflow exists,
  read it in full with `file_editor` before asking for information or answering
  any request that may inspect, modify, validate, test, or run the current
  workflow, including short contextual requests that presuppose it, and reuse
  the dataset/model identifiers and topology already in the file instead of
  asking the user to provide them again.
- Then invoke the matching listed skill, and read only the exact `references/`
  resource that the skill requires. For a local workflow edit, do not inspect
  general `knowledge/` before invoking the skill, and do not inspect it
  afterward unless the skill explicitly requires it.
- For requests that do not involve a current workflow, invoke a matching listed
  skill before searching the knowledge base.
- Data processing: invoke the `data-processing` skill for any dataset
  cleaning, preparation, or environment-dependent processing request; its
  SKILL.md routing table selects the processing paradigm (format-conversion
  for deterministic format/structure transformation, llm-pipeline for
  content-level processing with DataFlow operators or LLM). If the user
  explicitly names a mode ("DataFlow"/"脚本清洗"/"格式转换"), follow their
  choice. If the intent is genuinely ambiguous, ask the user before invoking.
- Treat any requested node, model, parameter, data, or topology change as a
  `generate-workflow-dsl` request, including phrases such as "换个模型跑一下"
  or "跑下 <model> 的效果". Modify and validate the DSL, then stop; do not
  invoke `debug-workflow` or `workflow_debug` in the same turn. Use
  `debug-workflow` only for an explicit test/debug request that contains no
  configuration change.
- For skill document lookup, use `invoke_skill` first, then `skills_read` only
  for `SKILL.md`, `references/**`, or assets explicitly referenced by the skill.
- For skill-linked resources, read the exact relative path from the skill root;
  do not manually search `references/`, `scripts/`, or `assets/` with grep.
- Treat `scripts/` as executable helpers. Do not read, grep, or summarize
  `scripts/` source during normal tasks; run the script or platform tool named
  by `SKILL.md` instead.

__PYROMIND_RUNTIME_CONTRACT__

Create and edit the workflow DSL at the relative path
`public_data/workflow_canvas/workflow.py` from the current working directory.
Prefer `apply_patch` for workflow changes. It accepts only the `patch` argument;
never pass a separate `path` argument. Put the full relative path in the patch
header, e.g.
`*** Add File: public_data/workflow_canvas/workflow.py` or
`*** Update File: public_data/workflow_canvas/workflow.py` — never use the
bare name `workflow.py`.

The separate `file_editor` tool does accept `path`. When using it for this file,
set `path` to `public_data/workflow_canvas/workflow.py`; the runtime resolves
conversation-relative paths to host-absolute paths. Do not hand-author long absolute
paths, and do not use `/workspace/...` or `workspace/conversations/...` as a
`file_editor.path` for the workflow file.
After creating or modifying the workflow file, stop normally; the server sends
the workflow to the frontend once the run finishes. Do not say the workflow has
been generated unless a tool call actually created or modified the workflow file.

- Invoke listed skills with `invoke_skill`. Do not invoke a workflow-generation
  skill for an article lookup alone.
- For skill documents, continue to use `skills_list` and `skills_read`; do not
  send skill-document lookup to the search subagent.
- Whenever a direct question or an intermediate step requires searching,
  grepping, or reading general `knowledge/` documentation, delegate the lookup
  to `subagent(type="search")`. The parent must not inspect those documents with
  grep, terminal, or file-reading tools.
- Make at most one `subagent(type="search")` call per parent-agent turn. Combine
  all related knowledge-base subquestions into one self-contained task. Do not
  split searches by topic, directory, expected source, or answer section, and
  never issue multiple or parallel search subagent calls in the same turn.
- Continue the parent task from the search subagent's sourced handoff. The
  subagent reads `knowledge/index.md` first and opens the selected source pages.
- Do not repeat the search subagent's work in the main conversation. Use its
  sourced result directly unless it reports that the knowledge base is missing
  the requested information.
- For a genuinely complex, multi-step workspace task whose intermediate reads,
  edits, and test output would clutter the main context, call `subagent` with
  `type="general_purpose"` and a self-contained task. The call is blocking and
  returns only the subagent's final handoff. Keep simple reads and edits in the
  main agent.
- For workflow generation, use the matching skill and its exact resources.
  Do not consult `knowledge/` before validation. Only when validation identifies
  a specific platform contract missing from those resources may you make one
  targeted `subagent(type="search")` call for that error.
""".replace("__PYROMIND_RUNTIME_CONTRACT__", PYROMIND_RUNTIME_CONTRACT.rstrip())


PYROMIND_PROFILE = AgentProfileSpec(
    name="pyromind",
    tag_value="pyromind",
    usage_id="pyromind-agent",
    system_prompt=None,
    skill_allowlist=(
        "cross-repo-testing",
        "data-processing",
        "debug-test-examples-workflow",
        "debug-workflow",
        "feature-release-rollout",
        "generate-workflow-dsl",
        "inference-evaluation",
        "label-studio",
        "manage-evals",
        "sandbox",
        "sdk-release",
        "training-analysis",
        "workflow-execution",
    ),
    tool_allowlist=PYROMIND_TOOL_ALLOWLIST,
)
