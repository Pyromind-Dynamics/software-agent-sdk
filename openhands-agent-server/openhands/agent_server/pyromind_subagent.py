"""Blocking, conversation-scoped Pyromind subagents."""

import fnmatch
import re
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from openhands.sdk import LLM, Agent, AgentContext, Tool, register_tool
from openhands.sdk.context.condenser import default_condenser
from openhands.sdk.logger import get_logger
from openhands.sdk.subagent.registry import AgentFactory
from openhands.sdk.subagent.schema import AgentDefinition
from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.grep import GrepTool
from openhands.tools.task.manager import TaskManager, TaskStatus
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState


_KNOWLEDGE_ROOT_CONFIG = ".pyromind_knowledge_root"
_MAX_SEARCH_MATCHES = 100
_MAX_READ_LINES = 400
_TEXT_SUFFIXES = frozenset({".json", ".md", ".mdx", ".py", ".txt", ".yaml", ".yml"})

SEARCH_AGENT_NAME = "search"
GENERAL_PURPOSE_AGENT_NAME = "general-purpose"

logger = get_logger(__name__)

SEARCH_AGENT_PROMPT = """\
You are the Pyromind knowledge-base retrieval specialist. Your only source of
platform facts is the read-only logical `knowledge/` tree exposed by your tools.

For every request:
1. Split the request into explicit subquestions.
2. Open `knowledge/index.md` first and select the smallest set of relevant pages
   from its titles, summaries, and tags. If the index is missing or has no useful
   entry, fall back to `knowledge_base_grep` across likely top-level sections.
3. Search only the candidate pages or sections using the user's terms and useful
   synonyms.
4. Open the relevant original pages with `knowledge_base_read`; never answer
   from filenames or grep snippets alone.
5. Cover relevant headings, tables, warnings, alternatives, and ordered steps.
6. Return a concise answer followed by sources as logical paths and headings.

Do not inspect the conversation workspace, modify files, use the web, or invent
missing platform behavior. If the knowledge base does not answer a subquestion,
state that explicitly.
"""

GENERAL_PURPOSE_AGENT_PROMPT = """\
You are a general-purpose Pyromind subagent for complex, multi-step work in the
conversation workspace. Complete the delegated task end-to-end using the smallest
useful set of reads, edits, commands, and tests.

Keep your work isolated from the parent conversation. Do not ask the parent to
repeat work you can complete yourself. When finished, return only a concise handoff:
- outcome and key decisions;
- files created or changed;
- tests or commands run and their final status;
- blockers or remaining risks.

Do not include a play-by-play, full command output, or large file contents unless
the parent explicitly requests them.
"""

SUBAGENT_TOOL_DESCRIPTION = """\
Run a complex task in an isolated, blocking subagent conversation and return only
its final handoff. Supported types:

- `search`: read-only, index-first Pyromind knowledge-base research. Use for
  platform documentation, SDK APIs, Studio, nodes, training, inference, or
  evaluation. Use it whenever the parent task needs to search, grep, or read
  general `knowledge/` documentation, including an intermediate lookup inside
  another task. Invoke it at most once per parent-agent turn: combine all related
  knowledge-base subquestions into one task instead of issuing multiple or
  parallel searches. Do not use it for skill instructions.
- `general_purpose`: multi-step workspace work with read/write tools, shell
  commands, and tests. Use it when delegating the work keeps the main context
  smaller; do not use it for a trivial single read or edit.

Pass a self-contained task. The call blocks until the subagent finishes, fails,
or reaches its run limit. Intermediate subagent events are logged separately and
are not returned to the main agent.
"""


class SubAgentType(StrEnum):
    """Pyromind subagent profiles exposed through the unified tool."""

    SEARCH = "search"
    GENERAL_PURPOSE = "general_purpose"


def _log_excerpt(value: str, limit: int = 200) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _log_knowledge_path(path: str) -> str:
    if path == "knowledge" or path.startswith("knowledge/"):
        return _log_excerpt(path)
    return "<invalid-logical-path>"


def configure_subagents(workspace_dir: Path, knowledge_root: Path) -> None:
    """Persist conversation-scoped resources used by Pyromind subagents."""
    resolved_root = knowledge_root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"Knowledge base path does not exist: {resolved_root}")
    config_path = workspace_dir.resolve() / _KNOWLEDGE_ROOT_CONFIG
    config_path.write_text(str(resolved_root), encoding="utf-8")
    config_path.chmod(0o600)


def _knowledge_root(conv_state: "ConversationState") -> Path:
    config_path = (
        Path(conv_state.workspace.working_dir).resolve() / _KNOWLEDGE_ROOT_CONFIG
    )
    try:
        root = Path(config_path.read_text(encoding="utf-8").strip()).resolve()
    except OSError as error:
        raise RuntimeError(
            "Pyromind knowledge-base search is not configured"
        ) from error
    if not root.is_dir():
        raise RuntimeError(f"Configured knowledge base no longer exists: {root}")
    return root


def _resolve_knowledge_path(root: Path, logical_path: str) -> Path:
    candidate = PurePosixPath(logical_path)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or candidate.parts[0] != "knowledge"
    ):
        raise ValueError("Knowledge paths must use the logical `knowledge/` root")
    relative_parts = candidate.parts[1:]
    resolved = root.joinpath(*relative_parts).resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes the knowledge base: {logical_path}")
    return resolved


def _logical_path(root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(root)
    return PurePosixPath("knowledge", *relative.parts).as_posix()


class KnowledgeBaseMatch(BaseModel):
    """One matching source line in the knowledge base."""

    model_config = ConfigDict(frozen=True)

    path: str
    line_number: int
    line: str


class KnowledgeBaseGrepAction(Action):
    """Search knowledge-base file contents with a regular expression."""

    pattern: str = Field(description="Case-insensitive regular expression to search")
    path: str = Field(
        default="knowledge",
        description="Logical knowledge path to search, rooted at `knowledge/`",
    )
    include: str | None = Field(
        default=None,
        description='Optional filename glob such as "*.mdx" or "*.md"',
    )


class KnowledgeBaseGrepObservation(Observation):
    """Structured knowledge-base search results."""

    matches: list[KnowledgeBaseMatch] = Field(default_factory=list)
    pattern: str
    search_path: str
    truncated: bool = False


class KnowledgeBaseGrepExecutor(
    ToolExecutor[KnowledgeBaseGrepAction, KnowledgeBaseGrepObservation]
):
    """Search only inside one configured knowledge root."""

    def __init__(self, root: Path):
        self.root = root

    def __call__(
        self,
        action: KnowledgeBaseGrepAction,
        conversation: "LocalConversation | None" = None,
    ) -> KnowledgeBaseGrepObservation:
        conversation_id = conversation.state.id if conversation else "unknown"
        logger.info(
            "[subagent-search] event=query_started conversation_id=%s "
            "path=%r include=%r pattern=%r",
            conversation_id,
            _log_knowledge_path(action.path),
            action.include,
            _log_excerpt(action.pattern),
        )
        try:
            regex = re.compile(action.pattern, re.IGNORECASE)
            search_path = _resolve_knowledge_path(self.root, action.path)
            if not search_path.exists():
                raise ValueError(f"Knowledge path does not exist: {action.path}")
            files = self._candidate_files(search_path, action.include)
            matches: list[KnowledgeBaseMatch] = []
            truncated = False
            for path in files:
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeDecodeError):
                    continue
                for line_number, line in enumerate(lines, start=1):
                    if not regex.search(line):
                        continue
                    if len(matches) == _MAX_SEARCH_MATCHES:
                        truncated = True
                        break
                    matches.append(
                        KnowledgeBaseMatch(
                            path=_logical_path(self.root, path),
                            line_number=line_number,
                            line=line,
                        )
                    )
                if truncated:
                    break
        except (ValueError, re.error) as error:
            logger.warning(
                "[subagent-search] event=query_failed conversation_id=%s "
                "path=%r error=%r",
                conversation_id,
                _log_knowledge_path(action.path),
                _log_excerpt(str(error)),
            )
            return KnowledgeBaseGrepObservation.from_text(
                text=str(error),
                matches=[],
                pattern=action.pattern,
                search_path=action.path,
                is_error=True,
            )

        if matches:
            rendered = "\n".join(
                f"{match.path}:{match.line_number}: {match.line}" for match in matches
            )
            text = f"Found {len(matches)} match(es):\n{rendered}"
            if truncated:
                text += "\n[Results truncated; narrow the path or pattern.]"
        else:
            text = (
                f"No matches found for {action.pattern!r} in {action.path!r}. "
                "Broaden the terms or search a neighboring category."
            )
        logger.info(
            "[subagent-search] event=query_completed conversation_id=%s "
            "path=%r matches=%d truncated=%s",
            conversation_id,
            _log_knowledge_path(action.path),
            len(matches),
            truncated,
        )
        return KnowledgeBaseGrepObservation.from_text(
            text=text,
            matches=matches,
            pattern=action.pattern,
            search_path=action.path,
            truncated=truncated,
        )

    def _candidate_files(self, path: Path, include: str | None) -> list[Path]:
        candidates = [path] if path.is_file() else list(path.rglob("*"))
        return sorted(
            candidate
            for candidate in candidates
            if candidate.is_file()
            and candidate.resolve().is_relative_to(self.root)
            and not any(
                part.startswith(".") for part in candidate.relative_to(self.root).parts
            )
            and candidate.suffix.lower() in _TEXT_SUFFIXES
            and (include is None or fnmatch.fnmatch(candidate.name, include))
        )


class KnowledgeBaseGrepTool(
    ToolDefinition[KnowledgeBaseGrepAction, KnowledgeBaseGrepObservation]
):
    """Read-only search tool for the configured Pyromind knowledge base."""

    @classmethod
    def create(
        cls, conv_state: "ConversationState"
    ) -> Sequence["KnowledgeBaseGrepTool"]:
        return [
            cls(
                action_type=KnowledgeBaseGrepAction,
                observation_type=KnowledgeBaseGrepObservation,
                description=(
                    "Search file contents under the read-only logical `knowledge/` "
                    "root. Results include logical paths and line numbers."
                ),
                annotations=ToolAnnotations(
                    title="knowledge_base_grep",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=KnowledgeBaseGrepExecutor(_knowledge_root(conv_state)),
            )
        ]

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=(), declared=True)


class KnowledgeBaseReadAction(Action):
    """Read a text file from the logical knowledge tree."""

    path: str = Field(description="Logical source path beginning with `knowledge/`")
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class KnowledgeBaseReadObservation(Observation):
    """A bounded section of one knowledge-base document."""

    path: str
    start_line: int
    end_line: int
    total_lines: int
    truncated: bool = False


class KnowledgeBaseReadExecutor(
    ToolExecutor[KnowledgeBaseReadAction, KnowledgeBaseReadObservation]
):
    """Read bounded text ranges without exposing host filesystem paths."""

    def __init__(self, root: Path):
        self.root = root

    def __call__(
        self,
        action: KnowledgeBaseReadAction,
        conversation: "LocalConversation | None" = None,
    ) -> KnowledgeBaseReadObservation:
        conversation_id = conversation.state.id if conversation else "unknown"
        logger.info(
            "[subagent-search] event=read_started conversation_id=%s "
            "path=%r start_line=%d end_line=%s",
            conversation_id,
            _log_knowledge_path(action.path),
            action.start_line,
            action.end_line or "EOF",
        )
        try:
            path = _resolve_knowledge_path(self.root, action.path)
            if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
                raise ValueError(f"Knowledge document does not exist: {action.path}")
            if action.end_line is not None and action.end_line < action.start_line:
                raise ValueError("end_line must be greater than or equal to start_line")
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError, ValueError) as error:
            logger.warning(
                "[subagent-search] event=read_failed conversation_id=%s "
                "path=%r error=%r",
                conversation_id,
                _log_knowledge_path(action.path),
                _log_excerpt(str(error)),
            )
            return KnowledgeBaseReadObservation.from_text(
                text=str(error),
                path=action.path,
                start_line=action.start_line,
                end_line=action.start_line,
                total_lines=0,
                is_error=True,
            )

        requested_end = action.end_line or len(lines)
        actual_end = min(requested_end, action.start_line + _MAX_READ_LINES - 1)
        actual_end = min(actual_end, len(lines))
        selected = lines[action.start_line - 1 : actual_end]
        rendered = "\n".join(
            f"{line_number:>6}\t{line}"
            for line_number, line in enumerate(selected, start=action.start_line)
        )
        truncated = actual_end < requested_end
        if truncated:
            rendered += f"\n[Output truncated; continue from line {actual_end + 1}.]"
        logger.info(
            "[subagent-search] event=read_completed conversation_id=%s "
            "path=%r lines=%d-%d total_lines=%d truncated=%s",
            conversation_id,
            _log_knowledge_path(action.path),
            action.start_line,
            actual_end,
            len(lines),
            truncated,
        )
        return KnowledgeBaseReadObservation.from_text(
            text=rendered or "The requested range is empty.",
            path=_logical_path(self.root, path),
            start_line=action.start_line,
            end_line=actual_end,
            total_lines=len(lines),
            truncated=truncated,
        )


class KnowledgeBaseReadTool(
    ToolDefinition[KnowledgeBaseReadAction, KnowledgeBaseReadObservation]
):
    """Read-only document viewer for the configured knowledge base."""

    @classmethod
    def create(
        cls, conv_state: "ConversationState"
    ) -> Sequence["KnowledgeBaseReadTool"]:
        return [
            cls(
                action_type=KnowledgeBaseReadAction,
                observation_type=KnowledgeBaseReadObservation,
                description=(
                    "Open a text document under the logical `knowledge/` root. "
                    "Use line ranges to continue through long documents."
                ),
                annotations=ToolAnnotations(
                    title="knowledge_base_read",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=KnowledgeBaseReadExecutor(_knowledge_root(conv_state)),
            )
        ]

    def declared_resources(self, action: Action) -> DeclaredResources:
        if not isinstance(action, KnowledgeBaseReadAction):
            raise TypeError(
                f"Expected KnowledgeBaseReadAction, got {type(action).__name__}"
            )
        return DeclaredResources(keys=(f"knowledge:{action.path}",), declared=True)


def _search_agent_factory() -> AgentFactory:
    definition = AgentDefinition(
        name=SEARCH_AGENT_NAME,
        description=(
            "Read-only specialist for thorough Pyromind knowledge-base retrieval "
            "with source paths and headings."
        ),
        model="inherit",
        tools=[KnowledgeBaseGrepTool.name, KnowledgeBaseReadTool.name],
        system_prompt=SEARCH_AGENT_PROMPT,
        permission_mode="never_confirm",
        max_iteration_per_run=12,
    )

    def create_agent(llm: LLM) -> Agent:
        return Agent(
            llm=llm,
            tools=[
                Tool(name=KnowledgeBaseGrepTool.name),
                Tool(name=KnowledgeBaseReadTool.name),
            ],
            agent_context=AgentContext(system_message_suffix=SEARCH_AGENT_PROMPT),
            condenser=default_condenser(
                llm.model_copy(update={"usage_id": "subagent-search-condenser"})
            ),
        )

    return AgentFactory(factory_func=create_agent, definition=definition)


def _general_purpose_agent_factory() -> AgentFactory:
    tool_names = [
        GrepTool.name,
        FileEditorTool.name,
        TerminalTool.name,
        TaskTrackerTool.name,
    ]
    definition = AgentDefinition(
        name=GENERAL_PURPOSE_AGENT_NAME,
        description=(
            "General-purpose specialist for complex workspace analysis, edits, "
            "commands, and tests."
        ),
        model="inherit",
        tools=tool_names,
        system_prompt=GENERAL_PURPOSE_AGENT_PROMPT,
        max_iteration_per_run=30,
    )

    def create_agent(llm: LLM) -> Agent:
        return Agent(
            llm=llm,
            tools=[Tool(name=name) for name in tool_names],
            agent_context=AgentContext(
                system_message_suffix=GENERAL_PURPOSE_AGENT_PROMPT
            ),
            condenser=default_condenser(
                llm.model_copy(
                    update={"usage_id": "subagent-general-purpose-condenser"}
                )
            ),
        )

    return AgentFactory(factory_func=create_agent, definition=definition)


def _subagent_factories() -> dict[SubAgentType, AgentFactory]:
    return {
        SubAgentType.SEARCH: _search_agent_factory(),
        SubAgentType.GENERAL_PURPOSE: _general_purpose_agent_factory(),
    }


class SubAgentAction(Action):
    """A self-contained task delegated to one blocking subagent profile."""

    type: SubAgentType = Field(description="Subagent profile to use")
    task: str = Field(description="Complete, self-contained task for the subagent")


class SubAgentObservation(Observation):
    """Compressed final handoff returned by a subagent."""

    task_id: str
    status: str
    type: SubAgentType
    child_conversation_id: str | None = None


class SubAgentExecutor(ToolExecutor[SubAgentAction, SubAgentObservation]):
    """Launch a fresh isolated subagent and block until its final handoff."""

    def __init__(
        self,
        manager: TaskManager,
        factories: dict[SubAgentType, AgentFactory],
    ):
        self.manager = manager
        self.factories = factories

    def __call__(
        self,
        action: SubAgentAction,
        conversation: "LocalConversation | None" = None,
    ) -> SubAgentObservation:
        if conversation is None:
            return SubAgentObservation.from_text(
                text="subagent requires an active parent conversation",
                task_id="unknown",
                status=TaskStatus.ERROR,
                type=action.type,
                is_error=True,
            )
        factory = self.factories[action.type]
        parent_conversation_id = conversation.state.id
        logger.info(
            "[pyromind-subagent] event=delegation_started "
            "parent_conversation_id=%s type=%s subagent=%s task=%r",
            parent_conversation_id,
            action.type,
            factory.definition.name,
            _log_excerpt(action.task),
        )
        task = self.manager.start_task_with_factory(
            prompt=action.task,
            factory=factory,
            description=action.type.value,
            conversation=conversation,
        )
        logger.info(
            "[pyromind-subagent] event=delegation_completed "
            "parent_conversation_id=%s child_conversation_id=%s task_id=%s "
            "type=%s subagent=%s status=%s",
            parent_conversation_id,
            task.conversation_id,
            task.id,
            action.type,
            factory.definition.name,
            task.status,
        )
        if task.status == TaskStatus.COMPLETED:
            return SubAgentObservation.from_text(
                text=task.result or "Subagent completed without a final handoff.",
                task_id=task.id,
                status=task.status,
                type=action.type,
                child_conversation_id=str(task.conversation_id),
            )
        return SubAgentObservation.from_text(
            text=task.error or "Subagent task failed.",
            task_id=task.id,
            status=task.status,
            type=action.type,
            child_conversation_id=str(task.conversation_id),
            is_error=True,
        )

    def close(self) -> None:
        self.manager.close()


class PyromindSubAgentTool(ToolDefinition[SubAgentAction, SubAgentObservation]):
    """Unified blocking launcher for Pyromind subagent profiles."""

    name = "subagent"

    @classmethod
    def create(
        cls, conv_state: "ConversationState"
    ) -> Sequence["PyromindSubAgentTool"]:
        _knowledge_root(conv_state)
        return [
            cls(
                action_type=SubAgentAction,
                observation_type=SubAgentObservation,
                description=SUBAGENT_TOOL_DESCRIPTION,
                annotations=ToolAnnotations(
                    title="subagent",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=SubAgentExecutor(TaskManager(), _subagent_factories()),
            )
        ]

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=(), declared=True)


register_tool(KnowledgeBaseGrepTool.name, KnowledgeBaseGrepTool)
register_tool(KnowledgeBaseReadTool.name, KnowledgeBaseReadTool)
register_tool(PyromindSubAgentTool.name, PyromindSubAgentTool)
