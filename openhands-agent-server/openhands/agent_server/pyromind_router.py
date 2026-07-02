"""Pyromind knowledge-base retrieval router.

This router wraps the lower-level conversation service, assembling the
system prompt, tools, and workspace on the server side so that the
frontend only needs to pass minimal configuration fields.
"""

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.dependencies import get_conversation_service
from openhands.agent_server.models import ConversationInfo
from openhands.sdk import LLM, TextContent, Tool
from openhands.sdk.conversation.request import (
    SendMessageRequest,
    StartConversationRequest,
)
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.preset.codex import get_codex_agent
from openhands.tools.preset.default import register_default_tools


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default knowledge base path — resolved relative to this file's package root;
# can be overridden via the PYROMIND_KNOWLEDGE_BASE_PATH environment variable.
_DEFAULT_KNOWLEDGE_BASE_PATH = os.environ.get(
    "PYROMIND_KNOWLEDGE_BASE_PATH",
    str(Path(__file__).resolve().parents[3] / "knowledge"),
)

# Default skills path — .agents/skills/ at the repo root
_DEFAULT_SKILLS_PATH = os.environ.get(
    "PYROMIND_SKILLS_PATH",
    str(Path(__file__).resolve().parents[3] / ".agents" / "skills"),
)

# Only load these skills for Pyromind (avoids loading unrelated SDK skills)
_PYROMIND_SKILL_NAMES = ["generate-workflow-dsl"]

# Knowledge-base retrieval guidance layered on top of the codex base prompt via
# get_codex_agent(custom_instructions=...). Kept lightweight: prefer invoking a
# matching skill first, and only fall back to grep + file_editor for free-form
# knowledge-base lookups.
PYROMIND_KB_INSTRUCTIONS = """\
The Pyromind platform knowledge base is at your working directory: \
{knowledge_base_path}

- If a listed skill fits the request (for example, generating a workflow), \
invoke that skill via `invoke_skill` first, before searching the knowledge base.
- Otherwise, for Pyromind questions, `grep` this directory for the relevant \
keywords, then open the matched files with `file_editor` to read their full \
content before answering. Do not restrict the search to a single file \
extension (such as .mdx/.md); the knowledge base contains various file types.\
"""


# ---------------------------------------------------------------------------
# Skill loading utilities
# ---------------------------------------------------------------------------


def _load_skills(
    skills_path: str, allow_list: list[str] | None = None
) -> list[dict[str, str]]:
    """Load SKILL.md files from the skills directory.

    Scans for both top-level markdown files and SKILL.md inside subdirectories.
    When *allow_list* is provided, only skills whose name is in the list are loaded.
    Returns a list of dicts with 'name' and 'content' keys.
    """
    skills: list[dict[str, str]] = []
    skills_dir = Path(skills_path)
    if not skills_dir.is_dir():
        logger.warning(f"Skills directory not found: {skills_path}")
        return skills

    for entry in sorted(skills_dir.iterdir()):
        name = entry.stem if entry.is_file() else entry.name
        # Filter by allow list if specified
        if allow_list and name not in allow_list:
            continue

        skill_file: Path | None = None
        if entry.is_dir():
            candidate = entry / "SKILL.md"
            if candidate.is_file():
                skill_file = candidate
        elif entry.is_file() and entry.suffix == ".md":
            skill_file = entry

        if skill_file:
            content = skill_file.read_text(encoding="utf-8")
            skills.append({"name": name, "content": content})
            logger.info(f"Loaded skill: {name}")

    return skills


def _build_skills_prompt(skills: list[dict[str, str]]) -> str:
    """Build the inner body for the codex ``<SKILLS>`` block.

    The codex template (`system_prompt_codex.j2`) already renders the
    ``<SKILLS>`` envelope and the ``invoke_skill(...)`` instructions, so this
    only returns the per-skill listing that goes inside it. Returns an empty
    string when there are no skills (the block is then omitted).
    """
    if not skills:
        return ""

    sections = []
    for skill in skills:
        sections.append(f"---\n{skill['content']}")

    return "\n\n".join(sections)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class PyromindLLMConfig(BaseModel):
    """LLM configuration passed from the frontend."""

    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None


class PyromindCreateConversationRequest(BaseModel):
    """Request body for creating a Pyromind knowledge-base conversation.

    Only essential fields are required; the server assembles the full agent
    configuration (system_prompt, tools, workspace) internally.

    The ``extra`` dict is intentionally open-ended to support future fields
    without breaking the API contract.
    """

    llm: PyromindLLMConfig
    message: str | None = Field(
        default=None,
        description="Optional initial user message to start the conversation.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extensible JSON object for additional configuration. "
            "Supported keys: "
            "'knowledge_base_path' (str) - override default KB path; "
            "'language' (str) - preferred response language; "
            "'custom_instructions' (str) - extra instructions appended to prompt."
        ),
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

pyromind_router = APIRouter(prefix="/pyromind", tags=["Pyromind"])


@pyromind_router.post("/conversations", response_model=ConversationInfo)
async def create_pyromind_conversation(
    request: PyromindCreateConversationRequest,
    response: Response,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationInfo:
    """Create a new conversation configured for Pyromind knowledge-base retrieval.

    The server assembles:
    - A codex-style base agent (prompt + tools) via ``get_codex_agent``
    - Pyromind KB-retrieval instructions layered on top via ``custom_instructions``
    - Tools: codex set (terminal + apply_patch + task_tracker) + grep and
      file_editor for KB search (grep finds files, file_editor views them)
    - Workspace pointing to the knowledge base directory
    """
    # Ensure the grep tool is registered (codex tools are registered by the preset).
    register_default_tools(enable_browser=False)

    # 1. Resolve knowledge base path (extra can override the default)
    knowledge_base_path = request.extra.get(
        "knowledge_base_path", _DEFAULT_KNOWLEDGE_BASE_PATH
    )
    knowledge_base_path = os.path.abspath(knowledge_base_path)

    if not os.path.isdir(knowledge_base_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Knowledge base path does not exist: {knowledge_base_path}",
        )

    # 2. Assemble the pyromind KB instructions (layered on the codex base prompt)
    custom_instructions = PYROMIND_KB_INSTRUCTIONS.format(
        knowledge_base_path=knowledge_base_path
    )

    # Append extra custom instructions from the request if provided
    extra_instructions = request.extra.get("custom_instructions")
    if extra_instructions:
        custom_instructions += f"\n\n补充说明：{extra_instructions}"

    # 3. Load available skills and render them into the codex <SKILLS> block
    skills_path = request.extra.get("skills_path", _DEFAULT_SKILLS_PATH)
    skills = _load_skills(skills_path, allow_list=_PYROMIND_SKILL_NAMES)
    skills_prompt = _build_skills_prompt(skills) or None

    # 4. Build LLM config
    llm = LLM(
        usage_id="pyromind-agent",
        model=request.llm.model,
        api_key=request.llm.api_key,
        base_url=request.llm.base_url,
    )

    # 5. Build the codex-style agent with the KB instructions + KB retrieval
    #    tools (grep to find files, file_editor to view their content).
    agent = get_codex_agent(
        llm=llm,
        cli_mode=True,
        available_skills_prompt=skills_prompt,
        custom_instructions=custom_instructions,
        extra_tools=[Tool(name="grep"), Tool(name="file_editor")],
    )

    # 6. Build workspace pointing to knowledge base
    workspace = LocalWorkspace(working_dir=knowledge_base_path)

    # 7. Assemble StartConversationRequest
    initial_message: SendMessageRequest | None = None
    if request.message:
        initial_message = SendMessageRequest(
            role="user",
            content=[TextContent(text=request.message)],
            run=True,
        )

    start_request = StartConversationRequest(
        agent=agent,
        workspace=workspace,
        initial_message=initial_message,
    )

    # 8. Delegate to conversation service
    info, is_new = await conversation_service.start_conversation(start_request)
    response.status_code = (
        status.HTTP_201_CREATED if is_new else status.HTTP_200_OK
    )
    return info
