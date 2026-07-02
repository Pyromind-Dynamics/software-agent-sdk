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
from openhands.sdk import LLM, Agent, TextContent, Tool
from openhands.sdk.conversation.request import (
    SendMessageRequest,
    StartConversationRequest,
)
from openhands.sdk.workspace import LocalWorkspace
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

PYROMIND_SYSTEM_PROMPT = """\
你是 Pyromind 平台的知识库助手。当用户询问关于 pyromind 平台的使用方法、节点概念或相关功能时，\
请使用 grep 工具在知识库目录中搜索相关关键信息，然后基于搜索结果回答用户的问题。

知识库目录路径: {knowledge_base_path}

工作流程：
1. 分析用户问题，提取关键词
2. 使用 grep 工具在知识库目录中搜索相关内容（可多次搜索不同关键词）
3. 综合搜索结果，用清晰易懂的语言回答用户问题
4. 如果知识库中没有找到相关信息，如实告知用户

注意：搜索时请使用中文和英文关键词都尝试，以获得更全面的结果。\
"""  # noqa: E501


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
    """Build the skills section to append to the system prompt."""
    if not skills:
        return ""

    sections = []
    sections.append("\n\n# 可用技能 (Available Skills)")
    sections.append("以下技能已加载，当用户需求匹配时请按照技能说明执行：\n")

    for skill in skills:
        sections.append(f"---\n{skill['content']}\n")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class PyromindLLMConfig(BaseModel):
    """LLM configuration passed from the frontend."""

    model: str = Field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o"),
    )
    api_key: str | None = Field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY"),
    )
    base_url: str | None = Field(
        default_factory=lambda: os.environ.get("LLM_BASE_URL"),
    )


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
    - A system prompt instructing the agent to use grep for KB search
    - Tools: terminal, file_editor, grep
    - Workspace pointing to the knowledge base directory
    """
    # Ensure grep tool is registered
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

    # 2. Assemble system prompt
    system_prompt = PYROMIND_SYSTEM_PROMPT.format(
        knowledge_base_path=knowledge_base_path
    )

    # 3. Load and append available skills
    skills_path = request.extra.get("skills_path", _DEFAULT_SKILLS_PATH)
    skills = _load_skills(skills_path, allow_list=_PYROMIND_SKILL_NAMES)
    skills_prompt = _build_skills_prompt(skills)
    if skills_prompt:
        system_prompt += skills_prompt

    # Append custom instructions from extra if provided
    custom_instructions = request.extra.get("custom_instructions")
    if custom_instructions:
        system_prompt += f"\n\n补充说明：{custom_instructions}"

    # 4. Build LLM config
    llm = LLM(
        usage_id="pyromind-agent",
        model=request.llm.model,
        api_key=request.llm.api_key,
        base_url=request.llm.base_url,
        native_tool_calling=False,
    )

    # 5. Build Agent with tools and system prompt
    agent = Agent(
        llm=llm,
        tools=[
            Tool(name="terminal"),
            Tool(name="file_editor"),
            Tool(name="grep"),
        ],
        system_prompt=system_prompt,
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
    response.status_code = status.HTTP_201_CREATED if is_new else status.HTTP_200_OK
    return info
