"""Pyromind knowledge-base retrieval router.

This router wraps the lower-level conversation service, assembling the
system prompt, tools, and workspace on the server side so that the
frontend only needs to pass minimal configuration fields.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field, SecretStr

from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.dependencies import (
    get_conversation_service,
    get_current_user_id,
    get_event_service,
    get_pyromind_jwt_token_from_request,
)
from openhands.agent_server.event_service import EventService
from openhands.agent_server.models import ConversationInfo, Success
from openhands.agent_server.pyromind_auth import (
    PYROMIND_AUTH_COOKIE_NAME,
    CurrentLoginUser,
    bind_debug_current_login_user_to_conversation,
    get_debug_current_login_user_by_conversation,
    is_dev,
)
from openhands.agent_server.pyromind_constants import (
    PYROMIND_APP_TAG_KEY,
    PYROMIND_APP_TAG_VALUE,
)
from openhands.sdk import LLM, AgentContext, TextContent, Tool
from openhands.sdk.conversation.request import (
    SendMessageRequest,
    StartConversationRequest,
)
from openhands.sdk.llm.message import Message
from openhands.sdk.secret import SecretSource, StaticSecret
from openhands.sdk.skills import Skill, load_skills_from_dir
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.preset.codex import get_codex_agent
from openhands.tools.preset.default import register_default_tools
from openhands.tools.pyromind_debug import get_debug_result_broker
from openhands.tools.workflow import DslToXyflowTool, ValidateWorkflowDslTool
from openhands.tools.workflow.validate_workflow_dsl import (
    PYROMIND_VALIDATE_AUTH_COOKIE_SECRET,
    PYROMIND_VALIDATE_HEADERS_STATE_KEY,
)


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
_PYROMIND_SKILL_NAMES = ["generate-workflow-dsl", "debug-workflow"]
_PYROMIND_VALIDATE_AUTHORIZATION_SECRET = "PYROMIND_VALIDATE_AUTHORIZATION"
_PYROMIND_VALIDATE_FORWARD_HEADERS = ("x-cluster", "accept-language")
_PYROMIND_DEBUG_URL_TIMEOUT_SECONDS = 30.0
_PYROMIND_DEBUG_RESPONSE_BODY_LIMIT = 20000

# Knowledge-base retrieval guidance layered on top of the codex base prompt via
# get_codex_agent(custom_instructions=...). Kept lightweight: prefer invoking a
# matching skill first, and only fall back to grep + file_editor for free-form
# knowledge-base lookups.
PYROMIND_KB_INSTRUCTIONS = """\
The Pyromind platform knowledge base is available at this absolute path:
{knowledge_base_path}

Knowledge base layout:
- basic/: platform basics
- jupyterlab/: JupyterLab and script-based training
- sdk/: Python SDK and script-based workflow APIs
- studio/: Studio workflow documentation
- nodes/<NodeType>/<NodeType>.md: node parameters, I/O, and ports
- dataset_processing_workflow.py: workflow DSL example

Your current working directory is this conversation's private workspace:
{working_dir}

Create and edit the workflow DSL at the relative path `workflow.py` from the
current working directory. Prefer `apply_patch` with `workflow.py` for workflow
changes. If you use `file_editor` for this file, set its `path` to `workflow.py`;
the runtime resolves workspace-relative paths to host-absolute paths. Do not
hand-author long absolute paths, and do not use `/workspace/...` or
`workspace/conversations/...` as a `file_editor.path` for `workflow.py`.
After creating or modifying `workflow.py`, stop normally; the server sends
the workflow to the frontend once the run finishes. Do not say the workflow has
been generated unless a tool call actually created or modified `workflow.py`.

- If a listed skill fits the request (for example, generating a workflow), \
invoke that skill via `invoke_skill` first, before searching the knowledge base.
- For Pyromind knowledge-base lookups, use `grep` with the absolute path above,
then open matched files with `file_editor` to read their full content before
answering or editing `workflow.py`.
"""


# ---------------------------------------------------------------------------
# Skill loading utilities
# ---------------------------------------------------------------------------


def _load_agent_skills(
    skills_path: str, allow_list: list[str] | None = None
) -> list[Skill]:
    """Load AgentSkills-format skills as :class:`Skill` objects.

    Returns real ``Skill`` objects (not prompt text) so they can be placed on
    an :class:`AgentContext`. This is what makes the SDK auto-attach
    ``InvokeSkillTool`` — passing skills as prompt text alone advertises
    ``invoke_skill(...)`` without ever attaching the tool, so the model cannot
    call it and falls back to grep.

    When *allow_list* is provided, only skills whose name is in the list are
    returned.
    """
    skills_dir = Path(skills_path)
    if not skills_dir.is_dir():
        logger.warning(f"Skills directory not found: {skills_path}")
        return []

    # load_skills_from_dir returns (repo_skills, knowledge_skills, agent_skills);
    # AgentSkills-format SKILL.md directories land in the third dict.
    _, _, agent_skills = load_skills_from_dir(skills_dir)

    selected: list[Skill] = []
    for name, skill in sorted(agent_skills.items()):
        if allow_list and name not in allow_list:
            continue
        selected.append(skill)
        logger.info(f"Loaded skill: {name}")

    return selected


# ---------------------------------------------------------------------------
# Tool loading utilities
# ---------------------------------------------------------------------------


def _build_workflow_validation_tool(
    http_request: Request,
    extra: dict[str, Any],
) -> tuple[Tool, dict[str, SecretSource]]:
    headers = {
        name: value
        for name in _PYROMIND_VALIDATE_FORWARD_HEADERS
        if name != "x-cluster" and (value := http_request.headers.get(name))
    }
    if cluster := _get_validation_cluster_header(http_request, extra):
        headers["x-cluster"] = cluster

    params: dict[str, Any] = {}
    endpoint_url = extra.get("workflow_validation_endpoint_url")
    if isinstance(endpoint_url, str) and endpoint_url:
        params["endpoint_url"] = endpoint_url
    if headers:
        params["headers"] = headers

    secrets: dict[str, SecretSource] = {}
    secret_headers: dict[str, str] = {}
    cookie_header = _get_validation_cookie_header(http_request)
    if cookie_header:
        secret_headers["cookie"] = PYROMIND_VALIDATE_AUTH_COOKIE_SECRET
        secrets[PYROMIND_VALIDATE_AUTH_COOKIE_SECRET] = StaticSecret(
            value=SecretStr(cookie_header)
        )

    authorization = http_request.headers.get("authorization")
    if authorization:
        secret_headers["authorization"] = _PYROMIND_VALIDATE_AUTHORIZATION_SECRET
        secrets[_PYROMIND_VALIDATE_AUTHORIZATION_SECRET] = StaticSecret(
            value=SecretStr(authorization)
        )

    if secret_headers:
        params["secret_headers"] = secret_headers

    return Tool(name=ValidateWorkflowDslTool.name, params=params), secrets


async def apply_pyromind_validation_context(
    event_service: EventService,
    current_user: CurrentLoginUser | None,
) -> None:
    if current_user is None:
        return
    if event_service.stored.tags.get(PYROMIND_APP_TAG_KEY) != PYROMIND_APP_TAG_VALUE:
        return

    if current_user.cookie:
        await event_service.update_secrets(
            {PYROMIND_VALIDATE_AUTH_COOKIE_SECRET: current_user.cookie}
        )
    if current_user.x_cluster:
        await event_service.update_agent_state(
            {PYROMIND_VALIDATE_HEADERS_STATE_KEY: {"x-cluster": current_user.x_cluster}}
        )


def _get_validation_cookie_header(http_request: Request) -> str | None:
    current_user = getattr(http_request.state, "current_user", None)
    if isinstance(current_user, CurrentLoginUser) and current_user.cookie:
        return current_user.cookie

    raw_cookie = http_request.headers.get("cookie")
    if raw_cookie:
        return raw_cookie

    auth_token = get_pyromind_jwt_token_from_request(http_request)
    if auth_token:
        return f"{PYROMIND_AUTH_COOKIE_NAME}={auth_token}"
    return None


def _get_validation_cluster_header(
    http_request: Request,
    extra: dict[str, Any],
) -> str | None:
    current_user = getattr(http_request.state, "current_user", None)
    if isinstance(current_user, CurrentLoginUser) and current_user.x_cluster:
        return current_user.x_cluster

    cluster = extra.get("x_cluster", extra.get("x-cluster"))
    if isinstance(cluster, str) and cluster:
        return cluster
    return http_request.headers.get("x-cluster")


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
    workflow_dsl: str | None = Field(
        default=None,
        description=(
            "Optional DSL of a workflow the user already had on the canvas "
            "before starting this conversation. If provided, it seeds "
            "workflow.py before the initial message is processed. `null` "
            "(the default) means no canvas state to seed from."
        ),
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


class PyromindSendMessageRequest(BaseModel):
    """Request body for sending a message in a Pyromind conversation.

    Unlike the generic ``POST /api/conversations/{id}/events`` endpoint, this
    also accepts the DSL of the workflow currently shown on the canvas, so
    that workflow.py is synced to what the user actually sees *before* the
    agent acts on this message. See "工作流同步链路" in the debug-loop plan.
    """

    text: str = Field(description="The user's message text.")
    workflow_dsl: str | None = Field(
        default=None,
        description=(
            "DSL of the workflow currently on the canvas. `null` means the "
            "frontend did not attach canvas state, so no sync is performed "
            "(workflow.py is left untouched). An empty string means the "
            "canvas is genuinely empty (the user cleared it)."
        ),
    )
    run: bool = Field(
        default=True,
        description="Whether the agent loop should run after this message.",
    )


class PyromindDebugCallbackRequest(BaseModel):
    """Webhook payload the debug platform posts when an async run finishes."""

    task_id: str = Field(description="The task id returned when the run was submitted.")
    status: Literal["passed", "failed"] = Field(description="Outcome of the run.")
    error_log: str | None = Field(
        default=None, description="Runtime error output when status='failed'."
    )


def _normalize_dsl(text: str) -> str:
    return text.strip()


def _sync_workflow_with_canvas(
    working_dir: Path, workflow_dsl: str | None
) -> TextContent | None:
    """Reconcile workflow.py with the DSL currently shown on the canvas.

    The user can edit the canvas between agent turns, so workflow.py must be
    re-synced from the canvas before each new user message is processed --
    otherwise the agent would keep editing a stale version. Returns a
    ``<system_reminder>`` TextContent to inject into the LLM's context (via
    ``extended_content``) when workflow.py actually changed as a result, or
    None when nothing needed to change.

    `workflow_dsl=None` means the caller attached no canvas state at all
    (e.g. a plain follow-up message) and is a deliberate no-op, distinct
    from `workflow_dsl=""` which means the canvas is genuinely empty.
    """
    if workflow_dsl is None:
        return None

    workflow_path = working_dir / "workflow.py"
    existed = workflow_path.is_file()
    current = workflow_path.read_text(encoding="utf-8") if existed else ""

    normalized_canvas = _normalize_dsl(workflow_dsl)
    normalized_current = _normalize_dsl(current)
    if normalized_canvas == normalized_current:
        return None  # Already in sync -- also covers the from-scratch case
        # where the canvas and workflow.py are both empty/missing.

    working_dir.mkdir(parents=True, exist_ok=True)

    if not normalized_canvas:
        # The user cleared the canvas. Remove workflow.py entirely (rather
        # than writing an empty file) so downstream logic -- the debug tool,
        # the generate-workflow-dsl skill's "does workflow.py exist?" check,
        # etc. -- treats this identically to "never created".
        workflow_path.unlink(missing_ok=True)
        return TextContent(
            text=(
                "<system_reminder>\n"
                "The user has cleared the workflow on the canvas. workflow.py "
                "has been removed to match. If asked to continue building a "
                "workflow, start fresh rather than assuming the previous "
                "workflow still exists.\n"
                "</system_reminder>"
            )
        )

    workflow_path.write_text(workflow_dsl, encoding="utf-8")
    if existed:
        reminder_text = (
            "<system_reminder>\n"
            "The user has modified the workflow on the canvas since your last "
            "turn. workflow.py has been overwritten to match the canvas "
            "exactly. Treat the current file content as the source of truth "
            "-- do not rely on your memory of a previous version when reading "
            "or editing it.\n"
            "</system_reminder>"
        )
    else:
        reminder_text = (
            "<system_reminder>\n"
            "The user already had a workflow on the canvas from before this "
            "message. It has been loaded into workflow.py as the current "
            "state -- read it before making further changes.\n"
            "</system_reminder>"
        )
    return TextContent(text=reminder_text)


class PyromindDebugUrlRequest(BaseModel):
    url: str = Field(description="HTTP or HTTPS URL to call with dev auth context.")
    conversation_id: UUID = Field(
        description=(
            "Conversation to scope the debug request to. The server verifies the "
            "conversation belongs to the current user before forwarding the "
            "current request's login context."
        ),
    )


class PyromindDebugUrlResponse(BaseModel):
    url: str
    context_source: str
    conversation_id: UUID | None = None
    status_code: int
    forwarded_headers: dict[str, str]
    response_headers: dict[str, str]
    body: str


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

pyromind_router = APIRouter(prefix="/pyromind", tags=["Pyromind"])


def _get_current_login_user(http_request: Request) -> CurrentLoginUser:
    current_user = getattr(http_request.state, "current_user", None)
    if not isinstance(current_user, CurrentLoginUser):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Pyromind user context is not available.",
        )
    return current_user


def _validate_debug_url(url: str) -> str:
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid URL: {exc}",
        ) from exc

    if parsed.scheme not in {"http", "https"} or not parsed.host:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL must be an absolute HTTP or HTTPS URL.",
        )
    return str(parsed)


def _build_debug_context_headers(current_user: CurrentLoginUser) -> dict[str, str]:
    headers: dict[str, str] = {}
    if current_user.cookie:
        headers["cookie"] = current_user.cookie
    if current_user.x_cluster:
        headers["x-cluster"] = current_user.x_cluster
    return headers


@pyromind_router.post(
    "/debug/request-url",
    response_model=PyromindDebugUrlResponse,
)
async def request_debug_url(
    request: PyromindDebugUrlRequest,
) -> PyromindDebugUrlResponse:
    if not is_dev():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    url = _validate_debug_url(request.url)
    conversation_user = get_debug_current_login_user_by_conversation(
        request.conversation_id
    )
    if conversation_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pyromind conversation login context is not available.",
        )
    headers = _build_debug_context_headers(conversation_user)

    try:
        async with httpx.AsyncClient(
            timeout=_PYROMIND_DEBUG_URL_TIMEOUT_SECONDS,
            follow_redirects=False,
        ) as client:
            response = await client.get(url, headers=headers)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to request debug URL: {type(exc).__name__}: {exc}",
        ) from exc

    return PyromindDebugUrlResponse(
        url=url,
        context_source="conversation",
        conversation_id=request.conversation_id,
        status_code=response.status_code,
        forwarded_headers=headers,
        response_headers=dict(response.headers),
        body=response.text[:_PYROMIND_DEBUG_RESPONSE_BODY_LIMIT],
    )


@pyromind_router.post("/conversations", response_model=ConversationInfo)
async def create_pyromind_conversation(
    http_request: Request,
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
    - Workspace pointing to a conversation-private directory
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

    # Generate the conversation id here so the workspace directory and persisted
    # conversation id stay aligned.
    conversation_id = uuid.uuid4()
    conversation_dir = conversation_service.conversations_dir / conversation_id.hex
    conversation_dir.mkdir(parents=True, exist_ok=True)

    # 2. Assemble the pyromind KB instructions (layered on the codex base prompt)
    custom_instructions = PYROMIND_KB_INSTRUCTIONS.format(
        knowledge_base_path=knowledge_base_path,
        working_dir=str(conversation_dir),
    )

    # Append extra custom instructions from the request if provided
    extra_instructions = request.extra.get("custom_instructions")
    if extra_instructions:
        custom_instructions += f"\n\n补充说明：{extra_instructions}"

    # 3. Load available skills as real Skill objects. Placing them on an
    #    AgentContext makes the SDK auto-attach InvokeSkillTool so the model can
    #    actually call invoke_skill(...) (prompt text alone does not attach it).
    skills_path = request.extra.get("skills_path", _DEFAULT_SKILLS_PATH)
    skills = _load_agent_skills(skills_path, allow_list=_PYROMIND_SKILL_NAMES)
    agent_context = AgentContext(skills=skills) if skills else None
    validation_tool, validation_secrets = _build_workflow_validation_tool(
        http_request, request.extra
    )

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
        agent_context=agent_context,
        custom_instructions=custom_instructions,
        extra_tools=[
            Tool(name="grep"),
            Tool(name="file_editor"),
            Tool(name="debug_workflow"),
            Tool(name=DslToXyflowTool.name),
            validation_tool,
        ],
    )

    # 6. Build a conversation-private workspace. The knowledge base is accessed
    #    separately via its absolute path in the prompt.
    workspace = LocalWorkspace(working_dir=str(conversation_dir))

    # Seed workflow.py from a canvas the user already had before starting this
    # conversation (e.g. they sketched something, then opened chat). No
    # system_reminder is needed here -- this is turn 1, so there is no
    # prior-turn workflow.py content for the agent to contrast against.
    if request.workflow_dsl:
        (conversation_dir / "workflow.py").write_text(
            request.workflow_dsl, encoding="utf-8"
        )

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
        conversation_id=conversation_id,
        initial_message=initial_message,
        secrets=validation_secrets,
        tags={PYROMIND_APP_TAG_KEY: PYROMIND_APP_TAG_VALUE},
        user_id=get_current_user_id(http_request),
    )

    # 8. Delegate to conversation service
    try:
        info, is_new = await conversation_service.start_conversation(start_request)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    current_user = getattr(http_request.state, "current_user", None)
    if is_dev() and isinstance(current_user, CurrentLoginUser):
        bind_debug_current_login_user_to_conversation(info.id, current_user)
    response.status_code = status.HTTP_201_CREATED if is_new else status.HTTP_200_OK
    return info


@pyromind_router.post(
    "/conversations/{conversation_id}/messages",
    response_model=Success,
    responses={404: {"description": "Conversation not found"}},
)
async def send_pyromind_message(
    request: PyromindSendMessageRequest,
    event_service: EventService = Depends(get_event_service),
) -> Success:
    """Send a user message, first syncing workflow.py to the canvas if needed.

    The frontend attaches the DSL currently shown on the canvas via
    ``workflow_dsl`` on every message. If it disagrees with workflow.py (the
    user edited the canvas, or cleared it), workflow.py is overwritten/
    removed to match and a ``<system_reminder>`` is injected into this
    turn's LLM context (not into the user's visible message) so the agent
    knows to treat the current file as authoritative.
    """
    conversation = event_service.get_conversation()
    working_dir = Path(conversation.workspace.working_dir)
    reminder = _sync_workflow_with_canvas(working_dir, request.workflow_dsl)

    message = Message(role="user", content=[TextContent(text=request.text)])
    await event_service.send_message(
        message,
        run=request.run,
        extended_content=[reminder] if reminder else None,
        workflow_dsl_snapshot=request.workflow_dsl,
    )
    return Success()


# ---------------------------------------------------------------------------
# Debug webhook router
#
# Deliberately a *separate* router from ``pyromind_router`` above and mounted
# directly on the app in api.py, bypassing the global
# ``Depends(check_session_api_key)`` applied to every other /api/* route.
# The caller here is the external debug platform (or, today, the in-process
# MockDebugPlatform's timer thread making a real HTTP call), not a logged-in
# user -- it has no session key and no Pyromind login cookie to present.
#
# Known trade-off (accepted for now): this endpoint has NO authentication.
# It is only safe to expose this server on a trusted/internal network. If
# the debug platform is ever reachable from a less trusted network, add a
# shared-secret header check here before going further.
# ---------------------------------------------------------------------------

pyromind_debug_webhook_router = APIRouter(prefix="/api/pyromind", tags=["Pyromind"])


@pyromind_debug_webhook_router.post("/debug/callback", response_model=Success)
async def pyromind_debug_callback(request: PyromindDebugCallbackRequest) -> Success:
    """Webhook the debug platform calls when an async debug run finishes.

    Resolves the in-process :class:`DebugResultBroker`, waking the
    ``debug_workflow`` tool-executor thread that is blocked waiting for this
    task's result. See ``openhands.tools.pyromind_debug`` for the tool side
    and ``MockDebugPlatform`` for the local stand-in used until the real
    platform integration is wired up.
    """
    broker = get_debug_result_broker()
    resolved = broker.resolve(
        request.task_id, status=request.status, error_log=request.error_log
    )
    if not resolved:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown or already-resolved debug task: {request.task_id}",
        )
    return Success()
