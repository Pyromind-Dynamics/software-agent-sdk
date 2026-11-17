from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from pyromind_agent_server.api.schemas import AgentLLMConfig

from openhands.agent_server.agent_profiles.pyromind import PYROMIND_KB_INSTRUCTIONS
from openhands.agent_server.event_service import EventService
from openhands.agent_server.pyromind_auth import (
    CurrentLoginUser,
)
from openhands.agent_server.pyromind_constants import (
    PYROMIND_APP_TAG_KEY,
    PYROMIND_APP_TAG_VALUE,
)
from openhands.agent_server.pyromind_router import (
    PYROMIND_AUTH_TOKEN_SECRET,
    PYROMIND_PRODUCT_USER_ID_SECRET,
    PYROMIND_X_CLUSTER_SECRET,
    _build_debug_context_headers,
    apply_pyromind_validation_context,
)
from openhands.tools.pyromind_dataset.definition import (
    PYROMIND_STORAGE_AUTH_COOKIE_SECRET,
    PYROMIND_STORAGE_HEADERS_STATE_KEY,
)
from openhands.tools.workflow.validate_workflow_dsl import (
    PYROMIND_VALIDATE_AUTH_COOKIE_SECRET,
    PYROMIND_VALIDATE_HEADERS_STATE_KEY,
)


def test_generate_workflow_skill_uses_progressive_reference_disclosure() -> None:
    repo_root = Path(__file__).parents[2]
    skill_root = repo_root / ".agents" / "skills" / "generate-workflow-dsl"
    skill_text = (skill_root / "SKILL.md").read_text(encoding="utf-8")
    references = {
        path.name: path.read_text(encoding="utf-8")
        for path in (skill_root / "references").glob("*.md")
    }

    frontmatter = skill_text.split("---", 2)[1]
    assert set(
        line.split(":", 1)[0]
        for line in frontmatter.splitlines()
        if line and not line.startswith(" ")
    ) == {"name", "description"}
    assert len(skill_text.splitlines()) <= 140
    assert "## 资料索引" in skill_text
    assert "## DSL 与资料边界" in skill_text
    assert "按 `<available_skills>` 的位置用 `read` 打开" in skill_text
    assert "<available_skills>" in skill_text
    assert "每次写入后调用 `validate_workflow_dsl()`" in skill_text
    assert "不写 import" in skill_text
    assert "生成任务不调用 `workflow_debug`、`submit_workflow`" in skill_text
    assert "不执行正式工作流" in frontmatter
    assert "retryable=false" in skill_text
    assert "单节点工作流" in frontmatter
    assert "ModelTrainSFTNode(" not in skill_text
    assert set(references) == {
        "custom-python-assets.md",
        "data-routing.md",
        "workflow-contracts.md",
    }
    assert all(f"`references/{name}`" in skill_text for name in references)
    assert not (skill_root / "references" / "example-workflows.md").exists()
    for removed in (
        "node-reference.md",
        "platform-contract-overrides.md",
        "stage-templates.md",
    ):
        assert not (skill_root / "references" / removed).exists()

    debug_skill_text = (
        repo_root / ".agents" / "skills" / "debug-workflow" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "使用 generate-workflow-dsl" in debug_skill_text
    assert "本轮不得调用 workflow_debug" in debug_skill_text


def test_generate_workflow_skill_rejects_module_level_literal_constants() -> None:
    repo_root = Path(__file__).parents[2]
    skill_text = (
        repo_root / ".agents" / "skills" / "generate-workflow-dsl" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert "顶层赋值只用于声明节点实例" in skill_text
    assert "禁止模块级字面量常量" in skill_text
    assert "字面量或上游节点输出" in skill_text
    assert "`DSL_PARSE_FAILED`" in skill_text


def test_pyromind_instructions_enforce_workflow_skill_reference_order() -> None:
    rendered = PYROMIND_KB_INSTRUCTIONS.format(
        knowledge_alias="knowledge",
        skills_alias=".agents/skills",
        working_dir="workspace/conversations/test",
    )

    assert "Then invoke the matching listed skill" in rendered
    assert "read only the exact `references/`\n  resource" in rendered
    assert "then `skills_read` only\n  for `SKILL.md`, `references/**`" in rendered
    assert "Treat `scripts/` as executable helpers" in rendered
    assert "Do not read, grep, or summarize\n  `scripts/` source" in rendered
    assert (
        "do not inspect\n  general `knowledge/` before invoking the skill" in rendered
    )
    assert "Treat any requested node, model, parameter" in rendered
    assert "do not\n  invoke `debug-workflow` or `workflow_debug`" in rendered
    assert "Pyromind platform nodes perform actual" in rendered
    assert "Use dedicated platform tools for preview/upload" in rendered
    assert "never use the local terminal as a substitute" in rendered
    assert "`public_data/` is the writable area" in rendered
    assert "every path you write or read starts with `public_data/`" in rendered
    assert "It accepts only the `patch` argument" in rendered
    assert "never pass a separate `path` argument" in rendered
    assert "The separate `file_editor` tool does accept `path`" in rendered
    assert "Each terminal call also starts at the conversation root" in rendered
    assert "never rely on a `cd`" in rendered
    assert "conversation-local auxiliary" in rendered
    assert "Do not consult `knowledge/` before validation" in rendered
    assert "Whenever a direct question or an intermediate step requires" in rendered
    assert "The parent must not inspect those documents" in rendered
    assert 'at most one `subagent(type="search")` call' in rendered
    assert "all related knowledge-base subquestions into one" in rendered
    assert "never issue multiple or parallel search subagent calls" in rendered
    assert "reads `knowledge/index.md` first" in rendered
    assert "Do not repeat the search subagent's work" in rendered
    assert '`type="general_purpose"`' in rendered
    assert "The call is blocking" in rendered
    assert "Do not run `pwd`, `cd`, or directory listings" not in rendered
    assert "Do not use `/dev/null`" not in rendered
    assert "Do not run `pwd`" not in rendered
    assert "do not access host-absolute paths, `/workspace`" not in rendered
    assert "workspace-discovery commands are rejected" not in rendered
    assert "exact, already-known conversation-local script" not in rendered
    assert "workspace/conversations/test" not in rendered


def test_pyromind_kb_instructions_keep_the_documented_knowledge_paths() -> None:
    """The KB instructions must keep pointing at the documented knowledge paths.

    Carried over from the stale duplicate that sat inside the source package
    (openhands-agent-server/tests/agent_server/test_pyromind_router.py), which
    no CI job ever ran. Its other content assertions had already rotted.
    """
    rendered = PYROMIND_KB_INSTRUCTIONS.format(
        knowledge_alias="knowledge",
        skills_alias=".agents/skills",
        working_dir="workspace/conversations/abc123",
    )

    assert "nodes/<NodeType>/<NodeType>.md" in rendered
    assert "dataset_processing_workflow.py" in rendered


def test_pyromind_llm_config_normalizes_chat_completions_base_url() -> None:
    config = AgentLLMConfig(
        model="openai/glm-5.2-fp8",
        api_key="test-key",
        base_url=" http://208.64.254.187:8000/v1/chat/completions/ ",
    )

    assert config.base_url == "http://208.64.254.187:8000/v1"


class _FakeEventService:
    def __init__(
        self,
        tags: dict[str, str],
        conversation_id: UUID | None = None,
    ) -> None:
        self.stored = type(
            "FakeStoredConversation",
            (),
            {
                "id": conversation_id or UUID("00000000-0000-0000-0000-000000000001"),
                "tags": tags,
            },
        )()
        self.secrets: dict[str, str] = {}
        self.agent_state: dict[str, object] = {}

    async def update_secrets(self, secrets: dict[str, str]) -> None:
        self.secrets.update(secrets)

    async def update_agent_state(self, values: dict[str, object]) -> None:
        self.agent_state.update(values)


def test_build_debug_context_headers_uses_current_user_context():
    current_user = CurrentLoginUser(
        username="debug-user-42",
        email="debug-user-42@example.test",
        user_id=42,
        cookie="auth_token=session-token; other=value",
        x_cluster="us-west-1#pre",
    )

    assert _build_debug_context_headers(current_user) == {
        "cookie": "auth_token=session-token; other=value",
        "x-cluster": "us-west-1#pre",
    }


@pytest.mark.asyncio
async def test_pyromind_validation_context_uses_websocket_user_headers():
    service = _FakeEventService({PYROMIND_APP_TAG_KEY: PYROMIND_APP_TAG_VALUE})
    current_user = CurrentLoginUser(
        username="debug-user-42",
        email="debug-user-42@example.test",
        user_id=42,
        cookie="auth_token=websocket-token; other=value",
        x_cluster="websocket-cluster",
    )

    await apply_pyromind_validation_context(
        cast(EventService, service), current_user, x_cluster="websocket-cluster"
    )

    assert service.secrets == {
        PYROMIND_VALIDATE_AUTH_COOKIE_SECRET: "auth_token=websocket-token; other=value",
        PYROMIND_AUTH_TOKEN_SECRET: "websocket-token",
        PYROMIND_STORAGE_AUTH_COOKIE_SECRET: "auth_token=websocket-token; other=value",
        PYROMIND_X_CLUSTER_SECRET: "websocket-cluster",
        PYROMIND_PRODUCT_USER_ID_SECRET: "42",
    }
    assert service.agent_state == {
        PYROMIND_VALIDATE_HEADERS_STATE_KEY: {"x-cluster": "websocket-cluster"},
        PYROMIND_STORAGE_HEADERS_STATE_KEY: {"x-cluster": "websocket-cluster"},
    }


def test_parse_auth_token_from_cookie_header_extracts_jwt():
    from openhands.agent_server.pyromind_auth import parse_auth_token_from_cookie_header

    assert (
        parse_auth_token_from_cookie_header("auth_token=jwt-token; other=value")
        == "jwt-token"
    )
    assert parse_auth_token_from_cookie_header(None) is None


@pytest.mark.asyncio
async def test_pyromind_validation_context_ignores_non_pyromind_conversations():
    service = _FakeEventService({})
    current_user = CurrentLoginUser(
        username="debug-user-42",
        email="debug-user-42@example.test",
        user_id=42,
        cookie="auth_token=websocket-token",
        x_cluster="websocket-cluster",
    )

    await apply_pyromind_validation_context(
        cast(EventService, service), current_user, x_cluster="websocket-cluster"
    )

    assert service.secrets == {}
    assert service.agent_state == {}
