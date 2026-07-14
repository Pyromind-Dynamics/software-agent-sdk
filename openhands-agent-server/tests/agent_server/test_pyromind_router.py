"""Tests for the Pyromind router codex-alignment helpers."""

from __future__ import annotations

from openhands.agent_server.pyromind_router import (
    PYROMIND_KB_INSTRUCTIONS,
    _load_agent_skills,
)


_REMOVED_WORKFLOW_TOOL = "publish" + "_workflow"


def test_load_agent_skills_missing_dir_returns_empty() -> None:
    assert _load_agent_skills("/nonexistent/skills/path") == []


def test_load_agent_skills_returns_skill_objects(tmp_path) -> None:
    """AgentSkills-format SKILL.md directories load as invocable Skill objects."""
    skill_dir = tmp_path / "generate-workflow-dsl"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: generate-workflow-dsl\n"
        "description: Generate a workflow DSL.\n---\n\nBody.\n",
        encoding="utf-8",
    )

    skills = _load_agent_skills(str(tmp_path), allow_list=["generate-workflow-dsl"])

    assert len(skills) == 1
    skill = skills[0]
    assert skill.name == "generate-workflow-dsl"
    # Must be model-invocable so the SDK auto-attaches InvokeSkillTool.
    assert skill.is_agentskills_format is True
    assert skill.disable_model_invocation is False


def test_load_agent_skills_respects_allow_list(tmp_path) -> None:
    for name in ("generate-workflow-dsl", "unrelated-skill"):
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name}.\n---\n\nBody.\n",
            encoding="utf-8",
        )

    skills = _load_agent_skills(str(tmp_path), allow_list=["generate-workflow-dsl"])

    assert [s.name for s in skills] == ["generate-workflow-dsl"]


def test_kb_instructions_format_injects_path() -> None:
    rendered = PYROMIND_KB_INSTRUCTIONS.format(
        knowledge_base_path="/kb/root",
        working_dir="workspace/conversations/abc123",
    )
    assert "/kb/root" in rendered
    assert "workspace/conversations/abc123" in rendered
    assert "workflow.py" in rendered
    assert "Pyromind" in rendered
    assert "nodes/<NodeType>/<NodeType>.md" in rendered
    assert "dataset_processing_workflow.py" in rendered
    assert "docs-mintlify/zh/docs" not in rendered
    assert _REMOVED_WORKFLOW_TOOL not in rendered
    assert "server sends" in rendered
    # Skill-first guidance must be present.
    assert "invoke_skill" in rendered
    assert "article lookup alone" in rendered
    assert "do not call `terminal`" in rendered
    assert 'include="*.mdx"' in rendered
    assert "`.` or `^`" in rendered
    assert "every relevant fact" in rendered
    assert "files you actually opened" in rendered
