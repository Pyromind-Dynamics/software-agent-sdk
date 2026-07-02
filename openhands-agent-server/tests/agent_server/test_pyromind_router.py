"""Tests for the Pyromind router codex-alignment helpers."""

from __future__ import annotations

from openhands.agent_server.pyromind_router import (
    PYROMIND_KB_INSTRUCTIONS,
    _build_skills_prompt,
)


def test_build_skills_prompt_empty() -> None:
    assert _build_skills_prompt([]) == ""


def test_build_skills_prompt_is_codex_body_only() -> None:
    """The body must not carry the old Chinese header; the codex template owns
    the ``<SKILLS>`` envelope + ``invoke_skill`` wording."""
    skills = [
        {"name": "a", "content": "SKILL A BODY"},
        {"name": "b", "content": "SKILL B BODY"},
    ]
    body = _build_skills_prompt(skills)

    assert "SKILL A BODY" in body
    assert "SKILL B BODY" in body
    assert "---" in body
    # Old header must be gone; envelope belongs to the template now.
    assert "可用技能" not in body
    assert "<SKILLS>" not in body


def test_kb_instructions_format_injects_path() -> None:
    rendered = PYROMIND_KB_INSTRUCTIONS.format(knowledge_base_path="/kb/root")
    assert "/kb/root" in rendered
    assert "Pyromind" in rendered
    # Must not restrict the search to a single file extension.
    assert "single" in rendered
    assert ".mdx" in rendered
