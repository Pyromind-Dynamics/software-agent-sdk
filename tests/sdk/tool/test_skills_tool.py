from pathlib import Path
from types import SimpleNamespace

from openhands.sdk.skills import Skill, SkillResources, SkillRuntime
from openhands.sdk.tool.builtins import (
    BUILT_IN_TOOL_CLASSES,
    SkillsListAction,
    SkillsListTool,
    SkillsReadAction,
    SkillsReadTool,
)


def make_runtime(tmp_path: Path) -> SkillRuntime:
    root = tmp_path / "reader"
    (root / "references").mkdir(parents=True)
    (root / "SKILL.md").write_text("# Reader", encoding="utf-8")
    (root / "references" / "guide.md").write_text("guide", encoding="utf-8")
    return SkillRuntime(
        [
            Skill(
                name="reader",
                content="# Reader",
                description="Read workflow references",
                source=str(root / "SKILL.md"),
                is_agentskills_format=True,
                resources=SkillResources(
                    skill_root=str(root), references=["guide.md"]
                ),
            )
        ]
    )


def test_tools_are_registered_and_create_without_runtime_params():
    assert BUILT_IN_TOOL_CLASSES["SkillsListTool"] is SkillsListTool
    assert BUILT_IN_TOOL_CLASSES["SkillsReadTool"] is SkillsReadTool
    assert len(SkillsListTool.create()) == 1
    assert len(SkillsReadTool.create()) == 1


def make_conversation(runtime: SkillRuntime):
    skills = [entry.skill for entry in runtime.list()]
    return SimpleNamespace(
        state=SimpleNamespace(
            agent=SimpleNamespace(
                agent_context=SimpleNamespace(skills=skills),
            )
        )
    )


def test_list_and_read_tools_execute(tmp_path):
    runtime = make_runtime(tmp_path)
    conversation = make_conversation(runtime)
    (list_tool,) = SkillsListTool.create()
    (read_tool,) = SkillsReadTool.create()

    listed = list_tool(SkillsListAction(query="workflow"), conversation=conversation)
    assert listed.skills == ["reader"]

    read = read_tool(
        SkillsReadAction(skill_name="reader", path="references/guide.md"),
        conversation=conversation,
    )
    assert read.skill_name == "reader"
    assert read.path == "references/guide.md"
    assert read.contents == "guide"


def test_tools_declare_read_only_resources(tmp_path):
    make_runtime(tmp_path)
    (list_tool,) = SkillsListTool.create()
    (read_tool,) = SkillsReadTool.create()

    assert list_tool.declared_resources(SkillsListAction()).declared is True
    resource = read_tool.declared_resources(
        SkillsReadAction(skill_name="reader", path="references/guide.md")
    )
    assert resource.declared is True
    assert resource.keys == ("skill:reader:references/guide.md",)
