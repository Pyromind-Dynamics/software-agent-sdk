from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel, Field

from openhands.sdk.skills import SkillRuntime
from openhands.sdk.tool.tool import (
    Action,
    DeclaredResources,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)


class SkillsListAction(Action):
    query: str | None = Field(default=None)
    limit: int = Field(default=5, ge=1, le=20)


class SkillsListObservation(Observation):
    skills: list[str] = Field(default_factory=list)


class SkillsReadAction(Action):
    skill_name: str
    path: str = Field(default="SKILL.md")


class SkillsReadObservation(Observation):
    skill_name: str = Field()
    path: str = Field()
    contents: str = Field()


class SkillsListExecutor(ToolExecutor):
    def __init__(self, runtime: SkillRuntime) -> None:
        self.runtime = runtime

    def __call__(self, action: SkillsListAction, conversation=None) -> SkillsListObservation:
        if action.query:
            selection = self.runtime.select(action.query, limit=action.limit)
            return SkillsListObservation(
                skills=[entry.name for entry in selection.candidate_entries]
            )
        return SkillsListObservation(skills=[entry.name for entry in self.runtime.list()])


class SkillsReadExecutor(ToolExecutor):
    def __init__(self, runtime: SkillRuntime) -> None:
        self.runtime = runtime

    def __call__(self, action: SkillsReadAction, conversation=None) -> SkillsReadObservation:
        result = self.runtime.read(action.skill_name, action.path)
        return SkillsReadObservation(
            skill_name=result.entry.name,
            path=result.handle.relative_path,
            contents=result.handle.contents,
        )


class SkillsListTool(ToolDefinition[SkillsListAction, SkillsListObservation]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence[Self]:
        runtime = params.pop("runtime", None)
        if params:
            raise ValueError("SkillsListTool doesn't accept parameters")
        if runtime is None:
            raise ValueError("SkillsListTool requires runtime")
        return [
            cls(
                action_type=SkillsListAction,
                observation_type=SkillsListObservation,
                description="List available skills, optionally filtering by query.",
                executor=SkillsListExecutor(runtime),
                annotations=ToolAnnotations(
                    title="skills_list",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


class SkillsReadTool(ToolDefinition[SkillsReadAction, SkillsReadObservation]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence[Self]:
        runtime = params.pop("runtime", None)
        if params:
            raise ValueError("SkillsReadTool doesn't accept parameters")
        if runtime is None:
            raise ValueError("SkillsReadTool requires runtime")
        return [
            cls(
                action_type=SkillsReadAction,
                observation_type=SkillsReadObservation,
                description="Read a skill document or bundled resource by skill name and relative path.",
                executor=SkillsReadExecutor(runtime),
                annotations=ToolAnnotations(
                    title="skills_read",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]
