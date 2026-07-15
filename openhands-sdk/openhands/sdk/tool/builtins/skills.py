from __future__ import annotations

from dataclasses import dataclass

from openhands.sdk.skills import SkillRuntime


@dataclass(frozen=True)
class SkillsListObservation:
    skills: list[str]


@dataclass(frozen=True)
class SkillsReadObservation:
    skill_name: str
    path: str
    contents: str


class SkillsBuiltinFacade:
    def __init__(self, runtime: SkillRuntime) -> None:
        self.runtime = runtime

    def list(self) -> SkillsListObservation:
        return SkillsListObservation(skills=[entry.name for entry in self.runtime.list()])

    def read(self, skill_name: str, path: str = "SKILL.md") -> SkillsReadObservation:
        result = self.runtime.read(skill_name, path)
        return SkillsReadObservation(
            skill_name=result.entry.name,
            path=result.handle.relative_path,
            contents=result.handle.contents,
        )
