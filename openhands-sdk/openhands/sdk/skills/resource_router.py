from __future__ import annotations

from pathlib import Path

from openhands.sdk.skills.catalog import (
    SkillCatalogEntry,
    SkillCatalogError,
    SkillResourceHandle,
)


class SkillResourceRouter:
    """Resolve skill-local resources without exposing workspace scanning."""

    def read(self, entry: SkillCatalogEntry, relative_path: str) -> SkillResourceHandle:
        root = entry.resource_root
        if root is None:
            raise SkillCatalogError(f"skill {entry.name} has no resource root")

        skill_root = Path(root).expanduser().resolve()
        candidate = (skill_root / relative_path).resolve()
        if skill_root not in candidate.parents and candidate != skill_root:
            raise SkillCatalogError(f"path escapes skill root: {relative_path}")
        if not candidate.is_file():
            raise SkillCatalogError(relative_path)
        return SkillResourceHandle(
            entry=entry,
            relative_path=relative_path,
            absolute_path=str(candidate),
            contents=candidate.read_text(encoding="utf-8"),
        )
