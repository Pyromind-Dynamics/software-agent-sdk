"""Three-table product version ledger; each unit of work owns its connection."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pyromind_runtime.domain.versions import (
    ArtifactVersion,
    Iteration,
    ProjectContext,
    RunSource,
    RunSpec,
    SaveVersion,
    Selection,
    SnapshotResult,
    VersionManifest,
    VersionProject,
    VersionRun,
    check_persistable,
    fingerprint,
    new_id,
)


SCHEMA = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL UNIQUE,
    owner_user_id TEXT NOT NULL, name TEXT NOT NULL,
    context_json TEXT NOT NULL CHECK(json_valid(context_json)),
    revision INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
    request_key TEXT NOT NULL, spec_json TEXT NOT NULL CHECK(json_valid(spec_json)),
    source_json TEXT NOT NULL CHECK(json_valid(source_json)), created_at INTEGER
    NOT NULL,
    UNIQUE(project_id, request_key), UNIQUE(project_id, id)
);
CREATE TABLE IF NOT EXISTS artifact_versions (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
    producer_run_id TEXT, kind TEXT NOT NULL, logical_key TEXT NOT NULL,
    version_no INTEGER NOT NULL CHECK(version_no > 0), registration_key TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('pending','ready','reference_only','failed')),
    manifest_json TEXT NOT NULL CHECK(json_valid(manifest_json)), created_at
    INTEGER NOT NULL,
    UNIQUE(project_id, kind, logical_key, version_no),
    UNIQUE(project_id, registration_key),
    FOREIGN KEY(project_id, producer_run_id) REFERENCES runs(project_id, id)
);
CREATE INDEX IF NOT EXISTS ix_runs_project ON runs(project_id, created_at);
CREATE INDEX IF NOT EXISTS ix_versions_project ON artifact_versions(project_id,
    kind, created_at);
CREATE INDEX IF NOT EXISTS ix_versions_producer ON artifact_versions(project_id,
    producer_run_id);
CREATE INDEX IF NOT EXISTS ix_versions_pending ON artifact_versions(state);
CREATE UNIQUE INDEX IF NOT EXISTS ix_runs_task ON runs(
    json_extract(source_json, '$.backend'), json_extract(source_json, '$.task_id')
) WHERE json_extract(source_json, '$.task_id') IS NOT NULL;
CREATE TRIGGER IF NOT EXISTS immutable_run BEFORE UPDATE ON runs
WHEN NEW.id IS NOT OLD.id OR NEW.project_id IS NOT OLD.project_id
 OR NEW.request_key IS NOT OLD.request_key OR NEW.spec_json IS NOT OLD.spec_json
 OR NEW.created_at IS NOT OLD.created_at
BEGIN SELECT RAISE(ABORT, 'run specification is immutable'); END;
CREATE TRIGGER IF NOT EXISTS immutable_version BEFORE UPDATE ON artifact_versions
WHEN OLD.state IN ('ready','reference_only')
 OR NEW.id IS NOT OLD.id OR NEW.project_id IS NOT OLD.project_id
 OR NEW.producer_run_id IS NOT OLD.producer_run_id OR NEW.kind IS NOT OLD.kind
 OR NEW.logical_key IS NOT OLD.logical_key OR NEW.version_no IS NOT OLD.version_no
 OR NEW.registration_key IS NOT OLD.registration_key OR NEW.created_at IS NOT
    OLD.created_at
BEGIN SELECT RAISE(ABORT, 'version identity or finalized content is immutable'); END;
CREATE TRIGGER IF NOT EXISTS retain_versions BEFORE DELETE ON artifact_versions
BEGIN SELECT RAISE(ABORT, 'version deletion is not supported'); END;
CREATE TRIGGER IF NOT EXISTS retain_runs BEFORE DELETE ON runs
BEGIN SELECT RAISE(ABORT, 'run deletion is not supported'); END;
PRAGMA user_version = 1;
COMMIT;
"""


def _project(row: sqlite3.Row) -> VersionProject:
    data = dict(row)
    data["context"] = ProjectContext.model_validate_json(data.pop("context_json"))
    return VersionProject.model_validate(data)


def _version(row: sqlite3.Row) -> ArtifactVersion:
    data = dict(row)
    data["manifest"] = VersionManifest.model_validate_json(data.pop("manifest_json"))
    return ArtifactVersion.model_validate(data)


def _run(row: sqlite3.Row) -> VersionRun:
    data = dict(row)
    data["spec"] = RunSpec.model_validate_json(data.pop("spec_json"))
    data["source"] = RunSource.model_validate_json(data.pop("source_json"))
    return VersionRun.model_validate(data)


class SQLiteVersionStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 1}:
                raise ValueError(f"unsupported version database schema: {version}")
            if version == 0:
                db.executescript(SCHEMA)

    @contextmanager
    def connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA busy_timeout=5000")
            db.execute("PRAGMA synchronous=FULL")
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if db.in_transaction:
                db.commit()
        except BaseException:
            if db.in_transaction:
                db.rollback()
            raise
        finally:
            db.close()

    def project(
        self, conversation_id: str, user_id: str, *, create: bool = False
    ) -> VersionProject | None:
        with self.connection(write=create) as db:
            row = db.execute(
                "SELECT * FROM projects WHERE conversation_id=?", (conversation_id,)
            ).fetchone()
            if row:
                project = _project(row)
                if project.owner_user_id != user_id:
                    raise PermissionError("project belongs to another user")
                return project
            if not create:
                return None
            iteration = Iteration()
            project = VersionProject(
                id=self._next_id(db, "p"),
                conversation_id=conversation_id,
                owner_user_id=user_id,
                context=ProjectContext(
                    current_iteration_id=iteration.id, iterations=[iteration]
                ),
            )
            db.execute(
                "INSERT INTO projects VALUES (?,?,?,?,?,?,?)",
                (
                    project.id,
                    conversation_id,
                    user_id,
                    project.name,
                    project.context.model_dump_json(by_alias=True),
                    0,
                    project.created_at,
                ),
            )
            return project

    @staticmethod
    def _next_id(db: sqlite3.Connection, prefix: Literal["p", "r", "v"]) -> str:
        # Call only inside BEGIN IMMEDIATE, before inserting the corresponding row.
        queries = {
            "p": "SELECT COALESCE(MAX(rowid), 0) + 1 FROM projects",
            "r": "SELECT COALESCE(MAX(rowid), 0) + 1 FROM runs",
            "v": "SELECT COALESCE(MAX(rowid), 0) + 1 FROM artifact_versions",
        }
        return f"{prefix}{db.execute(queries[prefix]).fetchone()[0]}"

    def bound_project(self, conversation_id: str) -> VersionProject | None:
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM projects WHERE conversation_id=?", (conversation_id,)
            ).fetchone()
            return _project(row) if row else None

    @staticmethod
    def _get_version(
        db: sqlite3.Connection, project_id: str, version_id: str
    ) -> ArtifactVersion:
        row = db.execute(
            "SELECT * FROM artifact_versions WHERE project_id=? AND id=?",
            (project_id, version_id),
        ).fetchone()
        if row is None:
            raise ValueError("unknown version in this project")
        return _version(row)

    def get_version(self, project_id: str, version_id: str) -> ArtifactVersion:
        with self.connection() as db:
            return self._get_version(db, project_id, version_id)

    def list_versions(self, project_id: str) -> list[ArtifactVersion]:
        with self.connection() as db:
            return [
                _version(row)
                for row in db.execute(
                    "SELECT * FROM artifact_versions WHERE project_id=? ORDER "
                    "BY created_at, version_no",
                    (project_id,),
                )
            ]

    def reserve_version(
        self, project: VersionProject, key: str, request: SaveVersion
    ) -> ArtifactVersion:
        value = request.model_dump(mode="json")
        check_persistable(value)
        digest = fingerprint(value)
        with self.connection(write=True) as db:
            row = db.execute(
                "SELECT * FROM artifact_versions WHERE project_id=? AND "
                "registration_key=?",
                (project.id, key),
            ).fetchone()
            if row:
                version = _version(row)
                if version.manifest.request_digest != digest:
                    raise ValueError(
                        "registration key already used with different content"
                    )
                return version
            for parent in request.parents:
                record = self._get_version(db, project.id, parent.version_id)
                if record.state not in {"ready", "reference_only"}:
                    raise ValueError("parent version is not finalized")
            for field in (
                "model_version_id",
                "dataset_version_id",
                "scoring_version_id",
                "inference_config_version_id",
            ):
                identity = request.summary.get(field)
                if identity is not None:
                    if not isinstance(identity, str):
                        raise ValueError("evaluation references must be version IDs")
                    self._get_version(db, project.id, identity)
            # Parents are finalized and immutable; a new node can only point
            # backwards to existing nodes, making cycles impossible.
            number = db.execute(
                "SELECT COALESCE(MAX(version_no),0)+1 FROM artifact_versions "
                "WHERE project_id=? AND kind=? AND logical_key=?",
                (project.id, request.kind, request.logical_key),
            ).fetchone()[0]
            identity = self._next_id(db, "v")
            version = ArtifactVersion(
                id=identity,
                project_id=project.id,
                producer_run_id=request.producer_run_id,
                kind=request.kind,
                logical_key=request.logical_key,
                version_no=number,
                registration_key=key,
                manifest=VersionManifest(
                    parents=request.parents,
                    output_role=request.output_role,
                    summary=request.summary,
                    source=request.source,
                    target=(
                        f"/.pyromind-agent/{project.conversation_id}/versions/"
                        f"{identity}/{new_id()}"
                    ),
                    request_digest=digest,
                ),
            )
            db.execute(
                "INSERT INTO artifact_versions VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    version.id,
                    project.id,
                    version.producer_run_id,
                    version.kind,
                    version.logical_key,
                    number,
                    key,
                    version.state,
                    version.manifest.model_dump_json(),
                    version.created_at,
                ),
            )
            return version

    def complete_version(
        self, project_id: str, version_id: str, result: SnapshotResult
    ) -> ArtifactVersion:
        check_persistable(result.model_dump(mode="json"))
        with self.connection(write=True) as db:
            version = self._get_version(db, project_id, version_id)
            if version.state in {"ready", "reference_only"}:
                if (version.manifest.storage_ref, version.manifest.digest) != (
                    result.storage_ref,
                    result.digest,
                ):
                    raise ValueError(
                        "snapshot completion differs from finalized version"
                    )
                return version
            if not result.reference_only and (
                not result.digest
                or not result.storage_ref.startswith(version.manifest.target + "/")
            ):
                raise ValueError(
                    "a ready version requires a digest and an isolated snapshot"
                )
            manifest = version.manifest.model_copy(
                update={
                    "storage_ref": result.storage_ref,
                    "manifest_ref": result.manifest_ref,
                    "digest": result.digest,
                    "error": None,
                    "source": version.manifest.source.model_copy(
                        update={"text": None, "storage_ref": result.storage_ref}
                    ),
                }
            )
            state = "reference_only" if result.reference_only else "ready"
            db.execute(
                "UPDATE artifact_versions SET state=?, manifest_json=? WHERE "
                "project_id=? AND id=?",
                (state, manifest.model_dump_json(), project_id, version_id),
            )
            return version.model_copy(update={"state": state, "manifest": manifest})

    def fail_version(self, project_id: str, version_id: str, error: str) -> None:
        with self.connection(write=True) as db:
            version = self._get_version(db, project_id, version_id)
            if version.state in {"ready", "reference_only"}:
                return
            manifest = version.manifest.model_copy(update={"error": error})
            db.execute(
                "UPDATE artifact_versions SET state='failed', manifest_json=? "
                "WHERE id=?",
                (manifest.model_dump_json(), version_id),
            )

    def select(
        self,
        project_id: str,
        slot: str,
        version_id: str,
        command_id: str,
        reason: str,
        revision: int,
    ) -> VersionProject:
        with self.connection(write=True) as db:
            project = self._get_project(db, project_id)
            for previous in project.context.selection_history:
                if previous.command_id == command_id:
                    if (previous.slot, previous.to, previous.reason) != (
                        slot,
                        version_id,
                        reason,
                    ):
                        raise ValueError(
                            "command id already used with different selection"
                        )
                    return project
            self._check_revision(project, revision)
            version = self._get_version(db, project_id, version_id)
            if version.state != "ready":
                raise ValueError("only a frozen, ready version can be selected")
            change = Selection.model_validate(
                {
                    "command_id": command_id,
                    "slot": slot,
                    "to": version_id,
                    "reason": reason,
                    "from": project.context.selected_versions.get(slot),
                }
            )
            context = project.context.model_copy(
                update={
                    "selected_versions": {
                        **project.context.selected_versions,
                        slot: version_id,
                    },
                    "selection_history": [*project.context.selection_history, change],
                }
            )
            return self._update_context(db, project, context)

    def new_iteration(
        self, project_id: str, goal: str, command_id: str, revision: int
    ) -> VersionProject:
        namespace = (
            uuid5(NAMESPACE_URL, project_id)
            if project_id.startswith("p")
            else UUID(project_id)
        )
        identity = str(uuid5(namespace, "iteration:" + command_id))
        with self.connection(write=True) as db:
            project = self._get_project(db, project_id)
            for item in project.context.iterations:
                if item.id == identity:
                    if item.goal != goal:
                        raise ValueError(
                            "command id already used with different iteration"
                        )
                    return project
            self._check_revision(project, revision)
            iteration = Iteration(
                id=identity, parent_id=project.context.current_iteration_id, goal=goal
            )
            return self._update_context(
                db,
                project,
                project.context.model_copy(
                    update={
                        "current_iteration_id": identity,
                        "iterations": [*project.context.iterations, iteration],
                    }
                ),
            )

    @staticmethod
    def _get_project(db: sqlite3.Connection, project_id: str) -> VersionProject:
        row = db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if row is None:
            raise ValueError("unknown project")
        return _project(row)

    @staticmethod
    def _check_revision(project: VersionProject, revision: int) -> None:
        if project.revision != revision:
            raise ValueError("project revision conflict; reload before retrying")

    @staticmethod
    def _update_context(
        db: sqlite3.Connection, project: VersionProject, context: ProjectContext
    ) -> VersionProject:
        check_persistable(context.model_dump(mode="json", by_alias=True))
        db.execute(
            "UPDATE projects SET context_json=?, revision=revision+1 WHERE id=? "
            "AND revision=?",
            (context.model_dump_json(by_alias=True), project.id, project.revision),
        )
        return project.model_copy(
            update={"context": context, "revision": project.revision + 1}
        )

    def reserve_run(
        self, project_id: str, key: str, spec: RunSpec, source: RunSource
    ) -> VersionRun:
        check_persistable(spec.model_dump(mode="json"))
        check_persistable(source.model_dump(mode="json"))
        with self.connection(write=True) as db:
            row = db.execute(
                "SELECT * FROM runs WHERE project_id=? AND request_key=?",
                (project_id, key),
            ).fetchone()
            if row:
                run = _run(row)
                if run.spec != spec or run.source.model_copy(
                    update={"task_id": None}
                ) != source.model_copy(update={"task_id": None}):
                    raise ValueError("request key already used with different run")
                return run
            project = self._get_project(db, project_id)
            if spec.iteration_id not in {
                item.id for item in project.context.iterations
            }:
                raise ValueError("unknown iteration in this project")
            for identity in spec.inputs.values():
                if self._get_version(db, project_id, identity).state not in {
                    "ready",
                    "reference_only",
                }:
                    raise ValueError("input version is not finalized")
            if (
                source.retry_of
                and not db.execute(
                    "SELECT 1 FROM runs WHERE project_id=? AND id=?",
                    (project_id, source.retry_of),
                ).fetchone()
            ):
                raise ValueError(
                    "retry source belongs to another project or is missing"
                )
            run = VersionRun(
                id=self._next_id(db, "r"),
                project_id=project_id,
                request_key=key,
                spec=spec,
                source=source,
            )
            db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?)",
                (
                    run.id,
                    project_id,
                    key,
                    spec.model_dump_json(),
                    source.model_dump_json(),
                    run.created_at,
                ),
            )
            return run

    def list_runs(self, project_id: str) -> list[VersionRun]:
        with self.connection() as db:
            return [
                _run(row)
                for row in db.execute(
                    "SELECT * FROM runs WHERE project_id=? ORDER BY created_at",
                    (project_id,),
                )
            ]

    def link_task(self, project_id: str, run_id: str, task_id: str) -> None:
        with self.connection(write=True) as db:
            row = db.execute(
                "SELECT * FROM runs WHERE project_id=? AND id=?", (project_id, run_id)
            ).fetchone()
            if row is None:
                raise ValueError("unknown run in this project")
            run = _run(row)
            if run.source.task_id not in {None, task_id}:
                raise ValueError("run already linked to another task")
            source = run.source.model_copy(update={"task_id": task_id})
            db.execute(
                "UPDATE runs SET source_json=? WHERE id=?",
                (source.model_dump_json(), run_id),
            )

    def recovery_scopes(self) -> list[str]:
        with self.connection() as db:
            return [
                row[0]
                for row in db.execute(
                    "SELECT conversation_id FROM projects WHERE id IN ("
                    "SELECT project_id FROM artifact_versions "
                    "WHERE state IN ('pending','failed') "
                    "UNION SELECT project_id FROM runs "
                    "WHERE json_extract(source_json, '$.task_id') IS NULL)"
                )
            ]

    def backup(self, destination: Path) -> None:
        with self.connection() as source, sqlite3.connect(destination) as target:
            source.backup(target)
