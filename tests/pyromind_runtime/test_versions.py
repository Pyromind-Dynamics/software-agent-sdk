import sqlite3
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4, uuid5

import pytest
from pyromind_runtime.application.versions import Versions, snapshot_result
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.versions import (
    RunSource,
    RunSpec,
    SaveVersion,
    SnapshotSource,
    VersionParent,
    VersionToolRequest,
)
from pyromind_runtime.infrastructure.file_artifact_snapshots import (
    FileArtifactSnapshots,
)
from pyromind_runtime.infrastructure.sqlite_version_store import SQLiteVersionStore


@pytest.fixture
def versions(tmp_path):
    store = SQLiteVersionStore(tmp_path / "versions.sqlite3")
    snapshots = FileArtifactSnapshots(tmp_path / "storage")
    service = Versions(store, snapshots)
    context = RequestContext(user_id="42", cookie="never-persist-this-cookie")
    project = service.project("conversation-a", context)
    return service, snapshots, project, context


def code_request(text="print(1)"):
    return SaveVersion(
        kind="code",
        logical_key="train",
        source=SnapshotSource(text=text, filename="train.py"),
    )


def test_three_tables_lazy_project_and_idempotent_save(versions):
    service, _, project, context = versions
    assert service.project("pure-chat", context, create=False) is None
    first = service.save(project, "save-1", code_request(), context)
    replay = service.save(project, "save-1", code_request(), context)
    assert project.id == "p1"
    assert first.id == "v1"
    assert first.id == replay.id
    with pytest.raises(ValueError, match="different content"):
        service.save(project, "save-1", code_request("changed"), context)
    with service.store.connection() as db:
        assert {
            r[0]
            for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        } == {"projects", "runs", "artifact_versions"}
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert b"never-persist-this-cookie" not in service.store.path.read_bytes()


def test_concurrent_numbers_and_same_key(versions):
    service, _, project, _ = versions

    def reserve(index):
        return service.store.reserve_version(
            project, f"save-{index}", code_request(str(index))
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(reserve, range(16)))
    assert sorted(v.version_no for v in rows) == list(range(1, 17))
    assert {v.id for v in rows} == {f"v{i}" for i in range(1, 17)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda _: reserve(0), range(16)))
    assert len({v.id for v in rows}) == 1


def test_project_and_run_ids_are_concurrent_and_survive_restart(versions):
    service, _, project, context = versions
    store = service.store

    def create_project(index):
        created = store.project(f"conversation-{index}", context.user_id, create=True)
        assert created is not None
        return created

    def reserve_run(index):
        return store.reserve_run(
            project.id,
            f"submit-{index}",
            RunSpec(kind="train", iteration_id=project.context.current_iteration_id),
            RunSource(kind="workflow", source_id=f"source-{index}"),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        projects = list(pool.map(create_project, range(16)))
        runs = list(pool.map(reserve_run, range(16)))
    assert {p.id for p in projects} == {f"p{i}" for i in range(2, 18)}
    assert {r.id for r in runs} == {f"r{i}" for i in range(1, 17)}
    store = SQLiteVersionStore(store.path)
    assert create_project(0) == projects[0]
    assert reserve_run(0) == runs[0]
    assert create_project(16).id == "p18"
    assert reserve_run(16).id == "r17"


def test_short_ids_preserve_legacy_uuid_project_and_iteration(versions):
    service, _, project, context = versions
    legacy_id = str(uuid4())
    with service.store.connection(write=True) as db:
        db.execute("UPDATE projects SET id=? WHERE id=?", (legacy_id, project.id))
    legacy = service.project(project.conversation_id, context)
    assert legacy.id == legacy_id
    changed = service.store.new_iteration(legacy_id, "improve", "next", 0)
    assert changed.context.current_iteration_id == str(
        uuid5(UUID(legacy_id), "iteration:next")
    )
    assert service.store.new_iteration(legacy_id, "improve", "next", 0) == changed
    version = service.save(legacy, "save", code_request(), context)
    assert version.id == "v1"
    assert version.project_id == legacy_id
    assert service.project("new-conversation", context).id == "p2"


def test_database_reset_does_not_reuse_old_snapshot(versions):
    service, snapshots, project, context = versions
    old = service.save(project, "save", code_request("old"), context)
    service.store.path.unlink()
    restarted = Versions(SQLiteVersionStore(service.store.path), snapshots)
    recreated = restarted.project(project.conversation_id, context)
    assert recreated is not None
    new = restarted.save(recreated, "save", code_request("new"), context)
    assert new.id == old.id == "v1"
    assert new.manifest.target != old.manifest.target
    assert snapshots.read_text(snapshot_result(new), context) == "new"
    assert snapshots.read_text(snapshot_result(old), context) == "old"


def test_snapshot_survives_source_change_restart_and_database_failure(
    versions, monkeypatch
):
    service, snapshots, project, context = versions
    source = snapshots.path("/scripts/train.py", context)
    source.parent.mkdir(parents=True)
    source.write_text("original")
    request = SaveVersion(
        kind="code",
        logical_key="train",
        source=SnapshotSource(storage_ref="/scripts/train.py"),
    )
    original = service.store.complete_version
    monkeypatch.setattr(
        service.store,
        "complete_version",
        lambda *args: (_ for _ in ()).throw(OSError("disk unavailable")),
    )
    with pytest.raises(ValueError, match="retry"):
        service.save(project, "save", request, context)
    source.write_text("changed after snapshot")
    monkeypatch.setattr(service.store, "complete_version", original)
    restarted = Versions(SQLiteVersionStore(service.store.path), snapshots)
    row = restarted.save(project, "save", request, context)
    assert row.state == "ready"
    assert snapshots.read_text(snapshot_result(row), context) == "original"
    assert len(service.store.list_versions(project.id)) == 1


def test_failed_snapshot_retains_number_and_can_recover(versions):
    service, snapshots, project, context = versions
    request = SaveVersion(
        kind="code",
        logical_key="train",
        source=SnapshotSource(storage_ref="/missing.py"),
    )
    with pytest.raises(ValueError, match="snapshot not completed"):
        service.save(project, "missing", request, context)
    second = service.save(project, "next", code_request(), context)
    assert second.version_no == 2
    snapshots.path("/missing.py", context).write_text("recovered")
    assert service.recover(project, context)[0]["state"] == "ready"


def test_cross_project_references_and_sql_producer_fk(versions):
    service, _, project, context = versions
    version = service.save(project, "save", code_request(), context)
    other = service.project("other", context)
    assert other is not None
    with pytest.raises(ValueError, match="unknown version"):
        service.store.reserve_version(
            other,
            "bad",
            code_request().model_copy(
                update={"parents": [VersionParent(version_id=version.id)]}
            ),
        )
    with pytest.raises(ValueError, match="unknown version"):
        service.store.reserve_run(
            other.id,
            "run",
            RunSpec(
                kind="train",
                iteration_id=other.context.current_iteration_id,
                inputs={"code": version.id},
            ),
            RunSource(kind="workflow", source_id="source"),
        )
    run = service.store.reserve_run(
        project.id,
        "run",
        RunSpec(kind="train", iteration_id=project.context.current_iteration_id),
        RunSource(kind="workflow", source_id="source"),
    )
    with pytest.raises(sqlite3.IntegrityError):
        service.store.reserve_version(
            other,
            "bad-producer",
            code_request().model_copy(update={"producer_run_id": run.id}),
        )
    with pytest.raises(PermissionError):
        service.project("conversation-a", RequestContext(user_id="someone-else"))


def test_final_versions_and_run_specs_are_immutable(versions):
    service, _, project, context = versions
    version = service.save(project, "save", code_request(), context)
    run = service.store.reserve_run(
        project.id,
        "run",
        RunSpec(
            kind="train",
            iteration_id=project.context.current_iteration_id,
            inputs={"code": version.id},
        ),
        RunSource(kind="workflow", source_id="source"),
    )
    with service.store.connection(write=True) as db:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute(
                "UPDATE artifact_versions SET manifest_json='{}' WHERE id=?",
                (version.id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute("UPDATE runs SET spec_json='{}' WHERE id=?", (run.id,))
    service.store.link_task(project.id, run.id, "task")
    with pytest.raises(ValueError, match="another task"):
        service.store.link_task(project.id, run.id, "other-task")


def test_selection_cas_replay_and_existing_run_stays_frozen(versions):
    service, _, project, context = versions
    first = service.save(project, "first", code_request(), context)
    second = service.save(project, "second", code_request("print(2)"), context)
    selected = service.store.select(project.id, "code", first.id, "s1", "baseline", 0)
    run = service.prepare_run(
        project.conversation_id,
        context,
        "submit",
        RunSource(kind="workflow", source_id="request"),
        {"script_path": "version://code"},
    )
    service.store.select(
        project.id, "code", second.id, "s2", "improve", selected.revision
    )
    assert first.id in run.spec.inputs.values()
    assert second.id not in run.spec.inputs.values()
    assert (
        service.store.select(project.id, "code", first.id, "s1", "baseline", 0).revision
        == 2
    )
    with pytest.raises(ValueError, match="revision conflict"):
        service.store.select(project.id, "code", first.id, "s3", "", 0)
    with pytest.raises(ValueError, match="different selection"):
        service.store.select(project.id, "code", second.id, "s1", "baseline", 0)


def test_missing_dependencies_are_reference_only_and_not_restorable(versions):
    service, snapshots, project, context = versions
    path = snapshots.path("/data.jsonl", context)
    path.parent.mkdir(parents=True)
    path.write_text('{"image":"/mutable/images/a.png"}')
    version = service.save(
        project,
        "dataset",
        SaveVersion(
            kind="dataset",
            logical_key="training",
            source=SnapshotSource(storage_ref="/data.jsonl"),
        ),
        context,
    )
    assert version.state == "reference_only"
    with pytest.raises(ValueError, match="reference-only"):
        service.call(
            project.conversation_id,
            VersionToolRequest(
                action="select",
                arguments={
                    "slot": "training_data",
                    "version_id": version.id,
                    "expected_revision": 0,
                },
            ),
            "select",
            context,
        )


def test_snapshot_integrity_and_path_isolation(versions, tmp_path):
    service, snapshots, project, context = versions
    version = service.save(project, "save", code_request(), context)
    path = snapshots.path(version.manifest.storage_ref, context)
    path.write_text("tampered")
    with pytest.raises(ValueError, match="digest changed"):
        snapshots.verify(snapshot_result(version), context)
    with pytest.raises(ValueError, match="Storage path"):
        snapshots.path("/../../secret", context)
    symlink = snapshots.path("/escape", context)
    symlink.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="namespace|symbolic"):
        snapshots.path("/escape/secret", context)


def test_outputs_dedupe_and_resume_guard(versions):
    service, _, project, context = versions
    run = service.prepare_run(
        project.conversation_id,
        context,
        "submit",
        RunSource(kind="pipeline", source_id="pipeline"),
        {"cpu": 1},
    )
    output = SaveVersion(
        kind="report",
        logical_key="report",
        source=SnapshotSource(text='{"score":0.8}', filename="report.json"),
    )
    pending = service.register_outputs(project, run, [output], None)
    assert pending[0].state == "pending"
    with pytest.raises(ValueError, match="before resume"):
        service.guard_resume(project, "pipeline")
    finished = service.register_outputs(project, run, [output], context)
    assert finished[0].id == pending[0].id
    assert (
        service.register_outputs(project, run, [output], context)[0].id == pending[0].id
    )
    service.guard_resume(project, "pipeline")
    assert finished[0].manifest.parents[0].version_id in run.spec.inputs.values()


def test_comparison_iteration_idempotency_and_backup(versions, tmp_path):
    service, _, project, context = versions
    first = service.save(project, "a", code_request("old\n"), context)
    second = service.save(project, "b", code_request("new\n"), context)
    assert "+new" in service.compare(project.id, first.id, second.id, context)["diff"]
    changed = service.store.new_iteration(project.id, "improve", "iteration", 0)
    assert service.store.new_iteration(project.id, "improve", "iteration", 0) == changed
    with pytest.raises(ValueError, match="different iteration"):
        service.store.new_iteration(project.id, "other", "iteration", 0)
    backup = tmp_path / "backup.sqlite3"
    service.store.backup(backup)
    assert len(SQLiteVersionStore(backup).list_versions(project.id)) == 2


def test_credentials_rejected_before_any_version_write(versions):
    service, _, project, context = versions
    with pytest.raises(ValueError, match="credential-bearing"):
        service.save(
            project,
            "secret",
            code_request().model_copy(update={"summary": {"authorization": "secret"}}),
            context,
        )
    assert service.store.list_versions(project.id) == []
