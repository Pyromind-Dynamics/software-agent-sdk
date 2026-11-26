import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest
from pyromind_sdk.client.client import PyroMindAPIClient
from pyromind_sdk.client.models import (
    SandboxExecResponse,
    SandboxResponse,
    SandboxType,
    VolumeMount,
)
from pyromind_sdk.client.sandbox import SandboxClient

from pyromind_agent_server.api.echomind_projects import _EvaluationStorageReader
from pyromind_agent_server.echomind_batch_archive import archive_tracer_batch
from pyromind_agent_server.echomind_projects import (
    ProjectGraph,
    ProjectInfo,
    ProjectRecord,
    ProjectStore,
)
from pyromind_agent_server.echomind_sandbox_storage import (
    SandboxBindingVerifier,
    SandboxEvaluationStorage,
    build_acquisition_storage,
    build_bound_evaluation_storage,
    build_project_storage_bindings,
)
from pyromind_runtime.application.project_storage import ProjectStorageBindings
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.project_storage import (
    ProjectStorageBinding,
    StorageProxyConfig,
)
from pyromind_runtime.infrastructure.file_project_storage import (
    FileProjectStorageBindingStore,
)
from pyromind_runtime.ports.storage import (
    EvaluationStorageMissingError,
    EvaluationStorageUnavailableError,
)


@pytest.fixture
def storage_setup(tmp_path):
    mount = tmp_path / "mounted-storage"
    mount.mkdir()
    client = Mock(spec=SandboxClient)
    client.get_sandbox.return_value = SandboxResponse(
        id="sb-test",
        name="Storage",
        type=SandboxType.CUSTOM,
        status="Running",
        volume_mounts=[VolumeMount(host_path="/workspace", mount_path=str(mount))],
    )

    def execute(_sandbox, command, *, timeout):
        result = subprocess.run(
            [sys.executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return SandboxExecResponse(
            output=result.stdout, stderr=result.stderr, returncode=result.returncode
        )

    def write(_sandbox, path, source):
        target = Path(path)
        target.write_bytes(source)
        return {"path": path, "size": len(source)}

    client.exec_command.side_effect = execute
    client.read_file.side_effect = lambda _sandbox, path: Path(path).read_bytes()
    client.write_file.side_effect = write
    binding = ProjectStorageBinding(
        project_id=str(uuid4()),
        owner_user_id="owner",
        cluster="cn-east-1",
        config=StorageProxyConfig(sandbox_id="sb-test", mount_root=str(mount)),
        status="ready",
        last_checked_at=datetime.now(UTC),
    )
    service = ProjectStorageBindings(
        FileProjectStorageBindingStore(tmp_path / "projects"),
        lambda _project, _owner: SandboxBindingVerifier("cn-east-1", lambda: client),
    )
    return mount, client, binding, service


def test_binding_is_explicit_local_get_and_stopped_sandbox_is_blocked(storage_setup):
    _, client, binding, service = storage_setup
    project, owner = binding.project_id, binding.owner_user_id
    assert service.get(project, owner) is None
    client.get_sandbox.assert_not_called()
    assert service.bind(project, owner, binding.config).status == "ready"
    client.get_sandbox.return_value.status = "Stopped"
    client.get_sandbox.reset_mock()
    assert service.get(project, owner).status == "ready"
    client.get_sandbox.assert_not_called()
    state = service.refresh(project, owner)
    assert state.status == "blocked" and "not running" in state.last_error
    assert state.config.sandbox_id == "sb-test"
    client.get_sandbox.side_effect = RuntimeError("secret-key must not be persisted")
    state = service.refresh(project, owner)
    assert state.status == "blocked"
    assert "secret-key" not in state.model_dump_json()


def test_binding_owner_and_cluster_are_not_silently_replaced(storage_setup):
    _, client, binding, service = storage_setup
    service.bind(binding.project_id, binding.owner_user_id, binding.config)
    client.get_sandbox.reset_mock()
    with pytest.raises(PermissionError):
        service.refresh(binding.project_id, "another-owner")
    changed = ProjectStorageBindings(
        service.store,
        lambda _project, _owner: SandboxBindingVerifier("us-west-2", lambda: client),
    )
    state = changed.refresh(binding.project_id, binding.owner_user_id)
    assert state.status == "blocked" and state.cluster == "cn-east-1"
    client.get_sandbox.assert_not_called()


def test_storage_reads_same_mount_with_aliases_and_distinguishes_missing(storage_setup):
    mount, client, binding, _ = storage_setup
    (mount / "reports").mkdir()
    (mount / "reports/report.json").write_bytes(b'{"status":"succeeded"}')
    storage = SandboxEvaluationStorage(client=client, binding=binding)
    assert storage.read("/workspace/reports/report.json") == storage.read(
        "/reports/report.json"
    )
    assert storage.list("/reports") == {"report.json"}
    with pytest.raises(EvaluationStorageMissingError):
        storage.read("/reports/missing.json")
    client.exec_command.side_effect = RuntimeError("private authorization value")
    with pytest.raises(EvaluationStorageUnavailableError, match="metadata"):
        storage.list("/reports")


def test_storage_rejects_traversal_and_symlink_escape(storage_setup, tmp_path):
    mount, client, binding, _ = storage_setup
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"private")
    (mount / "escape").symlink_to(outside)
    storage = SandboxEvaluationStorage(client=client, binding=binding)
    with pytest.raises(ValueError):
        storage.read("/workspace/../outside.txt")
    with pytest.raises(EvaluationStorageUnavailableError, match="outside"):
        storage.read("/escape")
    client.read_file.assert_not_called()


def test_storage_checks_size_before_download_and_writes_only_batch_prefix(
    storage_setup,
):
    mount, client, binding, _ = storage_setup
    (mount / "large").write_bytes(b"x" * 20)
    prefix = "/workspace/.pyromind-agent/project/project-batches"
    storage = SandboxEvaluationStorage(
        client=client, binding=binding, write_prefixes=(prefix,), max_read_bytes=10
    )
    with pytest.raises(EvaluationStorageUnavailableError, match="read limit"):
        storage.read("/large")
    client.read_file.assert_not_called()
    with pytest.raises(EvaluationStorageUnavailableError, match="restricted"):
        storage.write("/workspace/source.jsonl", b"bad")
    client.write_file.assert_not_called()
    storage.write(prefix + "/em-test/0/train.jsonl", b"{}\n")
    assert storage.read(prefix + "/em-test/0/train.jsonl") == b"{}\n"
    assert client.write_file.call_count == 1


@pytest.mark.parametrize("stored", [b"\n", None, b"different"])
def test_storage_reconciles_upload_error_by_reading_target(storage_setup, stored):
    _, client, binding, _ = storage_setup
    blocked = Mock()
    prefix = "/workspace/.pyromind-agent/project/project-batches"
    path = prefix + "/em-test/0/val.jsonl"
    storage = SandboxEvaluationStorage(
        client=client, binding=binding, write_prefixes=(prefix,), on_unavailable=blocked
    )

    def write(_sandbox, mapped, _payload):
        if stored is not None:
            Path(mapped).write_bytes(stored)
        raise TimeoutError("upload acknowledgment missing")

    client.write_file.side_effect = write
    if stored == b"\n":
        storage.write(path, b"\n")
        blocked.assert_not_called()
    else:
        with pytest.raises(EvaluationStorageUnavailableError, match="unconfirmed"):
            storage.write(path, b"\n")
        blocked.assert_called_once()
    client.write_file.assert_called_once()


@pytest.mark.parametrize("mount_root", ["/", "/./", "/workspace/../etc", "relative"])
def test_invalid_mount_root_is_rejected(mount_root):
    with pytest.raises(ValueError):
        StorageProxyConfig(sandbox_id="sb-test", mount_root=mount_root)


@pytest.fixture
def archived_batch_setup(storage_setup):
    mount, client, binding, _ = storage_setup
    source = mount / "EchoMind" / "em-test" / "data" / "spans" / "2"
    (source / "media" / "images").mkdir(parents=True)
    (source / "batch.json").write_text(
        json.dumps(
            {
                "index": 2,
                "state": "ready",
                "collected": 5,
                "ready_at": "2026-11-18T11:30:00Z",
            }
        )
    )
    (source / "spans_example.parquet").write_bytes(b"traj")
    (source / "media" / "images" / "task.jpg").write_bytes(b"photo")
    relative = Path(f".pyromind-agent/{uuid4().hex}/project-batches")
    storage = SandboxEvaluationStorage(
        client=client, binding=binding, write_prefixes=(f"/workspace/{relative}",)
    )
    archive = relative / "archived" / "batch-00000002"
    return storage, source, mount / archive, f"/workspace/{archive}"


def test_batch_archive_verifies_complete_files_after_original_is_released(
    archived_batch_setup,
):
    storage, source, destination, uri = archived_batch_setup
    args = {"tracer_resource_id": "em-test", "batch_index": 2, "archive_directory": uri}
    cutoff = datetime(2026, 11, 18, 11, 30, tzinfo=UTC)
    archive_tracer_batch(storage, **args, sealed_before=cutoff)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["tracer_resource_id"] == "em-test"
    assert manifest["batch_index"] == 2
    assert set(manifest["files"]) == {
        "batch.json",
        "spans_example.parquet",
        "media/images/task.jpg",
    }
    assert manifest["files"]["media/images/task.jpg"] == {
        "size": 5,
        "sha256": hashlib.sha256(b"photo").hexdigest(),
    }
    assert (destination / "batch/media/images/task.jpg").read_bytes() == b"photo"
    assert (source / "spans_example.parquet").read_bytes() == b"traj"
    shutil.rmtree(source)
    archive_tracer_batch(storage, **args, sealed_before=cutoff)
    with pytest.raises(EvaluationStorageUnavailableError, match="archive verification"):
        archive_tracer_batch(
            storage,
            **args,
            sealed_before=datetime(2026, 11, 18, 11, 29, tzinfo=UTC),
        )
    (destination / "batch/media/images/task.jpg").write_bytes(b"other")
    with pytest.raises(EvaluationStorageUnavailableError, match="archive verification"):
        archive_tracer_batch(storage, **args)


@pytest.mark.parametrize("problem", ["collecting", "missing_shard", "link", "reused"])
def test_batch_archive_rejects_unverified_source(archived_batch_setup, problem):
    storage, source, destination, uri = archived_batch_setup
    args = {"tracer_resource_id": "em-test", "batch_index": 2, "archive_directory": uri}
    if problem == "collecting":
        (source / "batch.json").write_text(
            json.dumps({"index": 2, "state": "collecting", "collected": 1})
        )
    elif problem == "missing_shard":
        (source / "spans_example.parquet").unlink()
    elif problem == "link":
        (source / "media" / "link").symlink_to(source / "batch.json")
    else:
        archive_tracer_batch(storage, **args)
        (source / "spans_example.parquet").write_bytes(b"new batch at the same index")
    with pytest.raises(EvaluationStorageUnavailableError, match="archive verification"):
        archive_tracer_batch(storage, **args)
    assert destination.exists() == (problem == "reused")


def test_batch_archive_rejects_destination_outside_write_prefix(archived_batch_setup):
    storage, _, _, _ = archived_batch_setup
    different_project = f"/workspace/.pyromind-agent/{uuid4().hex}/project-batches/2"
    with pytest.raises(ValueError, match="Project's batch Storage"):
        archive_tracer_batch(
            storage,
            tracer_resource_id="em-test",
            batch_index=2,
            archive_directory=different_project,
        )


@pytest.mark.parametrize(
    "ready_at", ["2026-11-18T11:31:00Z", "2026-11-18T11:29:00", None]
)
def test_batch_archive_rejects_unverified_seal_time(archived_batch_setup, ready_at):
    storage, source, destination, uri = archived_batch_setup
    metadata = json.loads((source / "batch.json").read_text())
    if ready_at is None:
        metadata.pop("ready_at")
    else:
        metadata["ready_at"] = ready_at
    (source / "batch.json").write_text(json.dumps(metadata))
    with pytest.raises(EvaluationStorageUnavailableError, match="archive verification"):
        archive_tracer_batch(
            storage,
            tracer_resource_id="em-test",
            batch_index=2,
            archive_directory=uri,
            sealed_before=datetime(2026, 11, 18, 11, 30, tzinfo=UTC),
        )
    assert not destination.exists()


def test_development_storage_is_transient_read_only_and_defers_to_binding(
    storage_setup, tmp_path, monkeypatch
):
    mount, client, binding, service = storage_setup
    monkeypatch.setattr(
        "pyromind_agent_server.echomind_sandbox_storage.build_project_storage_bindings",
        lambda **kwargs: service,
    )

    def read_only_proxy(config):
        return build_bound_evaluation_storage(
            projects_root=tmp_path,
            project_id=binding.project_id,
            context=RequestContext(
                user_id=binding.owner_user_id, x_cluster=binding.cluster
            ),
            credential_store=None,
            development_proxy=config,
        )

    client.get_sandbox.return_value.volume_mounts[0].read_only = True
    storage = read_only_proxy(binding.config)
    assert storage is not None
    (mount / "record.json").write_bytes(b"{}")
    assert storage.read("/record.json") == b"{}"
    with pytest.raises(EvaluationStorageUnavailableError, match="restricted"):
        storage.write("/record.json", b"changed")
    assert service.get(binding.project_id, binding.owner_user_id) is None
    client.get_sandbox.return_value.status = "Stopped"
    with pytest.raises(EvaluationStorageUnavailableError, match="not running"):
        read_only_proxy(binding.config)
    assert service.get(binding.project_id, binding.owner_user_id) is None

    client.get_sandbox.return_value.status = "Running"
    client.get_sandbox.return_value.volume_mounts[0].read_only = False
    assert (
        service.bind(binding.project_id, binding.owner_user_id, binding.config).status
        == "ready"
    )
    storage = read_only_proxy(StorageProxyConfig(sandbox_id="sb-other"))
    assert isinstance(storage, SandboxEvaluationStorage)
    assert storage.binding.config == binding.config


def test_bound_factory_checks_scope_and_closes_owner_sdk_clients(
    storage_setup, tmp_path, monkeypatch
):
    mount, client, binding, _ = storage_setup
    root = tmp_path / "factory-projects"
    conversation = uuid4().hex
    ProjectStore(root).save(
        ProjectInfo(
            project=ProjectRecord(
                project_id=binding.project_id,
                name="Storage test",
                description="test",
                skill="test",
                conversation_id=conversation,
                owner_user_id="owner",
                cluster="cn-east-1",
            ),
            graph=ProjectGraph(
                graph_id=str(uuid4()), project_id=binding.project_id, nodes=[], edges=[]
            ),
        )
    )
    platform = MagicMock(spec=PyroMindAPIClient)
    platform.__enter__.return_value = platform
    platform.sandboxes = client
    credential_store = Mock()
    credential_store.load.return_value = "owner-access-key"
    factory = Mock(return_value=platform)
    monkeypatch.setattr(
        "pyromind_agent_server.echomind_sandbox_storage.get_pyromind_api_client",
        factory,
    )
    kwargs = {
        "projects_root": root,
        "project_id": binding.project_id,
        "context": RequestContext(user_id="owner", x_cluster="cn-east-1"),
        "credential_store": credential_store,
    }
    assert build_bound_evaluation_storage(**kwargs) is None
    factory.assert_not_called()
    sim = uuid4()
    browser = Mock()
    monkeypatch.setattr(
        "pyromind_agent_server.echomind_sandbox_storage.build_evaluation_storage",
        browser,
    )
    assert (
        build_acquisition_storage(**kwargs, simulation_id=str(sim))
        is browser.return_value
    )
    browser.assert_called_once_with(kwargs["context"])
    service = build_project_storage_bindings(**kwargs)
    assert service.bind(binding.project_id, "owner", binding.config).status == "ready"
    storage = build_bound_evaluation_storage(**kwargs)
    assert storage is not None
    (mount / "report.json").write_bytes(b"{}")
    assert storage.read("/report.json") == b"{}"
    collection = build_acquisition_storage(**kwargs, simulation_id=sim.hex)
    prefix = f"/.pyromind-agent/{conversation}/tracer-batches/{sim}"
    collection.write(f"{prefix}/0/train.jsonl", b'{"id":"1"}\n')
    source_reader = _EvaluationStorageReader(collection)
    assert source_reader.read(f"{prefix.lstrip('/')}/0/train.jsonl") == b'{"id":"1"}\n'
    for elsewhere in (
        f"/.pyromind-agent/{conversation}/tracer-batches/{uuid4()}",
        f"/.pyromind-agent/{uuid4().hex}/tracer-batches/{sim}",
        f"/.pyromind-agent/{conversation}/project-batches",
    ):
        with pytest.raises(EvaluationStorageUnavailableError, match="restricted"):
            collection.write(f"{elsewhere}/0/train.jsonl", b"bad")
    (mount / prefix.lstrip("/") / "other").symlink_to(mount, target_is_directory=True)
    with pytest.raises(EvaluationStorageUnavailableError):
        collection.write(f"{prefix}/other/report.json", b"bad")
    assert (mount / "report.json").read_bytes() == b"{}"
    assert platform.__exit__.call_count == factory.call_count
    client.get_sandbox.return_value.status = "Stopped"
    with pytest.raises(EvaluationStorageUnavailableError, match="not running"):
        storage.read("/report.json")
    stopped = service.get(binding.project_id, "owner")
    assert stopped is not None and stopped.status == "blocked"
    with pytest.raises(ValueError, match="write prefix"):
        build_bound_evaluation_storage(
            **kwargs, write_prefixes=("/workspace/EchoMind",)
        )
