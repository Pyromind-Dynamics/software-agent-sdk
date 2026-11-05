"""Verified Storage files flow from a data task into training inputs."""

import hashlib
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path, PurePosixPath

import httpx
import pytest
from pyromind_runtime.domain.experiment_execution import RequestExperimentTask
from pyromind_runtime.infrastructure.sql_work_queue import StaleLeaseError, now, work

from openhands.tools.workflow import (
    ValidateWorkflowDslAction,
    ValidateWorkflowDslExecutor,
)

from . import test_standalone_experiments as fixtures


experiment = fixtures.experiment


@pytest.fixture
def delivery(experiment):
    service, owner, engine = experiment
    run = service.request(
        owner,
        RequestExperimentTask.model_validate(
            {
                "role_id": "cleaner",
                "request_id": "prepare-data",
                "task": {
                    "name": "Prepare training files",
                    "instruction": "Deliver messages JSONL and statistics",
                    "expected_outputs": ["train_data", "val_data", "data_stats"],
                },
            }
        ),
        ("cleaner",),
        execution_policy={
            "submission_tool": "df_submit_pipeline",
            "artifact_contract": "data",
        },
    )
    scope = fixtures.claim(service, owner, run)
    images = [
        f"/workspace/datasets/images/sample/{name}.png"
        for name in ("inspection", "difference", "reference")
    ]
    row = {
        "messages": [
            {"role": "system", "content": "Compare the three images in order."},
            {
                "role": "user",
                "content": [{"type": "image", "path": image} for image in images]
                + [{"type": "text", "text": "Inspect the first image."}],
            },
            {"role": "assistant", "content": '{"result":"pass"}'},
        ]
    }
    files = {
        "/datasets/clean/train.jsonl": (json.dumps(row) + "\n").encode(),
        "/datasets/clean/val.jsonl": (json.dumps(row) + "\n").encode(),
        "/datasets/clean/stats.json": b'{"train":1,"val":1}\n',
    }
    arguments = {
        "run_id": run["id"],
        "summary": "Delivered training, validation and statistics files.",
        "artifacts": [
            {"name": name, "uri": uri}
            for name, uri in zip(
                ("train_data", "val_data", "data_stats"), files, strict=True
            )
        ],
    }

    async def read(_owner, path):
        assert _owner == owner
        return files[path]

    async def inspect(_owner, path, kind):
        assert _owner == owner and kind == "file"
        body = files[path]
        if not body:
            raise ValueError("file is empty")
        return {
            "sha256": hashlib.sha256(body).hexdigest(),
            "scope": "complete file",
            "nonempty_lines": len(body.splitlines()),
        }

    async def execute(*_):
        raise AssertionError("file delivery must not start platform work")

    async def record(document=None, active_scope=None, active_owner=None):
        return await service.role_call(
            active_owner or owner,
            "cleaner",
            "record_experiment_result",
            document or arguments,
            active_scope or scope,
            execute,
            read,
            inspect,
        )

    return service, owner, engine, run, scope, files, arguments, record


@pytest.mark.parametrize("platform", [False, True])
async def test_data_files_complete_without_mandatory_report_and_bind_to_train(
    delivery, platform, tmp_path, monkeypatch
):
    service, owner, _, run, scope, files, arguments, record = delivery
    if platform:
        service.store.managed_submission(
            owner, run["id"], scope, {"mode": "full", "operation": "pipeline"}
        )
        service.store.managed_receipt(
            owner,
            run["id"],
            scope,
            "pipeline",
            {"task_id": "data-task", "output_dir": "/pipeline/unrelated"},
        )
        service.store.ingest(
            "data-complete", "data-task", "callback", {"status": "succeeded"}
        )
        service.store.apply_observations()
    assert await record() == {"run_id": run["id"], "report_state": "ready"}
    first = service.list(owner)[0]
    assert (first.state, first.report_state, first.report_uri) == (
        "succeeded",
        "ready",
        None,
    )
    assert {item["name"] for item in first.artifacts} == {
        "train_data",
        "val_data",
        "data_stats",
    }
    await record()
    assert service.list(owner)[0] == first
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
    training = service.request(
        owner,
        RequestExperimentTask.model_validate(
            {
                "role_id": "trainer",
                "request_id": "train",
                "task": {
                    "name": "Train",
                    "instruction": "Use the delivered messages files",
                    "artifact_refs": [
                        {
                            "input_name": input_name,
                            "task_run_id": run["id"],
                            "artifact_name": name,
                        }
                        for input_name, name in (
                            ("train_data_path", "train_data"),
                            ("val_data_path", "val_data"),
                        )
                    ],
                },
            }
        ),
        ("trainer",),
        execution_policy={
            "submission_tool": "submit_workflow",
            "artifact_contract": "model",
        },
    )
    inputs = training["task_spec"]["task"]["inputs"]
    for item in arguments["artifacts"][:2]:
        input_name = item["name"] + "_path"
        assert inputs[input_name] == item["uri"]
        assert (
            training["task_spec"]["input_artifacts"][input_name]["sha256"]
            == hashlib.sha256(files[item["uri"]]).hexdigest()
        )
    source = tmp_path / "messages.jsonl"
    source.write_bytes(files[inputs["train_data_path"]])
    skill = (
        Path(__file__).resolve().parents[2]
        / ".agents/skills/data-processing/scripts/cleaning"
    )
    validation_report = tmp_path / "validation.json"
    subprocess.run(
        [
            sys.executable,
            str(skill / "validate_format.py"),
            "--input",
            str(source),
            "--report",
            str(validation_report),
        ],
        check=True,
        capture_output=True,
    )
    assert json.loads(validation_report.read_text())["validation"]["status"] == "passed"
    content = json.loads(source.read_text())["messages"][1]["content"]
    assert [PurePosixPath(part["path"]).stem for part in content[:3]] == [
        "inspection",
        "difference",
        "reference",
    ]
    assert all(part["path"].startswith("/workspace/") for part in content[:3])
    dsl = (
        "kind = DatasetConfigBuilderMessageNode(id='kind', "
        "messages_field='messages')\n"
        "train = PathJoinNode(id='train', base_path='/workspace/', "
        f"subpath={inputs['train_data_path'].lstrip('/')!r})\n"
        "val = PathJoinNode(id='val', base_path='/workspace/', "
        f"subpath={inputs['val_data_path'].lstrip('/')!r})\n"
        "config = DatasetConfigBuilderNode(id='config', "
        "train_data_path=train.joined_path, val_data_path=val.joined_path, "
        "dataset_kind_config=kind.dataset_kind_config)\n"
    )
    received = []

    def validate_request(_url, **kwargs):
        received.append(kwargs["json"]["dsl"])
        return httpx.Response(200, json={"success": True, "data": {"valid": True}})

    monkeypatch.setattr(httpx, "post", validate_request)
    observation = ValidateWorkflowDslExecutor()(ValidateWorkflowDslAction(dsl=dsl))
    assert observation.valid and received == [dsl]
    assert [
        submission["task_id"]
        for submission in service.store.list(owner)[0]["submissions"]
    ] == (["data-task"] if platform else [])


@pytest.mark.parametrize(
    "problem",
    ["missing", "empty", "duplicate", "relative", "directory", "undeclared"],
)
async def test_invalid_delivery_is_visible_and_never_partially_registers(
    delivery, problem
):
    service, owner, _, run, _, files, arguments, record = delivery
    if problem == "missing":
        del files[arguments["artifacts"][1]["uri"]]
    elif problem == "empty":
        files[arguments["artifacts"][1]["uri"]] = b""
    elif problem == "duplicate":
        arguments["artifacts"][1]["name"] = "train_data"
    elif problem == "relative":
        arguments["artifacts"][1]["uri"] = "public_data/val.jsonl"
    elif problem == "directory":
        arguments["artifacts"][1]["kind"] = "directory"
    else:
        arguments["artifacts"] = []
    with pytest.raises(ValueError):
        await record()
    view = service.list(owner)[0]
    assert view.state == "preparing" and not view.artifacts
    assert view.waiting_reason
    if problem in {"missing", "empty", "relative", "directory"}:
        assert "val_data" in view.waiting_reason
    service.store.note_wait(owner, run["id"], "")
    assert service.list(owner)[0].waiting_reason == view.waiting_reason


async def test_report_failure_keeps_exact_error_and_can_retry_without_submission(
    delivery,
):
    service, owner, _, run, scope, files, arguments, record = delivery
    uri = arguments["artifacts"][1]["uri"]
    original = files.pop(uri)
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
    service.store.enqueue_report(owner, run["id"], command_id="inspect")
    item = service.store.queue.claim(kinds=("generate_report",))
    report_scope = {
        "run_id": run["id"],
        "work_id": item["id"],
        "generation": item["generation"],
    }
    with pytest.raises(ValueError, match="val_data"):
        await record(active_scope=report_scope)
    service.store.complete_agent_turn(
        owner, run["id"], report_scope, "Could not finish delivery.", report=True
    )
    failed = service.list(owner)[0]
    assert failed.report_state == "failed"
    assert uri in failed.waiting_reason and "val_data" in failed.summary
    assert service.store.pending_deliveries()
    service.store.queue.finish(item["id"], item["generation"], {})
    service.store.enqueue_report(owner, run["id"], command_id="retry")
    item = service.store.queue.claim(kinds=("generate_report",))
    report_scope.update(work_id=item["id"], generation=item["generation"])
    files[uri] = original
    await record(active_scope=report_scope)
    completed = service.list(owner)[0]
    assert completed.state == "succeeded" and completed.report_state == "ready"
    assert completed.waiting_reason is None
    assert not service.store.list(owner)[0]["submissions"]


async def test_changed_artifact_rolls_back_other_files_and_preserves_success(delivery):
    service, owner, _, _, _, files, arguments, record = delivery
    await record()
    original = service.list(owner)[0]
    files["/datasets/clean/extra.json"] = b"{}\n"
    files["/datasets/clean/val.jsonl"] += b'{"changed":true}\n'
    arguments["artifacts"].insert(
        0, {"name": "extra", "uri": "/datasets/clean/extra.json"}
    )
    with pytest.raises(ValueError, match="registered artifact changed: 'val_data'"):
        await record()
    assert service.list(owner)[0] == original


async def test_declarations_alone_do_not_complete_data_task(delivery):
    service, owner, _, run, scope, _, arguments, _ = delivery
    with pytest.raises(ValueError, match="every declared artifact must be verified"):
        service.store.record_result(owner, run["id"], scope, arguments, None)
    view = service.list(owner)[0]
    assert view.state != "succeeded" and not view.artifacts


async def test_report_conflict_rolls_back_files_and_success_together(delivery):
    service, owner, _, run, _, files, arguments, record = delivery
    report_uri = f"public_data/experiment-runs/{run['id']}/report.md"
    files[report_uri] = b"Delivery report."
    arguments["report_uri"] = report_uri
    arguments["artifacts"][0]["name"] = "agent_report"
    with pytest.raises(ValueError, match="registered artifact changed"):
        await record()
    view = service.list(owner)[0]
    assert view.state != "succeeded" and not view.artifacts


async def test_historical_data_name_remains_a_registered_reference(delivery):
    service, owner, _, _, _, _, arguments, record = delivery
    arguments["artifacts"] = [{**arguments["artifacts"][0], "name": "data"}]
    await record()
    assert service.list(owner)[0].artifacts[0]["name"] == "data"


@pytest.mark.parametrize("field", ["profile_id", "conversation_id"])
async def test_upstream_files_cannot_bind_across_owners(delivery, field):
    service, owner, _, run, _, _, _, record = delivery
    await record()
    with pytest.raises(ValueError, match="upstream task"):
        service.request(
            owner.model_copy(update={field: "another"}),
            RequestExperimentTask.model_validate(
                {
                    "role_id": "trainer",
                    "request_id": "foreign-training",
                    "task": {
                        "name": "Train",
                        "instruction": "Use another owner's files",
                        "artifact_refs": [
                            {
                                "input_name": "train_data_path",
                                "task_run_id": run["id"],
                                "artifact_name": "train_data",
                            }
                        ],
                    },
                }
            ),
            ("trainer",),
        )


@pytest.mark.parametrize("state", ["waiting_platform", "submission_uncertain"])
async def test_inflight_platform_submission_cannot_be_bypassed(delivery, state):
    service, owner, _, run, scope, _, _, record = delivery
    service.store.managed_submission(
        owner, run["id"], scope, {"mode": "full", "operation": "pipeline"}
    )
    service.store.managed_receipt(
        owner,
        run["id"],
        scope,
        "pipeline",
        {"task_id": "pending"} if state == "waiting_platform" else {"uncertain": True},
    )
    with pytest.raises(ValueError):
        await record()
    view = service.list(owner)[0]
    assert view.state == state and not view.artifacts


@pytest.mark.parametrize("field", ["user_id", "profile_id", "conversation_id", "lease"])
async def test_file_delivery_preserves_owner_and_lease_boundaries(delivery, field):
    service, owner, engine, _, scope, _, _, record = delivery
    if field == "lease":
        with engine.begin() as conn:
            conn.execute(
                work.update()
                .where(work.c.id == scope["work_id"])
                .values(lease_until=now() - timedelta(seconds=1))
            )
        foreign = owner
    else:
        foreign = owner.model_copy(update={field: "another-owner"})
    with pytest.raises((PermissionError, StaleLeaseError)):
        await record(active_owner=foreign)
    assert not service.list(owner)[0].artifacts
