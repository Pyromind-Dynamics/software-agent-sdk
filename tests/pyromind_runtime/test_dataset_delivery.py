"""Native Markdown handoffs feed downstream tasks without result registration."""

import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path, PurePosixPath

import httpx
import pytest
from PIL import Image
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
    return service, owner, engine, run, scope, files


def handoff(service, owner, run, scope, *, report=False, completed=True):
    return service.store.complete_agent_turn(
        owner,
        run["id"],
        scope,
        f"Report: `public_data/experiment-runs/{run['id']}/public_data/report.md`",
        report=report,
        completed=completed,
    )


@pytest.mark.parametrize("platform", [False, True])
def test_data_markdown_handoff_does_not_register_files(delivery, platform):
    service, owner, _, run, scope, _ = delivery
    if platform:
        service.store.managed_submission(
            owner, run["id"], scope, {"mode": "full", "operation": "pipeline"}
        )
        service.store.managed_receipt(
            owner,
            run["id"],
            scope,
            "pipeline",
            {"task_id": "data-task", "output_dir": "/pipeline/result"},
        )
        service.store.ingest(
            "data-complete", "data-task", "callback", {"status": "succeeded"}
        )
        service.store.apply_observations()
    report = f"Report: `public_data/experiment-runs/{run['id']}/public_data/report.md`"
    service.store.complete_agent_turn(
        owner, run["id"], scope, report, report=platform, completed=True
    )
    view = service.list(owner)[0]
    assert view.state == ("succeeded" if platform else "reported")
    assert view.report_state == "ready" and view.summary == report
    assert not view.artifacts


@pytest.mark.parametrize("platform", [False, True])
async def test_reported_data_paths_feed_training_without_copying(
    delivery, platform, tmp_path, monkeypatch
):
    service, owner, _, run, scope, files = delivery
    if platform:
        service.store.managed_submission(
            owner, run["id"], scope, {"mode": "full", "operation": "pipeline"}
        )
        service.store.managed_receipt(
            owner, run["id"], scope, "pipeline", {"task_id": "data-task"}
        )
        service.store.ingest(
            "data-complete", "data-task", "callback", {"status": "succeeded"}
        )
        service.store.apply_observations()
    handoff(service, owner, run, scope, report=platform)
    view = service.list(owner)[0]
    assert view.report_state == "ready" and view.summary
    assert not view.artifacts
    assert view.state == ("succeeded" if platform else "reported")
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
    paths = {
        "train_data_path": "/datasets/clean/train.jsonl",
        "val_data_path": "/datasets/clean/val.jsonl",
    }
    training = service.request(
        owner,
        RequestExperimentTask.model_validate(
            {
                "role_id": "trainer",
                "request_id": "train",
                "task": {
                    "name": "Train",
                    "instruction": "Use the paths checked in the Dataset report.",
                    "inputs": paths,
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
    assert inputs == paths and "input_artifacts" not in training["task_spec"]
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
    for index, part in enumerate(content[:3]):
        image_path = tmp_path / part["path"].lstrip("/")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (index * 80, 0, 0)).save(image_path)
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


@pytest.mark.parametrize("field", ["profile_id", "conversation_id"])
def test_reported_files_cannot_be_used_as_foreign_registered_artifacts(delivery, field):
    service, owner, _, run, scope, _ = delivery
    handoff(service, owner, run, scope)
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
def test_inflight_platform_submission_cannot_be_bypassed(delivery, state):
    service, owner, _, run, scope, _ = delivery
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
    handoff(service, owner, run, scope, report=True)
    view = service.list(owner)[0]
    assert view.state == state and view.report_state != "ready"
    assert not view.artifacts


@pytest.mark.parametrize("field", ["user_id", "profile_id", "conversation_id", "lease"])
def test_file_delivery_preserves_owner_and_lease_boundaries(delivery, field):
    service, owner, engine, run, scope, _ = delivery
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
        handoff(service, foreign, run, scope)
    assert service.list(owner)[0].report_state != "ready"
    assert not service.list(owner)[0].artifacts


def test_failed_handoff_retries_without_submitting_platform_work(delivery):
    service, owner, _, run, scope, _ = delivery
    handoff(service, owner, run, scope, completed=False)
    assert service.list(owner)[0].report_state == "failed"
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
    service.store.enqueue_report(owner, run["id"], command_id="retry-report")
    item = service.store.queue.claim(kinds=("generate_report",))
    assert item is not None
    retry_scope = {"work_id": item["id"], "generation": item["generation"]}
    handoff(service, owner, run, retry_scope, report=True)
    view = service.list(owner)[0]
    assert (view.state, view.report_state) == ("reported", "ready")
    assert not service.store.list(owner)[0]["submissions"]
