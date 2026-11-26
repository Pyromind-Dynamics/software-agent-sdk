from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from uuid import uuid4

import pytest

from pyromind_common.platform.credentials import materialize_runtime_environment
from pyromind_common.platform.environment import (
    EdpSubmitOptions,
    prepare_edp_shard,
    select_edp_manifests,
)
from pyromind_common.platform.preparation import StagingDirectory
from pyromind_common.platform.workflows import (
    WorkflowBatchSubmissionError,
    WorkflowSubmissionUncertain,
    submit_workflow,
    submit_workflow_batch,
)


def test_solver_shard_freezes_runner_without_credentials(
    edp_runtime, edp_staging, edp_run_id, monkeypatch
):
    monkeypatch.setenv("PYROMIND_API_KEY", "test-platform-key")
    monkeypatch.setenv("LLM_AUTH_TOKEN", "test-model-key")
    prepared = prepare_edp_shard(
        EdpSubmitOptions(manifest="/workspace/edp/batch-001/manifest.jsonl", limit=3),
        manifest="/edp/batch-001/manifest.jsonl",
        staging=edp_staging,
        runtime_dir=edp_runtime,
        run_id=edp_run_id,
        platform_env="pre",
        cluster="us-west-1#pre",
    )
    command = prepared.workflow["nodes"][0]["data"]["config"]["command"]
    assert prepared.output_dir == f"/workspace/edp/batch-001/{edp_run_id}"
    assert prepared.run_id == edp_run_id
    assert prepared.resumed is False
    assert "--manifest /target-workspace/edp/batch-001/manifest.jsonl" in command
    assert f"--output-dir /target-workspace/edp/batch-001/{edp_run_id}/run" in command
    assert "--limit 3" in command
    assert '--api-key "$PYROMIND_API_KEY"' in command
    assert '--set LLM_AUTH_TOKEN="$LLM_AUTH_TOKEN"' in command
    assert "PYROMIND_BASE_URL" in command
    frozen_inputs = json.loads((edp_staging.local / "edp-inputs.json").read_bytes())
    assert frozen_inputs["manifest"] == "/edp/batch-001/manifest.jsonl"
    assert "pyromind_common/__init__.py" in frozen_inputs["assets"]
    assert (edp_staging.local / "sandbox_runner.py").is_file()
    persisted = json.dumps(prepared.workflow) + "".join(
        path.read_text() for path in edp_staging.local.rglob("*") if path.is_file()
    )
    assert "test-platform-key" not in persisted
    assert "test-model-key" not in persisted
    assert "auth_token" not in persisted


def test_environment_only_shard_needs_no_model_credentials(
    edp_runtime, edp_staging, edp_run_id
):
    options = EdpSubmitOptions(
        manifest="/edp/batch-001/manifest.jsonl",
        profile_name="environment-only",
        dedup_by_image=True,
    )
    assert options.manifest is not None
    prepared = prepare_edp_shard(
        options,
        manifest=options.manifest,
        staging=edp_staging,
        runtime_dir=edp_runtime,
        run_id=edp_run_id,
        platform_env="prod",
        cluster="us-west-1",
    )
    bound = materialize_runtime_environment(
        prepared.workflow,
        {
            "PYROMIND_API_KEY": "platform-key",
            "PYROMIND_BASE_URL": "https://platform.example/api",
        },
    )
    command = bound["nodes"][0]["data"]["config"]["command"]
    assert "LLM_" not in command
    assert "export PYROMIND_API_KEY=platform-key" in command
    assert "--dedup-by-image" in command
    assert "platform-key" not in json.dumps(prepared.workflow)


def test_resume_uses_frozen_runtime_and_rejects_tampering(
    edp_runtime, edp_staging, edp_run_id
):
    options = EdpSubmitOptions(manifest="/edp/batch-001/manifest.jsonl")
    arguments = {
        "manifest": "/edp/batch-001/manifest.jsonl",
        "staging": edp_staging,
        "runtime_dir": edp_runtime,
        "run_id": edp_run_id,
        "platform_env": "prod",
        "cluster": "us-west-1",
    }
    original = prepare_edp_shard(options, **arguments)
    (edp_runtime / "sandbox_runner.py").write_text("print('upgraded runner')\n")
    resumed = prepare_edp_shard(
        options.model_copy(update={"run_id": edp_run_id}), **arguments
    )
    assert resumed.run_id == original.run_id
    assert resumed.output_dir == original.output_dir
    assert resumed.workflow["id"] != original.workflow["id"]
    assert resumed.resumed is True
    assert (edp_staging.local / "sandbox_runner.py").read_text() == (
        "print('frozen runner')\n"
    )
    (edp_staging.local / "sandbox_runner.py").write_text("print('tampered')\n")
    with pytest.raises(ValueError, match="runtime changed"):
        prepare_edp_shard(
            options.model_copy(update={"run_id": edp_run_id}), **arguments
        )


def test_profile_runtime_wheels_are_frozen_with_rewritten_storage_path(
    edp_runtime, edp_staging, edp_run_id
):
    profile = edp_runtime / "profiles/environment-only.json"
    profile.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "name": "exec",
                        "params": {"command": "pip install {edp_run_dir}/wheels/x.whl"},
                    }
                ]
            }
        )
    )
    options = EdpSubmitOptions(
        manifest="/edp/batch-001/manifest.jsonl", profile_name="environment-only"
    )
    assert options.manifest is not None
    with pytest.raises(ValueError, match="runtime wheels"):
        prepare_edp_shard(
            options,
            manifest=options.manifest,
            staging=edp_staging,
            runtime_dir=edp_runtime,
            run_id=edp_run_id,
            platform_env="prod",
            cluster="us-west-1",
        )
    (edp_runtime / "wheels").mkdir()
    (edp_runtime / "wheels/x.whl").write_bytes(b"wheel bytes")
    prepare_edp_shard(
        options,
        manifest=options.manifest,
        staging=edp_staging,
        runtime_dir=edp_runtime,
        run_id=edp_run_id,
        platform_env="prod",
        cluster="us-west-1",
    )
    assert (edp_staging.local / "wheels/x.whl").read_bytes() == b"wheel bytes"
    rewritten = (edp_staging.local / "environment-only.json").read_text()
    assert "{edp_run_dir}" not in rewritten
    assert f"/target-workspace/edp/batch-001/{edp_run_id}/wheels/x.whl" in rewritten


def test_shard_selection_preserves_order_and_limits():
    index = json.dumps(
        {
            "shards": [
                f"/workspace/edp/batch-{number}/manifest.jsonl" for number in range(6)
            ]
        }
    ).encode()
    assert select_edp_manifests(
        EdpSubmitOptions(shards="/workspace/edp/shards.json"), shard_index=index
    ) == ("/edp/batch-0/manifest.jsonl",)
    assert select_edp_manifests(
        EdpSubmitOptions(
            shards="/workspace/edp/shards.json", shard_offset=2, shard_count=3
        ),
        shard_index=index,
    ) == tuple(f"/edp/batch-{number}/manifest.jsonl" for number in range(2, 5))
    with pytest.raises(ValueError, match="select no shards"):
        select_edp_manifests(
            EdpSubmitOptions(shards="/edp/shards.json", shard_offset=6),
            shard_index=index,
        )
    with pytest.raises(ValueError):
        EdpSubmitOptions(shards="/edp/shards.json", shard_count=6)
    with pytest.raises(ValueError, match="unique"):
        select_edp_manifests(
            EdpSubmitOptions(shards="/edp/shards.json", shard_count=2),
            shard_index=b'{"shards":["/edp/a.jsonl","/workspace/edp/a.jsonl"]}',
        )


@pytest.mark.parametrize("profile_name", ["../outside", "/outside", "x/y", "x\n"])
def test_profile_name_cannot_escape_runtime(profile_name):
    with pytest.raises(ValueError):
        EdpSubmitOptions(manifest="/edp/manifest.jsonl", profile_name=profile_name)


def test_solver_dedup_and_wrong_output_directory_are_rejected(
    edp_runtime, edp_staging, edp_run_id, tmp_path: Path
):
    options = EdpSubmitOptions(
        manifest="/edp/batch-001/manifest.jsonl", dedup_by_image=True
    )
    arguments = {
        "manifest": "/edp/batch-001/manifest.jsonl",
        "runtime_dir": edp_runtime,
        "run_id": edp_run_id,
        "platform_env": "prod",
        "cluster": "us-west-1",
    }
    with pytest.raises(ValueError, match="agent-solving"):
        prepare_edp_shard(options, staging=edp_staging, **arguments)
    with pytest.raises(ValueError, match="beside its manifest"):
        prepare_edp_shard(
            options,
            staging=StagingDirectory(tmp_path / "wrong", "/workspace/other"),
            **arguments,
        )


@pytest.mark.parametrize("profile", ["tmax-validation", "environment-only"])
def test_prepared_shard_reaches_native_sdk_http_with_api_key_only(
    edp_runtime, edp_staging, edp_run_id, studio_http, direct_sdk, monkeypatch, profile
):
    monkeypatch.setenv("PYROMIND_API_KEY", "worker-api-key")
    monkeypatch.setenv("PYROMIND_BASE_URL", studio_http.url)
    monkeypatch.setenv("PYROMIND_AUTH_TOKEN", "request-cookie-must-not-travel")
    model_environment = {
        "LLM_BASE_URL": "https://model.example",
        "LLM_AUTH_TOKEN": "worker-model-key",
        "LLM_MODEL": "worker-model",
    }
    for key, value in model_environment.items():
        if profile == "tmax-validation":
            monkeypatch.setenv(key, value)
        else:
            monkeypatch.delenv(key, raising=False)
    options = EdpSubmitOptions(
        manifest="/edp/batch-001/manifest.jsonl", profile_name=profile
    )
    prepared = prepare_edp_shard(
        options,
        manifest="/edp/batch-001/manifest.jsonl",
        staging=edp_staging,
        runtime_dir=edp_runtime,
        run_id=edp_run_id,
        platform_env="pre",
        cluster="us-west-1#pre",
    )
    studio_http.responses.append(
        (
            200,
            {
                "success": True,
                "data": {
                    "task_id": "edp-native-101",
                    "name": "edp",
                    "status": "Pending",
                },
            },
        )
    )

    response = submit_workflow(
        direct_sdk, prepared.workflow, out_id="signed-edp-context"
    )

    assert response.task_id == "edp-native-101"
    assert response.status == "Pending"
    assert len(studio_http.requests) == 1
    path, payload = studio_http.requests[0]
    assert path == "/studio/tasks"
    headers = {name.lower(): value for name, value in studio_http.headers[0].items()}
    assert headers["authorization"] == "Bearer test-key"
    assert "cookie" not in headers
    assert payload["out_id"] == "signed-edp-context"
    command = payload["workflow"]["nodes"][0]["data"]["config"]["command"]
    assert "export PYROMIND_API_KEY=worker-api-key" in command
    assert '--api-key "$PYROMIND_API_KEY"' in command
    assert "{{pyromind.runtime:" not in command
    assert "request-cookie-must-not-travel" not in json.dumps(payload)
    assert "--auth-token" not in command
    if profile == "tmax-validation":
        assert "export LLM_AUTH_TOKEN=worker-model-key" in command
    else:
        assert "LLM_" not in command
    assert "worker-api-key" not in json.dumps(prepared.workflow)
    assert "worker-model-key" not in json.dumps(prepared.workflow)


def test_native_sdk_shard_batch_stops_after_failure_and_keeps_first_task(
    edp_runtime, tmp_path, studio_http, direct_sdk, monkeypatch
):
    monkeypatch.setenv("PYROMIND_API_KEY", "worker-api-key")
    monkeypatch.setenv("PYROMIND_BASE_URL", studio_http.url)
    options = EdpSubmitOptions(
        shards="/edp/shards.json", shard_count=3, profile_name="environment-only"
    )
    manifests = select_edp_manifests(
        options,
        shard_index=json.dumps(
            {"shards": [f"/edp/batch-{number}/manifest.jsonl" for number in range(3)]}
        ).encode(),
    )
    prepared = []
    for number, manifest in enumerate(manifests):
        run_id = uuid4()
        prepared.append(
            prepare_edp_shard(
                options,
                manifest=manifest,
                staging=StagingDirectory(
                    tmp_path / f"frozen-{number}",
                    str(PurePosixPath(manifest).parent / str(run_id)),
                ),
                runtime_dir=edp_runtime,
                run_id=run_id,
                platform_env="prod",
                cluster="us-west-1",
            )
        )
    studio_http.responses.extend(
        [
            (
                200,
                {
                    "success": True,
                    "data": {
                        "task_id": "accepted-shard-101",
                        "name": "edp",
                        "status": "Pending",
                    },
                },
            ),
            (503, {"message": "unavailable"}),
        ]
    )

    with pytest.raises(WorkflowBatchSubmissionError) as error:
        submit_workflow_batch(
            direct_sdk,
            (shard.workflow for shard in prepared),
            out_id="signed-edp-context",
        )

    assert [task.task_id for task in error.value.accepted] == ["accepted-shard-101"]
    assert isinstance(error.value.error, WorkflowSubmissionUncertain)
    assert len(studio_http.requests) == 2
    assert [payload["name"] for _, payload in studio_http.requests] == [
        shard.workflow["name"] for shard in prepared[:2]
    ]
    assert all("worker-api-key" not in json.dumps(shard.workflow) for shard in prepared)
