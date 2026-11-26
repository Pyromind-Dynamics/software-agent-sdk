import json
import logging
import os
import subprocess
from traceback import format_exception
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from pyromind_agent_server import echomind_dataset_backend, echomind_train_backend
from pyromind_agent_server.echomind_consolidation import ConsolidationContext
from pyromind_agent_server.echomind_consolidation_replay import _candidate
from pyromind_agent_server.echomind_dataset_backend import (
    PlatformDatasetWorkflowBackend,
)
from pyromind_agent_server.echomind_dataset_runtime import (
    DatasetWorkflowConfig,
    DatasetWorkflowSpec,
)
from pyromind_agent_server.echomind_runtime_contracts import (
    RuntimeCredentialUnavailableError,
    RuntimeSubmissionNotAttemptedError,
)
from pyromind_agent_server.echomind_train_runtime import TrainRuntimeError
from pyromind_common.platform.credentials import materialize_runtime_environment
from pyromind_common.tools.data_preparation.platform_submit import (
    _build_dataflow_command,
)
from pyromind_common.tools.data_preparation.recipe import (
    RECIPE_INPUT,
    RECIPE_OUTPUT,
    compose_recipe,
    recipe_asset,
)
from pyromind_runtime.application.consolidation_evidence import (
    ConsolidationEvidence,
    ConsolidationStageEvidence,
)
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.domain.task_correlation import (
    ProjectTaskReference,
    TaskCorrelation,
)
from pyromind_runtime.domain.workflow_execution import (
    DatasetRecipeReceipt,
    WorkflowExecutionRequest,
)
from pyromind_runtime.infrastructure.file_product_store import FileProductStore


_ENDPOINT = "https://vision.example.test/v1/chat/completions"
_MODEL = "vision-flash"
_KEY = "test-fixture-vision-key"


def _workflow(command: str, param: str) -> JsonObject:
    return {
        "nodes": [
            {
                "id": "processor",
                "data": {
                    "nodeType": "CustomCommandCPUNode",
                    "config": {"command": command, "param": param},
                },
            }
        ]
    }


def _command(model=_MODEL, endpoint=_ENDPOINT):
    return (
        'set -eu\nexport DF_API_KEY="$param"\nunset param\n'
        f"export DF_API_URL={endpoint}\nexport DF_MODEL_NAME={model}\n"
        f"# previously ran {_MODEL} at {_ENDPOINT}\n"
        "python3 -u - /workspace/raw /workspace/out <<'PCB_STAGE_PY'\n"
        'import os\nMODEL = os.environ["DF_MODEL_NAME"]\n'
        'CHAT_URL = os.environ["DF_API_URL"]\n'
        "PCB_STAGE_PY"
    )


@pytest.fixture
def label_backend(tmp_path, monkeypatch):
    cid = uuid4().hex
    conversation = tmp_path / cid
    conversation.mkdir()
    store = FileProductStore(conversation)
    store.create(
        ConversationSnapshot(
            conversation_id=cid, capabilities=HarnessCapabilities(cancel=True)
        ),
        user_id="owner",
        profile_id="echomind",
    )
    store.save_x_cluster("cn-east-1")
    for task_id, kind in [("1086", "data_preparation"), ("1087", "model_train")]:
        store.save_workflow_execution_request(
            WorkflowExecutionRequest(
                request_id=f"request-{task_id}",
                conversation_id=cid,
                workflow_path="workflow.py",
                fingerprint="revision",
                workflow=_workflow(
                    f"export DF_API_KEY={_KEY} DF_API_URL={_ENDPOINT} "
                    f"DF_MODEL_NAME={_MODEL}",
                    "",
                ),
                summary={},
                status="submitted",
                task_id=task_id,
                correlation=TaskCorrelation(conversation_id=cid, task_kind=kind),
            ),
            expected_revision=None,
        )
    spec = DatasetWorkflowSpec(
        task_name="dataset",
        conversation_id=cid,
        owner_user_id="owner",
        cluster="cn-east-1",
        workflow=_workflow(_command(), "DF_API_KEY"),
        dataflow_credential_task_id="1086",
        output_uri="/workspace/out",
        train_filename="train.jsonl",
        val_filename="val.jsonl",
    )
    client = Mock()
    client.studio.create.return_value.task_id = 102
    monkeypatch.setattr(echomind_train_backend, "_client", lambda **_: client)
    return PlatformDatasetWorkflowBackend(conversations_dir=tmp_path), spec, client


@pytest.mark.parametrize(
    "case", ["param", "declaration", "expired", "project", "foreign_parent", "endpoint"]
)
def test_sdk_dataset_key_comes_from_owners_signed_task(
    label_backend, monkeypatch, case
):
    backend, initial, client = label_backend
    store = FileProductStore(backend.conversations_dir / initial.conversation_id)
    project = "project"
    spec = initial.model_copy(
        update={
            "dataflow_credential_task_id": "102",
            "template_values": {"project_id": project},
        }
    )
    context = RequestContext(
        user_id="owner", cookie="session=owner", x_cluster=spec.cluster
    )
    command = "unset DF_API_KEY DF_API_URL DF_API_BASE_URL DF_MODEL_NAME\n" + _command()
    parent_ref = store.issue_task_correlation(
        TaskCorrelation(
            conversation_id=spec.conversation_id,
            task_kind="data_preparation",
            cluster=spec.cluster,
            execution=ProjectTaskReference(
                project_id=project,
                task_id="data",
                plan_sequence=4,
                attempt=1,
            ),
        )
    )
    declaration = (
        "{{pyromind.runtime:vision:DF_API_URL,DF_MODEL_NAME,DF_API_KEY}}\n"
        "python pipeline.py"
    )
    archived = _workflow(
        declaration if case == "declaration" else command, "DF_API_KEY"
    )
    if case == "declaration":
        remote = materialize_runtime_environment(
            archived,
            {
                "PYROMIND_DATAFLOW_PROFILES": json.dumps(
                    {
                        "vision": {
                            "DF_API_KEY": _KEY,
                            "DF_MODEL_NAME": _MODEL,
                            "DF_API_URL": _ENDPOINT,
                        },
                    }
                )
            },
        )
    else:
        remote = _workflow(command, _KEY)
    read = AsyncMock(return_value=remote)
    monkeypatch.setattr(echomind_dataset_backend, "read_accepted_workflow", read)
    for task_id in ("101", "102"):
        correlation = (
            parent_ref
            if task_id == "101"
            else store.issue_task_correlation(
                TaskCorrelation(
                    conversation_id=spec.conversation_id,
                    task_kind="data_preparation",
                    cluster=spec.cluster,
                )
            )
        )
        recipe = (
            DatasetRecipeReceipt(
                source_path="/workspace/source",
                output_directory="/workspace/output",
                command_template="prepare {source_batch} {output_uri}",
                assets=(),
                uses_labeling_gateway=True,
                replay_of_task_id="999" if case == "foreign_parent" else "101",
            )
            if task_id == "102"
            else None
        )
        store.save_workflow_execution_request(
            WorkflowExecutionRequest(
                request_id="sdk:" + task_id,
                conversation_id=spec.conversation_id,
                task_id=task_id,
                status="submitted",
                workflow_path="workflow.py",
                workflow=archived,
                fingerprint="hash",
                summary={},
                correlation=correlation,
                dataset_recipe=recipe,
            ),
            expected_revision=None,
        )
        store.append(
            ProductEvent.model_validate(
                {
                    "type": "external_task.submitted",
                    "conversation_id": spec.conversation_id,
                    "payload": {
                        "task_id": task_id,
                        "status": "succeeded",
                        "kind": "data_preparation",
                        "x_cluster": spec.cluster,
                        "submitted_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-01T00:00:00Z",
                        "correlation": correlation.model_dump(mode="json"),
                    },
                }
            )
        )
    store.append(
        ProductEvent.model_validate(
            {
                "type": "experiment_execution.changed",
                "conversation_id": spec.conversation_id,
                "payload": {
                    "execution": {
                        "run_id": "project:data:1",
                        "project_id": project,
                        "plan_sequence": 4,
                        "plan_attempt": 1,
                        "task_id": "data",
                        "task_name": "data",
                        "stage_snapshot": {"kind": "dataset"},
                        "role_id": "main",
                        "state": "succeeded",
                        "report_state": "ready",
                        "platform_task_id": "101",
                    }
                },
            }
        )
    )
    if case == "expired":
        with pytest.raises(RuntimeCredentialUnavailableError):
            backend.submit_workflow(spec=spec, request_context=None)
        read.assert_not_awaited()
    elif case in {"project", "foreign_parent", "endpoint"}:
        if case == "project":
            spec = spec.model_copy(update={"template_values": {"project_id": "other"}})
        if case == "endpoint":
            spec = spec.model_copy(
                update={
                    "workflow": _workflow(
                        _command(endpoint="https://other.test"), "DF_API_KEY"
                    )
                }
            )
        with pytest.raises(RuntimeSubmissionNotAttemptedError) as err:
            backend.submit_workflow(spec=spec, request_context=context)
        client.studio.create.assert_not_called()
        assert _KEY not in str(err.value)
        if case != "endpoint":
            read.assert_not_awaited()
    else:
        assert backend.submit_workflow(spec=spec, request_context=context) == "102"
        graph = client.studio.create.call_args.args[0].workflow
        assert graph["nodes"][0]["data"]["config"]["param"] == _KEY
        assert _KEY not in spec.model_dump_json()
        read.assert_awaited_once_with("102", context)


def test_only_platform_submission_receives_archived_dataflow_key(label_backend, caplog):
    backend, spec, client = label_backend
    before = spec.model_dump_json()
    assert backend.submit_workflow(spec=spec, request_context=None) == "102"
    transport = client.studio.create.call_args.args[0].workflow
    assert transport["nodes"] == _workflow(_command(), _KEY)["nodes"]
    assert _KEY not in before and spec.model_dump_json() == before

    def rejected(_request):
        logging.getLogger("pyromind_sdk").error("rejected param %s", _KEY)
        raise ValueError(f"invalid param {_KEY}")

    client.studio.create.side_effect = rejected
    with pytest.raises(TrainRuntimeError, match="REDACTED") as error:
        backend.submit_workflow(spec=spec, request_context=None)
    assert _KEY not in "".join(format_exception(error.value))
    assert _KEY not in caplog.text


@pytest.mark.parametrize(
    ("updates", "context"),
    [
        ({}, RequestContext(user_id="different-owner", x_cluster="cn-east-1")),
        ({}, RequestContext(user_id="owner", x_cluster="us-west-2")),
        ({"owner_user_id": "different-owner"}, None),
        ({"cluster": "us-west-2"}, None),
        ({"dataflow_credential_task_id": "1087"}, None),
        (
            {"workflow": _workflow(_command(model=_MODEL + "-other"), "DF_API_KEY")},
            None,
        ),
        (
            {
                "workflow": _workflow(
                    _command(endpoint=_ENDPOINT + "/other"), "DF_API_KEY"
                )
            },
            None,
        ),
    ],
)
def test_dataset_refuses_unrelated_credential(label_backend, updates, context):
    backend, spec, client = label_backend
    with pytest.raises(RuntimeSubmissionNotAttemptedError):
        backend.submit_workflow(
            spec=spec.model_copy(update=updates), request_context=context
        )
    client.studio.create.assert_not_called()


def test_development_recipe_submits_with_correct_model_profile_and_shanghai_index(
    label_backend,
    tmp_path,
):
    backend, first, client = label_backend
    client.studio.cluster = "cn-east-1"
    cid = first.conversation_id
    root = f"/.pyromind-agent/{cid}/data_preparation"
    asset_root = f"{root}/recipe-assets"
    local = tmp_path / "asset"
    local.write_text("fixture")
    assets = tuple(
        recipe_asset(role, local, asset_root)
        for role in (
            "pipeline",
            "prepare",
            "export",
            "support_file",
            "verifier",
        )
    )
    dataflow = _build_dataflow_command(
        input_path=RECIPE_INPUT,
        output_dir=RECIPE_OUTPUT,
        llm_env={},
        convert_format="none",
        support_file_name="job-spec.json",
        output_schema="structured",
        runtime_storage_dir=f"{root}/runtime-cache/{'a' * 64}",
        runtime_fingerprint="a" * 64,
    )
    recipe, archived = compose_recipe(
        input_path="/development/batch-one",
        output_dir=f"{root}/replay",
        assets=assets,
        dataflow_command=dataflow,
        fixed_validation_path="/workspace/fixed.jsonl",
        llm_env={
            "DF_MODEL_NAME": _MODEL,
            "DF_API_URL": _ENDPOINT,
            "DF_API_KEY": _KEY,
            "DF_API_BASE_URL": "https://vision.example.test/v1",
        },
        replay_of_task_id="1086",
    )
    store = FileProductStore(tmp_path / cid)
    source = WorkflowExecutionRequest(
        request_id="development",
        conversation_id=cid,
        task_id="1100",
        workflow_path="development.py",
        fingerprint="sample",
        summary={},
        correlation=TaskCorrelation(conversation_id=cid, task_kind="data_preparation"),
        status="submitted",
        dataset_recipe=recipe,
        workflow={**_workflow(archived, ""), "edges": []},
    )
    store.save_workflow_execution_request(source, expected_revision=None)
    stage = ConsolidationStageEvidence(
        task_id="1086",
        source="workflow_execution_request",
        definition_sha256="abc",
        revision=2,
    )
    source_batch = "/workspace/new round/batch-2"
    context = ConsolidationContext(
        project_id="p",
        evidence=ConsolidationEvidence(
            conversation_id=cid, dataset=stage, train=stage, eval=stage
        ),
        source_graph_sha256="draft",
        evidence_sha256="evidence",
        source_batch=source_batch,
        base_model="/workspace/model",
        validation_dataset="/workspace/eval.jsonl",
    )
    config = DatasetWorkflowConfig.model_validate(_candidate(source, context))
    workflow, output = config.render_workflow(
        inputs={"source_batch": source_batch},
        runtime_values={"graph_run_id": "round-2", "stage_run_id": "stage-1"},
    )
    spec = first.model_copy(
        update={
            "workflow": workflow,
            "output_uri": output,
            "dataflow_credential_task_id": config.dataflow_credential_task_id,
        }
    )
    assert _KEY not in config.model_dump_json() and _KEY not in spec.model_dump_json()
    backend.submit_workflow(spec=spec, request_context=None)
    node = client.studio.create.call_args.args[0].workflow["nodes"][0]["data"]["config"]
    command = node["command"]
    assert node["param"] == _KEY and _KEY not in command
    assert "cp " in command and '"$recipe_output/job-spec.json"' in command
    assert '"$recipe_output"/job-spec.json' in command
    assert "Replay requires" not in command
    assert "install --no-deps open-dataflow==" in command
    assert "requirements-cpu.txt" in command
    assert "/development/batch-one" not in command
    preamble = command.split('mkdir -p "$recipe_output"', 1)[0]
    check = (
        preamble
        + """
test -z "${param+x}" && test "$DF_API_KEY" = "$EXPECTED_KEY" &&
test "$DF_MODEL_NAME" = "$EXPECTED_MODEL" &&
test "$DF_API_URL" = "$EXPECTED_ENDPOINT" &&
test "$recipe_source" = '/target-workspace/new round/batch-2' &&
test "$recipe_output" = "$EXPECTED_OUTPUT" &&
test "$PIP_INDEX_URL" = 'https://mirrors.aliyun.com/pypi/simple/'
"""
    )
    subprocess.run(
        ["/bin/sh", "-c", check],
        check=True,
        env={
            **os.environ,
            "param": node["param"],
            "EXPECTED_KEY": _KEY,
            "EXPECTED_MODEL": _MODEL,
            "EXPECTED_ENDPOINT": _ENDPOINT,
            "EXPECTED_OUTPUT": output.replace("/workspace", "/target-workspace", 1),
        },
    )
