import json

import pytest

from pyromind_agent_server.echomind_dataset_consolidation import extract_dataset_recipe


def _workflow(command: str) -> dict:
    return {
        "nodes": [
            {
                "id": "1",
                "type": "default",
                "data": {
                    "nodeType": "CustomCommandCPUNode",
                    "config": {"command": command, "cpu": 2, "memory": 8},
                },
            }
        ],
        "edges": [],
    }


def test_archived_annotations_never_become_a_training_recipe() -> None:
    preparation = "/.pyromind-agent/current/data_preparation"
    old = f"{preparation}/old-run"
    cache = f"{preparation}/runtime-cache/v1"
    manifest = "/target-workspace/input/prelabel_manifest_1.jsonl"
    command = (
        f"DF_STATE_DIR=/target-workspace{old} "
        f"DF_OUTPUT_SCHEMA=structured DF_API_KEY=example-secret "
        f"PYTHONPATH=/target-workspace{cache} "
        f"python /target-workspace{old}/pipeline.py {manifest} "
        f"/target-workspace{old}/processed.jsonl\n"
        f"test -s /target-workspace{cache}/image_utils.py\n"
        "DF_API_KEY='/workspace/private/api-key-parts/pipeline.py'\n"
        "test -s /workspace/.pyromind-agent/another/runs/pipeline.py\n"
        "test -s /workspace/private/../other/pipeline.py\n"
        "test -s https://host/workspace/private/pipeline.py?token=example-secret\n"
    )
    result = extract_dataset_recipe(
        archive_workflow=_workflow(command),
        source_batch="/workspace/input/images",
        conversation_id="current",
    )

    assert result.runtime is None
    assert {asset.role: asset.path for asset in result.assets} == {
        "input_manifest": "/workspace/input/prelabel_manifest_1.jsonl",
        "pipeline_source": f"/workspace{old}/pipeline.py",
        "annotations": f"/workspace{old}/processed.jsonl",
        "runtime_file": f"/workspace{cache}/image_utils.py",
    }
    assert [blocker.code for blocker in result.blockers] == [
        "dataset_raw_input_binding_missing",
        "dataset_label_confirmation_missing",
        "dataset_sft_export_missing",
        "dataset_labeling_gateway_unbound",
    ]
    assert "example-secret" not in repr(result)


def test_sdk_preparation_command_exposes_source_references_for_repair() -> None:
    result = extract_dataset_recipe(
        archive_workflow={
            "nodes": [
                {
                    "data": {
                        "type": "DataPreparationNode",
                        "config": {
                            "command": (
                                "run /target-workspace/exp/pipeline.py "
                                "/target-workspace/incoming/source.jsonl"
                            )
                        },
                    }
                }
            ],
            "edges": [],
        },
        source_batch="/workspace/images",
    )
    assert [asset.role for asset in result.assets] == [
        "pipeline_source",
        "input_manifest",
    ]
    assert (
        result.runtime is None and result.blockers[0].code == "dataset_recipe_required"
    )


def _reviewed_runtime() -> dict:
    return {
        "workflow": _workflow(""),
        "command_binding_path": "nodes[0].data.config.command",
        "command_template": "convert {source_batch} {output_uri}",
        "output_uri": "/workspace/runs/{graph_run_id}/{stage_run_id}/dataset",
    }


def test_reviewed_contract_binds_input_and_run_scoped_artifacts() -> None:
    reviewed = _reviewed_runtime()
    result = extract_dataset_recipe(
        archive_workflow=_workflow("historical-only-command"),
        source_batch="/workspace/source with spaces",
        reviewed_runtime=reviewed,
    )

    assert result.runtime == reviewed
    assert result.runtime is not reviewed
    assert not result.blockers


@pytest.mark.parametrize(
    "change",
    [
        {"output_uri": "/workspace/static"},
        {"output_uri": "/workspace/{graph_run_id}/{project_id.missing}"},
        {"command_template": "convert /workspace/static {output_uri}"},
        {
            "command_template": (
                "DF_API_KEY=example-secret convert {source_batch} {output_uri}"
            )
        },
        {"workflow": {"nodes": [], "edges": []}},
    ],
)
def test_invalid_reviewed_contract_yields_safe_dataset_blocker(change: dict) -> None:
    result = extract_dataset_recipe(
        archive_workflow=_workflow("historical-only-command"),
        source_batch="/workspace/data",
        reviewed_runtime={**_reviewed_runtime(), **change},
    )

    assert result.runtime is None
    assert [blocker.code for blocker in result.blockers] == ["dataset_recipe_invalid"]
    assert "example-secret" not in json.dumps(result.blockers[0].model_dump())


def test_opaque_workflow_does_not_imply_source_type_or_label_policy() -> None:
    result = extract_dataset_recipe(
        archive_workflow=_workflow("convert /workspace/elsewhere/pipeline.py"),
        source_batch="/workspace/data",
    )

    assert result.runtime is None
    assert not result.assets
    assert [blocker.code for blocker in result.blockers] == ["dataset_recipe_required"]
