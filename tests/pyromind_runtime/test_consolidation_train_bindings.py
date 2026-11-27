import pytest

from pyromind_agent_server.echomind_consolidation import _validate_workflow_render
from pyromind_agent_server.echomind_consolidation_references import (
    _train_runtime_candidate,
)
from pyromind_agent_server.echomind_projects import ProjectNode
from pyromind_agent_server.echomind_train_runtime import (
    TrainWorkflowSpec,
    prepare_workflow_for_submission,
)
from pyromind_runtime.domain.sft_reader import archived_sft_input


@pytest.fixture
def vision_train():
    configs = {
        "data": ("DatasetConfigBuilderNode", {}),
        "path": ("PathJoinNode", {"base_path": "/workspace/old", "subpath": "train"}),
        "vision": ("DatasetConfigBuilderVisionNode", {"image_field": "images"}),
        "extra": ("DatasetExtraConfigBuilderNode", {}),
        "base": ("ModelConfigBuilderNode", {"model_path": "/workspace/base"}),
        "lora": ("LoraConfigBuilderNode", {"lora_rank": 16}),
        "training": ("TrainingConfigBuilderNode", {"num_epochs": 2}),
        "accelerate": ("AccelerateConfigBuilderNode", {}),
        "wandb": ("PyroMindWandbConfigBuilderNode", {"wandb_api_key": "WANDB_API_KEY"}),
        "train": ("ModelTrainSFTNode", {"output_path": "/workspace/adapter"}),
        "merge": (
            "ModelMergeLoraNode",
            {"model_path": "/workspace/base", "output_path": "/workspace/merged"},
        ),
    }
    links = [
        ("path", "joined_path", "data", "train_data_path"),
        ("vision", "dataset_kind_config", "data", "dataset_kind_config"),
        ("extra", "dataset_extra_config", "data", "dataset_extra_config"),
        ("data", "dataset_config", "train", "dataset_config"),
        ("base", "model_config", "train", "model_config"),
        ("lora", "lora_config", "train", "lora_config"),
        ("training", "training_config", "train", "training_config"),
        ("accelerate", "accelerate_config", "train", "accelerate_config"),
        ("wandb", "wandb_config", "train", "wandb_config"),
        ("train", "model_output_path", "merge", "lora_path"),
    ]
    return {
        "nodes": [
            {"id": node, "data": {"nodeType": kind, "config": config}}
            for node, (kind, config) in configs.items()
        ],
        "edges": [
            {"source": a, "sourceHandle": ap, "target": b, "targetHandle": bp}
            for a, ap, b, bp in links
        ],
    }


@pytest.mark.parametrize("has_validation", [False, True])
def test_train_binds_only_archived_dataset_inputs(vision_train, has_validation):
    if has_validation:
        vision_train["nodes"][0]["data"]["config"]["val_data_path"] = "/workspace/val"
    conversation = "a" * 32
    recipe = _train_runtime_candidate(vision_train, conversation, "21")
    assert recipe
    roles = {"base_model", "train_dataset"}
    if has_validation:
        roles.add("validation_dataset")
    spec = TrainWorkflowSpec.model_validate(
        {
            "task_name": "probe",
            "conversation_id": conversation,
            "cluster": "test",
            "workflow": recipe["workflow"],
            "bindings": recipe["workflow_bindings"],
            "workflow_templates": recipe["workflow_templates"],
            "inputs": {role: "/workspace/new/" + role for role in roles},
            "template_values": {"graph_run_id": "run", "stage_run_id": "stage"},
            "output_node_id": recipe["output_node_id"],
            "output_uri": "/workspace/new/merged",
        }
    )
    assert set(spec.bindings.values()) == roles
    assert spec.workflow
    effective = prepare_workflow_for_submission(spec.workflow, spec=spec)
    configs = {n["id"]: n["data"]["config"] for n in effective["nodes"]}
    assert configs["vision"] == {"image_field": "images"}
    assert configs["data"] == {
        "train_data_path": "/workspace/new/train_dataset",
        **(
            {"val_data_path": "/workspace/new/validation_dataset"}
            if has_validation
            else {}
        ),
    }
    assert configs["training"] == {"num_epochs": 2}
    assert configs["lora"] == {"lora_rank": 16}
    assert configs["train"]["output_path"].endswith("/run/stage/lora")
    assert configs["merge"]["output_path"] == "/workspace/new/merged"


def test_train_rejects_two_sources_for_one_port(vision_train):
    vision_train["nodes"][0]["data"]["config"]["train_data_path"] = "/workspace/other"
    assert _train_runtime_candidate(vision_train, "a" * 32, "21") is None


def test_text_dataset_with_direct_train_path(vision_train):
    vision_train["edges"].pop(0)
    vision_train["nodes"] = [n for n in vision_train["nodes"] if n["id"] != "path"]
    data, kind = vision_train["nodes"][:2]
    data["data"]["config"] = {"train_data_path": "/workspace/文本 训练/train.jsonl"}
    kind["data"] = {
        "nodeType": "DatasetConfigBuilderTextNode",
        "config": {"assistant_response_field": "answer"},
    }
    recipe = _train_runtime_candidate(vision_train, "a" * 32, "21")
    assert recipe
    bindings = recipe["workflow_bindings"]
    assert isinstance(bindings, dict)
    assert set(bindings) == {
        "nodes[0].data.config.train_data_path",
        "nodes[3].data.config.model_path",
        "nodes[9].data.config.model_path",
    }


def test_successful_reader_records_the_bound_file_and_fields(vision_train):
    data, join, vision = vision_train["nodes"][:3]
    data["data"]["config"] = {"val_data_path": "/workspace/固定验证/val.jsonl"}
    join["data"]["config"]["subpath"] = "/train_sft_messages.jsonl"
    mapping = {
        "image_field": "images",
        "user_prompt_field": "user_prompt",
        "assistant_response_field": "gt",
    }
    vision["data"]["config"] = mapping
    assert archived_sft_input(vision_train) == {
        "reader": "DatasetConfigBuilderVisionNode",
        "reader_version": None,
        "paths": {
            "train_data_path": "/workspace/old/train_sft_messages.jsonl",
            "val_data_path": "/workspace/固定验证/val.jsonl",
        },
        "fields": mapping,
    }
    vision["data"]["version"] = "1.7.4"
    reference = archived_sft_input(vision_train)
    assert reference and reference["reader_version"] == "1.7.4"

    vision_train["edges"].append(dict(vision_train["edges"][0]))
    assert archived_sft_input(vision_train) is None


def test_rendered_train_cannot_swap_verified_splits(vision_train):
    data, _, vision = vision_train["nodes"][:3]
    data["data"]["config"]["val_data_path"] = "/workspace/old/val"
    vision["data"]["config"].update(
        user_prompt_field="user_prompt", assistant_response_field="gt"
    )
    recipe = _train_runtime_candidate(vision_train, "a" * 32, "21")
    assert recipe
    node = ProjectNode.model_validate(
        {
            "id": "train",
            "type": "train",
            "title": "Train",
            "inputs": {
                key: {
                    "artifact_type": "dataset",
                    "schema_ref": "v1",
                    "binding": {"kind": "run_input", "name": key},
                }
                for key in ("train_dataset", "validation_dataset", "base_model")
            },
        }
    )
    _validate_workflow_render(node, config=recipe, inputs=node.inputs)
    bindings = recipe["workflow_bindings"]
    assert isinstance(bindings, dict)
    for key, role in list(bindings.items()):
        if role in ("train_dataset", "validation_dataset"):
            bindings[key] = (
                "train_dataset"
                if role == "validation_dataset"
                else "validation_dataset"
            )
    with pytest.raises(ValueError, match="输出端口不匹配"):
        _validate_workflow_render(node, config=recipe, inputs=node.inputs)
