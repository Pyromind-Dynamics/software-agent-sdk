"""Enforce dataset roles before preparing final-test inference."""

from typing import Any

from pyromind_runtime.application.checkpoint_evaluations import KIND, job_state
from pyromind_runtime.ports.product_store import ProductStore


def final_test_definition(
    store: ProductStore, conversation_id: str, definition: dict[str, Any]
) -> dict[str, Any]:
    inference = definition.get("inference", {})
    selection_id = inference.get("selection_run_id")
    if not selection_id:
        return definition
    parent = store.load_pipeline_runs().get(selection_id)
    if (
        parent is None
        or parent.conversation_id != conversation_id
        or parent.definition.get("kind") not in {"prediction_evaluation", KIND}
        or not parent.result.get("model")
    ):
        raise ValueError("final test requires a completed validation selection")
    continuous = parent.definition.get("kind") == KIND
    if continuous and job_state(parent).status != "completed":
        raise ValueError("final test requires the checkpoint selection to finish")
    manifest = parent.definition if continuous else parent.definition["source_manifest"]
    assert isinstance(manifest, dict)
    datasets = manifest.get("datasets", {})
    assert isinstance(datasets, dict)
    test = datasets.get("test")
    if not isinstance(test, dict) or not test.get("sha256"):
        raise ValueError("no test run: no independent test dataset was declared")
    for role in ("train", "validation"):
        entry = datasets.get(role)
        if not isinstance(entry, dict):
            raise ValueError("test independence requires train/validation provenance")
        if test["path"] == entry["path"] or test["sha256"] == entry["sha256"]:
            raise ValueError("test must be independent of training and validation")
    if definition["input_path"] != test["path"]:
        raise ValueError("final test input must match the frozen test dataset")
    model = parent.result["model"]
    assert isinstance(model, dict)
    source = definition["evaluation_config"].get("model_source", {})
    if (
        inference["model_path"] != model["path"]
        or source.get("type") != "checkpoint"
        or source.get("kind") != model["kind"]
        or (model["kind"] == "lora" and source.get("base_model") != model["base_model"])
    ):
        raise ValueError("final test must use the validation-selected checkpoint")
    if definition["evaluation_config"].get("generation", {}).get("limit", 0):
        raise ValueError("final test requires the complete test dataset")
    if continuous:
        template = parent.definition["template"]
        assert isinstance(template, dict)
        frozen_evaluation = template["evaluation_config"]
        assert isinstance(frozen_evaluation, dict)
        supplied = definition["evaluation_config"]
        if {k: v for k, v in supplied.items() if k != "model_source"} != {
            k: v for k, v in frozen_evaluation.items() if k != "model_source"
        }:
            raise ValueError(
                "final test must use frozen evaluation rules and generation settings"
            )
        scoring = template.get("business_scoring")
        return {
            **definition,
            "dataset_role": "test",
            "test_source": test,
            "dataset_provenance": {**test, "dataset_role": "test"},
            **(
                {
                    "business_scoring": {
                        **scoring,
                        "dataset_role": "test",
                        "dataset_source": test,
                        "selection_run_id": selection_id,
                    }
                }
                if isinstance(scoring, dict)
                else {}
            ),
        }
    return {
        **definition,
        "dataset_role": "test",
        "test_source": test,
        "business_scoring": {
            "script": parent.definition["scoring_script"],
            "spec": parent.definition["scoring_spec"],
            "selection_run_id": selection_id,
        },
    }
