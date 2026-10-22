"""Enforce dataset roles before preparing final-test inference."""

from typing import Any

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
        or parent.definition.get("kind") != "prediction_evaluation"
        or not parent.result.get("model")
    ):
        raise ValueError("final test requires a completed validation selection")
    manifest = parent.definition["source_manifest"]
    assert isinstance(manifest, dict)
    datasets = manifest.get("datasets", {})
    assert isinstance(datasets, dict)
    test = datasets.get("test")
    if not isinstance(test, dict) or not test.get("sha256"):
        raise ValueError(
            "train + validation only: reuse saved predictions; no test run"
        )
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
