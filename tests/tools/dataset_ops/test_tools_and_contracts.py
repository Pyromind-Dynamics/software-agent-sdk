from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pyromind_runtime.domain.snapshot import ExternalTaskState

from openhands.tools.data_preparation.platform_submit import (
    DataPreparationTaskAssociation,
    _build_dataflow_command,
)
from openhands.tools.dataset_ops.contracts import (
    AugmentationPlan,
    GapItem,
    GapPlan,
    HostSelector,
    SynthesisRequest,
)
from openhands.tools.dataset_ops.synthesis import default_strategy_registry
from openhands.tools.dataset_ops.tools import (
    DatasetSubmitAction,
    SubmitDatasetAnalysisTool,
)


def test_augmentation_requires_approved_gap() -> None:
    gap = GapPlan(
        analysis_run_id="analysis-1",
        gaps=(
            GapItem(
                dimension="condition",
                label="scratch",
                current_count=1,
                current_ratio=0.01,
                target_count=3,
                reason="business priority",
                recommended_strategy="avi_pcb.scratch",
            ),
        ),
    )
    with pytest.raises(ValidationError, match="approved gap plan"):
        AugmentationPlan(
            source_path="pcb",
            adapter="avi_pcb",
            gap_plan=gap,
            host_selector=HostSelector(),
            requests=(
                SynthesisRequest(
                    dimension="condition",
                    label="scratch",
                    strategy_id="avi_pcb.scratch",
                    count=2,
                ),
            ),
            output_schema="avi_pcb_v1",
        )


def test_submit_requires_three_sample_approval() -> None:
    with pytest.raises(ValidationError, match="three-record local sample"):
        DatasetSubmitAction(
            script_path="pipeline.py",
            input_path="/data/train.jsonl",
            spec_path="analysis_spec.json",
            runtime_profile="dataflow_text",
        )


def test_submit_schema_exposes_profiles_not_dependencies_or_resources() -> None:
    schema = SubmitDatasetAnalysisTool.create()[0].to_mcp_tool()["inputSchema"]
    properties = schema["properties"]
    assert set(properties["runtime_profile"]["enum"]) == {
        "dataflow_text",
        "dataflow_vision",
        "avi_pcb_cpu",
    }
    assert "cpu" not in properties
    assert "memory" not in properties
    assert "pip_dependencies" not in properties


def test_strategy_registry_rejects_generic_image_synthesis() -> None:
    registry = default_strategy_registry()
    assert registry.resolve("avi_pcb.dot").strategy_prefix == "avi_pcb"
    with pytest.raises(ValueError, match="generic pixel synthesis is not allowed"):
        registry.resolve("generic.draw_defect")


def test_dataset_command_uses_fixed_profile_and_contract() -> None:
    command = _build_dataflow_command(
        input_path="/datasets/input",
        output_dir="/outputs/run",
        llm_env={},
        convert_format="none",
        output_filename="synthesized.jsonl",
        python_packages=("opencv-python-headless==4.10.0.84",),
        support_file_name="job-spec.json",
        task_kind="data_synthesis",
    )
    assert "opencv-python-headless==4.10.0.84" in command
    assert "synthesized.jsonl" in command
    assert "job-spec.json" in command
    assert "dataset_job_runtime.py" in command
    assert "--kind data_synthesis" in command
    assert "source_fingerprint.py" in command
    assert "source_fingerprint_before" in command
    assert "source_fingerprint_after" in command
    assert "source_integrity_rc=90" in command


def test_source_fingerprint_detects_source_byte_changes(tmp_path: Path) -> None:
    runtime = Path(
        ".agents/skills/data-processing/scripts/preparation/source_fingerprint.py"
    )
    namespace: dict[str, object] = {"__name__": "test_runtime"}
    exec(runtime.read_text(encoding="utf-8"), namespace)
    source = tmp_path / "source"
    source.mkdir()
    item = source / "records.jsonl"
    item.write_text('{"id": "a"}\n', encoding="utf-8")
    before = namespace["source_fingerprint"](source)  # type: ignore[operator]

    item.write_text('{"id": "b"}\n', encoding="utf-8")

    after = namespace["source_fingerprint"](source)  # type: ignore[operator]
    assert before != after


def test_task_association_preserves_dataset_kind(tmp_path: Path) -> None:
    association = DataPreparationTaskAssociation(
        task_id="task",
        conversation_id="conversation",
        run_id="run",
        output_dir="/output",
        input_path="/input",
        script_path="pipeline.py",
        task_kind="dataset_analysis",
    )
    restored = DataPreparationTaskAssociation.from_dict(association.to_dict())
    assert restored.task_kind == "dataset_analysis"


@pytest.mark.parametrize("kind", ["dataset_analysis", "data_synthesis"])
def test_product_runtime_accepts_dataset_task_kinds(kind: str) -> None:
    state = ExternalTaskState(
        task_id="task",
        kind=kind,
        status="pending",
        submitted_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
    assert state.kind == kind


def test_runtime_finalizer_rejects_missing_distribution(tmp_path: Path) -> None:
    runtime = Path(
        ".agents/skills/data-processing/scripts/preparation/dataset_job_runtime.py"
    )
    namespace: dict[str, object] = {"__name__": "test_runtime"}
    exec(runtime.read_text(encoding="utf-8"), namespace)
    spec = tmp_path / "job-spec.json"
    taxonomy = {"dimensions": []}
    taxonomy_fingerprint = hashlib.sha256(
        json.dumps(
            taxonomy,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "adapter": "jsonl_text",
                "label_source": "existing_labels",
                "taxonomy": taxonomy,
                "taxonomy_fingerprint": taxonomy_fingerprint,
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source.jsonl"
    source.write_text('{"id": "a"}\n', encoding="utf-8")
    primary = tmp_path / "labels.jsonl"
    primary.write_text('{"sample_id": "a", "assignments": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="distribution_report"):
        namespace["finalize"](  # type: ignore[operator]
            "dataset_analysis", tmp_path, primary, spec, source
        )
