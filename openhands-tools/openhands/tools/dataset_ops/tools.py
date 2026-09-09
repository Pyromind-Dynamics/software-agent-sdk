"""Thin tools for gated dataset analysis and gap-driven synthesis jobs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import Field, model_validator
from rich.text import Text

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.data_preparation.definition import (
    _resolve_input_path,
    _resolve_output_path,
)
from openhands.tools.data_preparation.platform_submit import (
    DfSubmitPipelineAction,
    DfSubmitPipelineExecutor,
    DfSubmitPipelineObservation,
    ReuseAssessment,
    _normalize_storage_path,
)
from openhands.tools.data_preparation.progress import (
    DfCheckProgressAction,
    DfCheckProgressExecutor,
    DfCheckProgressObservation,
)
from openhands.tools.data_preparation.runner import (
    ProcessLocalSampleExecutor,
    build_dataflow_env,
    resolve_dataflow_python,
    runtime_public_names,
    validate_managed_image_pipeline,
)
from openhands.tools.data_preparation.stop_task import (
    DfStopTaskAction,
    DfStopTaskExecutor,
    DfStopTaskObservation,
)
from openhands.tools.data_preparation.workspace_paths import resolve_workspace_file
from openhands.tools.dataset_ops.adapters import create_adapter
from openhands.tools.dataset_ops.contracts import (
    AnalysisSpec,
    AugmentationPlan,
    DistributionReport,
    RuntimeProfile,
)


JobKind = Literal["dataset_analysis", "data_synthesis"]
_PROFILE_PACKAGES: dict[str, tuple[str, ...]] = {
    "dataflow_text": ("open-dataflow==1.0.10",),
    "dataflow_vision": ("open-dataflow==1.0.10",),
    "avi_pcb_cpu": (
        "numpy==1.26.4",
        "Pillow==12.1.1",
        "opencv-python-headless==4.10.0.84",
        "matplotlib==3.9.4",
    ),
}
_PROFILE_RESOURCES: dict[str, tuple[int, int]] = {
    "dataflow_text": (4, 32),
    "dataflow_vision": (8, 64),
    "avi_pcb_cpu": (8, 32),
}


def _validate_profile(kind: JobKind, profile: str, spec: Any) -> None:
    if profile == "avi_pcb_cpu" and getattr(spec, "adapter", None) != "avi_pcb":
        raise ValueError("avi_pcb_cpu is only valid with adapter='avi_pcb'")
    if kind == "data_synthesis" and getattr(spec, "adapter", None) == "avi_pcb":
        if profile != "avi_pcb_cpu":
            raise ValueError("AVI/PCB pixel synthesis requires avi_pcb_cpu")
    taxonomy = getattr(spec, "taxonomy", None)
    if taxonomy is not None and taxonomy.modality in {"image", "mixed"}:
        if profile == "dataflow_text":
            raise ValueError("image or mixed analysis requires dataflow_vision")


def _load_contract(path: Path, kind: JobKind) -> AnalysisSpec | AugmentationPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if kind == "dataset_analysis":
        return AnalysisSpec.model_validate(payload)
    return AugmentationPlan.model_validate(payload)


def _validate_pipeline_runtime(
    *,
    script: Path,
    profile: str,
    spec: AnalysisSpec | AugmentationPlan,
    runtime_dir: Path | None,
) -> None:
    needs_model = not (
        profile == "avi_pcb_cpu"
        or (isinstance(spec, AnalysisSpec) and spec.label_source == "existing_labels")
    )
    if needs_model and "LoggingLLMServing" not in script.read_text(encoding="utf-8"):
        raise ValueError("model pipelines must use LoggingLLMServing")
    if profile == "dataflow_vision":
        if runtime_dir is None:
            raise ValueError("dataflow_vision requires the managed runtime directory")
        image_utils = runtime_dir / "image_utils.py"
        validate_managed_image_pipeline(script, runtime_public_names(image_utils))


class DatasetRunAction(Action):
    script_path: str = Field(
        description="Workspace Python script: input, output, spec."
    )
    input_path: str = Field(description="Workspace source file or directory.")
    spec_path: str = Field(
        description="Workspace analysis_spec.json or augmentation_plan.json."
    )
    output_dir: str = Field(
        description="Workspace directory for the three-sample result."
    )
    runtime_profile: RuntimeProfile


class DatasetRunObservation(Observation):
    job_kind: JobKind
    output_dir: str | None = None
    output_path: str | None = None
    record_count: int = 0
    sample_records: list[dict[str, Any]] = Field(default_factory=list)
    exit_code: int = -1
    stdout_tail: str = ""
    stderr_tail: str = ""

    @property
    def visualize(self) -> Text:
        text = Text()
        text.append(f"{self.job_kind} sample: ", style="bold cyan")
        text.append("passed" if not self.is_error else "failed")
        return text


class DatasetRunExecutor(ToolExecutor[DatasetRunAction, DatasetRunObservation]):
    """Run an agent-authored job against a tool-owned maximum of three records."""

    def __init__(self, kind: JobKind, runtime_dir: Path | None = None) -> None:
        self._kind = kind
        self._runtime_dir = runtime_dir
        self._process = ProcessLocalSampleExecutor()

    def interrupt(self) -> None:
        self._process.interrupt()

    def __call__(
        self, action: DatasetRunAction, conversation: Any = None
    ) -> DatasetRunObservation:
        try:
            source = _resolve_input_path(conversation, action.input_path)
            script = resolve_workspace_file(conversation, action.script_path)
            spec_path = resolve_workspace_file(conversation, action.spec_path)
            if script.suffix.lower() != ".py" or spec_path.suffix.lower() != ".json":
                raise ValueError("script_path must be .py and spec_path must be .json")
            spec = _load_contract(spec_path, self._kind)
            _validate_profile(self._kind, action.runtime_profile, spec)
            _validate_pipeline_runtime(
                script=script,
                profile=action.runtime_profile,
                spec=spec,
                runtime_dir=self._runtime_dir,
            )
            declared_source = _resolve_input_path(conversation, spec.source_path)
            if declared_source.resolve() != source.resolve():
                raise ValueError("spec source_path must match input_path")
            source_fingerprint_before = _local_source_fingerprint(source)
            output_dir = _resolve_output_path(conversation, action.output_dir)
            try:
                output_dir.relative_to(source.resolve())
            except ValueError:
                pass
            else:
                raise ValueError("output_dir must not be inside the source directory")
            output_dir.mkdir(parents=True, exist_ok=True)
            sample_input, expected_records = _materialize_sample(
                source,
                spec.adapter,
                output_dir,
                limit=3,
                require_explicit_train=self._kind == "data_synthesis",
            )
            output_name = (
                "labels.jsonl"
                if self._kind == "dataset_analysis"
                else "synthesized.jsonl"
            )
            output_path = output_dir / output_name
            needs_model = not (
                action.runtime_profile == "avi_pcb_cpu"
                or (
                    isinstance(spec, AnalysisSpec)
                    and spec.label_source == "existing_labels"
                )
            )
            env = (
                build_dataflow_env(
                    conversation,
                    "vision" if action.runtime_profile == "dataflow_vision" else "text",
                )
                if needs_model
                else {}
            )
            env["DATASET_SAMPLE_LIMIT"] = "3"
            env["DATASET_RUNTIME_PROFILE"] = action.runtime_profile
            if self._runtime_dir is not None:
                existing = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = os.pathsep.join(
                    value for value in (str(self._runtime_dir), existing) if value
                )
            python = (
                resolve_dataflow_python(None)
                if action.runtime_profile.startswith("dataflow_")
                else os.environ.get("AVI_PCB_PYTHON", sys.executable)
            )
            rc, stdout, stderr = self._process.run(
                python,
                [str(script), str(sample_input), str(output_path), str(spec_path)],
                cwd=str(script.parent),
                env_extra=env,
                timeout=3600,
            )
            records = _read_jsonl(output_path, 3)
            if rc != 0 or not output_path.is_file() or len(records) != expected_records:
                raise RuntimeError(
                    stderr[-4000:]
                    or "dataset sample failed or did not produce one output per input"
                )
            if _local_source_fingerprint(source) != source_fingerprint_before:
                raise RuntimeError("source dataset changed during local sample")
            _validate_local_artifacts(
                output_dir=output_dir,
                spec=spec,
                output_records=records,
                expected_records=expected_records,
            )
            _write_local_validation(
                output_dir,
                self._kind,
                output_name,
                len(records),
                source_fingerprint_before,
            )
            return DatasetRunObservation.from_text(
                text=(
                    f"Local {self._kind} sample passed; review before full submission."
                ),
                job_kind=self._kind,
                output_dir=str(output_dir),
                output_path=str(output_path),
                record_count=len(records),
                sample_records=records,
                exit_code=0,
                stdout_tail=stdout[-4000:],
                stderr_tail=stderr[-4000:],
            )
        except Exception as exc:
            return DatasetRunObservation.from_text(
                text=str(exc), job_kind=self._kind, is_error=True, exit_code=1
            )


def _materialize_sample(
    source: Path,
    adapter_name: str,
    output_dir: Path,
    limit: int,
    *,
    require_explicit_train: bool,
) -> tuple[Path, int]:
    adapter = create_adapter(adapter_name)
    records = []
    for record in adapter.iter_records(source):
        if record.is_evaluation:
            continue
        if require_explicit_train and record.split != "train":
            raise ValueError(
                f"sample {record.sample_id!r} has no explicit train split; "
                "synthesis will not guess whether it belongs to validation"
            )
        records.append(record)
        if len(records) == limit:
            break
    if not records:
        raise ValueError("no unambiguous training samples were found")
    sample_root = Path(tempfile.mkdtemp(prefix=".dataset-sample-", dir=output_dir))
    if adapter_name != "avi_pcb":
        target = sample_root / "input.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for record in records:
                value = dict(record.value)
                if adapter_name == "vision_manifest":
                    value["images"] = [str(path) for path in record.media_paths]
                    value.pop("image_path", None)
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        return target, len(records)
    target = sample_root / "input"
    target.mkdir()
    for record in records:
        destination = target / record.source_path.name
        shutil.copytree(record.source_path, destination)
    return target, len(records)


def _read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("job output JSONL records must be objects")
            values.append(value)
            if len(values) > limit:
                break
    return values


def _validate_local_artifacts(
    *,
    output_dir: Path,
    spec: AnalysisSpec | AugmentationPlan,
    output_records: list[dict[str, Any]],
    expected_records: int,
) -> None:
    (output_dir / "failures.jsonl").touch(exist_ok=True)
    if isinstance(spec, AnalysisSpec):
        report_path = output_dir / "distribution_report.json"
        if not report_path.is_file():
            raise ValueError("analysis sample must write distribution_report.json")
        report = DistributionReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
        if report.analysis_mode != "exact" or report.total_records != expected_records:
            raise ValueError("local analysis report must exactly describe the sample")
        (output_dir / "taxonomy.json").write_text(
            spec.taxonomy.model_dump_json(indent=2), encoding="utf-8"
        )
        (output_dir / "analysis_spec.json").write_text(
            spec.model_dump_json(indent=2), encoding="utf-8"
        )
        return
    provenance = _read_jsonl(output_dir / "provenance.jsonl", expected_records)
    if len(provenance) != len(output_records):
        raise ValueError("synthesis sample needs one provenance row per output")
    if spec.adapter == "avi_pcb" and not (output_dir / "assets").is_dir():
        raise ValueError("AVI synthesis sample must write assets/")
    if spec.adapter != "avi_pcb":
        (output_dir / "assets").mkdir(exist_ok=True)
    (output_dir / "augmentation_plan.json").write_text(
        spec.model_dump_json(indent=2), encoding="utf-8"
    )


def _write_local_validation(
    output_dir: Path,
    kind: JobKind,
    output_name: str,
    count: int,
    source_fingerprint: str,
) -> None:
    payload = {
        "schema_version": 1,
        "kind": kind,
        "valid": True,
        "primary_output": output_name,
        "records": count,
        "source_fingerprint": source_fingerprint,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _local_source_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    files = (
        [path]
        if path.is_file()
        else sorted(item for item in path.rglob("*") if item.is_file())
    )
    for item in files:
        if path.is_dir():
            digest.update(item.relative_to(path).as_posix().encode())
            digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class DatasetSubmitAction(Action):
    script_path: str
    input_path: str = Field(description="Source path in Pyromind Storage.")
    spec_path: str
    runtime_profile: RuntimeProfile
    local_sample_approved: bool = False
    taxonomy_approved: bool = False
    mode: Literal["full", "resume"] = "full"
    resume_run_id: uuid.UUID | None = None
    reuse_assessment: ReuseAssessment | None = None

    @model_validator(mode="after")
    def require_sample_gate(self) -> DatasetSubmitAction:
        if not self.local_sample_approved:
            raise ValueError("the three-record local sample must be approved")
        if self.mode == "resume" and self.resume_run_id is None:
            raise ValueError("resume_run_id is required in resume mode")
        if self.mode == "full" and self.resume_run_id is not None:
            raise ValueError("resume_run_id is only valid in resume mode")
        return self


class DatasetSubmitExecutor(
    ToolExecutor[DatasetSubmitAction, DfSubmitPipelineObservation]
):
    def __init__(self, kind: JobKind, **params: Any) -> None:
        self._kind = kind
        self._params = params

    def __call__(
        self, action: DatasetSubmitAction, conversation: Any = None
    ) -> DfSubmitPipelineObservation:
        try:
            spec_path = resolve_workspace_file(conversation, action.spec_path)
            script_path = resolve_workspace_file(conversation, action.script_path)
            spec = _load_contract(spec_path, self._kind)
            _validate_profile(self._kind, action.runtime_profile, spec)
            runtime_value = self._params.get("runtime_dir")
            _validate_pipeline_runtime(
                script=script_path,
                profile=action.runtime_profile,
                spec=spec,
                runtime_dir=(Path(str(runtime_value)) if runtime_value else None),
            )
            if _normalize_storage_path(spec.source_path, "spec.source_path") != (
                _normalize_storage_path(action.input_path, "input_path")
            ):
                raise ValueError("spec source_path must match input_path")
            if action.mode == "resume" and spec.source_fingerprint is None:
                raise ValueError("resume requires source_fingerprint in the job spec")
            if (
                isinstance(spec, AnalysisSpec)
                and spec.label_source == "inferred_taxonomy"
                and not action.taxonomy_approved
            ):
                raise ValueError("taxonomy must be approved before full labeling")
        except Exception as exc:
            return DfSubmitPipelineObservation.from_text(
                text=str(exc), status="Failed", is_error=True
            )
        output_filename = (
            "labels.jsonl" if self._kind == "dataset_analysis" else "synthesized.jsonl"
        )
        delegate = DfSubmitPipelineExecutor(
            **self._params,
            task_kind=self._kind,
            output_namespace=self._kind,
            output_filename=output_filename,
            python_packages=_PROFILE_PACKAGES[action.runtime_profile],
            requires_llm=not (
                action.runtime_profile == "avi_pcb_cpu"
                or (
                    isinstance(spec, AnalysisSpec)
                    and spec.label_source == "existing_labels"
                )
            ),
        )
        cpu, memory = _PROFILE_RESOURCES[action.runtime_profile]
        return delegate(
            DfSubmitPipelineAction(
                script_path=action.script_path,
                support_file_path=action.spec_path,
                input_path=action.input_path,
                mode=action.mode,
                resume_run_id=action.resume_run_id,
                reuse_assessment=action.reuse_assessment,
                model_profile=(
                    "vision" if action.runtime_profile == "dataflow_vision" else "text"
                ),
                cpu=cpu,
                memory=memory,
            ),
            conversation,
        )


class DatasetCheckTaskAction(Action):
    output_dir: str
    job_kind: JobKind
    tail_lines: int = Field(default=5, ge=0, le=50)


class DatasetCheckTaskExecutor(
    ToolExecutor[DatasetCheckTaskAction, DfCheckProgressObservation]
):
    def __init__(self, **params: Any) -> None:
        self._params = params

    def __call__(
        self, action: DatasetCheckTaskAction, conversation: Any = None
    ) -> DfCheckProgressObservation:
        output_name = (
            "labels.jsonl"
            if action.job_kind == "dataset_analysis"
            else "synthesized.jsonl"
        )
        delegate = DfCheckProgressExecutor(**self._params, output_filename=output_name)
        return delegate(
            DfCheckProgressAction(
                output_dir=action.output_dir, tail_lines=action.tail_lines
            ),
            conversation,
        )


class DatasetStopTaskAction(DfStopTaskAction):
    pass


def _tool_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value is not None}


class _RunTool(ToolDefinition[DatasetRunAction, DatasetRunObservation]):
    job_kind: ClassVar[JobKind]

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> Sequence[Self]:  # noqa: ARG003
        runtime_value = params.pop("runtime_dir", None)
        runtime_dir = Path(str(runtime_value)) if runtime_value else None
        if params:
            raise ValueError(f"unexpected local dataset tool params: {sorted(params)}")
        return [
            cls(
                description=cls.__doc__ or "",
                action_type=DatasetRunAction,
                observation_type=DatasetRunObservation,
                executor=DatasetRunExecutor(cls.job_kind, runtime_dir),
                annotations=ToolAnnotations(title=cls.name, readOnlyHint=False),
            )
        ]


class RunDatasetAnalysisTool(_RunTool):
    """Run up to three analysis records; inferred labels need approved taxonomy."""

    name = "run_dataset_analysis"
    job_kind = "dataset_analysis"


class RunDataSynthesisTool(_RunTool):
    """Run at most three synthesis records locally for human review."""

    name = "run_data_synthesis"
    job_kind = "data_synthesis"


class _SubmitTool(ToolDefinition[DatasetSubmitAction, DfSubmitPipelineObservation]):
    job_kind: ClassVar[JobKind]

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> Sequence[Self]:  # noqa: ARG003
        return [
            cls(
                description=cls.__doc__ or "",
                action_type=DatasetSubmitAction,
                observation_type=DfSubmitPipelineObservation,
                executor=DatasetSubmitExecutor(cls.job_kind, **_tool_params(params)),
                annotations=ToolAnnotations(
                    title=cls.name, readOnlyHint=False, openWorldHint=True
                ),
            )
        ]


class SubmitDatasetAnalysisTool(_SubmitTool):
    """Submit approved full dataset analysis with a pinned runtime profile."""

    name = "submit_dataset_analysis"
    job_kind = "dataset_analysis"


class SubmitDataSynthesisTool(_SubmitTool):
    """Submit synthesis; its plan must contain an approved Gap."""

    name = "submit_data_synthesis"
    job_kind = "data_synthesis"


class CheckDatasetTaskTool(
    ToolDefinition[DatasetCheckTaskAction, DfCheckProgressObservation]
):
    name = "check_dataset_task"

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> Sequence[Self]:  # noqa: ARG003
        return [
            cls(
                description="Check analysis or synthesis progress and recent records.",
                action_type=DatasetCheckTaskAction,
                observation_type=DfCheckProgressObservation,
                executor=DatasetCheckTaskExecutor(**_tool_params(params)),
                annotations=ToolAnnotations(title=cls.name, readOnlyHint=True),
            )
        ]


class StopDatasetTaskTool(ToolDefinition[DatasetStopTaskAction, DfStopTaskObservation]):
    name = "stop_dataset_task"

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> Sequence[Self]:  # noqa: ARG003
        return [
            cls(
                description="Stop an owned dataset analysis or synthesis task.",
                action_type=DatasetStopTaskAction,
                observation_type=DfStopTaskObservation,
                executor=DfStopTaskExecutor(**_tool_params(params)),
                annotations=ToolAnnotations(
                    title=cls.name, readOnlyHint=False, destructiveHint=True
                ),
            )
        ]


for _name, _tool in (
    (RunDatasetAnalysisTool.name, RunDatasetAnalysisTool),
    (SubmitDatasetAnalysisTool.name, SubmitDatasetAnalysisTool),
    (RunDataSynthesisTool.name, RunDataSynthesisTool),
    (SubmitDataSynthesisTool.name, SubmitDataSynthesisTool),
    (CheckDatasetTaskTool.name, CheckDatasetTaskTool),
    (StopDatasetTaskTool.name, StopDatasetTaskTool),
):
    register_tool(_name, _tool)
