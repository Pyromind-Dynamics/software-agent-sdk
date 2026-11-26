from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from pyromind_agent_server.echomind_graph_runs import GraphRun, GraphRunStore
from pyromind_agent_server.echomind_projects import ProjectNode
from pyromind_agent_server.echomind_publication import PublicationStore, PublishedGraph
from pyromind_agent_server.echomind_stage_runs import ProjectRunStore, StageRun
from pyromind_agent_server.echomind_storage import (
    canonical_uuid,
    project_directory,
    read_model,
    write_model,
)
from pyromind_runtime.domain.echomind_model_reports import ModelEvaluationRecord
from pyromind_runtime.domain.storage_paths import normalize_workspace_path


class GraphModelReportStore:
    """Adapt immutable GraphRun evidence to the product model-report catalog."""

    def __init__(self, projects_root: Path | str) -> None:
        self.root = Path(projects_root)
        self.runs = GraphRunStore(self.root)
        self.stages = ProjectRunStore(self.root)
        self.publications = PublicationStore(self.root)

    def list_candidates(self, project_id: str) -> list[ModelEvaluationRecord]:
        records: list[ModelEvaluationRecord] = []
        for run in self.runs.list(project_id):
            if run.status != "succeeded":
                continue
            record = self._candidate(run)
            if record is None:
                continue
            try:
                cached = read_model(
                    self._cache_path(project_id, record.candidate_id),
                    ModelEvaluationRecord,
                )
            except FileNotFoundError:
                cached = None
            if cached is not None and (
                cached.project_id == record.project_id
                and cached.candidate_id == record.candidate_id
                and cached.release_digest == record.release_digest
                and cached.report_uri == record.report_uri
                and cached.model_uri == record.model_uri
                and cached.report_artifact_id == record.report_artifact_id
            ):
                record = cached.model_copy(
                    update={
                        "training_base_uri": record.training_base_uri,
                        "deployment_model_uri": record.deployment_model_uri,
                    }
                )
            records.append(record)
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def save_report(self, record: ModelEvaluationRecord) -> None:
        write_model(self._cache_path(record.project_id, record.candidate_id), record)

    def _cache_path(self, project_id: str, candidate_id: str) -> Path:
        return (
            project_directory(self.root, project_id)
            / "model_reports"
            / f"{canonical_uuid(candidate_id, kind='candidate')}.json"
        )

    def _candidate(self, run: GraphRun) -> ModelEvaluationRecord | None:
        stages = self.stages.list_for_graph_run(
            project_id=run.project_id, graph_run_id=run.run_id
        )
        successful = [stage for stage in stages if stage.status == "succeeded"]
        evaluators = [stage for stage in successful if stage.node_type == "evaluator"]
        if not evaluators:
            return None
        evaluation = evaluators[-1]
        published = self.publications.load(
            project_id=run.project_id, release_id=run.release_id
        )
        if published.content_sha256 != run.release_digest:
            raise ValueError("Model report release does not match GraphRun")
        node = next(node for node in published.nodes if node.id == evaluation.node_id)
        model = _input_uri(node, "candidate_model", run, successful)
        report_ref = next(
            (
                ref
                for ref in evaluation.output_refs
                if ref.artifact_type == "evaluation.report"
            ),
            None,
        )
        if model is None or report_ref is None:
            return None
        artifact_id = report_ref.metadata.get("artifact_id")
        if not isinstance(artifact_id, str):
            return None
        document = self.stages.load_artifact(
            project_id=run.project_id, artifact_id=artifact_id
        )
        if (
            document.artifact.run_id != evaluation.run_id
            or document.artifact.kind != "evaluation.report"
        ):
            raise ValueError("Evaluation report belongs to another stage")
        report_uri = document.content.get("uri")
        if not isinstance(report_uri, str):
            return None
        config = node.config.get("evaluator_runtime", {})
        if not isinstance(config, dict):
            config = {}
        training = _candidate_training_node(published, node)
        base_uri = (
            _input_uri(training, "base_model", run, successful) if training else None
        ) or _input_uri(node, "base_model", run, successful)
        deployable = _merged_model_uri(node, evaluation, run)
        if deployable is None and training is not None:
            runtime = training.config.get("train_runtime")
            if isinstance(runtime, dict) and runtime.get("artifact_selection") == (
                "merged_model"
            ):
                deployable = normalize_workspace_path(model)
        return ModelEvaluationRecord(
            candidate_id=run.run_id,
            project_id=canonical_uuid(run.project_id, kind="project"),
            run_id=run.run_id,
            release_id=run.release_id,
            release_digest=run.release_digest,
            round_no=run.round_no,
            execution_mode=run.execution_mode,
            source_batch_id=run.source_batch_id or _batch_id(run),
            model_uri=normalize_workspace_path(model),
            deployment_model_uri=deployable,
            training_base_uri=base_uri,
            report_uri=normalize_workspace_path(report_uri),
            report_artifact_id=artifact_id,
            evaluation_dataset_uri=_input_uri(
                node, "validation_dataset", run, successful
            ),
            evaluation_config_uri=_evaluation_config_uri(config),
            created_at=run.finished_at or run.created_at,
        )


def _batch_id(run: GraphRun) -> str | None:
    batch = run.inputs.get("tracer_batch")
    if isinstance(batch, dict) and isinstance(batch.get("batch_id"), str):
        return batch["batch_id"]
    return None


def _input_uri(
    node: ProjectNode, port: str, run: GraphRun, stages: list[StageRun]
) -> str | None:
    definition = node.inputs.get(port)
    if definition is None:
        return None
    binding = definition.binding
    if binding.kind == "run_input":
        value = run.inputs.get(binding.name or "")
        return value if isinstance(value, str) else None
    if binding.kind == "literal":
        return binding.value if isinstance(binding.value, str) else None
    if binding.kind == "node_output":
        for stage in reversed(stages):
            if stage.node_id == binding.node_id:
                for ref in stage.output_refs:
                    if ref.metadata.get("port") == binding.port:
                        uri = ref.metadata.get("source_uri")
                        return uri if isinstance(uri, str) else None
    return None


def _candidate_training_node(
    published: PublishedGraph, evaluator: ProjectNode
) -> ProjectNode | None:
    binding = evaluator.inputs["candidate_model"].binding
    if binding.kind != "node_output" or binding.port != "trained_model":
        return None
    return next(
        (
            node
            for node in published.nodes
            if node.id == binding.node_id
            and node.type == "train"
            and node.executor == "train_runtime"
        ),
        None,
    )


def _merged_model_uri(node: ProjectNode, stage: StageRun, run: GraphRun) -> str | None:
    config = node.config.get("evaluator_runtime")
    if not isinstance(config, dict):
        return None
    workflow = config.get("workflow")
    if not isinstance(workflow, dict):
        return None
    templates = config.get("workflow_templates", {})
    bindings = config.get("workflow_bindings", {})
    if not isinstance(templates, dict) or not isinstance(bindings, dict):
        return None
    for index, item in enumerate(workflow.get("nodes", [])):
        if not isinstance(item, dict):
            continue
        data = item.get("data", {})
        if not isinstance(data, dict) or data.get("nodeType") != "ModelMergeLoraNode":
            continue
        if bindings.get(f"nodes[{index}].data.config.lora_path") != "candidate_model":
            continue
        template = templates.get(f"nodes[{index}].data.config.output_path")
        if not isinstance(template, str):
            return None
        try:
            rendered = template.format(
                graph_run_id=run.run_id,
                stage_run_id=stage.run_id,
                project_id=run.project_id,
            )
            return normalize_workspace_path(rendered)
        except (KeyError, ValueError):
            return None
    return None


def _evaluation_config_uri(config: dict[str, Any]) -> str | None:
    command = config.get("command_template")
    if not isinstance(command, str):
        return None
    try:
        tokens = shlex.split(command)
        path = tokens[tokens.index("--evaluation-config") + 1]
        return normalize_workspace_path(path) if "{" not in path else None
    except (ValueError, IndexError):
        return None
