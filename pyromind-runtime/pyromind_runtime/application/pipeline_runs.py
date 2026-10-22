from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from pyromind_runtime.application.versions import Versions, snapshot_result
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.versions import (
    RunSource,
    SaveVersion,
    SnapshotSource,
    canonical,
)
from pyromind_runtime.ports.external_tasks import ExternalTaskRegistry
from pyromind_runtime.ports.product_store import ProductStore


class PipelineRuns:
    """Reserve immutable pipeline runs and coordinate production submissions."""

    def __init__(
        self,
        conversation_id: str,
        store: ProductStore,
        registry: ExternalTaskRegistry,
        versions: Versions | None = None,
        context: RequestContext | None = None,
    ) -> None:
        self.conversation_id = conversation_id
        self._store = store
        self._registry = registry
        self._versions = versions
        self._context = context

    def with_context(self, context: RequestContext) -> PipelineRuns:
        return PipelineRuns(
            self.conversation_id, self._store, self._registry, self._versions, context
        )

    def prepare_execution(
        self, run_id: str, workflow: dict[str, Any], dsl: str, script: str
    ) -> tuple[dict[str, Any], str]:
        if self._versions is None:
            return workflow, dsl
        if self._context is None:
            raise ValueError("versioned execution requires request credentials")
        run = self._store.load_pipeline_runs()[run_id]
        project = self._versions.project(self.conversation_id, self._context)
        assert project is not None
        prior = next(
            (
                item
                for item in reversed(self._versions.store.list_runs(project.id))
                if item.source.source_id == run_id
                and item.source.execution_revision < run.execution_revision
            ),
            None,
        )
        key = f"pipeline:{run_id}:{run.execution_revision}"
        existing = next(
            (
                item
                for item in self._versions.store.list_runs(project.id)
                if item.request_key == key
            ),
            None,
        )
        if existing is not None:
            for identity in existing.spec.inputs.values():
                version = self._versions.store.get_version(project.id, identity)
                if version.state == "ready":
                    self._versions.snapshots.verify(
                        snapshot_result(version), self._context
                    )
            saved = existing.spec.parameters["definition"]
            assert isinstance(saved, dict)
            saved_workflow = saved["workflow"]
            assert isinstance(saved_workflow, dict)
            code = self._versions.store.get_version(
                project.id, existing.spec.inputs["code"]
            )
            saved_dsl = self._versions.snapshots.read_text(
                snapshot_result(code), self._context
            )
            return saved_workflow, saved_dsl
        inputs: dict[str, str] = {}
        input_definition = run.definition
        evaluation = run.definition.get("evaluation_config", {})
        assert isinstance(evaluation, dict)
        source = evaluation.get("model_source", {})
        assert isinstance(source, dict)
        selected_checkpoint = source.get("type") == "checkpoint"
        generated_model = source.get("type") in {"clone", "download", "checkpoint"}
        original_inference = run.definition.get("inference", {})
        assert isinstance(original_inference, dict)
        if generated_model:
            input_definition = {
                **run.definition,
                "inference": {
                    key: value
                    for key, value in original_inference.items()
                    if key != "model_path"
                },
            }
        if selected_checkpoint:
            input_definition = {
                **input_definition,
                "evaluation_config": {
                    **evaluation,
                    "model_source": {
                        k: v for k, v in source.items() if k != "base_model"
                    },
                },
            }
            for role, reference in (
                ("selected_checkpoint", original_inference["model_path"]),
                ("base_model", source.get("base_model")),
            ):
                if isinstance(reference, str):
                    version = self._versions.save(
                        project,
                        key + ":" + role,
                        SaveVersion(
                            kind="model",
                            logical_key=role,
                            source=SnapshotSource(
                                storage_ref=reference.removeprefix("/workspace"),
                                reference_only=True,
                            ),
                        ),
                        self._context,
                    )
                    inputs[role] = version.id
        effective_definition = (
            input_definition
            if run.definition.get("kind")
            in {"prediction_evaluation", "evaluation_products", "badcase_review"}
            or run.definition.get("dataset_role") == "test"
            else self._versions.resolve_inputs(
                project, input_definition, inputs, self._context, key
            )
        )
        if selected_checkpoint:
            effective_definition["evaluation_config"] = evaluation
        effective_inference = effective_definition.get("inference", {})
        assert isinstance(effective_inference, dict)
        if generated_model:
            effective_inference["model_path"] = original_inference["model_path"]
        if prior is not None:
            previous_spec = prior.spec.parameters.get("definition")
            if (
                isinstance(previous_spec, dict)
                and previous_spec.get("definition") != effective_definition
            ):
                raise ValueError("selected inputs changed; use a new full run")
        replacements: dict[str, str] = {}
        for old, new in (
            (run.definition.get("input_path"), effective_definition.get("input_path")),
            (
                original_inference.get("model_path"),
                effective_inference.get("model_path"),
            ),
        ):
            if isinstance(old, str) and isinstance(new, str) and old != new:
                replacements[old] = new
        for role, filename, text, kind in (
            (
                "dataset_config",
                "dataset_config.json",
                canonical(effective_definition["dataset_config"]),
                "config",
            ),
            (
                "evaluation_config",
                "evaluation_config.json",
                canonical(effective_definition["evaluation_config"]),
                "config",
            ),
            ("runtime_script", "evaluate_inference.py", script, "code"),
            *(
                (
                    (
                        "scoring_script",
                        "score_eval.py",
                        str(effective_definition["scoring_script"]),
                        "code",
                    ),
                    (
                        "prediction_evaluation",
                        "prediction_evaluation.json",
                        canonical(
                            {
                                "source_manifest": effective_definition[
                                    "source_manifest"
                                ],
                                "scoring_spec": effective_definition["scoring_spec"],
                            }
                        ),
                        "config",
                    ),
                )
                if effective_definition.get("kind") == "prediction_evaluation"
                else ()
            ),
            *(
                (
                    (
                        "products_definition",
                        "products_definition.json",
                        canonical(effective_definition),
                        "config",
                    ),
                )
                if effective_definition.get("products_operation")
                else ()
            ),
            *(
                (
                    (
                        "business_scoring",
                        "business_scoring.json",
                        canonical(
                            {
                                **cast(dict, effective_definition["business_scoring"]),
                                "test_source": effective_definition["test_source"],
                            }
                        ),
                        "config",
                    ),
                )
                if effective_definition.get("business_scoring")
                else ()
            ),
        ):
            version = self._versions.save(
                project,
                key + ":" + role,
                SaveVersion.model_validate(
                    {
                        "kind": kind,
                        "logical_key": role,
                        "source": {"text": text, "filename": filename},
                    }
                ),
                self._context,
            )
            inputs[role] = version.id
            replacements[run.output_dir + "/" + filename] = (
                version.manifest.storage_ref or ""
            )

        def rewrite(value: Any) -> Any:
            if isinstance(value, dict):
                return {name: rewrite(item) for name, item in value.items()}
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            if isinstance(value, str):
                for old, new in sorted(
                    replacements.items(), key=lambda pair: -len(pair[0])
                ):
                    value = value.replace(old, new)
            return value

        workflow = rewrite(workflow)
        dsl = rewrite(dsl)
        record = self._versions.prepare_run(
            self.conversation_id,
            self._context,
            f"pipeline:{run_id}:{run.execution_revision}",
            RunSource(
                kind="pipeline",
                source_id=run_id,
                execution_revision=run.execution_revision,
                retry_of=prior.id if prior else None,
            ),
            {
                "definition": effective_definition,
                "workflow": workflow,
                "runtime_script": script,
            },
            code=dsl,
            kind="evaluate",
            resolved_inputs=inputs,
        )
        effective = record.spec.parameters["definition"]
        assert isinstance(effective, dict)
        effective_workflow = effective["workflow"]
        assert isinstance(effective_workflow, dict)
        return effective_workflow, dsl

    def resolve(self, run_id: str) -> dict[str, Any] | None:
        run = self._store.load_pipeline_runs().get(str(UUID(run_id)))
        if run is None:
            return None
        if run.conversation_id != self.conversation_id:
            raise ValueError("pipeline run belongs to another conversation")
        return run.model_dump(mode="json")

    def reserve(
        self, definition: dict[str, Any], resume_run_id: str | None
    ) -> dict[str, Any]:
        from pyromind_runtime.application.test_evaluation import final_test_definition
        from pyromind_runtime.domain.versions import check_persistable

        check_persistable(definition)
        definition = final_test_definition(
            self._store, self.conversation_id, definition
        )
        if definition.get("kind") == "prediction_evaluation":
            from pyromind_runtime.domain.training_evaluation import (
                CheckpointSelectionPolicy,
            )
            from pyromind_runtime.domain.versions import check_persistable

            check_persistable(definition)
            CheckpointSelectionPolicy.model_validate(
                definition["scoring_spec"].get("selection", {})
            )
        fingerprint = hashlib.sha256(
            json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        previous = self.resolve(resume_run_id) if resume_run_id else None
        if resume_run_id and previous is None:
            raise ValueError("unknown inference run in this conversation")
        expected_revision = None
        if previous is not None:
            if self._versions is not None and self._context is not None:
                project = self._versions.project(
                    self.conversation_id, self._context, create=False
                )
                if project is not None:
                    self._versions.guard_resume(project, str(previous["run_id"]))
            prior = PipelineRun.model_validate(previous)
            if prior.fingerprint != fingerprint:
                raise ValueError("evaluation configuration changed; use a new full run")
            if prior.submission_status in {"preparing", "uncertain"}:
                raise ValueError(
                    "submission is in progress or uncertain; reconcile it first"
                )
            if prior.task_id:
                task = self._registry.resolve(self.conversation_id, prior.task_id)
                projected = next(
                    (
                        item
                        for item in self._store.load_snapshot().external_tasks
                        if item.task_id == prior.task_id
                    ),
                    None,
                )
                status = (
                    projected.status
                    if projected is not None
                    else (task.get("status") if task else None)
                )
                if status not in {
                    "succeeded",
                    "failed",
                    "terminated",
                    "stopped",
                }:
                    raise ValueError(
                        "pipeline task is still active; stop it before resume"
                    )
            expected_revision = prior.revision
            run = prior.model_copy(
                update={
                    "revision": prior.revision + 1,
                    "execution_revision": prior.execution_revision + 1,
                    "task_id": None,
                    "submission_status": "preparing",
                }
            )
        else:
            run_id = uuid4()
            run = PipelineRun(
                run_id=run_id,
                conversation_id=self.conversation_id,
                definition=definition,
                fingerprint=fingerprint,
                output_dir=(
                    f"/.pyromind-agent/{self.conversation_id}/data_preparation/{run_id}"
                ),
            )
        self._store.save_pipeline_run(run, expected_revision=expected_revision)
        return run.model_dump(mode="json")

    def finish_submission(
        self,
        run_id: str,
        revision: int,
        task_id: str | None,
        status: Literal["submitted", "failed", "uncertain"],
    ) -> None:
        value = self.resolve(run_id)
        if value is None:
            raise ValueError("unknown pipeline run")
        run = PipelineRun.model_validate(value)
        if run.revision != revision or run.submission_status != "preparing":
            raise ValueError("pipeline submission has already changed")
        run = run.model_copy(
            update={
                "revision": revision + 1,
                "task_id": task_id,
                "submission_status": status,
            }
        )
        self._store.save_pipeline_run(run, expected_revision=revision)
        if task_id:
            now = datetime.now(UTC).isoformat()
            payload: JsonObject = {
                "task_id": task_id,
                "run_id": run_id,
                "kind": "data_preparation",
                "status": "pending",
                "output_dir": run.output_dir,
                "submitted_at": now,
                "updated_at": now,
                "resume_pending": False,
            }
            self._registry.register(self.conversation_id, payload)
            if self._versions is not None:
                project = self._versions.store.bound_project(self.conversation_id)
                if project is not None:
                    for record in self._versions.store.list_runs(project.id):
                        if (
                            record.request_key
                            == f"pipeline:{run_id}:{run.execution_revision}"
                        ):
                            self._versions.store.link_task(
                                project.id, record.id, task_id
                            )

    def task_for(
        self, task_id: str | None, run_id: str | None, output_dir: str | None
    ) -> str | None:
        for run in self._store.load_pipeline_runs().values():
            if run.conversation_id != self.conversation_id:
                continue
            if (
                (task_id and run.task_id == task_id)
                or (run_id and str(run.run_id) == run_id)
                or (output_dir and run.output_dir == output_dir.rstrip("/"))
            ):
                return run.task_id
        return None
