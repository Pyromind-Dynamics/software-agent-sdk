"""Validate independently executed review tasks using Product Store and Storage."""

import hashlib
import json
from typing import cast

from pyromind_runtime.application.training_evaluations import digest
from pyromind_runtime.domain.review import DatasetProfile, ReviewTask
from pyromind_runtime.ports.product_store import EvaluationCommitStore, ProductStore
from pyromind_runtime.ports.storage import EvaluationStorage


class ReviewTasks:
    def __init__(self, store: ProductStore, storage: EvaluationStorage):
        self.store = store
        self.storage = storage

    def ready(self, run_id: str) -> ReviewTask:
        run = self.store.load_pipeline_runs()[run_id]
        if run.definition.get("kind") != "badcase_review":
            raise ValueError("not an independent review task")
        frozen = run.definition["products_manifest"]
        assert isinstance(frozen, dict)
        binding = json.loads(self.storage.read(run.output_dir + "/run_manifest.json"))
        task = ReviewTask.model_validate_json(
            self.storage.read(run.output_dir + "/review_task.json")
        )
        profile = DatasetProfile.model_validate_json(
            self.storage.read(run.output_dir + "/dataset_profile.json")
        )
        source = frozen["source"]
        assert isinstance(source, dict)
        profile_ref = frozen["profile"]
        assert isinstance(profile_ref, dict)
        if (
            binding.get("run_id") != run_id
            or binding.get("kind") != "badcase_review"
            or binding.get("products_manifest_sha256") != digest(frozen)
            or task.review_id != run_id
            or task.source_kind != source["kind"]
            or task.source_path != source["path"]
            or task.source_sha256 != source["sha256"]
            or task.profile_sha256 != profile_ref["sha256"]
            or task.scoring_run_id != source.get("scoring_run_id")
            or task.scoring_spec_sha256
            != (source.get("scoring_spec_sha256") or digest(source))
            or binding.get("evaluation_config_sha256") != task.scoring_spec_sha256
        ):
            raise ValueError("review artifacts do not match the frozen task")
        original = self.storage.read(str(profile_ref["path"]))
        if (
            hashlib.sha256(original).hexdigest() != task.profile_sha256
            or DatasetProfile.model_validate_json(original) != profile
            or profile.status != "ready"
            or profile.issues
            or profile.datasets != frozen.get("datasets")
            or profile.label_mapping != frozen.get("label_mapping", {})
            or not {"train", "validation"} <= profile.distribution.keys()
        ):
            raise ValueError("review requires a complete matching dataset profile")
        result = task.model_dump(mode="json")
        if run.result != result:
            self.store.save_pipeline_run(
                run.model_copy(update={"result": result, "revision": run.revision + 1}),
                expected_revision=run.revision,
            )
        return task

    def refresh_context(self, run_id: str) -> None:
        """Update feedback evidence without re-rendering reports or reselecting."""
        from pyromind_runtime.application.evaluation_annotations import (
            EvaluationAnnotationService,
        )

        task = self.ready(run_id)
        runs = self.store.load_pipeline_runs()
        source = runs.get(task.scoring_run_id or "")
        if source is None or not source.definition.get("parent_run_id"):
            return
        parent = runs.get(str(source.definition["parent_run_id"]))
        if parent is None:
            return
        output = runs[run_id].output_dir
        # Annotation persistence remains in the existing commit protocol.
        service = EvaluationAnnotationService(
            storage=self.storage,
            commits=cast(EvaluationCommitStore, self.store),
        )
        state = service.load(run_id=run_id, output_dir=output)
        path = parent.output_dir + "/iteration_context.json"
        context = json.loads(self.storage.read(path))
        reviews = context.setdefault("human_review", {}).setdefault("by_task", {})
        reviews[run_id] = {
            "scoring_run_id": task.scoring_run_id,
            "scoring_spec_sha256": task.scoring_spec_sha256,
            "revision": state.revision,
            "reviewed": state.reviewed,
            "total_badcases": len(state.records),
            "coverage": state.reviewed / len(state.records) if state.records else 1.0,
            "records_path": output + f"/evaluation_annotations.r{state.revision}.jsonl",
            "conclusions": [
                {"case_id": row["case_id"], "annotation": row["human_annotation"]}
                for row in state.records
                if row["review_status"] == "reviewed"
            ][:20],
        }
        context["human_review"]["status"] = "available"
        self.storage.write(path, json.dumps(context, ensure_ascii=False).encode())
