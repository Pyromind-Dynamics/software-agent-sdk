"""Register structured tool and workspace deliveries against successful runs."""

import json
import logging
import re

from pyromind_runtime.application.versions import Versions
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.eval_delivery import validate_eval_delivery
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.domain.training_evaluation import CheckpointArtifact
from pyromind_runtime.domain.versions import (
    SaveVersion,
    SnapshotSource,
    VersionParent,
    VersionProject,
    VersionRun,
    canonical,
    fingerprint,
)
from pyromind_runtime.ports.storage import (
    EvaluationStorage,
    EvaluationStorageUnavailableError,
)


logger = logging.getLogger(__name__)


def register_training_artifacts(
    versions: Versions,
    project: VersionProject,
    run: VersionRun,
    nodes: list[JsonObject],
    context: RequestContext,
) -> int:
    count = 0
    parents = [
        VersionParent(version_id=identity, relation=role)
        for role, identity in run.spec.inputs.items()
    ]
    for node in nodes:
        node_id = node["training_node_id"]
        if not node.get("output_path"):
            continue
        root = to_storage_api_path(node["output_path"])
        for item in node.get("checkpoints", []):
            checkpoint = CheckpointArtifact.model_validate(item)
            if not checkpoint.complete or checkpoint.kind == "unknown":
                continue
            step = checkpoint.step
            path = to_storage_api_path(checkpoint.path)
            if path != f"{root}/checkpoint-{step}":
                raise ValueError("checkpoint path does not match the training output")
            prediction = next(
                (
                    row
                    for row in node.get("evaluations", [])
                    if row["step"] == step and not row.get("errors")
                ),
                None,
            )
            summary = {
                "training_node_id": node_id,
                "step": step,
                "kind": checkpoint.kind,
                "base_model": checkpoint.base_model,
            }
            if prediction and prediction.get("rows", 0) > 0:
                prediction_path = to_storage_api_path(prediction["path"])
                if prediction_path != f"{root}/eval_step_{step}.jsonl":
                    raise ValueError(
                        "prediction path does not match the training output"
                    )
                summary["prediction"] = {
                    "path": prediction_path,
                    "sha256": prediction["sha256"],
                    "rows": prediction["rows"],
                }
            key = fingerprint([run.id, node_id, step, path, summary])
            versions.save(
                project,
                "training-artifact:" + key,
                SaveVersion(
                    kind="model",
                    logical_key=f"checkpoint:{node_id}:{step}",
                    source=SnapshotSource(storage_ref=path),
                    producer_run_id=run.id,
                    output_role="checkpoint",
                    summary=summary,
                    parents=parents,
                ),
                context,
            )
            count += 1
    return count


def register_eval_deliveries(
    versions: Versions,
    project: VersionProject,
    successful_runs: dict[str, VersionRun],
    storage: EvaluationStorage,
    context: RequestContext,
) -> bool:
    changed = False
    directory = f"/.pyromind-agent/{project.conversation_id}/public_data"
    artifacts = versions.store.list_versions(project.id)
    checkpoints = [
        artifact
        for artifact in artifacts
        if artifact.registration_key.startswith("training-artifact:")
        and artifact.state in {"ready", "reference_only"}
    ]
    if not checkpoints:
        return False
    completed = {
        artifact.registration_key for artifact in artifacts if artifact.state == "ready"
    }
    for name in sorted(storage.list(directory)):
        if not re.fullmatch(r"eval_result-[A-Za-z0-9_-]+\.json", name):
            continue
        try:
            data = validate_eval_delivery(
                json.loads(storage.read(f"{directory}/{name}"))
            )
            run = successful_runs.get(data["task_id"])
            if run is None:
                raise ValueError(
                    "source training task did not succeed in this conversation"
                )
            node_id = data["training_node_id"]
            for step in data["steps"]:
                path = to_storage_api_path(step["checkpoint_path"])
                step["checkpoint_path"] = path
                if not any(
                    artifact.producer_run_id == run.id
                    and artifact.manifest.storage_ref == path
                    and artifact.manifest.summary.get("training_node_id") == node_id
                    and artifact.manifest.summary.get("step") == step["step"]
                    and isinstance(
                        prediction := artifact.manifest.summary.get("prediction"), dict
                    )
                    and prediction.get("sha256") == step["prediction_sha256"]
                    for artifact in checkpoints
                ):
                    raise ValueError(
                        "step does not match a registered checkpoint and prediction"
                    )
            data["report_path"] = to_storage_api_path(data["report_path"])
            if not data["report_path"].startswith(directory + "/"):
                raise ValueError("report must be in the conversation public workspace")
            identity = fingerprint([run.id, name, data])
            key = "eval-delivery:" + identity
            if key in completed:
                continue
            storage.read(data["report_path"])
            versions.save(
                project,
                key,
                SaveVersion(
                    kind="evaluation",
                    logical_key="eval-delivery",
                    source=SnapshotSource(
                        text=canonical(data), filename="eval_result.json"
                    ),
                    producer_run_id=run.id,
                    output_role="eval_delivery",
                    summary=data,
                ),
                context,
            )
            changed = True
        except (
            ValueError,
            OverflowError,
            LookupError,
            EvaluationStorageUnavailableError,
        ) as exc:
            logger.warning(
                "Evaluation delivery pending: %s (%s)", name, type(exc).__name__
            )
    return changed
