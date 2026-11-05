"""Validate declared outputs against platform provenance and inspected artifacts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath

from pyromind_runtime.domain.experiment_execution import RecordExperimentResult
from pyromind_runtime.domain.experiment_plans import PlanOwner


async def verify_task_artifacts(
    owner: PlanOwner,
    role: str,
    row: dict,
    result: RecordExperimentResult,
    artifact: dict | None,
    scope: dict,
    execute: Callable[..., Awaitable[dict]],
    inspect: Callable[..., Awaitable[dict]] | None,
) -> tuple[list[dict], dict] | None:
    policy = (row.get("task_spec") or {}).get("execution_policy") or {}
    contract = policy.get("artifact_contract", "evaluation")
    if (
        contract == "data"
        and row["task_spec"]["intent"] == "execute"
        and not result.blockers
    ):
        if row["state"] in {"failed", "cancelled"} and not result.artifacts:
            return None
        if inspect is None or not result.artifacts:
            raise ValueError("declare the Storage files produced by this task")
        if len({item.name for item in result.artifacts}) != len(result.artifacts):
            raise ValueError("artifact names must be unique")
        outputs = []
        for output in result.artifacts:
            path = PurePosixPath(output.uri)
            if (
                output.kind != "file"
                or not path.is_absolute()
                or ".." in path.parts
                or any(c in output.uri for c in ("\\", ":", "?", "#"))
            ):
                raise ValueError(
                    f"artifact {output.name!r} must reference a Storage file: "
                    f"{output.uri}"
                )
            try:
                verified = await inspect(owner, output.uri, output.kind)
            except (OSError, ValueError, KeyError) as exc:
                raise ValueError(
                    f"artifact {output.name!r} is not readable: {output.uri}"
                ) from exc
            if not verified.get("sha256") or verified.get("scope") != "complete file":
                raise ValueError(
                    f"artifact {output.name!r} requires a complete file digest: "
                    f"{output.uri}"
                )
            outputs.append(
                {
                    "name": output.name,
                    "uri": output.uri,
                    "sha256": verified["sha256"],
                    "metadata": verified,
                }
            )
        return outputs, {}
    if (
        contract == "model"
        and row["state"] == "waiting_artifacts"
        and not result.blockers
    ):
        if inspect is None or artifact is None:
            raise ValueError("verified outputs and a readable report are required")
        by_name = {item.name: item for item in result.artifacts}
        if set(by_name) != {contract} or len(result.artifacts) != 1:
            raise ValueError(
                "register the output required by this task's artifact contract"
            )
        output = by_name[contract]
        if (contract == "model") != (output.kind == "directory"):
            raise ValueError(
                "model outputs require a directory; data outputs require a file"
            )
        verified = await inspect(owner, output.uri, output.kind)
        report_artifact = {"name": "report", **artifact}
        metrics = {}
        if contract == "model":
            names = [
                str(name).rsplit("/", 1)[-1] for name in verified.get("entries", [])
            ]
            if not {"config.json", "adapter_config.json"}.intersection(
                names
            ) or not any(
                name.endswith((".safetensors", ".bin", ".pt")) for name in names
            ):
                raise ValueError("model output must contain config and actual weights")
            task_id = (row.get("submission") or {}).get("task_id")
            provenance = await execute(
                owner,
                role,
                scope,
                "training_analysis",
                {"operation": "artifacts", "task_id": task_id},
            )
            nodes = (
                (provenance.get("details") or {})
                .get("result", {})
                .get("training_nodes", [])
            )

            def storage_path(value):
                return PurePosixPath(str(value or ""))

            roots = [
                storage_path(node["output_path"])
                for node in nodes
                if node.get("output_path")
            ]
            if (
                provenance.get("is_error")
                or not roots
                or not any(
                    storage_path(output.uri).is_relative_to(root) for root in roots
                )
            ):
                raise ValueError("model is not an output of this platform task")
            source_model = row["task_spec"]["task"].get("inputs", {}).get("model_path")
            if output.uri == source_model:
                raise ValueError("the base model is not a training output")
        return (
            [
                report_artifact,
                {
                    "name": contract,
                    "uri": output.uri,
                    "sha256": verified["sha256"],
                    "metadata": verified,
                },
            ],
            metrics,
        )
    return None
