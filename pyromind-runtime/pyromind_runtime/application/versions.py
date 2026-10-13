"""Version use cases shared by all harnesses and execution entry points."""

from __future__ import annotations

import difflib
from typing import Any

from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.versions import (
    ArtifactVersion,
    RunSource,
    RunSpec,
    SaveVersion,
    SnapshotResult,
    SnapshotSource,
    VersionParent,
    VersionProject,
    VersionRun,
    VersionToolRequest,
    canonical,
    check_persistable,
    fingerprint,
)
from pyromind_runtime.ports.versions import ArtifactSnapshotPort, VersionStore


def snapshot_result(version: ArtifactVersion) -> SnapshotResult:
    if version.state not in {"ready", "reference_only"}:
        raise ValueError("version is not finalized")
    return SnapshotResult(
        storage_ref=version.manifest.storage_ref or "",
        manifest_ref=version.manifest.manifest_ref,
        digest=version.manifest.digest,
        reference_only=version.state == "reference_only",
    )


class Versions:
    def __init__(self, store: VersionStore, snapshots: ArtifactSnapshotPort) -> None:
        self.store = store
        self.snapshots = snapshots

    def project(
        self, conversation_id: str, context: RequestContext, *, create: bool = True
    ) -> VersionProject | None:
        return self.store.project(conversation_id, context.user_id, create=create)

    def save(
        self,
        project: VersionProject,
        key: str,
        request: SaveVersion,
        context: RequestContext,
    ) -> ArtifactVersion:
        # A directory copy alone cannot prove that a dataset's external media or
        # a model's base checkpoint were captured. Keep those imports explicit.
        if (
            request.kind in {"dataset", "annotation", "model"}
            and request.source.storage_ref
        ):
            request = request.model_copy(
                update={
                    "source": request.source.model_copy(update={"reference_only": True})
                }
            )
        version = self.store.reserve_version(project, key, request)
        return self.finish(version, context)

    def finish(
        self, version: ArtifactVersion, context: RequestContext
    ) -> ArtifactVersion:
        if version.state in {"ready", "reference_only"}:
            return version
        try:
            result = self.snapshots.save(
                version.manifest.source, version.manifest.target, context
            )
            return self.store.complete_version(version.project_id, version.id, result)
        except Exception as exc:
            self.store.fail_version(version.project_id, version.id, type(exc).__name__)
            raise ValueError(
                f"snapshot not completed ({type(exc).__name__}); "
                "retry the same operation"
            ) from None

    def call(
        self,
        conversation_id: str,
        request: VersionToolRequest,
        command_id: str,
        context: RequestContext,
    ) -> dict[str, Any]:
        project = self.project(
            conversation_id,
            context,
            create=request.action not in {"list", "compare", "recover"},
        )
        if project is None:
            return {"project": None, "versions": [], "runs": []}
        args = request.arguments
        if request.action == "list":
            return {
                "project": project.model_dump(mode="json", by_alias=True),
                "versions": [
                    self.describe(v) for v in self.store.list_versions(project.id)
                ],
                "runs": [
                    run.model_dump(mode="json")
                    for run in self.store.list_runs(project.id)
                ],
            }
        if request.action == "save":
            result = self.save(
                project,
                "manual:" + command_id,
                SaveVersion.model_validate(args),
                context,
            )
            return self.describe(result)
        if request.action == "select":
            version_id, slot = (
                self._string(args, "version_id"),
                self._string(args, "slot"),
            )
            version = self.store.get_version(project.id, version_id)
            self.snapshots.verify(snapshot_result(version), context)
            updated = self.store.select(
                project.id,
                slot,
                version_id,
                command_id,
                str(args.get("reason", "")),
                self._revision(args),
            )
            return updated.model_dump(mode="json", by_alias=True)
        if request.action == "new_iteration":
            updated = self.store.new_iteration(
                project.id, self._string(args, "goal"), command_id, self._revision(args)
            )
            return updated.model_dump(mode="json", by_alias=True)
        if request.action == "recover":
            return {"versions": self.recover(project, context)}
        return self.compare(
            project.id,
            self._string(args, "left_version_id"),
            self._string(args, "right_version_id"),
            context,
        )

    @staticmethod
    def _string(args: JsonObject, key: str) -> str:
        value = args.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a nonempty string")
        return value

    @staticmethod
    def _revision(args: JsonObject) -> int:
        value = args.get("expected_revision")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("expected_revision is required; list the project first")
        return value

    @staticmethod
    def describe(version: ArtifactVersion) -> dict[str, Any]:
        value = version.model_dump(mode="json")
        # Do not echo complete code/config sources into every tool result.
        value["manifest"].pop("source", None)
        return value

    def recover(
        self, project: VersionProject, context: RequestContext
    ) -> list[dict[str, Any]]:
        results = []
        for version in self.store.list_versions(project.id):
            if version.state in {"pending", "failed"}:
                try:
                    results.append(self.describe(self.finish(version, context)))
                except ValueError:
                    results.append({"id": version.id, "state": "failed"})
        return results

    def compare(
        self, project_id: str, left_id: str, right_id: str, context: RequestContext
    ) -> dict[str, Any]:
        left = self.store.get_version(project_id, left_id)
        right = self.store.get_version(project_id, right_id)
        if left.kind != right.kind:
            raise ValueError("compare versions of the same artifact kind")
        result: dict[str, Any] = {
            "left": self.describe(left),
            "right": self.describe(right),
        }
        if (
            left.kind in {"code", "workflow", "config"}
            and left.state == right.state == "ready"
        ):
            a = self.snapshots.read_text(snapshot_result(left), context)
            b = self.snapshots.read_text(snapshot_result(right), context)
            result["diff"] = "".join(
                difflib.unified_diff(
                    a.splitlines(True),
                    b.splitlines(True),
                    fromfile=left.id,
                    tofile=right.id,
                )
            )
        if left.kind == "evaluation":
            a, b = left.manifest.summary, right.manifest.summary
            keys = (
                "dataset_version_id",
                "scoring_version_id",
                "inference_config_version_id",
            )
            result["comparable"] = all(
                a.get(key) and a.get(key) == b.get(key) for key in keys
            )
            if result["comparable"]:
                result["comparable"] = all(
                    self.store.get_version(project_id, str(a[key])).state == "ready"
                    for key in keys
                )
        return result

    def prepare_run(
        self,
        conversation_id: str,
        context: RequestContext,
        key: str,
        source: RunSource,
        definition: JsonObject,
        *,
        code: str | None = None,
        kind: str = "workflow",
        resolved_inputs: dict[str, str] | None = None,
    ) -> VersionRun:
        check_persistable(definition)
        project = self.project(conversation_id, context)
        assert project is not None
        original_digest = fingerprint({"definition": definition, "code": code})
        for prior in self.store.list_runs(project.id):
            if prior.request_key == key:
                if prior.spec.parameters.get("original_digest") != original_digest:
                    raise ValueError(
                        "execution request already frozen with different content"
                    )
                for identity in prior.spec.inputs.values():
                    version = self.store.get_version(project.id, identity)
                    if version.state == "ready":
                        self.snapshots.verify(snapshot_result(version), context)
                return prior
        inputs: dict[str, str] = dict(resolved_inputs or {})
        effective = (
            definition
            if resolved_inputs is not None
            else self.resolve_inputs(project, definition, inputs, context, key)
        )
        config = self.save(
            project,
            key + ":config",
            SaveVersion(
                kind="config",
                logical_key=kind + "-config",
                source=SnapshotSource(
                    text=canonical(effective), filename="config.json"
                ),
            ),
            context,
        )
        inputs["config"] = config.id
        if code is not None:
            script = self.save(
                project,
                key + ":code",
                SaveVersion(
                    kind="code",
                    logical_key=kind + "-code",
                    source=SnapshotSource(text=code, filename="workflow.py"),
                ),
                context,
            )
            inputs["code"] = script.id
        return self.store.reserve_run(
            project.id,
            key,
            RunSpec(
                kind=kind,
                iteration_id=project.context.current_iteration_id,
                inputs=inputs,
                parameters={
                    "definition": effective,
                    "original_digest": original_digest,
                },
            ),
            source,
        )

    def resolve_inputs(
        self,
        project: VersionProject,
        definition: JsonObject,
        inputs: dict[str, str],
        context: RequestContext,
        key: str,
    ) -> JsonObject:
        roles = {
            "input_path": "training_data",
            "model_path": "model",
            "base_model": "base_model",
        }

        def visit(value: Any, trail: str, field: str = "") -> Any:
            if isinstance(value, dict):
                return {
                    name: visit(item, trail + "." + name, name)
                    for name, item in value.items()
                }
            if isinstance(value, list):
                return [
                    visit(item, trail + f".{index}") for index, item in enumerate(value)
                ]
            slot = (
                value.removeprefix("version://")
                if isinstance(value, str) and value.startswith("version://")
                else roles.get(field)
            )
            if slot and slot in project.context.selected_versions:
                selected = self.store.get_version(
                    project.id, project.context.selected_versions[slot]
                )
                result = snapshot_result(selected)
                self.snapshots.verify(result, context)
                inputs[trail] = selected.id
                return (
                    "/workspace"
                    if isinstance(value, str) and value.startswith("/workspace/")
                    else ""
                ) + result.storage_ref
            if isinstance(value, str) and value.startswith("version://"):
                raise ValueError(f"no selected version for {slot}")
            if (
                field in roles
                and isinstance(value, str)
                and value.startswith(("/", "storage/"))
            ):
                version = self.save(
                    project,
                    key + ":input:" + trail,
                    SaveVersion(
                        kind="dataset" if field == "input_path" else "model",
                        logical_key=roles[field],
                        source=SnapshotSource(
                            storage_ref=value.removeprefix("/workspace")
                            if value.startswith("/workspace/")
                            else value,
                            reference_only=True,
                        ),
                    ),
                    context,
                )
                inputs[trail] = version.id
            return value

        return visit(definition, "input")

    def register_outputs(
        self,
        project: VersionProject,
        run: VersionRun,
        outputs: list[SaveVersion],
        context: RequestContext | None,
    ) -> list[ArtifactVersion]:
        results = []
        for index, output in enumerate(outputs):
            output = output.model_copy(
                update={
                    "producer_run_id": run.id,
                    "parents": [
                        VersionParent(version_id=identity, relation=role)
                        for role, identity in run.spec.inputs.items()
                    ],
                }
            )
            if (
                output.kind in {"dataset", "annotation", "model"}
                and output.source.storage_ref
            ):
                output = output.model_copy(
                    update={
                        "source": output.source.model_copy(
                            update={"reference_only": True}
                        )
                    }
                )
            version = self.store.reserve_version(
                project, f"output:{run.id}:{index}", output
            )
            results.append(version)
        if context is not None:
            for index, version in enumerate(results):
                try:
                    results[index] = self.finish(version, context)
                except ValueError:
                    results[index] = self.store.get_version(project.id, version.id)
        return results

    def guard_resume(self, project: VersionProject, source_id: str) -> None:
        run_ids = {
            run.id
            for run in self.store.list_runs(project.id)
            if run.source.source_id == source_id
        }
        if any(
            v.producer_run_id in run_ids and v.state in {"pending", "failed"}
            for v in self.store.list_versions(project.id)
        ):
            raise ValueError("previous outputs must finish snapshotting before resume")
