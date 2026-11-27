from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from pyromind_agent_server.echomind_failure_evidence import redact_graph_failure_text
from pyromind_agent_server.echomind_graph_runs import GraphRunRepairHandoff
from pyromind_agent_server.echomind_projects import (
    ProjectInfo,
    ProjectNode,
    ProjectNotFoundError,
)
from pyromind_agent_server.echomind_publication import (
    ProjectLifecycleError,
    ProjectPublicationService,
    PublishedGraph,
    RuntimeGraphRepairProposal,
)
from pyromind_agent_server.echomind_repair_assets import (
    recipe_assets,
    replace_recipe_asset,
)
from pyromind_agent_server.echomind_sandbox_storage import (
    build_bound_evaluation_storage,
)
from pyromind_agent_server.echomind_storage import (
    ProjectLease,
    canonical_uuid,
    project_directory,
    read_model,
    write_model,
)
from pyromind_agent_server.evaluation_storage import build_evaluation_storage
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload


def repair_node_evidence(
    node: ProjectNode, context: RequestContext, conversation_id: str
) -> JsonObject:
    evidence: JsonObject = {
        "node_id": node.id,
        "type": node.type,
        "executor": node.executor,
        "inputs": list(node.inputs),
        "outputs": list(node.outputs),
    }
    config = node.config.get(node.executor)
    if not isinstance(config, dict):
        return evidence
    template = config.get("command_template")
    if (
        isinstance(template, str)
        and redact_graph_failure_text(template, context) == template
    ):
        evidence["command_template"] = template
    if node.executor == "dataset_runtime" and isinstance(template, str):
        evidence["recipe_assets"] = list(recipe_assets(template, conversation_id))
    for key in ("workflow_bindings", "workflow_templates"):
        bindings = config.get(key)
        if isinstance(bindings, dict):
            evidence[key] = {
                path: value
                for path, value in bindings.items()
                if isinstance(value, str)
                and redact_graph_failure_text(value, context) == value
            }
    workflow = config.get("workflow")
    parameters: JsonObject = {}
    if isinstance(workflow, dict):
        for part in workflow.get("nodes", []):
            if isinstance(part, dict) and isinstance(data := part.get("data"), dict):
                if isinstance(fields := data.get("config"), dict) and "id" in part:
                    parameters[str(part["id"])] = list(fields)
    evidence["workflow_parameters"] = parameters
    return evidence


class GraphRepairProposals:
    def __init__(self, projects_root: Path) -> None:
        self.projects_root = projects_root
        self.publication = ProjectPublicationService(projects_root)

    def _active(
        self,
        conversation_id: str,
        project_id: str,
        run_id: str,
        handoff_id: str,
        context: RequestContext,
    ) -> tuple[ProjectInfo, PublishedGraph, GraphRunRepairHandoff]:
        store = self.publication.project_store
        try:
            project = store.load(project_id, owner_user_id=context.user_id)
        except ProjectNotFoundError as exc:
            raise PermissionError("Project unavailable to this conversation") from exc
        if project.project.conversation_id != conversation_id:
            raise PermissionError("Repair requires the original Project conversation")
        store.require_cluster(project, context.x_cluster)
        try:
            run = self.publication.graph_run_store.load(
                project_id=project_id, run_id=run_id
            )
        except FileNotFoundError as exc:
            raise ProjectLifecycleError("Unknown debug repair run") from exc
        handoff = run.current_repair_handoff()
        if (
            run.execution_mode != "debug"
            or run.repair_restart is not None
            or handoff is None
            or handoff.handoff_id != handoff_id
            or handoff.delivered_at is None
        ):
            raise ProjectLifecycleError("No matching active debug repair handoff")
        source = self.publication.publication_store.load(
            project_id=project_id, release_id=run.release_id
        )
        if (
            source.content_sha256 != run.release_digest
            or handoff.release_digest != run.release_digest
        ):
            raise ProjectLifecycleError(
                "The debug snapshot no longer matches the handoff"
            )
        return project, source, handoff

    def inspect(
        self,
        *,
        conversation_id: str,
        project_id: str,
        run_id: str,
        handoff_id: str,
        context: RequestContext,
    ) -> JsonObject:
        _, source, handoff = self._active(
            conversation_id, project_id, run_id, handoff_id, context
        )
        return {
            "handoff_id": handoff_id,
            "run_id": run_id,
            "release_digest": source.content_sha256,
            "failed_node_id": handoff.node_id,
            "nodes": [
                repair_node_evidence(node, context, conversation_id)
                for node in source.nodes
            ],
        }

    def submit(
        self,
        *,
        conversation_id: str,
        project_id: str,
        run_id: str,
        handoff_id: str,
        proposal: JsonObject,
        context: RequestContext,
        read_asset: Callable[[str], bytes] | None = None,
    ) -> JsonObject:
        if not project_directory(self.projects_root, project_id).is_dir():
            raise PermissionError("Project unavailable to this conversation")
        with ProjectLease(self.projects_root, project_id).acquire():
            project, source, _ = self._active(
                conversation_id, project_id, run_id, handoff_id, context
            )
            validate_persisted_payload(proposal)
            reason = proposal.get("reason")
            if (
                isinstance(reason, str)
                and redact_graph_failure_text(reason, context) != reason
            ):
                raise ValueError(
                    "Remove credentials and URL query parameters from the reason"
                )
            if "recipe_asset_replacement" in proposal:
                proposal = dict(proposal)
                replacement = proposal.pop("recipe_asset_replacement")
                if proposal.get("runtime_config", {}) != {}:
                    raise ValueError(
                        "A recipe asset replacement must be the only runtime change"
                    )
                node = next(
                    (n for n in source.nodes if n.id == proposal.get("node_id")), None
                )
                if node is None:
                    raise ValueError("Unknown recipe node")
                if read_asset is None:
                    storage = build_bound_evaluation_storage(
                        projects_root=self.projects_root,
                        project_id=project_id,
                        context=context,
                        credential_store=None,
                    ) or build_evaluation_storage(context)
                    read_asset = storage.read
                command = replace_recipe_asset(
                    node, conversation_id, replacement, read_asset
                )
                proposal["runtime_config"] = {"command_template": command}
            try:
                candidate = RuntimeGraphRepairProposal.model_validate(proposal)
            except ValidationError as exc:
                errors = [
                    f"{'.'.join(map(str, item['loc']))}: {item['type']}"
                    for item in exc.errors(include_input=False)[:8]
                ]
                raise ValueError(
                    "Invalid repair proposal: " + ", ".join(errors)
                ) from exc
            self.publication.preview_repair_debug(project, source, candidate)
            write_model(self._path(project_id, run_id, handoff_id), candidate)
        return {
            "status": "reviewable",
            "proposal_sha256": candidate.content_sha256,
            "handoff_id": handoff_id,
            "node_id": candidate.node_id,
        }

    def load(
        self, project_id: str, run_id: str, handoff_id: str
    ) -> RuntimeGraphRepairProposal | None:
        path = self._path(project_id, run_id, handoff_id)
        return read_model(path, RuntimeGraphRepairProposal) if path.is_file() else None

    def _path(self, project_id: str, run_id: str, handoff_id: str) -> Path:
        return (
            project_directory(self.projects_root, project_id)
            / "graph_runs"
            / canonical_uuid(run_id, kind="graph run")
            / "repair_proposals"
            / (canonical_uuid(handoff_id, kind="repair handoff") + ".json")
        )
