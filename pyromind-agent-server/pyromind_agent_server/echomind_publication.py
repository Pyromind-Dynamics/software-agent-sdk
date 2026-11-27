from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from pyromind_agent_server.echomind_dataset_runtime import DatasetWorkflowConfig
from pyromind_agent_server.echomind_graph_runs import (
    ActiveGraphRunExistsError,
    GraphRun,
    GraphRunStore,
)
from pyromind_agent_server.echomind_projects import (
    ProjectEdge,
    ProjectGraph,
    ProjectInfo,
    ProjectNode,
    ProjectNodeInput,
    ProjectNodeInputBinding,
    ProjectNodeOutput,
    ProjectNodeType,
    ProjectNotFoundError,
    ProjectRecord,
    ProjectStore,
)
from pyromind_agent_server.echomind_runtime_slots import RuntimeSlotStore
from pyromind_agent_server.echomind_stage_runs import ProjectRunStore
from pyromind_agent_server.echomind_storage import (
    ProjectLease,
    canonical_sha256,
    canonical_uuid,
    project_directory,
    read_model,
    write_model,
)
from pyromind_agent_server.echomind_train_runtime import (
    TrainRuntimeError,
    render_workflow_template,
)
from pyromind_runtime.domain.graph_repair import (
    GraphRepairDraft,
    RuntimeGraphRepairChanges,
)
from pyromind_runtime.domain.storage_paths import (
    StoragePathError,
    normalize_workspace_path,
)


ExecutionProfile = Literal["echomind.flywheel/v1"]
PublicationMode = Literal["debug", "production"]
TracerProvisioningMode = Literal["attach_existing", "create_if_missing"]
PromotionDirection = Literal["maximize", "minimize"]
DEBUG_ONLY_EXECUTORS = frozenset({"artifact_replay", "contract_probe"})
_RUNTIME_EXECUTOR_TYPES: dict[str, ProjectNodeType] = {
    "dataset_runtime": "dataset",
    "train_runtime": "train",
    "evaluator_runtime": "evaluator",
}
RUNTIME_VALIDATION_GRAPH_TEMPLATE = "echomind.runtime-validation/v1"


def _utc_now() -> datetime:
    return datetime.now(UTC)


class RuntimeRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tracer: TracerProvisioningMode = "create_if_missing"
    serving_inference: Literal["attach_existing"] = "attach_existing"
    candidate_inference: Literal["evaluator_managed"] = "evaluator_managed"
    max_active_graph_runs: Literal[1] = 1


class PromotionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(min_length=1, max_length=256)
    direction: PromotionDirection = "maximize"
    minimum_gain: float = 0.0
    require_candidate_healthy: bool = True


class DebugExecutionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_project_id: str
    source_project_name: str
    run_id: str
    release_id: str
    release_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    finished_at: datetime | None
    reused_node_ids: list[str] = Field(default_factory=list)


class DebugExecutionChange(BaseModel):
    node_id: str
    changed_fields: list[str]


class DebugExecutionAdoptionPreview(BaseModel):
    source: DebugExecutionCandidate
    changes: list[DebugExecutionChange]
    content_sha256: str


class _AdoptedDebugExecution(BaseModel):
    source: DebugExecutionCandidate
    target_revision_sha256: str
    preview_sha256: str
    snapshot_release_id: str
    snapshot_sha256: str


class PublishedGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    release_id: str = Field(min_length=1, max_length=128)
    project_id: str
    source_graph_id: str = Field(min_length=1, max_length=128)
    source_graph_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    execution_profile: ExecutionProfile = "echomind.flywheel/v1"
    publication_mode: PublicationMode = "debug"
    runtime_requirements: RuntimeRequirements
    promotion_policy: PromotionPolicy
    nodes: list[ProjectNode] = Field(min_length=1)
    edges: list[ProjectEdge] = Field(default_factory=list)
    content_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    published_at: datetime
    published_by: str = Field(min_length=1, max_length=256)
    adopted_debug_run: DebugExecutionCandidate | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_published_content(self) -> PublishedGraph:
        ProjectGraph(
            graph_id=self.source_graph_id,
            project_id=self.project_id,
            lifecycle="frozen",
            nodes=self.nodes,
            edges=self.edges,
        )
        if any(node.launch_mode != "executor" for node in self.nodes):
            raise ValueError("published graph nodes must use executor launch mode")
        if self.publication_mode == "production" and any(
            node.executor in DEBUG_ONLY_EXECUTORS for node in self.nodes
        ):
            raise ValueError("production publication cannot use debug executors")
        if self.content_sha256 != self.compute_content_sha256():
            raise ValueError("published graph content digest does not match content")
        return self

    @classmethod
    def build(
        cls,
        *,
        release_id: str,
        project_id: str,
        source_graph_id: str,
        source_graph_digest: str,
        runtime_requirements: RuntimeRequirements,
        promotion_policy: PromotionPolicy,
        publication_mode: PublicationMode = "debug",
        nodes: list[ProjectNode],
        edges: list[ProjectEdge],
        published_by: str,
        published_at: datetime | None = None,
        adopted_debug_run: DebugExecutionCandidate | None = None,
    ) -> PublishedGraph:
        published_at = published_at or _utc_now()
        draft = cls.model_construct(
            schema_version=1,
            release_id=release_id,
            project_id=project_id,
            source_graph_id=source_graph_id,
            source_graph_digest=source_graph_digest,
            execution_profile="echomind.flywheel/v1",
            publication_mode=publication_mode,
            runtime_requirements=runtime_requirements,
            promotion_policy=promotion_policy,
            nodes=nodes,
            edges=edges,
            content_sha256="sha256:" + "0" * 64,
            published_at=published_at,
            published_by=published_by,
            adopted_debug_run=adopted_debug_run,
        )
        return cls.model_validate(
            {
                **draft.model_dump(mode="json"),
                "content_sha256": draft.compute_content_sha256(),
            }
        )

    def compute_content_sha256(self) -> str:
        excluded = {"content_sha256"}
        if self.adopted_debug_run is None:
            excluded.add("adopted_debug_run")
        return canonical_sha256(self.model_dump(mode="json", exclude=excluded))


class PublishedGraphManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    release_id: str
    project_id: str
    source_graph_id: str
    execution_profile: ExecutionProfile
    publication_mode: PublicationMode
    content_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    published_at: datetime
    published_by: str
    adopted_debug_run: DebugExecutionCandidate | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ActiveRelease(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    project_id: str
    release_id: str
    content_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    activated_at: datetime = Field(default_factory=_utc_now)


class ProjectLifecycleError(ValueError):
    pass


class PublicationImmutableError(RuntimeError):
    pass


class PublicationStore:
    def __init__(self, projects_root: Path | str) -> None:
        self.projects_root = Path(projects_root)

    def save(self, graph: PublishedGraph) -> PublishedGraph:
        path = self._release_graph_path(graph.project_id, graph.release_id)
        if path.is_file():
            existing = read_model(path, PublishedGraph)
            if existing != graph:
                raise PublicationImmutableError(
                    f"Published graph already exists: {graph.release_id}"
                )
        else:
            write_model(path, graph)
        manifest_path = path.with_name("manifest.json")
        if manifest_path.is_file():
            return graph
        write_model(
            manifest_path,
            PublishedGraphManifest(
                release_id=graph.release_id,
                project_id=graph.project_id,
                source_graph_id=graph.source_graph_id,
                execution_profile=graph.execution_profile,
                publication_mode=graph.publication_mode,
                content_sha256=graph.content_sha256,
                published_at=graph.published_at,
                published_by=graph.published_by,
                adopted_debug_run=graph.adopted_debug_run,
            ),
        )
        return graph

    def activate(self, graph: PublishedGraph) -> ActiveRelease:
        persisted = self.save(graph)
        active = ActiveRelease(
            project_id=persisted.project_id,
            release_id=persisted.release_id,
            content_sha256=persisted.content_sha256,
        )
        write_model(self._active_release_path(persisted.project_id), active)
        return active

    def load(self, *, project_id: str, release_id: str) -> PublishedGraph:
        path = self._release_graph_path(project_id, release_id)
        if not path.is_file():
            raise FileNotFoundError(f"published graph not found: {release_id}")
        graph = read_model(path, PublishedGraph)
        if canonical_uuid(graph.project_id, kind="project") != canonical_uuid(
            project_id,
            kind="project",
        ):
            raise ValueError("published graph belongs to another project")
        return graph

    def load_active(self, project_id: str) -> PublishedGraph:
        active_path = self._active_release_path(project_id)
        if not active_path.is_file():
            raise FileNotFoundError(f"active release not found: {project_id}")
        active = read_model(active_path, ActiveRelease)
        if canonical_uuid(active.project_id, kind="project") != canonical_uuid(
            project_id,
            kind="project",
        ):
            raise ValueError("active release belongs to another project")
        graph = self.load(project_id=project_id, release_id=active.release_id)
        if graph.content_sha256 != active.content_sha256:
            raise ValueError("active release digest does not match published graph")
        return graph

    def list(self, project_id: str) -> list[PublishedGraph]:
        releases_root = (
            project_directory(self.projects_root, project_id) / "published_graphs"
        )
        if not releases_root.is_dir():
            return []
        graphs = [
            read_model(release_dir / "graph.json", PublishedGraph)
            for release_dir in releases_root.iterdir()
            if (release_dir / "graph.json").is_file()
        ]
        return sorted(graphs, key=lambda graph: (graph.published_at, graph.release_id))

    def _release_graph_path(self, project_id: str, release_id: str) -> Path:
        return (
            project_directory(self.projects_root, project_id)
            / "published_graphs"
            / canonical_uuid(release_id, kind="release")
            / "graph.json"
        )

    def _active_release_path(self, project_id: str) -> Path:
        return (
            project_directory(
                self.projects_root,
                project_id,
            )
            / "active_release.json"
        )


class ProjectPublicationService:
    def __init__(self, projects_root: Path | str) -> None:
        self.projects_root = Path(projects_root)
        self.project_store = ProjectStore(self.projects_root)
        self.publication_store = PublicationStore(self.projects_root)
        self.graph_run_store = GraphRunStore(self.projects_root)
        self.stage_run_store = ProjectRunStore(self.projects_root)

    def list_debug_execution_candidates(
        self, project: ProjectInfo
    ) -> list[DebugExecutionCandidate]:
        if project.project.lifecycle != "development":
            return []
        candidates: list[DebugExecutionCandidate] = []
        for summary in self.project_store.list(
            owner_user_id=project.project.owner_user_id
        ):
            if (
                summary.conversation_id != project.project.conversation_id
                or summary.cluster != project.project.cluster
                or summary.owner_user_id != project.project.owner_user_id
            ):
                continue
            try:
                source = self._debug_source_project(project, summary.project_id)
                runs = self.graph_run_store.list(summary.project_id)
            except (RuntimeImplementationAdoptionError, OSError, ValueError):
                continue
            for run in runs:
                if run.execution_mode != "debug" or run.status != "succeeded":
                    continue
                try:
                    candidate, _ = self._debug_candidate(project, source, run)
                except (RuntimeImplementationAdoptionError, OSError, ValueError):
                    continue
                candidates.append(candidate)
        return sorted(
            candidates,
            key=lambda candidate: (
                candidate.finished_at or datetime.min.replace(tzinfo=UTC),
                candidate.run_id,
            ),
            reverse=True,
        )

    def preview_debug_execution_adoption(
        self, *, project: ProjectInfo, source_project_id: str, run_id: str
    ) -> DebugExecutionAdoptionPreview:
        preview, _ = self._prepare_debug_execution_adoption(
            project=project, source_project_id=source_project_id, run_id=run_id
        )
        return preview

    def _prepare_debug_execution_adoption(
        self, *, project: ProjectInfo, source_project_id: str, run_id: str
    ) -> tuple[DebugExecutionAdoptionPreview, PublishedGraphDraft]:
        if project.project.lifecycle != "development":
            raise RuntimeImplementationAdoptionError(
                "Reopen the Project before adopting a Debug execution."
            )
        source = self._debug_source_project(project, source_project_id)
        candidate, snapshot = self._debug_candidate(
            project, source, self._debug_run(source, run_id)
        )
        pinned = self._compile_execution(
            project, snapshot.nodes, snapshot.edges, "debug", candidate
        )
        current = self.runtime_contract_draft(project)
        changes = _debug_execution_changes(current.nodes, snapshot.nodes)
        selection = self._adopted_debug_execution(project)
        digest = canonical_sha256(
            {
                "target_project_id": project.project.project_id,
                "target_revision_sha256": _debug_adoption_revision(project),
                "compared_execution_sha256": current.content_sha256,
                "current_selection": selection.model_dump(mode="json")
                if selection is not None
                else None,
                "source": candidate.model_dump(mode="json"),
                "changes": [change.model_dump(mode="json") for change in changes],
            }
        )
        return DebugExecutionAdoptionPreview(
            source=candidate, changes=changes, content_sha256=digest
        ), pinned

    def adopt_debug_execution(
        self,
        *,
        project: ProjectInfo,
        source_project_id: str,
        run_id: str,
        expected_preview_sha256: str,
        reviewed_by: str,
    ) -> RuntimeContractUpgradeDraft:
        project_id = project.project.project_id
        with ProjectLease(self.projects_root, project_id).acquire():
            project = self.project_store.load(
                project_id, owner_user_id=project.project.owner_user_id
            )
            self._require_no_active_run(project_id)
            preview, pinned = self._prepare_debug_execution_adoption(
                project=project, source_project_id=source_project_id, run_id=run_id
            )
            current = self._adopted_debug_execution(project)
            revision = _debug_adoption_revision(project)
            if (
                current is not None
                and current.source == preview.source
                and current.target_revision_sha256 == revision
                and current.preview_sha256 == expected_preview_sha256
            ):
                return self.runtime_contract_draft(project)
            if preview.content_sha256 != expected_preview_sha256:
                raise RuntimeImplementationAdoptionError(
                    "Debug execution preview changed. Review the latest preview."
                )
            snapshot = self.publication_store.save(
                self._get_or_build_snapshot(pinned, published_by=reviewed_by)
            )
            write_model(
                self._debug_adoption_path(project),
                _AdoptedDebugExecution(
                    source=preview.source,
                    target_revision_sha256=revision,
                    preview_sha256=expected_preview_sha256,
                    snapshot_release_id=snapshot.release_id,
                    snapshot_sha256=snapshot.content_sha256,
                ),
            )
            return self.runtime_contract_draft(project)

    def runtime_contract_draft(
        self, project: ProjectInfo, validation_project_id: str | None = None
    ) -> RuntimeContractUpgradeDraft:
        validation = (
            load_runtime_validation_project(
                store=self.project_store,
                project=project,
                validation_project_id=validation_project_id,
            )
            if validation_project_id is not None
            else None
        )
        if validation is not None:
            return build_runtime_contract_draft(project, validation_project=validation)
        adoption = self._adopted_debug_execution(project)
        if adoption is None:
            return build_runtime_contract_draft(project)
        try:
            if adoption.target_revision_sha256 != _debug_adoption_revision(project):
                raise RuntimeImplementationAdoptionError(
                    "The Project changed after this Debug execution was adopted."
                )
            snapshot = self.publication_store.load(
                project_id=project.project.project_id,
                release_id=adoption.snapshot_release_id,
            )
            if (
                snapshot.publication_mode != "debug"
                or snapshot.content_sha256 != adoption.snapshot_sha256
                or snapshot.adopted_debug_run != adoption.source
            ):
                raise RuntimeImplementationAdoptionError(
                    "The adopted Debug snapshot changed."
                )
        except (RuntimeImplementationAdoptionError, OSError, ValueError) as exc:
            draft = build_runtime_contract_draft(project)
            return _with_runtime_draft_updates(
                draft,
                adopted_debug_run=adoption.source,
                issues=[
                    *draft.issues,
                    PublicationIssue(
                        code="adopted_debug_execution_stale",
                        message=f"{exc} Review and adopt a Debug execution again.",
                    ),
                ],
            )
        draft = RuntimeContractUpgradeDraft.model_construct(
            source_project_id=project.project.project_id,
            source_graph_id=project.graph.graph_id,
            source_graph_digest=_source_graph_digest(project.graph),
            nodes=snapshot.nodes,
            edges=snapshot.edges,
            content_sha256="",
            adopted_debug_run=adoption.source,
        )
        return _with_runtime_draft_updates(draft)

    def _debug_adoption_path(self, project: ProjectInfo) -> Path:
        return project_directory(self.projects_root, project.project.project_id) / (
            "adopted_debug_execution.json"
        )

    def _adopted_debug_execution(
        self, project: ProjectInfo
    ) -> _AdoptedDebugExecution | None:
        try:
            return read_model(
                self._debug_adoption_path(project), _AdoptedDebugExecution
            )
        except FileNotFoundError:
            return None

    def _debug_source_project(
        self, target: ProjectInfo, source_project_id: str
    ) -> ProjectInfo:
        try:
            source = self.project_store.load(
                source_project_id, owner_user_id=target.project.owner_user_id
            )
        except ProjectNotFoundError as exc:
            raise RuntimeImplementationAdoptionError(
                "Debug Project was not found."
            ) from exc
        if (
            source.project.owner_user_id != target.project.owner_user_id
            or source.project.conversation_id != target.project.conversation_id
            or source.project.cluster != target.project.cluster
        ):
            raise RuntimeImplementationAdoptionError(
                "Debug execution must belong to the same owner, conversation and "
                "cluster."
            )
        return source

    def _debug_run(self, source: ProjectInfo, run_id: str) -> GraphRun:
        try:
            return self.graph_run_store.load(
                project_id=source.project.project_id, run_id=run_id
            )
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeImplementationAdoptionError(
                "Debug execution was not found."
            ) from exc

    def _debug_candidate(
        self, target: ProjectInfo, source: ProjectInfo, run: GraphRun
    ) -> tuple[DebugExecutionCandidate, PublishedGraph]:
        if run.status != "succeeded" or run.execution_mode != "debug":
            raise RuntimeImplementationAdoptionError(
                "Only successful Debug executions can be adopted."
            )
        snapshot = self.publication_store.load(
            project_id=source.project.project_id, release_id=run.release_id
        )
        if (
            snapshot.publication_mode != "debug"
            or snapshot.content_sha256 != run.release_digest
        ):
            raise RuntimeImplementationAdoptionError(
                "Debug execution snapshot does not match its recorded digest."
            )
        if _topology(target.graph.nodes, target.graph.edges) != _topology(
            snapshot.nodes, snapshot.edges
        ):
            raise RuntimeImplementationAdoptionError(
                "Debug execution nodes or edges do not match the target Project."
            )
        stages = self.stage_run_store.list_for_graph_run(
            project_id=source.project.project_id, graph_run_id=run.run_id
        )
        successful_stages = {
            stage.node_id: stage for stage in stages if stage.status == "succeeded"
        }
        if set(successful_stages) != {node.id for node in snapshot.nodes}:
            raise RuntimeImplementationAdoptionError(
                "Debug execution is missing successful stage records."
            )
        for node in snapshot.nodes:
            stage = successful_stages[node.id]
            if (
                stage.node_type != node.type
                or stage.executor != node.executor
                or stage.implementation_ref != node.implementation_ref
                or stage.implementation_digest != node.implementation_digest
            ):
                raise RuntimeImplementationAdoptionError(
                    "Debug stage execution does not match its snapshot."
                )
        return DebugExecutionCandidate(
            source_project_id=source.project.project_id,
            source_project_name=source.project.name,
            run_id=run.run_id,
            release_id=snapshot.release_id,
            release_digest=snapshot.content_sha256,
            finished_at=run.finished_at,
            reused_node_ids=sorted(
                stage.node_id
                for stage in successful_stages.values()
                if stage.reused_from_stage_run_id is not None
            ),
        ), snapshot

    def _compile_execution(
        self,
        project: ProjectInfo,
        nodes: list[ProjectNode],
        edges: list[ProjectEdge],
        publication_mode: PublicationMode,
        adopted_debug_run: DebugExecutionCandidate | None = None,
    ) -> PublishedGraphDraft:
        working = _with_lifecycle(project, lifecycle="development")
        return compile_working_graph(
            ProjectInfo(
                project=working.project,
                graph=working.graph.model_copy(update={"nodes": nodes, "edges": edges}),
            ),
            publication_mode=publication_mode,
            require_production_executors=publication_mode == "debug",
            adopted_debug_run=adopted_debug_run,
        )

    def prepare_debug(
        self,
        *,
        project: ProjectInfo,
        reviewed_by: str,
        expected_contract_sha256: str,
        validation_project_id: str | None = None,
    ) -> PublishedGraph:
        project_id = project.project.project_id
        with ProjectLease(self.projects_root, project_id).acquire():
            project = self.project_store.load(
                project_id, owner_user_id=project.project.owner_user_id
            )
            draft = self._compile_reviewed_execution(
                project=project,
                expected_contract_sha256=expected_contract_sha256,
                validation_project_id=validation_project_id,
                publication_mode="debug",
            )
            return self.publication_store.save(
                self._get_or_build_snapshot(draft, published_by=reviewed_by)
            )

    def resource_setup_release(self, project: ProjectInfo) -> PublishedGraph:
        if project.project.lifecycle == "frozen":
            release = self.publication_store.load_active(project.project.project_id)
            if release.publication_mode != "production":
                raise ProjectLifecycleError(
                    "Runtime resources require a production plan."
                )
            return release
        contract = self.runtime_contract_draft(project)
        _require_reviewed_runtime_contract(contract, contract.content_sha256)
        candidate = self._compile_execution(
            project,
            contract.nodes,
            contract.edges,
            "production",
            contract.adopted_debug_run,
        )
        return self._get_or_build_snapshot(
            candidate, published_by=project.project.owner_user_id or "system"
        )

    def prepare_repair_debug(
        self,
        *,
        project: ProjectInfo,
        release_id: str,
        proposal: RuntimeGraphRepairProposal,
        reviewed_by: str,
    ) -> PublishedGraph:
        project_id = project.project.project_id
        with ProjectLease(self.projects_root, project_id).acquire():
            project = self.project_store.load(
                project_id, owner_user_id=project.project.owner_user_id
            )
            source = self.publication_store.load(
                project_id=project_id, release_id=release_id
            )
            draft = self.preview_repair_debug(project, source, proposal)
            return self.publication_store.save(
                self._get_or_build_snapshot(draft, published_by=reviewed_by)
            )

    def preview_repair_debug(
        self,
        project: ProjectInfo,
        source: PublishedGraph,
        proposal: RuntimeGraphRepairProposal,
    ) -> PublishedGraphDraft:
        if (
            project.project.lifecycle != "development"
            or source.publication_mode != "debug"
        ):
            raise ProjectLifecycleError("Graph repair requires a development debug run")
        repaired = proposal.repaired_node(source)
        return self._compile_execution(
            project,
            [repaired if node.id == repaired.id else node for node in source.nodes],
            source.edges,
            "debug",
            source.adopted_debug_run,
        )

    def publish(
        self,
        *,
        project: ProjectInfo,
        published_by: str,
        publication_mode: PublicationMode = "debug",
        expected_draft_sha256: str | None = None,
        expected_contract_sha256: str | None = None,
        validation_project_id: str | None = None,
        require_ready_resources: bool = False,
    ) -> PublishedGraph:
        if validation_project_id is not None and expected_contract_sha256 is None:
            raise ProjectLifecycleError("Review the runtime contract before publishing")
        if expected_contract_sha256 is not None and publication_mode != "production":
            raise ProjectLifecycleError(
                "A reviewed runtime contract uses production publication"
            )
        project_id = project.project.project_id
        with ProjectLease(self.projects_root, project_id).acquire():
            project = self.project_store.load(
                project_id,
                owner_user_id=project.project.owner_user_id,
            )
            if (
                project.project.lifecycle == "development"
                and expected_contract_sha256 is None
                and self._adopted_debug_execution(project) is not None
            ):
                raise ProjectLifecycleError(
                    "Review the adopted Debug execution before publishing"
                )
            reviewed = (
                self._compile_reviewed_execution(
                    project=project,
                    expected_contract_sha256=expected_contract_sha256,
                    validation_project_id=validation_project_id,
                    publication_mode=publication_mode,
                )
                if expected_contract_sha256 is not None
                else None
            )
            if project.project.lifecycle == "frozen":
                active = self.publication_store.load_active(project_id)
                if active.publication_mode != publication_mode:
                    raise ProjectLifecycleError(
                        "Reopen the Project before changing its publication mode"
                    )
                _require_publication_digest(active, expected_draft_sha256)
                if (
                    reviewed is not None
                    and reviewed.candidate_release_id != active.release_id
                ):
                    raise ProjectLifecycleError(
                        "Reopen the Project before publishing a changed contract"
                    )
                return active
            draft = reviewed or compile_working_graph(
                project,
                publication_mode=publication_mode,
            )
            _require_publication_digest(draft, expected_draft_sha256)
            self._require_no_active_run(project_id)
            if require_ready_resources and publication_mode == "production":
                slot = RuntimeSlotStore(self.projects_root).load_optional(project_id)
                if (
                    slot is None
                    or slot.status != "ready"
                    or slot.active_release_id != draft.candidate_release_id
                ):
                    raise ProjectLifecycleError(
                        "先完成当前执行方案的 Inference 和 Tracer 搭建及健康检查。"
                    )
            published = self._get_or_build_snapshot(draft, published_by=published_by)
            self.publication_store.activate(published)
            self.project_store.save(
                _with_lifecycle(
                    project,
                    lifecycle="frozen",
                    active_release_id=published.release_id,
                )
            )
            return published

    def _compile_reviewed_execution(
        self,
        *,
        project: ProjectInfo,
        expected_contract_sha256: str,
        validation_project_id: str | None,
        publication_mode: PublicationMode,
    ) -> PublishedGraphDraft:
        draft = self.runtime_contract_draft(project, validation_project_id)
        _require_reviewed_runtime_contract(draft, expected_contract_sha256)
        return self._compile_execution(
            project,
            draft.nodes,
            draft.edges,
            publication_mode,
            draft.adopted_debug_run,
        )

    def _get_or_build_snapshot(
        self, draft: PublishedGraphDraft, *, published_by: str
    ) -> PublishedGraph:
        try:
            published = self.publication_store.load(
                project_id=draft.project_id, release_id=draft.candidate_release_id
            )
        except FileNotFoundError:
            return PublishedGraph.build(
                release_id=draft.candidate_release_id,
                project_id=draft.project_id,
                source_graph_id=draft.source_graph_id,
                source_graph_digest=draft.source_graph_digest,
                runtime_requirements=draft.runtime_requirements,
                promotion_policy=draft.promotion_policy,
                publication_mode=draft.publication_mode,
                nodes=draft.nodes,
                edges=draft.edges,
                published_by=published_by,
                adopted_debug_run=draft.adopted_debug_run,
            )
        if canonical_sha256(_draft_content_payload(published)) != draft.content_sha256:
            raise PublicationImmutableError(
                f"Existing release has different content: {published.release_id}"
            )
        return published

    def reopen(self, *, project: ProjectInfo) -> ProjectInfo:
        project_id = project.project.project_id
        with ProjectLease(self.projects_root, project_id).acquire():
            project = self.project_store.load(
                project_id,
                owner_user_id=project.project.owner_user_id,
            )
            if project.project.lifecycle != "frozen":
                raise ProjectLifecycleError("only frozen Projects can be reopened")
            self._require_no_active_run(project_id)
            return self.project_store.save(
                _with_lifecycle(project, lifecycle="development")
            )

    def _require_no_active_run(self, project_id: str) -> None:
        active_run = self.graph_run_store.active(project_id)
        if active_run is not None:
            raise ActiveGraphRunExistsError(
                f"Project already has an active GraphRun: {active_run.run_id}"
            )


def _require_publication_digest(
    graph: PublishedGraphDraft | PublishedGraph, expected_draft_sha256: str | None
) -> None:
    if expected_draft_sha256 is not None and expected_draft_sha256 != canonical_sha256(
        _draft_content_payload(graph)
    ):
        raise PublicationValidationError(
            [
                PublicationIssue(
                    code="publication_draft_stale",
                    message="Publication draft is stale; compile it before publishing.",
                )
            ]
        )


class PublicationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)
    node_id: str | None = Field(default=None, max_length=128)


class PublicationPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    project_id: str
    source_graph_id: str
    source_graph_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    publication_mode: PublicationMode = "debug"
    ready: bool
    issues: list[PublicationIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ready(self) -> PublicationPreflight:
        if self.ready != (not self.issues):
            raise ValueError("publication readiness must match the issue list")
        return self


class PublishedGraphDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    candidate_release_id: str = Field(min_length=1, max_length=128)
    project_id: str
    source_graph_id: str = Field(min_length=1, max_length=128)
    source_graph_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    execution_profile: ExecutionProfile = "echomind.flywheel/v1"
    publication_mode: PublicationMode = "debug"
    runtime_requirements: RuntimeRequirements
    promotion_policy: PromotionPolicy
    nodes: list[ProjectNode] = Field(min_length=1)
    edges: list[ProjectEdge] = Field(default_factory=list)
    content_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    adopted_debug_run: DebugExecutionCandidate | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_draft(self) -> PublishedGraphDraft:
        ProjectGraph(
            graph_id=self.source_graph_id,
            project_id=self.project_id,
            lifecycle="development",
            nodes=self.nodes,
            edges=self.edges,
        )
        if any(node.launch_mode != "executor" for node in self.nodes):
            raise ValueError("published draft nodes must use executor launch mode")
        if self.content_sha256 != self.compute_content_sha256():
            raise ValueError("published draft digest does not match content")
        return self

    def compute_content_sha256(self) -> str:
        return canonical_sha256(_draft_content_payload(self))


class PublicationValidationError(ValueError):
    def __init__(self, issues: list[PublicationIssue]) -> None:
        self.issues = issues
        super().__init__(
            "Project cannot be compiled for publication: "
            + "; ".join(issue.message for issue in issues)
        )


class RuntimeGraphRepairProposal(GraphRepairDraft):
    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))

    def repaired_node(self, source: PublishedGraph) -> ProjectNode:
        if source.content_sha256 != self.release_digest:
            raise ProjectLifecycleError("Repair proposal is for a different snapshot")
        node = next((item for item in source.nodes if item.id == self.node_id), None)
        if node is None:
            raise ProjectLifecycleError("Repair proposal references an unknown stage")
        return _repair_executor_config(node, self.runtime_config)


def _repair_executor_config(
    node: ProjectNode, changes: RuntimeGraphRepairChanges
) -> ProjectNode:
    current = node.config.get(node.executor)
    if not isinstance(current, dict):
        raise ProjectLifecycleError(
            "Repair stage has no registered executor configuration"
        )
    updated = deepcopy(current)
    updated.update(
        changes.model_dump(exclude_none=True, exclude={"workflow_parameter_updates"})
    )
    credential_terms = ("api_key", "token", "secret", "password", "credential")
    paths = [*(changes.workflow_bindings or {}), *(changes.workflow_templates or {})]
    if any(term in path.lower() for path in paths for term in credential_terms):
        raise ProjectLifecycleError(
            "Workflow credential bindings are managed separately"
        )
    values = dict.fromkeys(
        (
            *node.inputs,
            "project_id",
            "release_id",
            "graph_run_id",
            "stage_run_id",
            "node_id",
            "output_uri",
        ),
        "validation",
    )
    try:
        for template in (changes.workflow_templates or {}).values():
            render_workflow_template(template, values=values)
        if changes.command_template is not None:
            render_workflow_template(changes.command_template, values=values)
    except TrainRuntimeError as exc:
        raise ProjectLifecycleError(str(exc)) from exc
    if changes.workflow_parameter_updates:
        workflow = updated.get("workflow")
        if not isinstance(workflow, dict) or not isinstance(
            workflow.get("nodes"), list
        ):
            raise ProjectLifecycleError(
                "Workflow parameter repair requires an inline workflow"
            )
        nodes = {
            str(item["id"]): item["data"]["config"]
            for item in workflow["nodes"]
            if isinstance(item, dict)
            and "id" in item
            and isinstance(item.get("data"), dict)
            and isinstance(item["data"].get("config"), dict)
        }
        for node_id, parameters in changes.workflow_parameter_updates.items():
            target = nodes.get(node_id)
            if target is None or not parameters.keys() <= target.keys():
                raise ProjectLifecycleError(
                    "Repair references an unknown workflow parameter"
                )
            if any(
                term in key.lower() for key in parameters for term in credential_terms
            ):
                raise ProjectLifecycleError(
                    "Workflow credential parameters are managed separately"
                )
            target.update(parameters)
    if updated == current:
        raise ProjectLifecycleError(
            "Repair proposal does not change the executable stage"
        )
    return node.model_copy(update={"config": {**node.config, node.executor: updated}})


class RuntimeContractAssumption(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)
    node_id: str | None = Field(default=None, max_length=128)


class RuntimeValidationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    name: str


class RuntimeContractUpgradeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    source_project_id: str
    source_graph_id: str
    source_graph_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    nodes: list[ProjectNode] = Field(min_length=1)
    edges: list[ProjectEdge] = Field(default_factory=list)
    assumptions: list[RuntimeContractAssumption] = Field(default_factory=list)
    issues: list[PublicationIssue] = Field(default_factory=list)
    validation_candidates: list[RuntimeValidationCandidate] = Field(
        default_factory=list
    )
    validation_project_id: str | None = None
    resource_release_id: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    content_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    adopted_debug_run: DebugExecutionCandidate | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @property
    def ready(self) -> bool:
        return not self.issues

    @model_validator(mode="after")
    def validate_draft(self) -> RuntimeContractUpgradeDraft:
        self._validate_graph()
        if self.content_sha256 != self.compute_content_sha256():
            raise ValueError("runtime contract draft digest does not match content")
        return self

    def _validate_graph(self) -> None:
        ProjectGraph(
            graph_id=self.source_graph_id,
            project_id=self.source_project_id,
            lifecycle="development",
            nodes=self.nodes,
            edges=self.edges,
        )

    def compute_content_sha256(self) -> str:
        excluded = {"content_sha256", "validation_candidates", "resource_release_id"}
        if self.validation_project_id is None:
            excluded.add("validation_project_id")
        if self.adopted_debug_run is None:
            excluded.add("adopted_debug_run")
        return canonical_sha256(self.model_dump(mode="json", exclude=excluded))


class RuntimeContractApplication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    source_project_id: str
    draft_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    project: ProjectInfo


class RuntimeImplementationAdoption(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    source_project_id: str
    validation_project_id: str
    project: ProjectInfo


class RuntimeImplementationAdoptionError(ValueError):
    pass


def _debug_adoption_revision(project: ProjectInfo) -> str:
    return canonical_sha256(
        {
            "source_graph_digest": _source_graph_digest(project.graph),
            "requirements": [
                requirement.model_dump(mode="json")
                for requirement in project.project.requirements
            ],
        }
    )


def _topology(
    nodes: list[ProjectNode], edges: list[ProjectEdge]
) -> tuple[dict[str, ProjectNodeType], list[tuple[str, str]]]:
    return (
        {node.id: node.type for node in nodes},
        sorted((edge.source, edge.target) for edge in edges),
    )


def _debug_execution_changes(
    current: list[ProjectNode], proposed: list[ProjectNode]
) -> list[DebugExecutionChange]:
    current_by_id = {node.id: node for node in current}
    changes: list[DebugExecutionChange] = []
    fields = (
        "executor",
        "launch_mode",
        "inputs",
        "outputs",
        "implementation_ref",
        "implementation_digest",
        "policies",
    )
    for node in proposed:
        before = current_by_id[node.id].model_dump(mode="json")
        after = node.model_dump(mode="json")
        changed = [field for field in fields if before[field] != after[field]]
        previous_config = before["config"]
        next_config = after["config"]
        changed.extend(
            f"config.{key}"
            for key in sorted(previous_config.keys() | next_config.keys())
            if key not in previous_config
            or key not in next_config
            or previous_config[key] != next_config[key]
        )
        if changed:
            changes.append(
                DebugExecutionChange(node_id=node.id, changed_fields=changed)
            )
    return changes


def _with_runtime_draft_updates(
    draft: RuntimeContractUpgradeDraft, **updates: Any
) -> RuntimeContractUpgradeDraft:
    draft = draft.model_copy(update=updates)
    return RuntimeContractUpgradeDraft.model_validate(
        {
            **draft.model_dump(mode="json"),
            "content_sha256": draft.compute_content_sha256(),
        }
    )


_REQUIRED_INPUTS: dict[
    str,
    dict[str, tuple[str, frozenset[str], str | None, str | None]],
] = {
    "dataset": {
        "source_batch": (
            "source_batch",
            frozenset({"run_input", "runtime_slot", "literal"}),
            None,
            None,
        ),
    },
    "train": {
        "train_dataset": (
            "train_messages",
            frozenset({"node_output"}),
            "dataset",
            "train_messages",
        ),
        "validation_dataset": (
            "val_messages",
            frozenset({"node_output"}),
            "dataset",
            "val_messages",
        ),
        "base_model": (
            "base_model",
            frozenset({"run_input", "runtime_slot", "literal"}),
            None,
            None,
        ),
    },
    "evaluator": {
        "candidate_model": (
            "trained_model",
            frozenset({"node_output"}),
            "train",
            "trained_model",
        ),
        "validation_dataset": (
            "val_messages",
            frozenset({"node_output", "literal"}),
            "dataset",
            "val_messages",
        ),
    },
}

_REQUIRED_OUTPUTS = {
    "dataset": ("train_messages", "val_messages"),
    "train": ("trained_model",),
    "evaluator": ("eval_report",),
}


_RESOURCE_CONTRACTS: dict[str, tuple[str, str]] = {
    "data": (
        "dataset.source",
        "echomind://schemas/dataset-source/v1",
    ),
    "base_model": (
        "model.checkpoint",
        "echomind://schemas/model/v1",
    ),
}

_OUTPUT_CONTRACTS: dict[str, tuple[str, str]] = {
    "train_messages": (
        "dataset.messages.train",
        "echomind://schemas/messages-dataset/v1",
    ),
    "val_messages": (
        "dataset.messages.validation",
        "echomind://schemas/messages-dataset/v1",
    ),
    "trained_model": (
        "model.checkpoint",
        "echomind://schemas/model/v1",
    ),
    "eval_report": (
        "evaluation.report",
        "echomind://schemas/evaluation-report/v1",
    ),
}


def build_runtime_contract_draft(
    project: ProjectInfo,
    *,
    validation_project: ProjectInfo | None = None,
) -> RuntimeContractUpgradeDraft:
    nodes = (
        _preview_runtime_nodes(project, validation_project)
        if validation_project is not None
        else project.graph.nodes
    )
    assumptions: list[RuntimeContractAssumption] = []
    issues: list[PublicationIssue] = []
    task_to_stage = _task_to_stage(project)
    resources = _resources_by_id(project)
    upgraded_nodes: list[ProjectNode] = []

    for node in nodes:
        tasks = _node_tasks(node)
        if not tasks:
            issues.append(
                PublicationIssue(
                    code="runtime_task_missing",
                    message=f"Node has no execution task configuration: {node.id}",
                    node_id=node.id,
                )
            )
        inputs, outputs = node.inputs, node.outputs
        if "development_evidence" not in node.config:
            inputs = {
                **_derive_node_inputs(
                    node=node,
                    tasks=tasks,
                    task_to_stage=task_to_stage,
                    resources=resources,
                    assumptions=assumptions,
                    issues=issues,
                ),
                **inputs,
            }
            outputs = {
                **_derive_node_outputs(node=node, tasks=tasks, issues=issues),
                **outputs,
            }
        implementation_ref, implementation_digest = _implementation_binding(
            node=node,
            assumptions=assumptions,
            issues=issues,
        )
        policies = _derive_policies(
            node=node,
            assumptions=assumptions,
            issues=issues,
        )
        if node.launch_mode != "executor":
            assumptions.append(
                RuntimeContractAssumption(
                    code="executor_launch_derived",
                    message=(
                        f"{node.id} will publish with executor launch mode for "
                        "runtime execution."
                    ),
                    node_id=node.id,
                )
            )
        upgraded_nodes.append(
            node.model_copy(
                update={
                    "status": "idle",
                    "launch_mode": "executor",
                    "implementation_ref": implementation_ref,
                    "implementation_digest": implementation_digest,
                    "inputs": inputs,
                    "outputs": outputs,
                    "policies": policies,
                }
            )
        )

    if validation_project is not None:
        for node in upgraded_nodes:
            _validate_executor(node, "production", issues)

    edges = sorted(project.graph.edges, key=lambda edge: (edge.source, edge.target))
    draft = RuntimeContractUpgradeDraft.model_construct(
        schema_version=1,
        source_project_id=project.project.project_id,
        source_graph_id=project.graph.graph_id,
        source_graph_digest=_source_graph_digest(project.graph),
        nodes=upgraded_nodes,
        edges=edges,
        assumptions=assumptions,
        issues=issues,
        validation_project_id=(
            validation_project.project.project_id
            if validation_project is not None
            else None
        ),
        content_sha256="sha256:" + "0" * 64,
    )
    if validation_project is not None:
        try:
            draft._validate_graph()
        except ValidationError as exc:
            raise RuntimeImplementationAdoptionError(
                "Validation Project bindings do not match the source Project's "
                "runtime graph."
            ) from exc
    digest = draft.compute_content_sha256()
    return RuntimeContractUpgradeDraft.model_validate(
        {
            **draft.model_dump(mode="json"),
            "content_sha256": digest,
        }
    )


def adopt_runtime_implementation(
    *,
    store: ProjectStore,
    project: ProjectInfo,
    validation_project_id: str | None = None,
) -> RuntimeImplementationAdoption:
    if validation_project_id is None:
        candidates = list_runtime_validation_candidates(store=store, project=project)
        if not candidates:
            raise RuntimeImplementationAdoptionError(
                "No runtime validation recipe was found for this Project."
            )
        if len(candidates) > 1:
            raise RuntimeImplementationAdoptionError(
                "Multiple runtime validation recipes were found; select one explicitly."
            )
        validation_project_id = candidates[0].project_id
    validation = load_runtime_validation_project(
        store=store,
        project=project,
        validation_project_id=validation_project_id,
    )
    draft = build_runtime_contract_draft(project, validation_project=validation)
    if draft.issues:
        raise RuntimeImplementationAdoptionError(
            "Runtime implementation contract is invalid: "
            + ", ".join(issue.code for issue in draft.issues)
        )
    return RuntimeImplementationAdoption(
        source_project_id=project.project.project_id,
        validation_project_id=validation.project.project_id,
        project=ProjectInfo(
            project=project.project,
            graph=project.graph.model_copy(update={"nodes": draft.nodes}),
        ),
    )


def _preview_runtime_nodes(
    project: ProjectInfo, validation: ProjectInfo
) -> list[ProjectNode]:
    _require_matching_validation_project(project, validation)
    validation_nodes = {node.id: node for node in validation.graph.nodes}
    runtime_nodes: list[ProjectNode] = []
    for node in project.graph.nodes:
        validation_node = validation_nodes.get(node.id)
        if validation_node is None or validation_node.type != node.type:
            raise RuntimeImplementationAdoptionError(
                f"Validation Project does not provide a matching {node.id} node."
            )
        executor = validation_node.executor
        if (
            validation_node.launch_mode != "executor"
            or _RUNTIME_EXECUTOR_TYPES.get(executor) != node.type
        ):
            raise RuntimeImplementationAdoptionError(
                f"Validation node {node.id} is not backed by a compatible "
                "runtime executor."
            )
        recipe_keys = {executor, "runtime_input_defaults"}
        config = {
            key: value for key, value in node.config.items() if key not in recipe_keys
        }
        config.update(
            {
                key: deepcopy(validation_node.config[key])
                for key in recipe_keys
                if key in validation_node.config
            }
        )
        runtime_nodes.append(
            node.model_copy(
                update={
                    "launch_mode": validation_node.launch_mode,
                    "executor": executor,
                    "config": config,
                    "inputs": {**node.inputs, **validation_node.inputs},
                }
            )
        )
    return runtime_nodes


def list_runtime_validation_candidates(
    *, store: ProjectStore, project: ProjectInfo
) -> list[RuntimeValidationCandidate]:
    candidates: list[RuntimeValidationCandidate] = []
    for summary in store.list(owner_user_id=project.project.owner_user_id):
        if (
            summary.project_id == project.project.project_id
            or summary.conversation_id != project.project.conversation_id
            or summary.cluster != project.project.cluster
            or summary.lifecycle != "development"
        ):
            continue
        candidate = store.load(
            summary.project_id, owner_user_id=project.project.owner_user_id
        )
        if candidate.graph.template != RUNTIME_VALIDATION_GRAPH_TEMPLATE:
            continue
        try:
            draft = build_runtime_contract_draft(project, validation_project=candidate)
        except RuntimeImplementationAdoptionError:
            continue
        if draft.ready:
            candidates.append(
                RuntimeValidationCandidate(
                    project_id=summary.project_id, name=summary.name
                )
            )
    return candidates


def load_runtime_validation_project(
    *, store: ProjectStore, project: ProjectInfo, validation_project_id: str
) -> ProjectInfo:
    try:
        validation = store.load(
            validation_project_id,
            owner_user_id=project.project.owner_user_id,
        )
    except ProjectNotFoundError as exc:
        raise RuntimeImplementationAdoptionError(
            f"Validation Project not found: {validation_project_id}"
        ) from exc
    _require_matching_validation_project(project, validation)
    return validation


def _require_matching_validation_project(
    project: ProjectInfo, validation: ProjectInfo
) -> None:
    if validation.project.project_id == project.project.project_id:
        raise RuntimeImplementationAdoptionError(
            "A Project cannot select itself as a runtime recipe."
        )
    if validation.project.owner_user_id != project.project.owner_user_id:
        raise RuntimeImplementationAdoptionError(
            "Validation Project belongs to another owner."
        )
    if validation.project.conversation_id != project.project.conversation_id:
        raise RuntimeImplementationAdoptionError(
            "Validation Project belongs to another conversation."
        )
    if validation.project.cluster != project.project.cluster:
        raise RuntimeImplementationAdoptionError(
            "Validation Project belongs to another cluster."
        )
    if (
        validation.project.lifecycle != "development"
        or validation.graph.template != RUNTIME_VALIDATION_GRAPH_TEMPLATE
    ):
        raise RuntimeImplementationAdoptionError(
            "Runtime implementation must come from a development Validation Project."
        )


def _reviewed_runtime_contract(
    *,
    store: ProjectStore,
    project: ProjectInfo,
    expected_contract_sha256: str,
    validation_project_id: str | None,
) -> RuntimeContractUpgradeDraft:
    validation = (
        load_runtime_validation_project(
            store=store,
            project=project,
            validation_project_id=validation_project_id,
        )
        if validation_project_id is not None
        else None
    )
    draft = build_runtime_contract_draft(project, validation_project=validation)
    _require_reviewed_runtime_contract(draft, expected_contract_sha256)
    return draft


def _require_reviewed_runtime_contract(
    draft: RuntimeContractUpgradeDraft, expected_contract_sha256: str
) -> None:
    if draft.content_sha256 != expected_contract_sha256:
        raise PublicationValidationError(
            [
                PublicationIssue(
                    code="runtime_contract_draft_stale",
                    message=(
                        "Runtime contract draft no longer matches the source "
                        "Project graph; review the updated runtime contract."
                    ),
                )
            ]
        )
    if draft.issues:
        raise PublicationValidationError(draft.issues)


def apply_runtime_contract_draft(
    *,
    store: ProjectStore,
    project: ProjectInfo,
    draft: RuntimeContractUpgradeDraft,
) -> RuntimeContractApplication:
    draft = _reviewed_runtime_contract(
        store=store,
        project=project,
        expected_contract_sha256=draft.content_sha256,
        validation_project_id=draft.validation_project_id,
    )
    project_id = str(
        uuid5(
            NAMESPACE_URL,
            (
                "echomind-runtime-contract:"
                f"{project.project.project_id}:{draft.content_sha256}"
            ),
        )
    )
    with ProjectLease(store.root, project_id).acquire():
        try:
            existing = store.load(
                project_id,
                owner_user_id=project.project.owner_user_id,
            )
        except ProjectNotFoundError:
            existing = None
        if existing is not None:
            return RuntimeContractApplication(
                source_project_id=project.project.project_id,
                draft_sha256=draft.content_sha256,
                project=existing,
            )

        record = ProjectRecord(
            project_id=project_id,
            name=f"{project.project.name} (Runtime Contract)",
            description=project.project.description,
            skill=project.project.skill,
            requirements=project.project.requirements,
            owner_user_id=project.project.owner_user_id,
            conversation_id=project.project.conversation_id,
            cluster=project.project.cluster,
            lifecycle="development",
            source_project_id=project.project.project_id,
            runtime_contract_sha256=draft.content_sha256,
            validation_project_id=draft.validation_project_id,
            created_at=_utc_now(),
        )
        graph = ProjectGraph(
            graph_id=str(
                uuid5(
                    NAMESPACE_URL,
                    f"echomind-runtime-graph:{project_id}:{draft.content_sha256}",
                )
            ),
            project_id=project_id,
            template="echomind.runtime-contract/v1",
            lifecycle="development",
            nodes=draft.nodes,
            edges=draft.edges,
        )
        derived = store.save(ProjectInfo(project=record, graph=graph))
        return RuntimeContractApplication(
            source_project_id=project.project.project_id,
            draft_sha256=draft.content_sha256,
            project=derived,
        )


def _task_to_stage(project: ProjectInfo) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for node in project.graph.nodes:
        for task in _node_tasks(node):
            task_id = task.get("id")
            stage_id = task.get("stage_id")
            if isinstance(task_id, str) and isinstance(stage_id, str):
                mapping[task_id] = stage_id
    return mapping


def _resources_by_id(project: ProjectInfo) -> dict[str, dict[str, Any]]:
    resources: dict[str, dict[str, Any]] = {}
    for node in project.graph.nodes:
        raw = node.config.get("resources")
        if not isinstance(raw, list):
            continue
        for resource in raw:
            if not isinstance(resource, dict):
                continue
            resource_id = resource.get("id")
            if isinstance(resource_id, str):
                resources.setdefault(resource_id, resource)
    return resources


def _node_tasks(node: ProjectNode) -> list[dict[str, Any]]:
    raw = node.config.get("tasks")
    if not isinstance(raw, list):
        return []
    return [task for task in raw if isinstance(task, dict)]


def _derive_node_inputs(
    *,
    node: ProjectNode,
    tasks: list[dict[str, Any]],
    task_to_stage: dict[str, str],
    resources: dict[str, dict[str, Any]],
    assumptions: list[RuntimeContractAssumption],
    issues: list[PublicationIssue],
) -> dict[str, ProjectNodeInput]:
    inputs: dict[str, ProjectNodeInput] = {}
    for task in tasks:
        raw_inputs = task.get("inputs")
        if not isinstance(raw_inputs, list):
            continue
        for raw_input in raw_inputs:
            if not isinstance(raw_input, dict):
                continue
            input_name = raw_input.get("name")
            if not isinstance(input_name, str) or not input_name:
                continue
            resource_id = raw_input.get("resource_id")
            source_task_id = raw_input.get("task_id")
            output_name = raw_input.get("output_name")
            if isinstance(resource_id, str) and resource_id:
                contract = _RESOURCE_CONTRACTS.get(
                    resource_id,
                    ("runtime.resource", "echomind://schemas/resource/v1"),
                )
                port_name = (
                    "source_batch"
                    if node.type == "dataset" and resource_id == "data"
                    else input_name
                )
                resource = resources.get(resource_id, {})
                assumptions.append(
                    RuntimeContractAssumption(
                        code="resource_mapped_to_run_input",
                        message=(
                            f"{node.id}.{input_name} maps resource {resource_id} "
                            "to a per-run input."
                        ),
                        node_id=node.id,
                    )
                )
                if resource.get("uri"):
                    assumptions.append(
                        RuntimeContractAssumption(
                            code="resource_default_recorded",
                            message=(
                                f"{resource_id} currently points to "
                                f"{resource['uri']}; each GraphRun may override it."
                            ),
                            node_id=node.id,
                        )
                    )
                inputs[port_name] = ProjectNodeInput(
                    artifact_type=contract[0],
                    schema_ref=contract[1],
                    binding=ProjectNodeInputBinding(
                        kind="run_input",
                        name=input_name,
                    ),
                )
                continue
            if (
                isinstance(source_task_id, str)
                and isinstance(output_name, str)
                and source_task_id
                and output_name
            ):
                source_node_id = task_to_stage.get(source_task_id)
                if source_node_id == node.id:
                    continue
                if source_node_id is None:
                    issues.append(
                        PublicationIssue(
                            code="runtime_input_source_missing",
                            message=(
                                f"{node.id}.{input_name} references unknown task "
                                f"{source_task_id}."
                            ),
                            node_id=node.id,
                        )
                    )
                    continue
                source_contract = _OUTPUT_CONTRACTS.get(
                    output_name,
                    ("runtime.artifact", "echomind://schemas/artifact/v1"),
                )
                port_name = _input_port_name(
                    node_type=node.type,
                    output_name=output_name,
                    fallback=input_name,
                )
                inputs[port_name] = ProjectNodeInput(
                    artifact_type=source_contract[0],
                    schema_ref=source_contract[1],
                    binding=ProjectNodeInputBinding(
                        kind="node_output",
                        node_id=source_node_id,
                        port=output_name,
                    ),
                )
    return inputs


def _derive_node_outputs(
    *,
    node: ProjectNode,
    tasks: list[dict[str, Any]],
    issues: list[PublicationIssue],
) -> dict[str, ProjectNodeOutput]:
    output_names: list[str] = []
    for task in tasks:
        raw_outputs = task.get("outputs")
        if not isinstance(raw_outputs, list):
            continue
        for raw_output in raw_outputs:
            if not isinstance(raw_output, dict):
                continue
            output_name = raw_output.get("name")
            if isinstance(output_name, str) and output_name:
                output_names.append(output_name)
    if not output_names:
        output_names.extend(_REQUIRED_OUTPUTS.get(node.type, ()))
    outputs: dict[str, ProjectNodeOutput] = {}
    for output_name in dict.fromkeys(output_names):
        contract = _OUTPUT_CONTRACTS.get(
            output_name,
            ("runtime.artifact", "echomind://schemas/artifact/v1"),
        )
        outputs[output_name] = ProjectNodeOutput(
            artifact_type=contract[0],
            schema_ref=contract[1],
        )
    if not outputs:
        issues.append(
            PublicationIssue(
                code="runtime_output_missing",
                message=f"No runtime outputs could be derived for node {node.id}.",
                node_id=node.id,
            )
        )
    return outputs


def _input_port_name(
    *,
    node_type: str,
    output_name: str,
    fallback: str,
) -> str:
    if node_type == "train" and output_name == "train_messages":
        return "train_dataset"
    if node_type == "train" and output_name == "val_messages":
        return "validation_dataset"
    if node_type == "evaluator" and output_name == "val_messages":
        return "validation_dataset"
    if node_type == "evaluator" and output_name == "trained_model":
        return "candidate_model"
    return fallback


def _implementation_binding(
    *,
    node: ProjectNode,
    assumptions: list[RuntimeContractAssumption],
    issues: list[PublicationIssue],
) -> tuple[str | None, str | None]:
    if node.executor not in DEBUG_ONLY_EXECUTORS and (
        node.executor not in _RUNTIME_EXECUTOR_TYPES
    ):
        _validate_executor(node, "debug", issues)
        return None, None
    if node.executor != "contract_probe":
        assumptions.append(
            RuntimeContractAssumption(
                code="executor_implementation_bound",
                message=(
                    f"{node.id} uses registered executor {node.executor}; "
                    "its builtin implementation digest will be frozen."
                ),
                node_id=node.id,
            )
        )
    implementation_ref = f"builtin://echomind/{node.executor}/v1"
    implementation_digest = canonical_sha256(
        {
            "kind": "builtin_executor",
            "name": node.executor,
            "schema_version": 1,
        }
    )
    return implementation_ref, implementation_digest


def _derive_policies(
    *,
    node: ProjectNode,
    assumptions: list[RuntimeContractAssumption],
    issues: list[PublicationIssue],
) -> dict[str, Any]:
    policies = dict(node.policies)
    if node.type != "evaluator":
        return policies
    if not isinstance(policies.get("promotion"), dict):
        policies["promotion"] = {
            "metric": "accuracy",
            "direction": "maximize",
            "minimum_gain": 0.0,
            "require_candidate_healthy": True,
        }
        assumptions.append(
            RuntimeContractAssumption(
                code="promotion_policy_defaulted",
                message=(
                    f"{node.id} will publish with accuracy/maximize promotion "
                    "policy. Confirm the metric before applying."
                ),
                node_id=node.id,
            )
        )
    if not isinstance(policies.get("candidate_inference"), dict):
        policies["candidate_inference"] = {
            "create": "evaluator",
            "health_check": "required",
            "cleanup": "finalizer",
        }
        assumptions.append(
            RuntimeContractAssumption(
                code="candidate_inference_policy_defaulted",
                message=(
                    f"{node.id} will create and clean up candidate inference "
                    "inside Evaluator execution."
                ),
                node_id=node.id,
            )
        )
    try:
        PromotionPolicy.model_validate(policies["promotion"])
    except ValueError as exc:
        issues.append(
            PublicationIssue(
                code="promotion_policy_invalid",
                message=f"Evaluator promotion policy is invalid: {exc}",
                node_id=node.id,
            )
        )
    return policies


def build_publication_preflight(
    project: ProjectInfo,
    *,
    publication_mode: PublicationMode = "debug",
) -> PublicationPreflight:
    issues = validate_for_publication(
        project,
        publication_mode=publication_mode,
    )
    return PublicationPreflight(
        project_id=project.project.project_id,
        source_graph_id=project.graph.graph_id,
        source_graph_digest=_source_graph_digest(project.graph),
        publication_mode=publication_mode,
        ready=not issues,
        issues=issues,
    )


def compile_working_graph(
    project: ProjectInfo,
    *,
    publication_mode: PublicationMode = "debug",
    require_production_executors: bool = False,
    adopted_debug_run: DebugExecutionCandidate | None = None,
) -> PublishedGraphDraft:
    preflight = build_publication_preflight(
        project,
        publication_mode=(
            "production" if require_production_executors else publication_mode
        ),
    )
    if preflight.issues:
        raise PublicationValidationError(preflight.issues)

    nodes = _publication_nodes(project.graph)
    evaluator = next(node for node in nodes if node.type == "evaluator")
    draft = PublishedGraphDraft.model_construct(
        schema_version=1,
        candidate_release_id="pending",
        project_id=project.project.project_id,
        source_graph_id=project.graph.graph_id,
        source_graph_digest=preflight.source_graph_digest,
        execution_profile="echomind.flywheel/v1",
        publication_mode=publication_mode,
        runtime_requirements=RuntimeRequirements(),
        promotion_policy=PromotionPolicy.model_validate(
            evaluator.policies["promotion"]
        ),
        nodes=nodes,
        edges=sorted(project.graph.edges, key=lambda edge: (edge.source, edge.target)),
        content_sha256="sha256:" + "0" * 64,
        adopted_debug_run=adopted_debug_run,
    )
    digest = draft.compute_content_sha256()
    release_scope = (
        "echomind-draft-debug"
        if require_production_executors and publication_mode == "debug"
        else "echomind-release"
    )
    return PublishedGraphDraft.model_validate(
        {
            **draft.model_dump(mode="json"),
            "candidate_release_id": str(
                uuid5(NAMESPACE_URL, f"{release_scope}:{digest}")
            ),
            "content_sha256": digest,
        }
    )


def validate_for_publication(
    project: ProjectInfo,
    *,
    publication_mode: PublicationMode = "debug",
) -> list[PublicationIssue]:
    issues: list[PublicationIssue] = []
    if project.project.lifecycle != "development":
        issues.append(
            PublicationIssue(
                code="project_not_development",
                message="Project must be in development before publication",
            )
        )
    for requirement in project.project.requirements:
        if requirement.status != "confirmed":
            issues.append(
                PublicationIssue(
                    code="requirement_unconfirmed",
                    message=f"Requirement is not confirmed: {requirement.key}",
                )
            )

    nodes_by_type: dict[str, list[ProjectNode]] = {}
    for node in project.graph.nodes:
        nodes_by_type.setdefault(node.type, []).append(node)
        if node.launch_mode != "executor":
            issues.append(
                PublicationIssue(
                    code="agent_launch_mode",
                    message=f"Node must use executor launch mode: {node.id}",
                    node_id=node.id,
                )
            )
        _validate_executor(node, publication_mode, issues)
        if not node.implementation_ref or not node.implementation_digest:
            issues.append(
                PublicationIssue(
                    code="implementation_binding_missing",
                    message=f"Node has no immutable implementation binding: {node.id}",
                    node_id=node.id,
                )
            )

    for node_type, label in (
        ("dataset", "Dataset"),
        ("train", "Train"),
        ("evaluator", "Evaluator"),
    ):
        count = len(nodes_by_type.get(node_type, []))
        if count != 1:
            issues.append(
                PublicationIssue(
                    code="node_cardinality",
                    message=(
                        f"Published graph requires exactly one {label}, found {count}"
                    ),
                )
            )
    for node in nodes_by_type.get("inference", []):
        issues.append(
            PublicationIssue(
                code="inference_is_runtime_resource",
                message=(
                    "Inference is a RuntimeSlot resource and cannot be a "
                    f"published business node: {node.id}"
                ),
                node_id=node.id,
            )
        )

    _validate_ports(project, issues)
    _validate_profile_topology(project, nodes_by_type, issues)
    _validate_evaluator_policies(nodes_by_type, issues)
    return issues


def _validate_executor(
    node: ProjectNode,
    publication_mode: PublicationMode,
    issues: list[PublicationIssue],
) -> None:
    executor_type = _RUNTIME_EXECUTOR_TYPES.get(node.executor)
    if node.executor in DEBUG_ONLY_EXECUTORS:
        if publication_mode == "production":
            issues.append(
                PublicationIssue(
                    code="debug_executor_in_production",
                    message=f"Node {node.id} uses debug-only executor {node.executor}.",
                    node_id=node.id,
                )
            )
        return
    if executor_type != node.type:
        issues.append(
            PublicationIssue(
                code="unknown_executor"
                if executor_type is None
                else "executor_type_mismatch",
                message=(
                    f"Executor {node.executor} cannot run {node.type} node {node.id}."
                ),
                node_id=node.id,
            )
        )
        return
    if publication_mode == "debug":
        return

    config = node.config.get(node.executor)
    try:
        if node.executor == "dataset_runtime":
            if config is None:
                raise ValueError(
                    "Dataset requires a full platform processing workflow. "
                    "Select a recipe that includes full Dataset processing."
                )
            DatasetWorkflowConfig.model_validate(config).validate_workflow_inputs(
                input_ports=node.inputs.keys()
            )
        else:
            _validate_platform_workflow_source(config)
    except (ValueError, RuntimeError) as exc:
        if isinstance(exc, ValidationError):
            detail = "; ".join(
                f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
                if error["loc"]
                else error["msg"]
                for error in exc.errors(include_input=False)
            )
        else:
            detail = str(exc)
        issues.append(
            PublicationIssue(
                code="executor_configuration_invalid",
                message=(
                    f"Node {node.id} has invalid {node.executor} configuration: "
                    f"{detail[:1700]}"
                ),
                node_id=node.id,
            )
        )


def _validate_platform_workflow_source(config: Any) -> None:
    if not isinstance(config, dict):
        raise ValueError("Expected a runtime workflow configuration.")
    workflow = config.get("workflow")
    path = config.get("workflow_path")
    if (workflow is None) == (path is None):
        raise ValueError("Configure exactly one of workflow or workflow_path.")
    if workflow is not None:
        if (
            not isinstance(workflow, dict)
            or not isinstance(workflow.get("nodes"), list)
            or not workflow["nodes"]
        ):
            raise ValueError("The inline workflow requires at least one node.")
        _validate_workflow_wandb_config(workflow)
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise ValueError("The workflow path must be a non-empty string.")


def _validate_workflow_wandb_config(workflow: dict[str, Any]) -> None:
    nodes = {
        str(node["id"]): node["data"]
        for node in workflow["nodes"]
        if isinstance(node, dict)
        and "id" in node
        and isinstance(node.get("data"), dict)
    }
    raw_edges = workflow.get("edges")
    edges = raw_edges if isinstance(raw_edges, list) else []
    connected = {
        str(edge.get("target"))
        for edge in edges
        if isinstance(edge, dict)
        and edge.get("targetHandle") == "wandb_config"
        and str(edge.get("source")) in nodes
    }
    for node_id, data in nodes.items():
        node_type = data.get("nodeType")
        if node_type not in (
            "ModelTrainSFTNode",
            "ModelTrainDPONode",
            "ModelTrainGRPONode",
        ):
            continue
        config = data.get("config")
        wandb = config.get("wandb_config") if isinstance(config, dict) else None
        if node_id in connected or (isinstance(wandb, str) and wandb.strip()):
            continue
        raise ValueError(
            f"Workflow node {node_id} ({node_type}) requires wandb_config. "
            "Connect PyroMindWandbConfigBuilderNode.wandb_config."
        )


def _validate_ports(project: ProjectInfo, issues: list[PublicationIssue]) -> None:
    nodes_by_id = {node.id: node for node in project.graph.nodes}
    train_only = any(
        node.type == "dataset" and node.config.get("batch_validation_source") == "none"
        for node in project.graph.nodes
    )
    for node in project.graph.nodes:
        for input_name, (
            required_port,
            allowed_kinds,
            expected_source_type,
            expected_source_port,
        ) in _REQUIRED_INPUTS.get(node.type, {}).items():
            node_input = node.inputs.get(input_name)
            if (
                train_only
                and node.type == "train"
                and input_name == "validation_dataset"
            ):
                if node_input is not None:
                    issues.append(
                        PublicationIssue(
                            code="input_binding_invalid",
                            message=(
                                "Train cannot bind a validation split "
                                "with a train-only Dataset."
                            ),
                            node_id=node.id,
                        )
                    )
                continue
            if node_input is None:
                issues.append(
                    PublicationIssue(
                        code="required_input_missing",
                        message=(
                            f"Node is missing required input: {node.id}.{input_name}"
                        ),
                        node_id=node.id,
                    )
                )
                continue
            binding = node_input.binding
            if binding.kind not in allowed_kinds:
                issues.append(
                    PublicationIssue(
                        code="input_binding_invalid",
                        message=(
                            f"Node input {node.id}.{input_name} must use "
                            f"one of: {', '.join(sorted(allowed_kinds))}"
                        ),
                        node_id=node.id,
                    )
                )
                continue
            if expected_source_type is None:
                continue
            if binding.kind == "literal":
                if not _is_fixed_validation_dataset(node_input):
                    issues.append(
                        PublicationIssue(
                            code="input_binding_source_invalid",
                            message=(
                                f"Node input {node.id}.{input_name} must bind "
                                "a fixed validation JSONL file under /workspace/."
                            ),
                            node_id=node.id,
                        )
                    )
                continue
            source = nodes_by_id.get(binding.node_id or "")
            if (
                source is None
                or source.type != expected_source_type
                or binding.port != expected_source_port
            ):
                issues.append(
                    PublicationIssue(
                        code="input_binding_source_invalid",
                        message=(
                            f"Node input {node.id}.{input_name} must bind "
                            f"{expected_source_type}.{expected_source_port}"
                        ),
                        node_id=node.id,
                    )
                )
                continue
            source_output = source.outputs.get(expected_source_port or required_port)
            if source_output is None:
                issues.append(
                    PublicationIssue(
                        code="required_output_missing",
                        message=f"Node is missing required output: "
                        f"{source.id}.{expected_source_port or required_port}",
                        node_id=source.id,
                    )
                )
                continue
            if (
                source_output.artifact_type != node_input.artifact_type
                or source_output.schema_ref != node_input.schema_ref
            ):
                issues.append(
                    PublicationIssue(
                        code="port_contract_mismatch",
                        message=(
                            f"Port contract mismatch: "
                            f"{source.id}.{expected_source_port} "
                            f"does not match {node.id}.{input_name}"
                        ),
                        node_id=node.id,
                    )
                )

        for required_output in _REQUIRED_OUTPUTS.get(node.type, ()):
            if required_output in node.outputs:
                continue
            issues.append(
                PublicationIssue(
                    code="required_output_missing",
                    message=(
                        f"Node is missing required output: {node.id}.{required_output}"
                    ),
                    node_id=node.id,
                )
            )


def _is_fixed_validation_dataset(port: ProjectNodeInput) -> bool:
    value = port.binding.value
    if not isinstance(value, str) or not value.endswith(".jsonl"):
        return False
    try:
        return (
            normalize_workspace_path(value) == value
            and (
                port.artifact_type,
                port.schema_ref,
            )
            == _OUTPUT_CONTRACTS["val_messages"]
        )
    except StoragePathError:
        return False


def _validate_profile_topology(
    project: ProjectInfo,
    nodes_by_type: dict[str, list[ProjectNode]],
    issues: list[PublicationIssue],
) -> None:
    dataset = _single_node(nodes_by_type, "dataset")
    train = _single_node(nodes_by_type, "train")
    evaluator = _single_node(nodes_by_type, "evaluator")
    if dataset is None or train is None or evaluator is None:
        return
    expected_edges = {
        (dataset.id, train.id),
        (train.id, evaluator.id),
    }
    actual_edges = [(edge.source, edge.target) for edge in project.graph.edges]
    if len(actual_edges) != len(expected_edges) or set(actual_edges) != expected_edges:
        issues.append(
            PublicationIssue(
                code="profile_topology_mismatch",
                message="echomind.flywheel/v1 requires Dataset -> Train -> Evaluator",
            )
        )


def _validate_evaluator_policies(
    nodes_by_type: dict[str, list[ProjectNode]],
    issues: list[PublicationIssue],
) -> None:
    evaluator = _single_node(nodes_by_type, "evaluator")
    if evaluator is None:
        return
    candidate_policy = evaluator.policies.get("candidate_inference")
    if not isinstance(candidate_policy, dict) or not all(
        candidate_policy.get(key) for key in ("create", "health_check", "cleanup")
    ):
        issues.append(
            PublicationIssue(
                code="candidate_inference_policy_missing",
                message=(
                    "Evaluator must define Candidate Inference create, "
                    "health_check and cleanup policy"
                ),
                node_id=evaluator.id,
            )
        )
    try:
        PromotionPolicy.model_validate(evaluator.policies.get("promotion"))
    except ValueError as exc:
        issues.append(
            PublicationIssue(
                code="promotion_policy_invalid",
                message=f"Evaluator promotion policy is invalid: {exc}",
                node_id=evaluator.id,
            )
        )


def _single_node(
    nodes_by_type: dict[str, list[ProjectNode]],
    node_type: str,
) -> ProjectNode | None:
    nodes = nodes_by_type.get(node_type, [])
    return nodes[0] if len(nodes) == 1 else None


def _source_graph_digest(graph: ProjectGraph) -> str:
    return canonical_sha256(
        {
            "schema_version": graph.schema_version,
            "graph_id": graph.graph_id,
            "project_id": graph.project_id,
            "template": graph.template,
            "nodes": [
                node.model_dump(mode="json") for node in _publication_nodes(graph)
            ],
            "edges": [
                edge.model_dump(mode="json")
                for edge in sorted(
                    graph.edges,
                    key=lambda edge: (edge.source, edge.target),
                )
            ],
        }
    )


def _publication_nodes(graph: ProjectGraph) -> list[ProjectNode]:
    return [
        node.model_copy(update={"status": "idle"})
        for node in sorted(graph.nodes, key=lambda node: node.id)
    ]


def _draft_content_payload(
    draft: PublishedGraphDraft | PublishedGraph,
) -> dict[str, object]:
    payload = {
        "project_id": draft.project_id,
        "source_graph_id": draft.source_graph_id,
        "source_graph_digest": draft.source_graph_digest,
        "execution_profile": draft.execution_profile,
        "publication_mode": draft.publication_mode,
        "runtime_requirements": draft.runtime_requirements.model_dump(mode="json"),
        "promotion_policy": draft.promotion_policy.model_dump(mode="json"),
        "nodes": [node.model_dump(mode="json") for node in draft.nodes],
        "edges": [edge.model_dump(mode="json") for edge in draft.edges],
    }
    if draft.adopted_debug_run is not None:
        payload["adopted_debug_run"] = draft.adopted_debug_run.model_dump(mode="json")
    return payload


def _with_lifecycle(
    project: ProjectInfo,
    *,
    lifecycle: Literal["development", "frozen"],
    active_release_id: str | None = None,
) -> ProjectInfo:
    frozen_at = _utc_now() if lifecycle == "frozen" else None
    return ProjectInfo(
        project=project.project.model_copy(
            update={
                "lifecycle": lifecycle,
                "active_release_id": active_release_id,
                "frozen_at": frozen_at,
            }
        ),
        graph=project.graph.model_copy(update={"lifecycle": lifecycle}),
    )
