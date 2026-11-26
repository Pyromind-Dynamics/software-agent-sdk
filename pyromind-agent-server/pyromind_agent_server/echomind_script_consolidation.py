from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from pyromind_agent_server.echomind_consolidation import (
    ConsolidationBlocker,
    ConsolidationContext,
    ConsolidationDatasetReplay,
    ConsolidationProposal,
    ConsolidationRecipe,
    ConsolidationReview,
    ConsolidationStageName,
    ConsolidationStageStatus,
    compilation_path,
    compile_consolidation,
    consolidation_context,
    validate_consolidation_stage,
)
from pyromind_agent_server.echomind_consolidation_references import (
    _workflow_receipt,
    consolidation_evaluation_reference,
    consolidation_references,
)
from pyromind_agent_server.echomind_consolidation_replay import (
    compatible_train_reader,
    verify_dataset_replays,
)
from pyromind_agent_server.echomind_dataset_consolidation import extract_dataset_recipe
from pyromind_agent_server.echomind_failure_evidence import redact_graph_failure_text
from pyromind_agent_server.echomind_projects import ProjectInfo, ProjectStore
from pyromind_agent_server.echomind_storage import (
    ProjectLease,
    canonical_sha256,
    read_model,
    write_model,
)
from pyromind_runtime.application.consolidation_evidence import (
    successful_dataset_recipe_runs,
)
from pyromind_runtime.application.versions import Versions
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.workflow_execution import DatasetRecipeReceipt
from pyromind_runtime.infrastructure.file_product_store import FileProductStore


class ConsolidationCompilation(BaseModel):
    evidence_sha256: str
    proposal: ConsolidationProposal | None = None
    dataset_replay: ConsolidationDatasetReplay | None = None
    stages: tuple[ConsolidationStageStatus, ...]
    blockers: tuple[ConsolidationBlocker, ...]
    assets: JsonObject = Field(default_factory=dict)
    issue: str | None = None

    def review(
        self, project: ProjectInfo, context: ConsolidationContext
    ) -> ConsolidationReview:
        if self.evidence_sha256 != context.evidence_sha256:
            raise ValueError("执行方案的研发证据已改变，请重新生成。")
        if self.proposal is not None and self.dataset_replay is None:
            return ConsolidationReview(
                context=context,
                proposal_path=context.proposal_path,
                issue="历史方案已通过静态校验，请生成并验证换批结果后采用。",
                assets=self.assets,
            )
        proposal = self.proposal
        return ConsolidationReview(
            context=context,
            proposal_path=context.proposal_path,
            stages=self.stages,
            blockers=self.blockers,
            assets=self.assets,
            issue=self.issue,
            proposal_sha256=(
                canonical_sha256(proposal.model_dump(mode="json")) if proposal else None
            ),
            draft=compile_consolidation(project, context, proposal)
            if proposal
            else None,
        )


def load_script_compilation(
    store: ProjectStore, context: ConsolidationContext
) -> ConsolidationCompilation | None:
    try:
        return read_model(compilation_path(store, context), ConsolidationCompilation)
    except FileNotFoundError:
        return None


def save_script_compilation(
    store: ProjectStore,
    project: ProjectInfo,
    context: ConsolidationContext,
    built: ConsolidationCompilation,
    conversations_root: Path,
) -> ConsolidationReview:
    project_id = project.project.project_id
    with ProjectLease(store.root, project_id).acquire():
        current = store.load(project_id, owner_user_id=project.project.owner_user_id)
        fresh = consolidation_context(
            current,
            conversations_root,
            task_ids={
                "dataset": context.evidence.dataset.task_id,
                "train": context.evidence.train.task_id,
                "eval": context.evidence.eval.task_id,
            },
        )
        review = built.review(current, fresh)
        write_model(compilation_path(store, fresh), built)
        return review


def build_script_compilation(
    project: ProjectInfo,
    context: ConsolidationContext,
    conversations_root: Path,
    request_context: RequestContext,
    *,
    versions: Versions | None,
    recipe: ConsolidationRecipe | None = None,
    read_replay_file: Callable[[str], bytes] | None = None,
    check_replay: Callable[[DatasetRecipeReceipt], bytes] | None = None,
) -> ConsolidationCompilation:
    supplied = recipe or ConsolidationRecipe(evidence_sha256=context.evidence_sha256)
    if supplied.evidence_sha256 != context.evidence_sha256:
        raise ValueError("配方与所选研发任务不匹配，请核对证据摘要。")
    evidence = context.evidence
    store = FileProductStore(conversations_root / evidence.conversation_id)
    references = consolidation_references(store, evidence)
    dataset = _workflow_receipt(
        tuple(store.load_workflow_execution_requests().values()),
        evidence.dataset,
        evidence.conversation_id,
    )
    receipts = successful_dataset_recipe_runs(store, evidence)
    replay = verify_dataset_replays(context, receipts, read_replay_file, check_replay)
    source = extract_dataset_recipe(
        archive_workflow=dataset.workflow,
        source_batch=context.source_batch,
        reviewed_runtime=replay.runtime or supplied.dataset_runtime,
        conversation_id=evidence.conversation_id,
    )
    evaluator = (
        consolidation_evaluation_reference(versions, evidence, request_context)
        if versions is not None and supplied.evaluator_runtime is None
        else {}
    )
    train = references.get("train")
    train_candidate = (
        train.get("runtime_candidate") if isinstance(train, dict) else None
    )
    eval_candidate = evaluator.get("runtime_candidate")
    candidates: dict[ConsolidationStageName, JsonObject | None] = {
        "dataset": source.runtime,
        "train": supplied.train_runtime
        if supplied.train_runtime is not None
        else (train_candidate if isinstance(train_candidate, dict) else None),
        "eval": supplied.evaluator_runtime
        if supplied.evaluator_runtime is not None
        else (eval_candidate if isinstance(eval_candidate, dict) else None),
    }
    assets: JsonObject = {
        "dataset": [asset.model_dump(mode="json") for asset in source.assets],
        "eval": evaluator.get("assets", {}),
    }
    problems = {
        "dataset": "缺少已验证的 Dataset 运行配方。",
        "train": "成功任务的训练工作流未匹配已知绑定，请补充 Train 配方。",
        "eval": _error_text(
            evaluator.get("runtime_blocker"),
            request_context,
            "缺少可验证的评测工作流或版本资产，请补充 Eval 配方。",
        ),
    }
    verified: dict[ConsolidationStageName, JsonObject] = {}
    blockers: list[ConsolidationBlocker] = []
    if replay.proof:
        if (
            context.training_validation
            and context.training_validation != replay.proof.fixed_validation_path
        ):
            blockers.append(
                ConsolidationBlocker(
                    stage="dataset",
                    code="dataset_policy_conflict",
                    message="回放使用的训练中验证集与 Train 在 Plan 中确认的文件不同。",
                )
            )
        node = next(node for node in project.graph.nodes if node.type == "dataset")
        for field, value in replay.proof.model_dump(
            include={
                "batch_label_source",
                "batch_validation_source",
                "fixed_validation_path",
            }
        ).items():
            if field == "batch_label_source" and value is None:
                continue
            if node.config.get(field) is not None and node.config[field] != value:
                blockers.append(
                    ConsolidationBlocker(
                        stage="dataset",
                        code="dataset_policy_conflict",
                        message=f"换批策略 {field} 与 Project 确认值不符，请核对。",
                    )
                )
        assets["dataset_replay"] = replay.proof.model_dump(mode="json")
    for stage, candidate in candidates.items():
        if stage == "dataset" and any(b.stage == stage for b in blockers):
            continue
        if stage == "dataset" and replay.blocker:
            blockers.append(replay.blocker)
            continue
        if (
            stage == "dataset"
            and candidate is not None
            and (
                replay.proof is None
                or (
                    supplied.dataset_runtime is not None
                    and supplied.dataset_runtime != replay.runtime
                )
            )
        ):
            blockers.append(
                ConsolidationBlocker(
                    stage=stage,
                    code="dataset_replay_required",
                    message="候选配方仅通过静态校验。请使用已确认的独立小样执行同一配方并验证。",
                )
            )
            continue
        if candidate is None:
            if stage == "dataset" and source.blockers:
                blockers.extend(source.blockers)
            else:
                blockers.append(
                    ConsolidationBlocker(
                        stage=stage,
                        code="recipe_required",
                        message=problems[stage],
                    )
                )
            continue
        try:
            rendered = validate_consolidation_stage(project, context, stage, candidate)
        except (ValueError, RuntimeError, TypeError) as exc:
            blockers.append(
                ConsolidationBlocker(
                    stage=stage,
                    code="binding_invalid",
                    message=_configuration_error(exc, request_context),
                )
            )
            continue
        if (
            stage == "train"
            and rendered is not None
            and replay.proof
            and not compatible_train_reader({"workflow": rendered})
        ):
            blockers.append(
                ConsolidationBlocker(
                    stage=stage,
                    code="train_consumer_unsupported",
                    message=(
                        "本次回放校验 messages SFT。"
                        "当前 Train 读取器或字段映射不同，请核对消费契约。"
                    ),
                )
            )
            continue
        verified[stage] = candidate
    stages = tuple(
        ConsolidationStageStatus(
            stage=stage,
            status="ready" if stage in verified else "blocked",
            verification=(
                "replayed"
                if stage == "dataset" and stage in verified
                else "static"
                if stage in verified
                else None
            ),
            verification_task_ids=(
                replay.proof.task_ids
                if stage == "dataset" and stage in verified and replay.proof
                else ()
            ),
        )
        for stage in candidates
    )
    if blockers:
        return ConsolidationCompilation(
            evidence_sha256=context.evidence_sha256,
            stages=stages,
            blockers=tuple(blockers),
            assets=assets,
        )
    assert replay.proof is not None
    proposal = ConsolidationProposal(
        batch_label_source=replay.proof.batch_label_source,
        batch_validation_source=replay.proof.batch_validation_source,
        fixed_validation_path=replay.proof.fixed_validation_path,
        evidence_sha256=context.evidence_sha256,
        dataset_runtime=verified["dataset"],
        train_runtime=verified["train"],
        evaluator_runtime=verified["eval"],
    )
    try:
        compile_consolidation(project, context, proposal)
    except ValueError as exc:
        return ConsolidationCompilation(
            evidence_sha256=context.evidence_sha256,
            stages=stages,
            blockers=(),
            issue=_configuration_error(exc, request_context),
            assets=assets,
        )
    return ConsolidationCompilation(
        evidence_sha256=context.evidence_sha256,
        stages=stages,
        blockers=(),
        proposal=proposal,
        dataset_replay=replay.proof,
        assets=assets,
    )


def _configuration_error(error: Exception, context: RequestContext) -> str:
    if isinstance(error, ValidationError):
        fields = ", ".join(
            ".".join(map(str, issue["loc"]))
            for issue in error.errors(include_input=False)[:4]
        )
        return _error_text(fields, context, "配方字段无效。")
    return redact_graph_failure_text(str(error), context)[:500]


def _error_text(value: object, context: RequestContext, fallback: str) -> str:
    return (
        redact_graph_failure_text(value, context)[:500]
        if isinstance(value, str) and value
        else fallback
    )
