"""Issue one scoped repair for the first incomplete development stage."""

import json
from datetime import timedelta
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel

from pyromind_runtime.application.consolidation_evidence import Stage
from pyromind_runtime.domain.base import utc_now
from pyromind_runtime.domain.commands import CommandReceipt, UserMessageCommand
from pyromind_runtime.domain.content import JsonObject, TextContent
from pyromind_runtime.domain.snapshot import ConversationSnapshot


class ConsolidationRepair(BaseModel):
    stage: Stage
    command_id: str
    status: Literal["working", "needs_input", "blocked"]
    message: str


def repair_recipe_path(evidence: str, stage: Stage) -> str:
    return (
        f"public_data/execution-recipes/{evidence.removeprefix('sha256:')}-{stage}.json"
    )


def conversation_has_work(snapshot: ConversationSnapshot) -> bool:
    return snapshot.status in {"running", "waiting_for_external_task"} or any(
        item.kind == "operation"
        and item.category == "subagent"
        and item.status == "running"
        for item in snapshot.timeline
    )


def plan_repair(
    *,
    stage: Stage,
    evidence: str,
    facts: JsonObject,
    issues: list[str],
    revision: str,
    receipts: tuple[CommandReceipt, ...],
    snapshot: ConversationSnapshot,
    retry_command_id: str | None,
) -> tuple[ConsolidationRepair, UserMessageCommand | None]:
    prefix = f"consolidation-{evidence.removeprefix('sha256:')}-{stage}-"
    previous = [
        receipt for receipt in receipts if receipt.command_id.startswith(prefix)
    ]
    previous.sort(key=lambda item: item.recorded_at)
    latest = previous[-1] if previous else None
    latest_id = latest.command_id if latest else ""
    if (
        snapshot.pending_inputs
        or snapshot.pending_permissions
        or snapshot.status
        in {
            "waiting_for_confirmation",
            "waiting_for_input",
        }
    ):
        return ConsolidationRepair(
            stage=stage,
            command_id=latest_id,
            status="needs_input",
            message=(
                "请在当前会话查看待授权的操作，处理后将继续校验。"
                if snapshot.status == "waiting_for_confirmation"
                or snapshot.pending_permissions
                else "请先在当前对话回答待确认的问题，工作台会继续校验。"
            ),
        ), None
    if conversation_has_work(snapshot) or (
        latest is not None
        and latest.status != "failed"
        and utc_now() - latest.recorded_at < timedelta(seconds=15)
    ):
        return ConsolidationRepair(
            stage=stage,
            command_id=latest_id,
            status="working",
            message=(
                "当前会话正在等待平台任务回调，随后重新校验。"
                if snapshot.status == "waiting_for_external_task"
                else "当前会话正在处理，结束后将重新校验。"
            ),
        ), None

    payload = json.dumps(facts, ensure_ascii=False, sort_keys=True)
    digest = sha256(
        json.dumps([issues, payload, revision], ensure_ascii=False).encode()
    ).hexdigest()[:20]
    same = [
        item for item in previous if item.command_id.startswith(f"{prefix}{digest}-")
    ]
    manual_retry = latest is not None and retry_command_id == latest_id
    if latest and not manual_retry and (same or len(previous) >= 2):
        return ConsolidationRepair(
            stage=stage,
            command_id=latest_id,
            status="blocked",
            message=f"{stage} 仍缺可验证的交付。核对下方原因后可重新交给 Agent。",
        ), None
    command_id = f"{prefix}{digest}-{len(same) + 1}"
    path = repair_recipe_path(evidence, stage)
    field = "evaluator_runtime" if stage == "eval" else "train_runtime"
    stage_instruction = (
        "按 data-processing 的 SDK 路径使用 prepare_pipeline、"
        "validate_workflow_dsl 和 submit_workflow，"
        "复用原研发任务的 pipeline、"
        "model_profile 和 output_schema。"
        "用 prepare_script_path 接入实际执行的原始输入转换，用 sft_export_script_path "
        "接入训练数据导出。固定训练中验证集使用 plan 确认的 fixed_validation_path，"
        "若原 Train 未绑定验证集则全批训练，不切验证样本，正式 Eval 集单独绑定。"
        "按已确认策略设置 batch_label_source，"
        "取值 dataset 或 collection。"
        "回放时 replay_of_task_id 填 development_task.task_id，"
        "从 PYROMIND_TASK_CONTEXTS 读取 data_preparation 的签名 out_id，独立登记回放。"
        "每次回放在 /workspace/.pyromind-agent/<当前会话ID>/data_preparation/ "
        "下创建独立运行目录。"
        "已有平台任务先核对其终态与冻结的 dataset-recipe.json，旧版 SDK 回执缺失时，"
        "用 product_versions 的 bind_execution_metadata 动作，"
        "将实际校验图和冻结配方关联至原小样任务号。"
        "将同一确认来源内的 1–5 条样本放在独立的 input_path，保留来源身份和媒体。"
        "有已封存的 Tracer 样本时，在换批验收中覆盖其实际输入布局。"
        "先核对已有回放证据；仅补所缺的小样，使同一配方在两个不同输入上完成 CPU 回放。"
        "每份平台执行图都应实际运行输入准备、处理、SFT 导出和 verifier。"
        "旧任务只做过预标注时复用其脚本和样本，另用完整配方跑缺少的小样。"
        "对照研发任务和已确认的样本身份与标签策略，缺少业务决定时使用 ask_user。"
        "编译器直接读取工具生成的回执，手写验证结果不能作为证据。"
        if stage == "dataset"
        else (
            "复用该研发任务的工作流、执行器端口和已归档的版本化资产。"
            f"保存到 {path}，JSON 只含 evidence_sha256 和 {field}。"
        )
    )
    prompt = (
        f"请在主会话直接补齐当前 Project 的 {stage} 固化缺口，"
        "引用原研发任务和已有回执，在该阶段记录验证结果及产物。\n"
        + "\n".join(f"- {issue}" for issue in issues)
        + f"\n{stage_instruction}\n"
        "可以验证有界的小样，不得启动全量任务、训练、正式评测、部署或修改 Project。"
        "使用本轮的输入绑定和独立输出目录，不得保存密钥；已有可验证的节点无需重写。"
        "\n以下为该 Project 已核对的输入及资产引用。\n" + payload
    )
    if any(issue.startswith("dataset_policy_conflict") for issue in issues):
        prompt = (
            "当前 Dataset 的回放与 Project 已确认的业务策略冲突。请通过 ask_user "
            "列明具体差异，让用户确认保留哪一种标签/切分策略。等待决定后再调整；"
            "在此之前不要重复提交回放或覆盖 Project。\n"
            + "\n".join(issues)
            + "\n"
            + payload
        )
    return ConsolidationRepair(
        stage=stage,
        command_id=command_id,
        status="working",
        message=f"主 Agent 正在补齐 {stage}，完成后自动校验。",
    ), UserMessageCommand(command_id=command_id, content=(TextContent(text=prompt),))
