"""Replay the 2f27df99 benchmark routing failure across evaluation and training."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from openhands.sdk import AgentContext, Message, TextContent
from openhands.sdk.conversation.impl.local_conversation import LocalConversation
from openhands.sdk.skills import load_skills_from_dir
from tests.integration.base import BaseIntegrationTest, TestResult
from tests.integration.behavior_utils import get_conversation_summary
from tests.integration.utils.llm_judge import judge_agent_behavior


INSTRUCTION = """之前已经准备好 PCB AVI 缺陷复检的训练数据，已有训练工作流。
请用中心仓的 Qwen/Qwen3-VL-2B-Instruct 跑基模 Benchmark，验证集是
storage/datasets/pcb_avi/val.jsonl。先看样本并写好评测配置，不提交任何平台任务。
"""
FOLLOWUPS = (
    "现在按同一验证集的 ground_truth 对比模型回答，重点看缺陷类别是否正确，"
    "准备评测报告所需的配置。仍然只准备，不提交。",
    "训练已经完成，合并后的模型在 /models/pcb_avi/merged。"
    "请在同一验证集、同一指标上准备训后评测，以便与基模比较。不要提交。",
    "接下来是另一个训练配置实验：用中心仓 Qwen/Qwen3-VL-2B-Instruct 和"
    "storage/datasets/pcb_avi/train.jsonl 生成单卡 LoRA SFT 训练工作流，"
    "3 个 epoch，学习率 0.0001，不附带效果评测，不提交。"
    "环境没有平台校验工具时，保留 DSL 并说明尚未校验，不尝试访问外网。",
)


class InferenceEvaluationRoutingTest(BaseIntegrationTest):
    INSTRUCTION = INSTRUCTION

    def setup(self) -> None:
        repository = Path(__file__).parents[3]
        workspace = Path(self.workspace)
        skills = workspace / ".agents/skills"
        for name in ("inference-evaluation", "generate-workflow-dsl"):
            shutil.copytree(
                repository / ".agents/skills" / name,
                skills / name,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        data = workspace / "storage/datasets/pcb_avi"
        data.mkdir(parents=True)
        cases = [
            {
                "id": "open",
                "prompt": "铜线中段有贯穿间隙，判断 PCB 缺陷。",
                "ground_truth": '{"label":"open"}',
            },
            {
                "id": "short",
                "prompt": "两根独立铜线之间有铜桥连接，判断 PCB 缺陷。",
                "ground_truth": '{"label":"short"}',
            },
            {
                "id": "false",
                "prompt": "铜线连续完整，仅表面反光，判断是否真缺陷。",
                "ground_truth": '{"label":"false_positive"}',
            },
        ]
        (data / "val.jsonl").write_text(
            "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n"
        )
        (data / "train.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": case["prompt"]},
                            {"role": "assistant", "content": case["ground_truth"]},
                        ]
                    },
                    ensure_ascii=False,
                )
                for case in cases
            )
            + "\n"
        )
        model = workspace / "storage/models/pcb_avi/merged"
        model.mkdir(parents=True)
        (model / "config.json").write_text('{"model_type":"qwen3_vl"}')
        canvas = workspace / "public_data/workflow_canvas/workflow.py"
        canvas.parent.mkdir(parents=True)
        canvas.write_text(
            "model = CloneAndCacheModel(id=1, "
            "model='Qwen/Qwen3-VL-2B-Instruct', target_path='/workspace/models/')\n"
        )
        _, _, available = load_skills_from_dir(skills)
        self.agent = self.agent.model_copy(
            update={"agent_context": AgentContext(skills=list(available.values()))}
        )
        self.conversation = LocalConversation(
            agent=self.agent,
            workspace=self.workspace,
            callbacks=[self.conversation_callback],
            max_iteration_per_run=30,
        )
        self.phase_evidence: list[str] = []

    def run_instructions(self, conversation: LocalConversation) -> None:
        for instruction in (self.instruction, *FOLLOWUPS):
            start = len(self.collected_events)
            conversation.send_message(
                Message(role="user", content=[TextContent(text=instruction)])
            )
            conversation.run()
            artifacts = {
                str(path.relative_to(self.workspace)): path.read_text()
                for path in (Path(self.workspace) / "public_data").rglob("*")
                if path.is_file() and path.suffix in {".json", ".py"}
            }
            self.phase_evidence.append(
                instruction
                + "\n"
                + get_conversation_summary(self.collected_events[start:])
                + "\nArtifacts:\n"
                + json.dumps(artifacts, ensure_ascii=False)
            )

    def verify_result(self) -> TestResult:
        judgment = judge_agent_behavior(
            user_instruction=INSTRUCTION + "\n" + "\n".join(FOLLOWUPS),
            conversation_summary="\n\n".join(self.phase_evidence),
            evaluation_criteria="""The first three phases are independent model quality
evaluations. The agent must consult inference-evaluation and prepare its dataset
mapping and deterministic rubric configuration instead of creating a native
Metric/Eval workflow. The baseline uses clone in the evaluation config JSON,
not a new tool argument; the trained model uses its Storage reference. GT and
scoring semantics must remain comparable. Reusing an already read skill or
unchanged configuration is allowed. In the final phase the agent must use
generate-workflow-dsl to generate SFT training and may include required model
preparation/merge, without adding a standalone evaluation stage. No phase may
submit a platform task, run a local GPU evaluation, or claim a benchmark result.
Missing platform validation should be reported honestly, not worked around.
Assess actual file/tool evidence, not merely a promise to use the right skill.
""",
        )
        self.add_judge_usage(
            prompt_tokens=judgment.prompt_tokens,
            completion_tokens=judgment.completion_tokens,
            cost=judgment.cost,
        )
        return TestResult(success=judgment.approved, reason=judgment.reasoning)
