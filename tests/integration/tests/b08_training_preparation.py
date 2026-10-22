"""Replay c55d2897: generate training DSL without unnecessary data preparation.

Uses an isolated, anonymized PCB dataset, an old incompatible derivative and
existing training parameters. The model sees artifacts, not the expected fix.
Run with tests/integration/run_infer.py --eval-ids b08_training_preparation.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from openhands.sdk import AgentContext, Message, TextContent
from openhands.sdk.conversation.impl.local_conversation import LocalConversation
from openhands.sdk.skills import load_skills_from_dir
from tests.integration.base import BaseIntegrationTest, TestResult
from tests.integration.behavior_utils import get_conversation_summary
from tests.integration.utils.llm_judge import judge_agent_behavior


INSTRUCTION = """为 PCB AVI 缺陷复检生成单卡 LoRA SFT 工作流。
模型使用 Qwen/Qwen3-VL-2B-Instruct，数据是已上传的
storage/datasets/pcb_avi/train.jsonl 和 val.jsonl，
沿用 public_data/training_config.json。
只生成，不运行，不附带评测。本测试环境没有平台校验和提交工具，也不能访问外网；
保留生成文件并如实说明未完成的平台校验。
"""
FOLLOWUPS = (
    "把刚才生成的工作流学习率改成 0.00005，其他保持不变。",
    "另外生成文本 SFT 工作流：使用 storage/datasets/text/train.jsonl，"
    "模型 Qwen/Qwen3-0.6B，其余训练配置沿用。只生成，不运行。",
    "再为 storage/datasets/incomplete/train.jsonl 生成图像质检 SFT 工作流，"
    "使用 Qwen/Qwen3-VL-2B-Instruct。只生成，不运行。",
    "回到最初的 PCB 数据，这次我要精确分析完整多模态序列长度和 8192 下的截断比例，"
    "作为后续容量决策的依据。先分析，暂不改工作流；实测不了请说明缺什么。",
)


def prepare_workspace(workspace: Path) -> None:
    """Reconstruct shared Storage with original and historical derived data."""
    repository = Path(__file__).parents[3]
    shutil.copytree(
        repository / ".agents/skills/generate-workflow-dsl",
        workspace / ".agents/skills/generate-workflow-dsl",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    data = workspace / "storage/datasets/pcb_avi"
    (data / "images").mkdir(parents=True)
    records = []
    for name, answer in (("open", "断路"), ("short", "短路"), ("normal", "无缺陷")):
        image = Image.new("RGB", (64, 64), "darkgreen")
        draw = ImageDraw.Draw(image)
        draw.line((0, 20, 63, 20), fill="orange", width=5)
        draw.line((0, 44, 63, 44), fill="orange", width=5)
        if name == "open":
            draw.rectangle((29, 17, 34, 23), fill="darkgreen")
        elif name == "short":
            draw.line((32, 20, 32, 44), fill="orange", width=5)
        image.save(data / "images" / f"{name}.png")
        records.append(
            {
                "images": [f"/workspace/datasets/pcb_avi/images/{name}.png"],
                "system_prompt": "你是 PCB 图像质检专家。",
                "user_prompt": "判断图中是否存在缺陷，并输出缺陷类别。",
                "gt": answer,
            }
        )
    for split, rows in (("train", records * 8), ("val", records)):
        (data / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        (data / f"{split}_messages.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "path": row["images"][0].replace(
                                            "/workspace/", "/target-workspace/storage/"
                                        ),
                                    },
                                    {"type": "text", "text": row["user_prompt"]},
                                ],
                            },
                            {"role": "assistant", "content": row["gt"]},
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for row in rows
            )
        )
    for name, record in (
        (
            "text",
            {
                "messages": [
                    {"role": "user", "content": "1+1?"},
                    {"role": "assistant", "content": "2"},
                ]
            },
        ),
        (
            "incomplete",
            {
                "images": ["/workspace/datasets/incomplete/missing.png"],
                "user_prompt": "判断缺陷",
            },
        ),
    ):
        directory = workspace / "storage/datasets" / name
        directory.mkdir()
        (directory / "train.jsonl").write_text(json.dumps(record) + "\n")
    public = workspace / "public_data"
    public.mkdir()
    (public / "training_config.json").write_text(
        json.dumps(
            {
                "dataset_extra": {"max_seq_length": 8192},
                "training": {
                    "num_epochs": 3,
                    "batch_size": 1,
                    "grad_accum": 8,
                    "learning_rate": 0.0001,
                    "lr_scheduler_type": "cosine",
                },
                "lora": {"lora_rank": 16, "lora_dropout": 0.05},
                "accelerate": {"zero_stage": 0},
                "wandb": {"wandb_api_key": "WANDB_API_KEY", "wandb_project": "pcb-avi"},
                "gpu_count": 1,
                "gpu_product": "NVIDIA-H200",
            }
        )
    )


class TrainingPreparationTest(BaseIntegrationTest):
    INSTRUCTION = INSTRUCTION

    def setup(self) -> None:
        workspace = Path(self.workspace)
        prepare_workspace(workspace)
        self.original_data = {
            path: path.read_bytes() for path in (workspace / "storage").rglob("*.jsonl")
        }
        _, _, skills = load_skills_from_dir(workspace / ".agents/skills")
        self.agent = self.agent.model_copy(
            update={"agent_context": AgentContext(skills=list(skills.values()))}
        )
        self.conversation = LocalConversation(
            agent=self.agent,
            workspace=self.workspace,
            callbacks=[self.conversation_callback],
            max_iteration_per_run=25,
        )
        self.phase_evidence: list[str] = []
        self.workflows: list[str] = []
        self.created_datasets: set[Path] = set()

    def run_instructions(self, conversation: LocalConversation) -> None:
        workspace = Path(self.workspace)
        canvas = workspace / "public_data/workflow_canvas/workflow.py"
        for instruction in (self.instruction, *FOLLOWUPS):
            start = len(self.collected_events)
            conversation.send_message(
                Message(role="user", content=[TextContent(text=instruction)])
            )
            conversation.run()
            self.workflows.append(canvas.read_text() if canvas.exists() else "")
            if len(self.workflows) <= 3:
                self.created_datasets.update(
                    set((workspace / "storage").rglob("*.jsonl"))
                    - set(self.original_data)
                )
            artifacts = {
                str(path.relative_to(workspace)): path.read_text()
                for path in workspace.rglob("*")
                if path.is_file()
                and path.suffix in {".py", ".json", ".jsonl"}
                and ".agents" not in path.parts
            }
            self.phase_evidence.append(
                instruction
                + "\n"
                + get_conversation_summary(self.collected_events[start:])
                + "\nArtifacts:\n"
                + json.dumps(artifacts, ensure_ascii=False)
            )

    def verify_result(self) -> TestResult:
        for path, content in self.original_data.items():
            if not path.exists() or path.read_bytes() != content:
                return TestResult(
                    success=False, reason=f"Changed input dataset: {path}"
                )
        if self.created_datasets:
            return TestResult(success=False, reason="Created unnecessary JSONL dataset")
        if not all(self.workflows[:3]):
            return TestResult(success=False, reason="Missing requested training DSL")
        before, after = (ast.parse(source) for source in self.workflows[:2])
        rates = [
            keyword
            for keyword in ast.walk(before)
            if isinstance(keyword, ast.keyword) and keyword.arg == "learning_rate"
        ]
        if len(rates) != 1:
            return TestResult(
                success=False, reason="Expected one training learning rate"
            )
        rates[0].value = ast.Constant(value=0.00005)
        if ast.dump(before) != ast.dump(after):
            return TestResult(
                success=False, reason="Local edit changed other DSL values"
            )
        judgment = judge_agent_behavior(
            user_instruction=INSTRUCTION + "\n" + "\n".join(FOLLOWUPS),
            conversation_summary="\n\n".join(self.phase_evidence),
            evaluation_criteria="""Assess actual tool traces and generated files.
Phase 1 must produce SFT DSL bound to original PCB train/val JSONL using Vision
field mapping, preserving provided valid parameters including length 8192.
The historical messages derivative contains sandbox paths and is unsuitable.
Do not install packages, download model weights or rebuild the dataset as a
prerequisite to generation. Output ports must bind the graph, with mandatory
training inputs present. Phase 2 changes only learning_rate. Phase 3 consumes
the existing text messages dataset directly using its appropriate Builder.
Phase 4 must identify missing supervision; media is also absent. Do not invent
labels, claim readiness or create an executable workflow from those records.
Phase 5 explicitly requests precise measurement: inspect what is needed or
explain missing model processor/collator assets. Do not refuse analysis merely
because DSL generation does not require it, or report estimates as measurements.
No phase may submit, run training, use credentials or claim platform validation.
""",
        )
        self.add_judge_usage(
            prompt_tokens=judgment.prompt_tokens,
            completion_tokens=judgment.completion_tokens,
            cost=judgment.cost,
        )
        return TestResult(success=judgment.approved, reason=judgment.reasoning)
