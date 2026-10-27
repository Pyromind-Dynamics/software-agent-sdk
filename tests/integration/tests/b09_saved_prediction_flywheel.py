"""Real LLM replay of AVI checkpoint analysis, with isolated CPU platform transport.

The business rule and mixed artifacts reproduce the decision ambiguities from
training review: three raw labels, binary acceptance, an unsaved best eval,
a worse final checkpoint and changing output schemas. Only platform transport
is replaced; scoring code is authored by the real Agent and executed by the
production worker in a subprocess. No production service/GPU is contacted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

from pyromind_runtime.application.product_instructions import DATA_FLYWHEEL_INSTRUCTIONS
from pyromind_runtime.application.training_evaluations import TrainingEvaluations
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_product_store import FileProductStore

from openhands.sdk import AgentContext
from openhands.sdk.conversation.impl.local_conversation import LocalConversation
from openhands.sdk.skills import load_skills_from_dir
from openhands.sdk.tool import Tool, ToolExecutor, register_tool
from openhands.tools.data_preparation.evaluation_submit import (
    DfSubmitEvaluationAction,
    DfSubmitEvaluationObservation,
    DfSubmitEvaluationTool,
)
from openhands.tools.data_preparation.inference import validate_pipeline_topology
from openhands.tools.data_preparation.prediction_evaluation import (
    build_prediction_evaluation_dsl,
    prepare_definition,
)
from openhands.tools.training_analysis import (
    TrainingAnalysisAction,
    TrainingAnalysisObservation,
    TrainingAnalysisTool,
)
from openhands.tools.workflow.dsl_to_xyflow import convert_dsl_to_xyflow
from tests.integration.base import BaseIntegrationTest, TestResult
from tests.integration.behavior_utils import get_conversation_summary
from tests.integration.utils.llm_judge import judge_agent_behavior


ROOT = Path(__file__).parents[3]
INSTRUCTION = """分析刚完成的 PCB AVI 训练任务 1051，选模型并准备 badcase 复核。
业务规则：pass 与 defect 属于可放行，fail 必须拦截；GT=defect、预测=pass 和
GT=pass、预测=defect 都算通过，GT=fail、预测=defect 不通过。以业务通过率选优，
希望达到 95%。目前没有独立测试集。请交付实际指标、模型引用、复核报告及下一轮建议。
本环境的 training_analysis 和 df_submit_evaluation 使用隔离的历史任务回放，CPU
提交同步返回实际评分结果。文件仍通过 storage/ 挂载，禁止访问生产平台或外网。
"""


class LocalStorage:
    def __init__(self, root):
        self.root = root

    def read(self, path):
        return (self.root / path.lstrip("/")).read_bytes()

    def write(self, path, payload):
        target = self.root / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    def list(self, prefix):
        return {p.name for p in (self.root / prefix.lstrip("/")).iterdir()}


class ReplayTraining(ToolExecutor[TrainingAnalysisAction, TrainingAnalysisObservation]):
    def __init__(self, test):
        self.test = test

    def __call__(self, action, conversation=None):
        if action.operation == "artifacts":
            path = (
                ROOT / ".agents/skills/training-analysis/scripts/training_artifacts.py"
            )
            spec = importlib.util.spec_from_file_location("replay_artifacts", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result = module.discover("/training/1051", str(self.test.mount))
            result.update(
                task_id="1051",
                training_node_id="sft",
                datasets={
                    "train": "/datasets/train.jsonl",
                    "validation": "/datasets/val.jsonl",
                },
                generation={"temperature": 0, "max_tokens": 128},
            )
        else:
            result = {
                "status": "unavailable",
                "reason": "historical telemetry unavailable",
            }
        return TrainingAnalysisObservation.from_text(
            text=json.dumps(result), operation=action.operation, result=result
        )


class ReplaySubmit(
    ToolExecutor[DfSubmitEvaluationAction, DfSubmitEvaluationObservation]
):
    def __init__(self, test):
        self.test = test

    def __call__(self, action, conversation=None):
        self.test.submissions.append(action.model_dump(mode="json"))
        if action.operation in {"profile", "review", "report"}:
            return self._products(action)
        if action.operation != "score":
            return DfSubmitEvaluationObservation.from_text(
                text="Recorded unnecessary inference request",
                status="Failed",
                is_error=True,
            )
        try:
            definition = prepare_definition(
                action.request().options,
                SimpleNamespace(
                    read_text=lambda path: (
                        Path(self.test.workspace) / path
                    ).read_text()
                ),
                "replay-runtime",
                input_path=action.input_path,
                cpu=action.request().cpu,
                memory=action.request().memory,
            )
            run_id = str(uuid4())
            output_ref = f"/evaluation/{run_id}"
            output = self.test.mount / output_ref.lstrip("/")
            output.mkdir(parents=True)
            (output / "score_eval.py").write_text(definition["scoring_script"])
            (output / "prediction_evaluation.json").write_text(json.dumps(definition))
            dsl = build_prediction_evaluation_dsl(definition, output_ref, run_id)
            validate_pipeline_topology(convert_dsl_to_xyflow(dsl, name="replay"))
            worker = ROOT / ".agents/skills/inference-evaluation/scripts"
            training = ROOT / ".agents/skills/training-analysis/scripts"
            code = (
                "import sys,json; from pathlib import Path; "
                f"sys.path[:0] = {[str(worker), str(training)]!r}; "
                "from evaluate_predictions import evaluate_batch; "
                f"p=Path({str(output)!r}); "
                'd=json.loads((p/"prediction_evaluation.json").read_text()); '
                f'evaluate_batch(d["source_manifest"],d["scoring_spec"],p/"score_eval.py",p,{run_id!r},'
                f"Path({str(self.test.mount)!r}))"
            )
            subprocess.run(
                [sys.executable, "-c", code],
                check=True,
                capture_output=True,
                timeout=60,
            )
            self.test.store.save_pipeline_run(
                PipelineRun(
                    run_id=UUID(run_id),
                    conversation_id="replay",
                    definition=definition,
                    fingerprint="replay",
                    output_dir=output_ref,
                    submission_status="submitted",
                ),
                expected_revision=None,
            )
            result = TrainingEvaluations(
                self.test.store, LocalStorage(self.test.mount)
            ).finalize(run_id)
            self.test.selections.append(result)
            return DfSubmitEvaluationObservation.from_text(
                text=json.dumps(
                    {
                        "selection": result,
                        "report": output_ref + "/evaluation_report.html",
                    }
                ),
                task_id="replay-cpu",
                run_id=run_id,
                output_dir=output_ref,
                status="Succeeded",
            )
        except Exception as exc:
            return DfSubmitEvaluationObservation.from_text(
                text=f"{type(exc).__name__}: {exc}", status="Failed", is_error=True
            )

    def _products(self, action):
        from openhands.tools.data_preparation.evaluation_products import (
            build_products_dsl,
            prepare_products,
        )

        try:
            definition = prepare_products(
                action.request().options,
                SimpleNamespace(
                    read_text=lambda path: (
                        Path(self.test.workspace) / path
                    ).read_text()
                ),
                "replay-runtime",
                input_path=action.input_path,
                cpu=action.request().cpu,
                memory=action.request().memory,
            )
            run_id = str(uuid4())
            output_ref = f"/evaluation/{run_id}"
            output = self.test.mount / output_ref.lstrip("/")
            output.mkdir(parents=True)
            (output / "products_definition.json").write_text(json.dumps(definition))
            if definition.get("extraction_script"):
                (output / "extract_dataset.py").write_text(
                    definition["extraction_script"]
                )
            validate_pipeline_topology(
                convert_dsl_to_xyflow(
                    build_products_dsl(definition, output_ref, run_id), name="replay"
                ),
                require_inference=False,
            )
            worker = ROOT / ".agents/skills/inference-evaluation/scripts"
            training = ROOT / ".agents/skills/training-analysis/scripts"
            code = (
                "import sys,json; from pathlib import Path; "
                f"sys.path[:0] = {[str(worker), str(training)]!r}; "
                "from evaluation_products import run_products; "
                f"p=Path({str(output)!r}); "
                'd=json.loads((p/"products_definition.json").read_text()); '
                f"run_products(d,p,{run_id!r},Path({str(self.test.mount)!r}))"
            )
            subprocess.run(
                [sys.executable, "-c", code],
                check=True,
                capture_output=True,
                timeout=60,
            )
            self.test.store.save_pipeline_run(
                PipelineRun(
                    run_id=UUID(run_id),
                    conversation_id="replay",
                    definition=definition,
                    fingerprint="replay-products",
                    output_dir=output_ref,
                    submission_status="submitted",
                ),
                expected_revision=None,
            )
            return DfSubmitEvaluationObservation.from_text(
                text=(output / "report.json").read_text(),
                task_id="replay-cpu-products",
                run_id=run_id,
                output_dir=output_ref,
                status="Succeeded",
            )
        except Exception as exc:
            return DfSubmitEvaluationObservation.from_text(
                text=f"{type(exc).__name__}: {exc}", status="Failed", is_error=True
            )


class SavedPredictionFlywheelTest(BaseIntegrationTest):
    INSTRUCTION = INSTRUCTION

    def setup(self):
        workspace = Path(self.workspace)
        self.mount = workspace / "storage"
        data = self.mount / "datasets"
        training = self.mount / "training/1051"
        data.mkdir(parents=True)
        training.mkdir(parents=True)
        labels = ["pass", "defect", "fail"] * 20
        cases = [
            {"sample_key": f"pcb-{i}", "label": label, "prompt": f"Inspect board {i}"}
            for i, label in enumerate(labels)
        ]
        train = [dict(row, sample_key=f"train-{i}") for i, row in enumerate(cases * 4)]
        for name, rows in [("train", train), ("val", cases)]:
            (data / f"{name}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
        for step in (20, 100, 200, 300):
            predictions = []
            for i, row in enumerate(cases):
                answer = "fail" if row["label"] == "fail" else "pass"
                if (
                    step == 300
                    and row["label"] == "fail"
                    or step == 100
                    and i < 5
                    or step == 20
                    and i < 10
                ):
                    answer = "pass" if answer == "fail" else "fail"
                if step == 100:
                    prediction = {
                        "case": row,
                        "model_response": json.dumps({"verdict": answer}),
                    }
                else:
                    prediction = {**row, "completion": answer}
                predictions.append(prediction)
            (training / f"eval_step_{step}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in predictions)
            )
            if step != 200:
                cp = training / f"checkpoint-{step}"
                cp.mkdir()
                (cp / "adapter_config.json").write_text(
                    '{"base_model_name_or_path":"Qwen/base"}'
                )
                if step != 20:
                    (cp / "adapter_model.safetensors").write_bytes(
                        b"isolated-fixture-weights"
                    )
        self.original = {
            p: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.mount.rglob("*")
            if p.is_file()
        }
        private = workspace / "replay-product"
        private.mkdir()
        self.store = FileProductStore(private)
        self.store.create(
            ConversationSnapshot(
                conversation_id="replay", capabilities=HarnessCapabilities()
            ),
            user_id="replay",
        )
        self.submissions, self.selections = [], []
        for name in ("inference-evaluation", "training-analysis"):
            shutil.copytree(
                ROOT / ".agents/skills" / name,
                workspace / ".agents/skills" / name,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        _, _, skills = load_skills_from_dir(workspace / ".agents/skills")
        submit = DfSubmitEvaluationTool.create()[0].model_copy(
            update={"executor": ReplaySubmit(self)}
        )
        analysis = TrainingAnalysisTool.create()[0].model_copy(
            update={"executor": ReplayTraining(self)}
        )
        register_tool("ReplaySubmit", submit)
        register_tool("ReplayTraining", analysis)
        self.agent = self.agent.model_copy(
            update={
                "tools": [
                    *self.agent.tools,
                    Tool(name="ReplaySubmit"),
                    Tool(name="ReplayTraining"),
                ],
                "agent_context": AgentContext(
                    skills=list(skills.values()),
                    system_message_suffix=DATA_FLYWHEEL_INSTRUCTIONS,
                ),
            }
        )
        self.conversation = LocalConversation(
            agent=self.agent,
            workspace=self.workspace,
            callbacks=[self.conversation_callback],
            max_iteration_per_run=45,
        )

    def verify_result(self):
        if any(
            hashlib.sha256(path.read_bytes()).hexdigest() != digest
            for path, digest in self.original.items()
        ):
            return TestResult(
                success=False, reason="Changed original training artifacts"
            )
        if any(item.get("inference") for item in self.submissions):
            return TestResult(
                success=False, reason="Requested unnecessary GPU inference"
            )
        if not self.selections or self.selections[-1]["selected_step"] != 100:
            return TestResult(
                success=False,
                reason="Did not select the best saved complete checkpoint",
            )
        if self.selections[-1]["meets_target"] is not False:
            return TestResult(
                success=False, reason="Confused best available with acceptance target"
            )
        judgment = judge_agent_behavior(
            user_instruction=INSTRUCTION,
            conversation_summary=get_conversation_summary(self.collected_events),
            evaluation_criteria="""Inspect actual tool ordering: read/discover real
samples of BOTH schemas before authoring the scorer. Score ALL four evals with
the same script, preserve raw labels, merge pass/defect only for business scoring.
Explain the incomplete step 20, unsaved step 200, worse final step 300, and 100
being best available yet below 95%. Show train/validation label support before
badcase review; avoid unsupported root-cause claims and extra inference.
A tool response containing actual samples counts as reading them.
Do not reward promises without artifacts.""",
        )
        self.add_judge_usage(
            prompt_tokens=judgment.prompt_tokens,
            completion_tokens=judgment.completion_tokens,
            cost=judgment.cost,
        )
        return TestResult(success=judgment.approved, reason=judgment.reasoning)
