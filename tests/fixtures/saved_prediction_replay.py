"""CPU transport fixtures for saved-prediction evaluation."""

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

from pyromind_common.sdk.tool import ToolExecutor
from pyromind_common.tools.data_preparation.evaluation_submit import (
    DfSubmitEvaluationAction,
    DfSubmitEvaluationObservation,
)
from pyromind_common.tools.data_preparation.inference import validate_pipeline_topology
from pyromind_common.tools.data_preparation.prediction_evaluation import (
    PredictionEvaluationOptions,
    build_prediction_evaluation_dsl,
    prepare_definition,
)
from pyromind_common.tools.training_analysis import (
    TrainingAnalysisAction,
    TrainingAnalysisObservation,
)
from pyromind_common.tools.workflow.dsl_to_xyflow import convert_dsl_to_xyflow
from pyromind_runtime.application.training_evaluations import TrainingEvaluations
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_product_store import FileProductStore


ROOT = Path(__file__).parents[2]
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
            options = action.request().options
            assert isinstance(options, PredictionEvaluationOptions)
            definition = prepare_definition(
                options,
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
        from pyromind_common.tools.data_preparation.evaluation_products import (
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


class SavedPredictionReplay:
    workspace: str
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
