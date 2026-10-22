"""Real-agent rubric revision: existing predictions, distributions, fresh reviews."""

import json

from tests.integration.base import TestResult
from tests.integration.tests.b09_saved_prediction_flywheel import (
    INSTRUCTION,
    SavedPredictionFlywheelTest,
)


class IndependentBadcaseReviewTest(SavedPredictionFlywheelTest):
    INSTRUCTION = (
        INSTRUCTION
        + """
先完成原规则评分与独立复核任务，再模拟本轮业务口径修订：第二版只接受原始 GT 与预测
严格相同（pass/defect 不再合并）。复用原始预测，用新评分脚本重算全部 step，创建第二版
独立复核任务，绑定完整 train/validation 数据分布。不要修改第一版规则文件或任务产物，
不要复制旧人工标注，不要生成新的标注 UI，不要运行模型。交付两个任务及 badcase 变化。
"""
    )

    def verify_result(self):
        if any(item.get("inference") for item in self.submissions):
            return TestResult(success=False, reason="Unnecessary model inference")
        runs = list(self.store.load_pipeline_runs().values())
        scores = [
            run for run in runs if run.definition.get("kind") == "prediction_evaluation"
        ]
        reviews = [
            run for run in runs if run.definition.get("kind") == "badcase_review"
        ]
        if len(scores) < 2 or len(reviews) < 2:
            return TestResult(
                success=False, reason="Missing independent scoring/review runs"
            )
        hashes, memberships = set(), []
        for run in reviews:
            output = self.mount / run.output_dir.lstrip("/")
            task = json.loads((output / "review_task.json").read_text())
            profile = json.loads((output / "dataset_profile.json").read_text())
            if profile["status"] != "ready":
                return TestResult(
                    success=False, reason="Review opened without distribution"
                )
            hashes.add(task["scoring_spec_sha256"])
            records = [
                json.loads(line)
                for line in (output / "evaluation_annotations.r0.jsonl")
                .read_text()
                .splitlines()
            ]
            if any(record["human_annotation"] is not None for record in records):
                return TestResult(success=False, reason="Inherited human annotations")
            memberships.append({record["case_id"] for record in records})
        if len(hashes) < 2 or all(cases == memberships[0] for cases in memberships):
            return TestResult(
                success=False, reason="Rubric revision did not change badcases"
            )
        return TestResult(
            success=True, reason="Independent fresh reviews from CPU rescoring"
        )
