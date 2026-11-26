"""Display configuration for the project observations sidebar."""

from pyromind_runtime.domain.observations import (
    ObservationContentBlock,
    ObservationDatasetBlock,
    ObservationDetailForm,
    ObservationDetailSection,
    ObservationMetricGroup,
    ObservationMetricsBlock,
    ObservationSelectBlock,
    ObservationStatusBlock,
)


BASELINE_METRIC_GROUPS = (
    ObservationMetricGroup(
        title="整体表现",
        labels={
            "label_accuracy": "标签准确率",
            "macro_f1": "Macro-F1",
            "evaluated_samples": "评估样本",
            "api_or_evaluation_errors": "调用或评估错误",
            "mean_overall_score": "综合评分",
            "mean_latency_seconds": "平均时延",
        },
    ),
    ObservationMetricGroup(
        title="定位表现",
        labels={
            "box_f1": "定位 F1",
            "box_precision": "定位精确率",
            "box_recall": "定位召回率",
        },
    ),
    ObservationMetricGroup(
        title="分类表现",
        labels={
            "pass_precision": "Pass 精确率",
            "pass_recall": "Pass 召回率",
            "defect_precision": "Defect 精确率",
            "defect_recall": "Defect 召回率",
            "fail_precision": "Fail 精确率",
            "fail_recall": "Fail 召回率",
        },
    ),
)


def observation_detail_form(
    node_id: str, kind: str, title: str
) -> ObservationDetailForm:
    report_versions = ObservationSelectBlock(label="Report version", min_options=2)
    if kind == "init":
        section = ObservationDetailSection(
            id="report",
            label="Orchestrator report",
            selection="report_version",
            blocks=(
                ObservationSelectBlock(label="Report version"),
                ObservationStatusBlock(
                    source="report",
                    subject="基线评估",
                    loading_message="正在读取基线评估报告",
                    unavailable_message="暂时无法读取所选版本的基线评估报告",
                ),
                ObservationMetricsBlock(
                    source="report",
                    groups=BASELINE_METRIC_GROUPS,
                    empty_message="所选报告未提供结构化指标",
                ),
                ObservationContentBlock(type="findings"),
                ObservationContentBlock(type="artifacts"),
            ),
        )
        return ObservationDetailForm(project_node_id=node_id, sections=(section,))
    if kind == "dataset":
        section = ObservationDetailSection(
            id="report",
            label="Dataset report",
            selection="analysis_version",
            blocks=(
                report_versions,
                ObservationStatusBlock(
                    source="report",
                    subject="数据准备",
                    loading_message="正在读取已登记的数据集详情",
                    unavailable_message="暂时无法读取数据集详情",
                ),
                ObservationDatasetBlock(
                    empty_message="数据集报告已登记，未提供文件路径"
                ),
            ),
        )
        return ObservationDetailForm(project_node_id=node_id, sections=(section,))
    if kind == "eval":
        section = ObservationDetailSection(
            id="analysis",
            label="Evaluation report",
            selection="analysis_version",
            blocks=(
                ObservationSelectBlock(label="Report version"),
                ObservationStatusBlock(source="analysis", subject="评估"),
                ObservationMetricsBlock(
                    source="analysis",
                    empty_message="所选版本未提供结构化指标",
                ),
            ),
        )
        return ObservationDetailForm(project_node_id=node_id, sections=(section,))

    label, subject = {
        "train": ("Train", "训练"),
        "tracer": ("Tracer", "Tracer"),
        "inference": ("Inference", "推理"),
    }.get(kind, (title[:140], title[:160]))
    analysis = ObservationDetailSection(
        id="analysis",
        label=f"{label} report",
        selection="analysis_version",
        blocks=(
            report_versions,
            ObservationStatusBlock(source="analysis", subject=subject),
            ObservationContentBlock(type="summary"),
            ObservationMetricsBlock(source="analysis"),
            ObservationContentBlock(type="findings"),
            ObservationContentBlock(type="artifacts"),
            ObservationContentBlock(type="analysis_version"),
        ),
    )
    sections = (analysis,)
    if kind == "train":
        curves = ObservationDetailSection(
            id="curves",
            label="Train training metrics",
            selection="task_id",
            blocks=(
                ObservationSelectBlock(label="Task ID"),
                ObservationContentBlock(type="training_curves"),
            ),
        )
        sections = (curves, analysis)
    return ObservationDetailForm(project_node_id=node_id, sections=sections)
