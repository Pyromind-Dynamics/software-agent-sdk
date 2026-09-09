"""Dataset distribution analysis and gap-driven synthesis primitives."""

from openhands.tools.dataset_ops.adapters import (
    AviPcbAdapter,
    DatasetAdapter,
    DatasetRecord,
    JsonlTextAdapter,
    VisionManifestAdapter,
    create_adapter,
)
from openhands.tools.dataset_ops.analysis import (
    analyze_existing_labels,
    build_distribution_report,
    stable_sample,
    wilson_interval,
)
from openhands.tools.dataset_ops.contracts import (
    MAX_INFERRED_ANALYSIS_SAMPLES,
    AnalysisSpec,
    AnalyzedSample,
    AugmentationPlan,
    DatasetProgress,
    DatasetValidationReport,
    DistributionReport,
    GapPlan,
    ProvenanceRecord,
    Taxonomy,
)
from openhands.tools.dataset_ops.labeling import FailureLedger, parse_label_response
from openhands.tools.dataset_ops.synthesis import (
    AviPcbSynthesisStrategy,
    SynthesisArtifact,
    SynthesisStrategy,
    SynthesisStrategyRegistry,
    default_strategy_registry,
    execute_avi_plan,
)


__all__ = [
    "MAX_INFERRED_ANALYSIS_SAMPLES",
    "AnalysisSpec",
    "AnalyzedSample",
    "AugmentationPlan",
    "AviPcbAdapter",
    "DatasetAdapter",
    "DatasetRecord",
    "DistributionReport",
    "DatasetProgress",
    "DatasetValidationReport",
    "GapPlan",
    "ProvenanceRecord",
    "JsonlTextAdapter",
    "Taxonomy",
    "VisionManifestAdapter",
    "build_distribution_report",
    "analyze_existing_labels",
    "create_adapter",
    "stable_sample",
    "wilson_interval",
    "AviPcbSynthesisStrategy",
    "SynthesisArtifact",
    "SynthesisStrategy",
    "SynthesisStrategyRegistry",
    "default_strategy_registry",
    "execute_avi_plan",
    "FailureLedger",
    "parse_label_response",
]
