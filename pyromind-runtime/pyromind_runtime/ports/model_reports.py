from typing import Protocol

from pyromind_runtime.domain.echomind_model_reports import ModelEvaluationRecord


class ModelReportReader(Protocol):
    def read(self, path: str) -> bytes: ...


class ModelReportStore(Protocol):
    def list_candidates(self, project_id: str) -> list[ModelEvaluationRecord]: ...

    def save_report(self, record: ModelEvaluationRecord) -> None: ...
