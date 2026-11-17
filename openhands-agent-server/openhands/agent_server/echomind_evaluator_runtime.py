from __future__ import annotations

from pyromind_runtime.domain.context import RequestContext

from openhands.agent_server.echomind_train_runtime import (
    TrainRuntimeError,
    TrainRuntimeExecutor,
    TrainTaskState,
    TrainWorkflowBackend,
    TrainWorkflowSpec,
)


EvaluatorRuntimeError = TrainRuntimeError
EvaluatorTaskState = TrainTaskState
EvaluatorWorkflowBackend = TrainWorkflowBackend
EvaluatorWorkflowSpec = TrainWorkflowSpec


class EvaluatorRuntimeExecutor(TrainRuntimeExecutor):
    def __init__(
        self,
        *,
        projects_root: str,
        backend: EvaluatorWorkflowBackend,
        request_context: RequestContext | None = None,
    ) -> None:
        super().__init__(
            projects_root=projects_root,
            backend=backend,
            request_context=request_context,
            provider="pyromind.studio.evaluation",
            output_port="eval_report",
            config_key="evaluator_runtime",
            task_name_prefix="echomind-eval",
            pending_task_name="echomind-eval-pending",
            runtime_name="Evaluator",
            output_input_name="eval_report_uri",
        )
