"""Controlled Pi bridge for the existing training-analysis runtime."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from pydantic import Field, model_validator

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.workflow.validate_workflow_dsl import (
    PYROMIND_VALIDATE_HEADERS_STATE_KEY,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|authorization|cookie|password|secret|token|"
    r"credential|private[_-]?key|access[_-]?key|refresh[_-]?token)",
    re.IGNORECASE,
)
_REDACTED_KEY = re.compile(
    r"(?:api[_-]?key|authorization|cookie|password|secret|token|"
    r"credential|private[_-]?key|access[_-]?key|refresh[_-]?token|cluster)",
    re.IGNORECASE,
)
_INLINE_CREDENTIAL = re.compile(
    r"((?:api[_-]?key|authorization|cookie|password|secret|token|cluster)"
    r"\s*[:=]\s*)"
    r"([^\s,;\"']+)",
    re.IGNORECASE,
)
_PRE_API_BASE = "https://pre-api-portal.pyromind.ai/std2/studio_api/"
_PROD_API_BASE = "https://api-portal.pyromind.ai/std2/studio_api/"
_PROD_APP_ENVS = {"prod", "production", "online"}


def _default_api_base(configured: str | None = None) -> str:
    if configured and configured.strip():
        return configured.strip()
    from_environment = os.environ.get("PYROMIND_API_BASE", "").strip()
    if from_environment:
        return from_environment
    app_env = os.environ.get("APP_ENV", "").strip().lower()
    return _PROD_API_BASE if app_env in _PROD_APP_ENVS else _PRE_API_BASE


def _safe_headers(headers: dict[str, Any]) -> dict[str, str]:
    """Keep non-secret forwarding headers in the persisted tool params."""
    return {
        str(name): str(value)
        for name, value in headers.items()
        if value is not None and not _SENSITIVE_KEY.search(str(name))
    }


def _payload_secrets(payload: Any, key: str = "") -> tuple[str, ...]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for name, child in payload.items():
            name_text = str(name)
            if _REDACTED_KEY.search(name_text) and isinstance(child, str) and child:
                found.add(child)
            else:
                found.update(_payload_secrets(child, name_text))
    elif isinstance(payload, list):
        for child in payload:
            found.update(_payload_secrets(child, key))
    return tuple(sorted(found, key=len, reverse=True))


def _redact_text(value: str, payload: Any) -> str:
    for secret in _payload_secrets(payload):
        value = value.replace(secret, "[REDACTED]")
    return _INLINE_CREDENTIAL.sub(r"\1[REDACTED]", value)


def _redact_value(value: Any, payload: Any, key: str = "", depth: int = 0) -> Any:
    if _REDACTED_KEY.search(key):
        return "[REDACTED]"
    if depth >= 10:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        return {
            str(name): _redact_value(child, payload, str(name), depth + 1)
            for name, child in list(value.items())[:200]
        }
    if isinstance(value, list):
        return [
            _redact_value(child, payload, key, depth + 1) for child in value[:1000]
        ]
    if isinstance(value, str):
        return _redact_text(value, payload)[:100_000]
    return value


class TrainingAnalysisAction(Action):
    operation: Literal["probe", "analyze", "report"] = Field(
        description="Analysis operation to run."
    )
    task_id: str = Field(
        min_length=1,
        description="Pyromind training task id used to resolve the training run.",
    )
    run_url: str | None = Field(
        default=None,
        description="Optional W&B run URL hint used while resolving task_id.",
    )
    data_source: str | None = Field(
        default=None,
        description="Optional data-source adapter name; omit to detect it from task.",
    )
    metric: str | None = Field(
        default=None,
        description="Optional primary metric key for analyze or report.",
    )
    keys: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Optional metric keys to inspect; at most 20.",
    )
    output_path: str | None = Field(
        default=None,
        description=(
            "Report-only workspace-relative path under "
            "public_data/training-analysis/."
        ),
    )

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if not self.task_id.strip():
            raise ValueError("task_id is required")
        if self.operation != "report" and self.output_path is not None:
            raise ValueError("output_path is only valid for operation=report")
        if self.output_path is not None and not self.output_path.strip():
            raise ValueError("output_path must not be empty")
        return self


class TrainingAnalysisObservation(Observation):
    operation: Literal["probe", "analyze", "report"]
    target: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    report_path: str | None = None
    failure_stage: str | None = None
    error_code: str | None = None
    error_message: str | None = None


TOOL_DESCRIPTION = """Analyze a Pyromind training run through a controlled worker.

Use this tool for W&B metric discovery, loss/NaN/divergence/overfitting analysis,
four-stage reports, and run comparisons. Call operation=probe to discover keys,
operation=analyze for structured diagnostics, and operation=report to write a
Markdown report. Compare runs by analyzing each task separately. Credentials are
resolved from the platform and never returned. Do not run the skill's Python
scripts or access W&B through terminal.
"""


class TrainingAnalysisExecutor(
    ToolExecutor[TrainingAnalysisAction, TrainingAnalysisObservation]
):
    def __init__(
        self,
        *,
        runtime_dir: str,
        api_base: str | None = None,
        headers: dict[str, str] | None = None,
        secret_headers: dict[str, str] | None = None,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.runtime_dir = Path(runtime_dir).resolve()
        self.api_base = _default_api_base(api_base)
        self.headers = dict(headers or {})
        self.secret_headers = dict(secret_headers or {})
        timeout = float(timeout_seconds)
        if timeout <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        self.timeout_seconds = min(timeout, 600.0)
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    def __call__(
        self,
        action: TrainingAnalysisAction,
        conversation: BaseConversation | None = None,
    ) -> TrainingAnalysisObservation:
        if conversation is None:
            return self._error(
                action.operation,
                "context",
                "missing_context",
                "training_analysis requires a conversation",
            )
        payload: dict[str, Any]
        try:
            payload = self._payload(action, conversation)
        except ValueError as exc:
            message = str(exc)
            if "required secret header" in message:
                return self._error(
                    action.operation,
                    "credentials",
                    "training_credentials_unavailable",
                    message,
                )
            return self._error(
                action.operation,
                "input_resolution",
                "invalid_training_target",
                message,
            )
        worker = self.runtime_dir / "training_analysis_worker.py"
        if not worker.is_file():
            return self._error(
                action.operation,
                "runtime_dependency",
                "training_worker_missing",
                f"Training worker does not exist: {worker}",
            )
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                [sys.executable, str(worker)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(cast(Any, conversation).workspace.working_dir),
                env=_worker_environment(),
                start_new_session=True,
            )
            with self._lock:
                self._process = process
            stdout, stderr = process.communicate(
                json.dumps(payload, ensure_ascii=False), timeout=self.timeout_seconds
            )
        except OSError as exc:
            return self._error(
                action.operation,
                "runtime_dependency",
                "training_worker_start_failed",
                f"Could not start training worker: {type(exc).__name__}: {exc}",
            )
        except subprocess.TimeoutExpired:
            self.interrupt()
            # Drain pipes after terminating the process group so a worker child
            # cannot keep the parent's descriptors or become a zombie.
            if process is not None:
                try:
                    process.communicate(timeout=1)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    self.interrupt()
            return self._error(
                action.operation,
                "timeout",
                "training_analysis_timeout",
                f"Training analysis exceeded {self.timeout_seconds:g} seconds",
            )
        finally:
            with self._lock:
                self._process = None
        if process is None:
            return self._error(
                action.operation,
                "runtime_dependency",
                "training_worker_start_failed",
                "Training worker process was not created",
            )
        try:
            value = json.loads(stdout)
        except (json.JSONDecodeError, TypeError):
            return self._error(
                action.operation,
                "worker_execution",
                "invalid_worker_output",
                _redact_text(
                    str(stderr or "Training worker returned invalid JSON"), payload
                )[:2000],
            )
        if not isinstance(value, dict):
            return self._error(
                action.operation,
                "worker_execution",
                "invalid_worker_output",
                "Training worker returned a non-object result",
            )
        if process.returncode != 0 or value.get("ok") is not True:
            return self._error(
                action.operation,
                _redact_text(
                    str(value.get("failure_stage") or "worker_execution"), payload
                )[:100],
                _redact_text(
                    str(value.get("error_code") or "training_analysis_failed"),
                    payload,
                )[:200],
                _redact_text(
                    str(
                        value.get("error_message")
                        or stderr
                        or "Training analysis failed"
                    ),
                    payload,
                )[:2000],
            )
        target = (
            _redact_value(value.get("target"), payload)
            if isinstance(value.get("target"), dict)
            else {}
        )
        result = (
            _redact_value(value.get("result"), payload)
            if isinstance(value.get("result"), dict)
            else {}
        )
        raw_report_path = value.get("report_path")
        report_path = (
            _redact_text(raw_report_path, payload)[:1000]
            if isinstance(raw_report_path, str)
            else None
        )
        text = json.dumps(
            {
                "operation": action.operation,
                "target": target,
                "result": result,
                "report_path": report_path,
            },
            ensure_ascii=False,
            indent=2,
        )
        return TrainingAnalysisObservation.from_text(
            text=text,
            operation=action.operation,
            target=target,
            result=result,
            report_path=report_path,
        )

    def interrupt(self) -> None:
        with self._lock:
            process = self._process
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def close(self) -> None:
        self.interrupt()

    def _payload(
        self, action: TrainingAnalysisAction, conversation: BaseConversation
    ) -> dict[str, Any]:
        workspace = Path(cast(Any, conversation).workspace.working_dir).resolve()
        output_path = self._output_path(action, workspace)
        state = cast("ConversationState", conversation.state)
        headers = _safe_headers(self.headers)
        raw_headers = (getattr(state, "agent_state", None) or {}).get(
            PYROMIND_VALIDATE_HEADERS_STATE_KEY
        )
        if isinstance(raw_headers, dict):
            headers.update(_safe_headers(raw_headers))
        secret_registry = getattr(state, "secret_registry", None)
        for name, secret_name in self.secret_headers.items():
            value = (
                secret_registry.get_secret_value(secret_name)
                if secret_registry is not None
                else None
            )
            if not value:
                raise ValueError(f"required secret header is unavailable: {name}")
            headers[name] = value
        return {
            "operation": action.operation,
            "task_id": action.task_id,
            "run_url": action.run_url,
            "data_source": action.data_source,
            "metric": action.metric,
            "keys": action.keys,
            "output_path": str(output_path) if output_path else None,
            "output_relative": (
                output_path.relative_to(workspace).as_posix() if output_path else None
            ),
            "api_base": self.api_base,
            "headers": headers,
        }

    @staticmethod
    def _output_path(action: TrainingAnalysisAction, workspace: Path) -> Path | None:
        if action.operation != "report":
            return None
        identifier = (
            "".join(
                char if char.isalnum() or char in "-_" else "-"
                for char in (action.task_id or action.run_url or "run")
            )[-120:]
            or "run"
        )
        relative = Path(
            action.output_path
            or f"public_data/training-analysis/{identifier}/report.md"
        )
        if relative.is_absolute():
            raise ValueError("output_path must be workspace-relative")
        if ".." in relative.parts:
            raise ValueError(
                "output_path must stay under public_data/training-analysis"
            )
        current = workspace
        for part in relative.parts:
            if part == ".":
                continue
            if part == "..":
                current = current.parent
                continue
            current /= part
            if current.is_symlink():
                raise ValueError(
                    "output_path must not traverse symbolic links under "
                    "public_data/training-analysis"
                )
        root = (workspace / "public_data" / "training-analysis").resolve()
        target = (workspace / relative).resolve()
        if (
            not root.is_relative_to(workspace)
            or target == root
            or not target.is_relative_to(workspace)
            or not target.is_relative_to(root)
        ):
            raise ValueError(
                "output_path must stay under public_data/training-analysis"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _error(
        operation: Literal["probe", "analyze", "report"],
        stage: str,
        code: str,
        message: str,
    ) -> TrainingAnalysisObservation:
        message = message[:2000]
        return TrainingAnalysisObservation.from_text(
            text=f"failure_stage={stage}\nerror_code={code}\nerror_message={message}",
            is_error=True,
            operation=operation,
            failure_stage=stage,
            error_code=code,
            error_message=message,
        )


class TrainingAnalysisTool(
    ToolDefinition[TrainingAnalysisAction, TrainingAnalysisObservation]
):
    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        runtime_dir = params.pop("runtime_dir", "")
        if not runtime_dir:
            runtime_dir = str(
                Path(__file__).resolve().parents[3]
                / ".agents"
                / "skills"
                / "training-analysis"
                / "scripts"
            )
        executor = TrainingAnalysisExecutor(
            runtime_dir=str(runtime_dir),
            api_base=params.pop("api_base", None),
            headers=params.pop("headers", None),
            secret_headers=params.pop("secret_headers", None),
            timeout_seconds=float(params.pop("timeout_seconds", 180.0)),
        )
        if params:
            raise ValueError(
                f"TrainingAnalysisTool got unknown params: {', '.join(sorted(params))}"
            )
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=TrainingAnalysisAction,
                observation_type=TrainingAnalysisObservation,
                executor=executor,
                annotations=ToolAnnotations(
                    title="training_analysis",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(TrainingAnalysisTool.name, TrainingAnalysisTool)


def _worker_environment() -> dict[str, str]:
    """Keep worker credentials exclusively in its short-lived stdin payload."""
    sensitive_fragments = (
        "API_KEY",
        "APIKEY",
        "AUTHORIZATION",
        "COOKIE",
        "PASSWORD",
        "SECRET",
        "TOKEN",
        "CREDENTIAL",
        "PRIVATE_KEY",
        "ACCESS_KEY",
        "REFRESH_TOKEN",
        "CLUSTER",
    )
    return {
        name: value
        for name, value in os.environ.items()
        if not any(fragment in name.upper() for fragment in sensitive_fragments)
    }
