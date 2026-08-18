from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from pyromind_runtime.contracts.content import JsonObject, TextContentBlock
from pyromind_runtime.contracts.sandbox import SandboxRef, WorkspaceRef
from pyromind_runtime.contracts.tools import ToolResult, ToolRiskLevel, ToolSpec


logger = logging.getLogger(__name__)


class ToolHostRegistrationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    session_id: str
    user_id: str
    workspace: WorkspaceRef
    sandbox: SandboxRef


class ToolHandler(Protocol):
    async def __call__(
        self,
        arguments: JsonObject,
        context: ToolExecutionContext,
    ) -> ToolResult: ...


@dataclass(frozen=True, slots=True)
class ToolRiskPolicy:
    allowed: frozenset[ToolRiskLevel] = frozenset({"low"})

    def permits(self, risk_level: ToolRiskLevel) -> bool:
        return risk_level in self.allowed


@dataclass(frozen=True, slots=True)
class _RegisteredTool:
    spec: ToolSpec
    handler: ToolHandler
    validator: Draft202012Validator


class PythonToolHost:
    def __init__(
        self,
        risk_policy: ToolRiskPolicy | None = None,
        *,
        max_result_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        if max_result_bytes <= 0:
            raise ValueError("max_result_bytes must be greater than zero")
        self._risk_policy = risk_policy or ToolRiskPolicy()
        self._max_result_bytes = max_result_bytes
        self._tools: dict[str, _RegisteredTool] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if spec.name in self._tools:
            raise ToolHostRegistrationError(f"tool is already registered: {spec.name}")
        try:
            Draft202012Validator.check_schema(spec.input_schema)
        except SchemaError as exc:
            raise ToolHostRegistrationError(
                f"tool input schema is invalid: {spec.name}"
            ) from exc
        self._tools[spec.name] = _RegisteredTool(
            spec=spec,
            handler=handler,
            validator=Draft202012Validator(spec.input_schema),
        )

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec for tool in self._tools.values())

    async def execute(
        self,
        name: str,
        arguments: JsonObject,
        context: ToolExecutionContext,
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return self._error("tool_not_found", f"Unknown tool: {name}")
        if not self._risk_policy.permits(tool.spec.risk_level):
            return self._error(
                "tool_risk_denied",
                f"Tool risk level is not allowed: {tool.spec.risk_level}",
            )
        try:
            tool.validator.validate(arguments)
        except ValidationError as exc:
            return ToolResult(
                content=(TextContentBlock(text="Tool arguments are invalid."),),
                details={"path": [str(item) for item in exc.absolute_path]},
                is_error=True,
                error_code="invalid_tool_arguments",
            )

        try:
            async with asyncio.timeout(tool.spec.timeout_seconds):
                result = await tool.handler(arguments, context)
        except TimeoutError:
            return self._error("tool_timeout", "Tool execution timed out.")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Python business tool failed: %s (session=%s)",
                name,
                context.session_id,
            )
            return self._error("tool_handler_failed", "Tool execution failed.")

        if len(result.model_dump_json().encode("utf-8")) > self._max_result_bytes:
            return self._error(
                "tool_result_too_large",
                "Tool result exceeded the configured output limit.",
            )
        return result

    @staticmethod
    def _error(error_code: str, message: str) -> ToolResult:
        return ToolResult(
            content=(TextContentBlock(text=message),),
            is_error=True,
            error_code=error_code,
        )
