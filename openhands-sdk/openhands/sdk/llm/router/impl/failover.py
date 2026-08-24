from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from pydantic import Field, PrivateAttr

from openhands.sdk.llm.exceptions import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMContentPolicyViolationError,
    LLMContextWindowExceedError,
    LLMMalformedConversationHistoryError,
)
from openhands.sdk.llm.llm import LLM
from openhands.sdk.llm.message import Message
from openhands.sdk.llm.router.base import RouterLLM
from openhands.sdk.logger import get_logger


if TYPE_CHECKING:
    from openhands.sdk.llm.llm import LLMCallContext
    from openhands.sdk.llm.llm_response import LLMResponse
    from openhands.sdk.llm.streaming import (
        AnyTokenCallbackType,
        TokenCallbackType,
    )
    from openhands.sdk.tool.tool import ToolDefinition

logger = get_logger(__name__)

# Failures that switching providers cannot fix: keep the provider eligible so a
# fixed configuration takes effect immediately instead of waiting out a cooldown.
_NON_TRANSIENT_EXCEPTIONS: tuple[type[Exception], ...] = (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMContentPolicyViolationError,
    LLMContextWindowExceedError,
    LLMMalformedConversationHistoryError,
)


class FailoverRouter(RouterLLM):
    """Routes each request to the first healthy LLM in ``llms_for_routing`` order.

    When the active LLM fails with a transient error (connection, timeout,
    rate limit, 5xx), it is put into cooldown and the request moves on to the
    next healthy LLM. A provider in cooldown becomes eligible again after
    ``cooldown_seconds``, so recovery is automatic once the endpoint stabilizes.
    """

    router_name: str = "failover_router"
    cooldown_seconds: float = Field(
        default=300.0,
        ge=0,
        description=(
            "Seconds a provider stays suspended after a transient failure "
            "before it becomes eligible again."
        ),
    )
    multimodal_llms: dict[str, LLM] = Field(
        default_factory=dict,
        description=(
            "Vision-capable providers used for image-bearing requests. "
            "Requests without images always use ``llms_for_routing``."
        ),
    )
    _cooldowns_until: dict[str, float] = PrivateAttr(default_factory=dict)

    def select_llm(self, messages: list[Message]) -> str:  # noqa: ARG002
        """Return the highest-priority healthy LLM name."""
        return self._routing_order(messages)[0]

    def uses_responses_api(self) -> bool:
        """Delegate the API choice to the first LLM that handles the call."""
        return self.llms_for_routing[self._routing_order()[0]].uses_responses_api()

    def _routing_order(self, messages: Sequence[Message] | None = None) -> list[str]:
        """Return LLM names in priority order, excluding cooled-down providers.

        Text-only requests use ``llms_for_routing``. Image-bearing requests use
        the ``multimodal_llms`` cohort when one is configured, so a text-only
        fallback is never given image content it cannot handle. When every
        provider in the selected cohort is cooling down, all of them are
        returned so the request still goes out.
        """
        now = time.monotonic()
        for name in list(self._cooldowns_until):
            if self._cooldowns_until[name] <= now:
                del self._cooldowns_until[name]
        has_images = bool(messages) and any(msg.contains_image for msg in messages)
        cohort = [
            name
            for name in self._cohort_names(has_images)
            if name not in self._cooldowns_until
        ]
        if has_images and not self.multimodal_llms:
            vision_cohort = [
                name for name in cohort if self._provider(name).vision_is_active()
            ]
            if vision_cohort:
                cohort = vision_cohort
        return cohort or [name for name in self._cohort_names(has_images)]

    def _mark_failure(self, name: str) -> None:
        self._cooldowns_until[name] = time.monotonic() + self.cooldown_seconds

    def _mark_success(self, name: str) -> None:
        self._cooldowns_until.pop(name, None)

    @staticmethod
    def _is_transient(error: Exception) -> bool:
        return not isinstance(error, _NON_TRANSIENT_EXCEPTIONS)

    def _provider(self, name: str) -> LLM:
        """Look up a provider across the two cohorts."""
        llm = self.llms_for_routing.get(name) or self.multimodal_llms.get(name)
        assert llm is not None, f"unknown provider {name!r}"
        return llm

    def _cohort_names(self, multimodal: bool) -> list[str]:
        if multimodal and self.multimodal_llms:
            return list(self.multimodal_llms)
        return list(self.llms_for_routing)

    def _route(
        self,
        messages: Sequence[Message] | None,
        try_provider: Callable[[str], LLMResponse],
    ) -> LLMResponse:
        """Run ``try_provider`` over healthy LLMs until one succeeds."""
        last_error: Exception | None = None
        for name in self._routing_order(messages):
            self.active_llm = self._provider(name)
            metrics_before = self.active_llm.metrics.deep_copy()
            try:
                try:
                    result = try_provider(name)
                finally:
                    self.metrics.merge(self.active_llm.metrics.diff(metrics_before))
                self._mark_success(name)
                logger.info(f"[FailoverRouter] provider '{name}' succeeded")
                return result
            except Exception as error:
                last_error = error
                if self._is_transient(error):
                    self._mark_failure(name)
                    logger.warning(
                        "[FailoverRouter] provider '%s' failed with %s; "
                        "switching to next provider",
                        name,
                        type(error).__name__,
                    )
                else:
                    logger.warning(
                        "[FailoverRouter] provider '%s' failed with non-transient %s",
                        name,
                        type(error).__name__,
                    )
        assert last_error is not None
        raise last_error

    async def _aroute(
        self,
        messages: Sequence[Message] | None,
        try_provider: Callable[[str], Awaitable[LLMResponse]],
    ) -> LLMResponse:
        """Async variant of :meth:`_route`."""
        last_error: Exception | None = None
        for name in self._routing_order(messages):
            self.active_llm = self._provider(name)
            metrics_before = self.active_llm.metrics.deep_copy()
            try:
                try:
                    result = await try_provider(name)
                finally:
                    self.metrics.merge(self.active_llm.metrics.diff(metrics_before))
                self._mark_success(name)
                logger.info(f"[FailoverRouter] provider '{name}' succeeded")
                return result
            except Exception as error:
                last_error = error
                if self._is_transient(error):
                    self._mark_failure(name)
                    logger.warning(
                        "[FailoverRouter] provider '%s' failed with %s; "
                        "switching to next provider",
                        name,
                        type(error).__name__,
                    )
                else:
                    logger.warning(
                        "[FailoverRouter] provider '%s' failed with non-transient %s",
                        name,
                        type(error).__name__,
                    )
        assert last_error is not None
        raise last_error

    def completion(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None = None,
        add_security_risk_prediction: bool = False,
        on_token: TokenCallbackType | None = None,
        call_context: LLMCallContext | None = None,
        **kwargs,
    ) -> LLMResponse:
        def try_provider(name: str) -> LLMResponse:
            return self._provider(name).completion(
                messages=messages,
                tools=tools,
                add_security_risk_prediction=add_security_risk_prediction,
                on_token=on_token,
                call_context=call_context,
                **kwargs,
            )

        return self._route(messages, try_provider)

    def responses(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None = None,
        include: list[str] | None = None,
        store: bool | None = None,
        add_security_risk_prediction: bool = False,
        on_token: TokenCallbackType | None = None,
        call_context: LLMCallContext | None = None,
        **kwargs,
    ) -> LLMResponse:
        def try_provider(name: str) -> LLMResponse:
            return self._provider(name).responses(
                messages=messages,
                tools=tools,
                include=include,
                store=store,
                add_security_risk_prediction=add_security_risk_prediction,
                on_token=on_token,
                call_context=call_context,
                **kwargs,
            )

        return self._route(messages, try_provider)

    async def acompletion(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None = None,
        add_security_risk_prediction: bool = False,
        on_token: AnyTokenCallbackType | None = None,
        call_context: LLMCallContext | None = None,
        **kwargs,
    ) -> LLMResponse:
        async def try_provider(name: str) -> LLMResponse:
            return await self._provider(name).acompletion(
                messages=messages,
                tools=tools,
                add_security_risk_prediction=add_security_risk_prediction,
                on_token=on_token,
                call_context=call_context,
                **kwargs,
            )

        return await self._aroute(messages, try_provider)

    async def aresponses(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None = None,
        include: list[str] | None = None,
        store: bool | None = None,
        add_security_risk_prediction: bool = False,
        on_token: AnyTokenCallbackType | None = None,
        call_context: LLMCallContext | None = None,
        **kwargs,
    ) -> LLMResponse:
        async def try_provider(name: str) -> LLMResponse:
            return await self._provider(name).aresponses(
                messages=messages,
                tools=tools,
                include=include,
                store=store,
                add_security_risk_prediction=add_security_risk_prediction,
                on_token=on_token,
                call_context=call_context,
                **kwargs,
            )

        return await self._aroute(messages, try_provider)
