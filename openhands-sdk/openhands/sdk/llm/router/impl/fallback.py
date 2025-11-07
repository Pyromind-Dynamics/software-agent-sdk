from collections.abc import Sequence
from typing import ClassVar

from pydantic import model_validator

from openhands.sdk.llm.llm_response import LLMResponse
from openhands.sdk.llm.message import Message
from openhands.sdk.llm.router.base import RouterLLM
from openhands.sdk.logger import get_logger
from openhands.sdk.tool.tool import ToolDefinition


logger = get_logger(__name__)


class FallbackRouter(RouterLLM):
    """
    A RouterLLM implementation that provides fallback capability across multiple
    language models. When the primary model fails due to rate limits, timeouts,
    or service unavailability, it automatically falls back to secondary models.

    Models are tried in order: primary -> fallback1 -> fallback2 -> ...
    If all models fail, the exception from the last model is raised.

    Example:
        >>> primary = LLM(model="gpt-4", usage_id="primary")
        >>> fallback = LLM(model="gpt-3.5-turbo", usage_id="fallback")
        >>> router = FallbackRouter(
        ...     usage_id="fallback-router",
        ...     llms_for_routing={"primary": primary, "fallback": fallback}
        ... )
        >>> # Will try primary first, then fallback if primary fails
        >>> response = router.completion(messages)
    """

    router_name: str = "fallback_router"

    PRIMARY_MODEL_KEY: ClassVar[str] = "primary"

    def select_llm(self, messages: list[Message]) -> str:  # noqa: ARG002
        """
        For fallback router, we always start with the primary model.
        The fallback logic is implemented in the completion() method.
        """
        return self.PRIMARY_MODEL_KEY

    def completion(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None = None,
        return_metrics: bool = False,
        add_security_risk_prediction: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """
        Try models in order until one succeeds. Falls back to next model
        on retry-able exceptions (rate limits, timeouts, service errors).
        """
        # Get ordered list of model keys
        model_keys = list(self.llms_for_routing.keys())
        last_exception = None

        for i, model_key in enumerate(model_keys):
            llm = self.llms_for_routing[model_key]
            is_last_model = i == len(model_keys) - 1

            try:
                logger.info(
                    f"FallbackRouter: Attempting completion with model "
                    f"'{model_key}' ({llm.model})"
                )
                self.active_llm = llm

                response = llm.completion(
                    messages=messages,
                    tools=tools,
                    _return_metrics=return_metrics,
                    add_security_risk_prediction=add_security_risk_prediction,
                    **kwargs,
                )

                logger.info(
                    f"FallbackRouter: Successfully completed with model '{model_key}'"
                )
                return response

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"FallbackRouter: Model '{model_key}' failed with "
                    f"{type(e).__name__}: {str(e)}"
                )

                if is_last_model:
                    logger.error(
                        "FallbackRouter: All models failed. Raising last exception."
                    )
                    raise
                else:
                    next_model = model_keys[i + 1]
                    logger.info(f"FallbackRouter: Falling back to '{next_model}'...")

        # This should never happen, but satisfy type checker
        assert last_exception is not None
        raise last_exception

    @model_validator(mode="after")
    def _validate_llms_for_routing(self) -> "FallbackRouter":
        """Ensure required primary model is present in llms_for_routing."""
        if self.PRIMARY_MODEL_KEY not in self.llms_for_routing:
            raise ValueError(
                f"Primary LLM key '{self.PRIMARY_MODEL_KEY}' not found "
                "in llms_for_routing."
            )
        return self
