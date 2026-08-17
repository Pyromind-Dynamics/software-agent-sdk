import time
from unittest.mock import patch

import pytest
from litellm.exceptions import APIConnectionError, BadRequestError
from litellm.types.utils import (
    Choices,
    Message as LiteLLMMessage,
    ModelResponse,
    Usage,
)
from pydantic import SecretStr

from openhands.sdk.llm import LLM, FailoverRouter, ImageContent, Message, TextContent
from openhands.sdk.llm.exceptions import LLMServiceUnavailableError


def _get_mock_response(content: str = "ok", model: str = "gpt-4o") -> ModelResponse:
    return ModelResponse(
        id="resp-1",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=LiteLLMMessage(content=content, role="assistant"),
            )
        ],
        created=1,
        model=model,
        object="chat.completion",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _get_llm(model: str, usage_id: str) -> LLM:
    return LLM(
        model=model,
        api_key=SecretStr("k"),
        usage_id=usage_id,
        num_retries=0,
        retry_min_wait=0,
        retry_max_wait=0,
    )


def _get_router(*models: str) -> FailoverRouter:
    return FailoverRouter(
        llms_for_routing={
            model: _get_llm(model, usage_id=f"failover-{model}") for model in models
        },
        cooldown_seconds=300.0,
    )


_MSGS = [Message(role="user", content=[TextContent(text="hi")])]


def _assert_content(resp, expected_text: str) -> None:
    assert isinstance(resp.message.content[0], TextContent)
    assert resp.message.content[0].text == expected_text


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_primary_succeeds_fallback_not_tried(mock_comp):
    mock_comp.return_value = _get_mock_response("primary ok", model="primary")
    router = _get_router("primary", "fallback")

    _assert_content(router.completion(_MSGS), "primary ok")
    # One call per inner LLM; fallback never invoked.
    assert mock_comp.call_count == 1
    assert mock_comp.call_args.kwargs["model"] == "primary"


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_primary_transient_failure_routes_to_fallback(mock_comp):
    def side_effect(**kwargs):
        if kwargs.get("model") == "primary":
            raise APIConnectionError(
                message="connection reset", llm_provider="openai", model="primary"
            )
        return _get_mock_response("fallback ok", model="fallback")

    mock_comp.side_effect = side_effect
    router = _get_router("primary", "fallback")

    _assert_content(router.completion(_MSGS), "fallback ok")
    assert sorted(c.kwargs["model"] for c in mock_comp.call_args_list) == [
        "fallback",
        "primary",
    ]
    # The failed primary is now in cooldown.
    assert "primary" in router._cooldowns_until


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_all_providers_fail_raises(mock_comp):
    mock_comp.side_effect = APIConnectionError(
        message="down", llm_provider="openai", model="primary"
    )
    router = _get_router("primary", "fallback")

    # Inner LLMs map APIConnectionError to LLMServiceUnavailableError.
    with pytest.raises(LLMServiceUnavailableError):
        router.completion(_MSGS)


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_cooldown_skips_failed_provider_on_next_call(mock_comp):
    def side_effect(**kwargs):
        if kwargs.get("model") == "primary":
            raise APIConnectionError(
                message="down", llm_provider="openai", model="primary"
            )
        return _get_mock_response("fallback ok", model="fallback")

    mock_comp.side_effect = side_effect
    router = _get_router("primary", "fallback")

    _assert_content(router.completion(_MSGS), "fallback ok")
    primary_calls = sum(
        c.kwargs["model"] == "primary" for c in mock_comp.call_args_list
    )
    assert primary_calls == 1

    # Second call goes straight to the healthy fallback without retrying primary.
    _assert_content(router.completion(_MSGS), "fallback ok")
    assert sum(c.kwargs["model"] == "primary" for c in mock_comp.call_args_list) == 1


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_cooldown_expiry_reactivates_provider(mock_comp):
    mock_comp.side_effect = APIConnectionError(
        message="down", llm_provider="openai", model="primary"
    )
    router = _get_router("primary", "fallback")
    with pytest.raises(LLMServiceUnavailableError):
        router.completion(_MSGS)

    # Simulate the cooldown window passing; the failed call now succeeds.
    router._cooldowns_until["primary"] = time.monotonic() - 1
    mock_comp.side_effect = None
    mock_comp.return_value = _get_mock_response("primary ok", model="primary")

    _assert_content(router.completion(_MSGS), "primary ok")
    assert mock_comp.call_args.kwargs["model"] == "primary"


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_nontransient_error_does_not_cool_down(mock_comp):
    def side_effect(**kwargs):
        if kwargs.get("model") == "primary":
            raise BadRequestError(
                message="bad model name", llm_provider="openai", model="primary"
            )
        return _get_mock_response("fallback ok", model="fallback")

    mock_comp.side_effect = side_effect
    router = _get_router("primary", "fallback")

    _assert_content(router.completion(_MSGS), "fallback ok")
    # BadRequest maps to LLMBadRequestError: no cooldown for the primary.
    assert "primary" not in router._cooldowns_until

    # Primary stays first in line for the next request.
    mock_comp.side_effect = None
    mock_comp.return_value = _get_mock_response("primary ok", model="primary")
    _assert_content(router.completion(_MSGS), "primary ok")
    assert mock_comp.call_args.kwargs["model"] == "primary"


@patch("openhands.sdk.llm.llm.litellm_acompletion")
async def test_acompletion_falls_back_to_healthy_provider(mock_acomp):
    async def side_effect(**kwargs):
        if kwargs.get("model") == "primary":
            raise APIConnectionError(
                message="connection reset", llm_provider="openai", model="primary"
            )
        return _get_mock_response("fallback ok", model="fallback")

    mock_acomp.side_effect = side_effect
    router = _get_router("primary", "fallback")

    resp = await router.acompletion(_MSGS)
    _assert_content(resp, "fallback ok")
    assert {c.kwargs["model"] for c in mock_acomp.call_args_list} == {
        "primary",
        "fallback",
    }


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_image_request_skips_non_vision_provider(mock_comp):
    mock_comp.return_value = _get_mock_response("vision ok", model="gpt-4o")
    router = _get_router("primary", "gpt-4o")
    router.llms_for_routing["primary"].disable_vision = True
    # A text-only provider is never given image content.
    image_msgs = [
        Message(
            role="user",
            content=[
                TextContent(text="what is this?"),
                ImageContent(image_urls=["https://x/a.png"]),
            ],
        )
    ]

    _assert_content(router.completion(image_msgs), "vision ok")
    assert {c.kwargs["model"] for c in mock_comp.call_args_list} == {"gpt-4o"}


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_image_request_uses_all_providers_when_none_vision_capable(mock_comp):
    def side_effect(**kwargs):
        if kwargs.get("model") == "primary":
            raise APIConnectionError(
                message="down", llm_provider="openai", model="primary"
            )
        return _get_mock_response("fallback ok", model="fallback")

    mock_comp.side_effect = side_effect
    router = _get_router("primary", "fallback")
    router.llms_for_routing["primary"].disable_vision = True
    router.llms_for_routing["fallback"].disable_vision = True
    image_msgs = [
        Message(role="user", content=[ImageContent(image_urls=["https://x/a.png"])])
    ]

    # No provider claims vision support, so the request still goes out
    # against the full list instead of failing with an empty candidate set.
    _assert_content(router.completion(image_msgs), "fallback ok")
    assert {c.kwargs["model"] for c in mock_comp.call_args_list} == {
        "primary",
        "fallback",
    }
