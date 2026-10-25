from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest
from forecasting_tools.ai_models import general_llm as general_llm_module
from forecasting_tools.ai_models.general_llm import TextTokenCostResponse
from litellm import ModelResponse
from litellm.exceptions import APIError

from metaculus_bot.credit_telemetry import LITELLM_GATEWAY_KEY_ALIAS, PERSONAL_KEY_ALIAS
from metaculus_bot.fallback_openrouter import (
    _LITELLM_GATEWAY_PERMANENT_FAILURES,
    LiteLlmGatewayFallback,
    build_litellm_gateway_with_openrouter_fallback,
)
from metaculus_bot.llm_configs import GATEWAY_OPENROUTER_FALLBACKS


@pytest.fixture(autouse=True)
def clear_permanent_gateway_failures() -> None:
    _LITELLM_GATEWAY_PERMANENT_FAILURES.clear()


def _model(model: str = "nova-micro") -> LiteLlmGatewayFallback:
    return build_litellm_gateway_with_openrouter_fallback(
        gateway_model=model,
        openrouter_model=GATEWAY_OPENROUTER_FALLBACKS[model],
        role=f"forecaster:{model}",
        temperature=None,
        max_tokens=512,
        timeout=3,
        allowed_tries=1,
    )


def _llm_response(text: str) -> TextTokenCostResponse:
    return TextTokenCostResponse(
        data=text,
        prompt_tokens_used=1,
        completion_tokens_used=1,
        total_tokens_used=2,
        model="mock",
        cost=0.0,
    )


@pytest.mark.parametrize(("gateway_model", "openrouter_model"), GATEWAY_OPENROUTER_FALLBACKS.items())
def test_gateway_roster_slot_has_gateway_primary_and_openrouter_fallback(
    monkeypatch: pytest.MonkeyPatch,
    gateway_model: str,
    openrouter_model: str,
) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "gateway-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    llm = build_litellm_gateway_with_openrouter_fallback(
        gateway_model=gateway_model,
        openrouter_model=openrouter_model,
        role=f"forecaster:{gateway_model}",
        allowed_tries=1,
    )

    assert llm.model == f"openai/{gateway_model}"
    assert llm.litellm_kwargs["base_url"] == "https://litellm.vision.aivar.app/v1"
    assert llm.litellm_kwargs["api_key"] == "gateway-secret"
    assert llm.litellm_kwargs["metadata"] == {
        "role": f"forecaster:{gateway_model}",
        "key_alias": LITELLM_GATEWAY_KEY_ALIAS,
    }
    assert llm._openrouter_llm is not None
    assert llm._openrouter_llm.model == f"openrouter/{openrouter_model}"
    assert llm._openrouter_llm.litellm_kwargs["api_key"] == "router-secret"
    assert llm._openrouter_llm.litellm_kwargs["metadata"]["key_alias"] == PERSONAL_KEY_ALIAS


@pytest.mark.asyncio
async def test_gateway_success_does_not_invoke_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "gateway-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    llm = _model()
    assert llm._openrouter_llm is not None
    gateway_invoke = AsyncMock(return_value=_llm_response("gateway"))
    fallback = AsyncMock(return_value="unexpected fallback")
    monkeypatch.setattr(llm, "_invoke_with_request_cost_time_and_token_limits_and_retry", gateway_invoke)
    monkeypatch.setattr(llm._openrouter_llm, "invoke", fallback)
    assert await llm.invoke("prompt") == "gateway"
    gateway_invoke.assert_awaited_once()
    fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_framework_sends_gateway_base_url_key_and_plain_alias_to_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "gateway-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    llm = _model()
    response = ModelResponse(
        choices=[{"message": {"role": "assistant", "content": "gateway answer"}}],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
    acompletion = AsyncMock(return_value=response)
    monkeypatch.setattr(general_llm_module, "acompletion", acompletion)

    result = await llm._mockable_direct_call_to_model("hello")

    assert result.data == "gateway answer"
    request = acompletion.await_args.kwargs
    assert request["model"] == "openai/nova-micro"
    assert request["base_url"] == "https://litellm.vision.aivar.app/v1"
    assert request["api_key"] == "gateway-secret"
    assert request["api_key"] != "router-secret"


@pytest.mark.asyncio
async def test_gateway_failure_falls_back_once_without_changing_logical_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "gateway-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    llm = _model()
    assert llm._openrouter_llm is not None
    primary = AsyncMock(side_effect=APIError(status_code=503, message="gateway unavailable", llm_provider="openai", model="nova-micro"))
    fallback = AsyncMock(return_value="router answer")
    monkeypatch.setattr(llm, "_invoke_with_request_cost_time_and_token_limits_and_retry", primary)
    monkeypatch.setattr(llm._openrouter_llm, "invoke", fallback)

    assert await llm.invoke("prompt") == "router answer"
    assert primary.await_count == 1
    fallback.assert_awaited_once_with("prompt", None)
    assert llm.model == "openai/nova-micro"


@pytest.mark.asyncio
async def test_missing_gateway_key_skips_gateway_and_uses_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    llm = _model()
    assert llm._openrouter_llm is not None
    fallback = AsyncMock(return_value="router answer")
    monkeypatch.setattr(llm._openrouter_llm, "invoke", fallback)

    assert await llm.invoke("prompt") == "router answer"
    fallback.assert_awaited_once_with("prompt", None)


@pytest.mark.asyncio
async def test_permanent_gateway_auth_failure_is_not_retried_for_later_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "gateway-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    llm = _model()
    assert llm._openrouter_llm is not None
    primary = AsyncMock(
        side_effect=APIError(status_code=401, message="invalid gateway key", llm_provider="openai", model="nova-micro")
    )
    fallback = AsyncMock(side_effect=["first router answer", "second router answer"])
    monkeypatch.setattr(llm, "_invoke_with_request_cost_time_and_token_limits_and_retry", primary)
    monkeypatch.setattr(llm._openrouter_llm, "invoke", fallback)

    assert await llm.invoke("question one") == "first router answer"
    assert await llm.invoke("question two") == "second router answer"
    another_model = _model("nova-lite")
    assert another_model._openrouter_llm is not None
    another_fallback = AsyncMock(return_value="third router answer")
    monkeypatch.setattr(another_model._openrouter_llm, "invoke", another_fallback)
    assert await another_model.invoke("question three") == "third router answer"
    assert primary.await_count == 1
    assert fallback.await_count == 2
    another_fallback.assert_awaited_once_with("question three", None)


@pytest.mark.asyncio
async def test_no_gateway_or_fallback_credentials_fails_without_leaking_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    llm = _model()

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not configured") as exc_info:
        await llm.invoke("prompt")
    assert "gateway-secret" not in str(exc_info.value)
    assert "router-secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_fallback_failure_is_redacted(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "gateway-secret-value")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret-value")
    llm = _model()
    assert llm._openrouter_llm is not None
    monkeypatch.setattr(
        llm,
        "_invoke_with_request_cost_time_and_token_limits_and_retry",
        AsyncMock(side_effect=RuntimeError("gateway rejected gateway-secret-value")),
    )
    monkeypatch.setattr(llm._openrouter_llm, "invoke", AsyncMock(side_effect=RuntimeError("router-secret-value rejected")))

    with (
        caplog.at_level(logging.WARNING, logger="metaculus_bot.fallback_openrouter"),
        pytest.raises(RuntimeError, match="OpenRouter fallback failed") as exc_info,
    ):
        await llm.invoke("prompt")
    assert "gateway-secret-value" not in str(exc_info.value)
    assert "router-secret-value" not in str(exc_info.value)
    assert "gateway-secret-value" not in caplog.text
    assert "router-secret-value" not in caplog.text
