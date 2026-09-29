from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metaculus_bot import gemini_routing
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.gemini_routing import GeminiRouteUnavailable, VertexFirstGeminiLlm


@pytest.fixture(autouse=True)
def clear_route_caches() -> None:
    gemini_routing._PERMANENT_VERTEX_FAILURES.clear()
    gemini_routing._PERMANENT_OPENROUTER_FAILURES.clear()
    gemini_routing._vertex_client.cache_clear()


def _llm() -> VertexFirstGeminiLlm:
    llm = build_llm_with_openrouter_fallback(
        "openrouter/google/gemini-3.1-pro-preview",
        role="forecaster:google",
        temperature=None,
        timeout=5,
        max_tokens=512,
        allowed_tries=1,
    )
    assert isinstance(llm, VertexFirstGeminiLlm)
    return llm


def _vertex_client(response: object | None = None, *, error: BaseException | None = None) -> MagicMock:
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(
        return_value=response or SimpleNamespace(text="vertex answer", model_version="gemini-3.1-pro-preview"),
        side_effect=error,
    )
    return client


def _openrouter_response(content: str = "fallback answer", *, cost: float | None = None) -> SimpleNamespace:
    hidden = {"provider": "Google Vertex AI"}
    if cost is not None:
        hidden["response_cost"] = cost
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        _hidden_params=hidden,
    )


@pytest.mark.asyncio
async def test_vertex_key_reaches_explicit_vertex_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCP_API_KEY_1", "vertex-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    client = _vertex_client()
    with patch("metaculus_bot.gemini_routing.genai.Client", return_value=client) as factory:
        assert await _llm().invoke("forecast prompt") == "vertex answer"

    assert factory.call_args.kwargs["vertexai"] is True
    assert factory.call_args.kwargs["api_key"] == "vertex-secret"
    assert client.aio.models.generate_content.await_args.kwargs["model"] == "gemini-3.1-pro-preview"


@pytest.mark.asyncio
async def test_vertex_failure_falls_back_once_with_google_vertex_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCP_API_KEY_1", "vertex-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    client = _vertex_client(error=RuntimeError("503 temporary upstream failure"))
    completion = _openrouter_response(cost=0.0012)

    with (
        patch("metaculus_bot.gemini_routing.genai.Client", return_value=client),
        patch("metaculus_bot.gemini_routing.litellm.acompletion", new_callable=AsyncMock, return_value=completion) as fallback,
    ):
        result = await _llm().invoke("forecast prompt")

    assert result == "fallback answer"
    assert client.aio.models.generate_content.await_count == 1
    assert fallback.await_count == 1
    kwargs = fallback.await_args.kwargs
    assert kwargs["api_key"] == "router-secret"
    assert kwargs["api_key"] != "vertex-secret"
    assert kwargs["extra_body"] == {"provider": {"only": ["google-vertex"]}}
    assert kwargs["model"] == "openrouter/google/gemini-3.1-pro-preview"
    assert kwargs["num_retries"] == 0


@pytest.mark.asyncio
async def test_ai_studio_key_never_enables_native_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GCP_API_KEY_1", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "ai-studio-secret")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch("metaculus_bot.gemini_routing.genai.Client") as client_factory:
        with pytest.raises(GeminiRouteUnavailable, match="vertex_key_missing"):
            await _llm().invoke("forecast prompt")
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_permanent_vertex_auth_failure_is_not_retried_for_later_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GCP_API_KEY_1", "vertex-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    client = _vertex_client(error=RuntimeError("401 invalid api key"))
    with (
        patch("metaculus_bot.gemini_routing.genai.Client", return_value=client),
        patch(
            "metaculus_bot.gemini_routing.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=_openrouter_response(),
        ) as fallback,
    ):
        assert await _llm().invoke("question one") == "fallback answer"
        assert await _llm().invoke("question two") == "fallback answer"

    assert client.aio.models.generate_content.await_count == 1
    assert fallback.await_count == 2


@pytest.mark.asyncio
async def test_restricted_openrouter_failure_does_not_retry_or_loosen_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GCP_API_KEY_1", "vertex-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    client = _vertex_client(error=RuntimeError("403 permission denied"))
    with (
        patch("metaculus_bot.gemini_routing.genai.Client", return_value=client),
        patch(
            "metaculus_bot.gemini_routing.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=RuntimeError("No allowed providers for Google Vertex"),
        ) as fallback,
        pytest.raises(GeminiRouteUnavailable, match="google_vertex_not_allowed_or_unavailable"),
    ):
        await _llm().invoke("forecast prompt")

    assert fallback.await_count == 1
    assert fallback.await_args.kwargs["extra_body"] == {"provider": {"only": ["google-vertex"]}}


@pytest.mark.asyncio
async def test_route_logs_redact_provider_credentials(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("GCP_API_KEY_1", "vertex-secret-value")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret-value")
    client = _vertex_client(error=RuntimeError("401 rejected vertex-secret-value"))
    with (
        patch("metaculus_bot.gemini_routing.genai.Client", return_value=client),
        patch(
            "metaculus_bot.gemini_routing.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=_openrouter_response(),
        ),
        caplog.at_level(logging.INFO, logger="metaculus_bot.gemini_routing"),
    ):
        assert await _llm().invoke("forecast prompt") == "fallback answer"

    assert "vertex-secret-value" not in caplog.text
    assert "router-secret-value" not in caplog.text
    assert "configured_upstream=google-vertex" in caplog.text
    assert "observed_upstream=Google Vertex AI" in caplog.text
