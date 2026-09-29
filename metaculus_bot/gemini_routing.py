"""Direct Vertex-first Gemini generation with a Vertex-only OpenRouter fallback."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
from typing import Any

import litellm
from forecasting_tools import GeneralLlm
from google import genai
from google.genai import types as genai_types

from metaculus_bot.constants import (
    GCP_API_KEY_1_ENV,
    GEMINI_OPENROUTER_MODEL,
    GEMINI_OPENROUTER_MODEL_ENV,
    GEMINI_VERTEX_MODEL,
    GEMINI_VERTEX_MODEL_ENV,
    OPENROUTER_API_KEY_ENV,
)
from metaculus_bot.credit_telemetry import PERSONAL_KEY_ALIAS, llm_call_metadata
from metaculus_bot.research.gemini_client_config import build_gemini_http_options

logger = logging.getLogger(__name__)

_PERMANENT_VERTEX_FAILURES: set[str] = set()
_PERMANENT_OPENROUTER_FAILURES: set[str] = set()


class GeminiRouteUnavailable(RuntimeError):
    """Both eligible Gemini routes failed; contains only redacted route reasons."""


def _failure_status(exc: BaseException) -> int | None:
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _vertex_failure_reason(exc: BaseException) -> tuple[str, bool]:
    status = _failure_status(exc)
    message = str(exc).lower()
    if status in (401, 403) or any(term in message for term in ("invalid api key", "permission denied", "not authorized")):
        return "vertex_auth_or_permission", True
    if status == 404 or any(term in message for term in ("model not found", "not found", "unsupported model")):
        return "vertex_model_unavailable", True
    if any(term in message for term in ("quota exceeded", "quota_failure", "resource exhausted", "billing disabled")):
        return "vertex_quota_or_billing", True
    if status == 429 or "rate limit" in message:
        return "vertex_rate_limited", False
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return "vertex_timeout", False
    return "vertex_upstream_error", False


def _openrouter_failure_reason(exc: BaseException) -> tuple[str, bool]:
    status = _failure_status(exc)
    message = str(exc).lower()
    if any(term in message for term in ("no allowed providers", "no endpoints", "provider is not allowed")):
        return "google_vertex_not_allowed_or_unavailable", True
    if status in (401, 403, 404):
        return "openrouter_auth_or_route_unavailable", True
    if status == 429 or "rate limit" in message:
        return "openrouter_rate_limited", False
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return "openrouter_timeout", False
    return "openrouter_upstream_error", False


def _message_parts(prompt: Any, system_prompt: str | None, llm: GeneralLlm) -> tuple[str, list[dict[str, str]]]:
    messages = llm.model_input_to_message(prompt, system_prompt)
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user")).lower()
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError("Vertex Gemini routing currently accepts text messages only")
        if role in {"system", "developer"}:
            system_parts.append(content)
        else:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": [{"text": content}]})
    if not contents:
        raise ValueError("Gemini request contains no user or assistant content")
    return "\n\n".join(system_parts), contents


def _response_text(response: Any) -> str:
    value = getattr(response, "text", None)
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError("Gemini returned no text content")


def _observed_openrouter_provider(response: Any) -> str:
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        for name in ("provider", "custom_llm_provider", "upstream_provider"):
            value = hidden.get(name)
            if isinstance(value, str) and value:
                return value
    return "unknown"


def _observed_cost(response: Any) -> str:
    hidden = getattr(response, "_hidden_params", None)
    cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
    return f"{cost:.8f}" if isinstance(cost, (int, float)) else "unknown"


@functools.lru_cache(maxsize=2)
def _vertex_client(api_key: str, timeout_ms: int) -> genai.Client:
    return genai.Client(
        vertexai=True,
        api_key=api_key,
        http_options=build_gemini_http_options(timeout_ms=timeout_ms, attempts=1),
    )


class VertexFirstGeminiLlm(GeneralLlm):
    """A single GeneralLlm member that routes Vertex first and Vertex-only on fallback."""

    def __init__(self, *, model: str, role: str | None, **kwargs: Any) -> None:
        vertex_model = os.getenv(GEMINI_VERTEX_MODEL_ENV, GEMINI_VERTEX_MODEL).strip()
        openrouter_model = os.getenv(GEMINI_OPENROUTER_MODEL_ENV, GEMINI_OPENROUTER_MODEL).strip()
        if not vertex_model or not openrouter_model.startswith("google/"):
            raise ValueError("Gemini routes require GEMINI_VERTEX_MODEL and GEMINI_OPENROUTER_MODEL=google/<model>")
        super().__init__(
            model=model,
            metadata=llm_call_metadata(role, PERSONAL_KEY_ALIAS),
            **kwargs,
        )
        self._vertex_model = vertex_model
        self._openrouter_model = openrouter_model
        self._role = role or "gemini_generation"

    async def invoke(self, prompt: Any, system_prompt: str | None = None) -> str:  # type: ignore[override]
        messages = self.model_input_to_message(prompt, system_prompt)
        timeout_s = float(self.litellm_kwargs.get("timeout") or 480.0)
        deadline = asyncio.get_running_loop().time() + timeout_s
        direct_key = os.getenv(GCP_API_KEY_1_ENV)
        vertex_reason = "vertex_key_missing"

        if direct_key and self._vertex_model not in _PERMANENT_VERTEX_FAILURES:
            started = time.monotonic()
            try:
                system_instruction, vertex_contents = _message_parts(messages, None, self)
                response = await asyncio.wait_for(
                    self._invoke_vertex(direct_key, system_instruction, vertex_contents, timeout_s),
                    timeout=timeout_s,
                )
                text = _response_text(response)
                elapsed_ms = round((time.monotonic() - started) * 1000)
                usage = getattr(response, "usage_metadata", None)
                logger.info(
                    "GEMINI_ROUTE: role=%s requested_model=%s transport=vertex configured_upstream=google-vertex "
                    "observed_model=%s observed_upstream=google-vertex fallback_reason=none latency_ms=%d "
                    "cost_usd=unknown status=ok",
                    self._role,
                    self._vertex_model,
                    getattr(response, "model_version", None) or "unknown",
                    elapsed_ms,
                )
                logger.info(
                    "GEMINI_VERTEX_USAGE: role=%s model=%s prompt_tokens=%s output_tokens=%s cost_usd=unknown",
                    self._role,
                    getattr(response, "model_version", None) or self._vertex_model,
                    getattr(usage, "prompt_token_count", "unknown"),
                    getattr(usage, "candidates_token_count", "unknown"),
                )
                return text
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001  # one direct attempt then an explicitly restricted route
                vertex_reason, permanent = _vertex_failure_reason(exc)
                if permanent:
                    _PERMANENT_VERTEX_FAILURES.add(self._vertex_model)
                logger.warning(
                    "GEMINI_ROUTE: role=%s requested_model=%s transport=vertex configured_upstream=google-vertex "
                    "observed_upstream=unknown fallback_reason=%s latency_ms=%d cost_usd=unknown status=failed "
                    "error_type=%s http_status=%s",
                    self._role,
                    self._vertex_model,
                    vertex_reason,
                    round((time.monotonic() - started) * 1000),
                    type(exc).__name__,
                    _failure_status(exc) or "unknown",
                )
        elif self._vertex_model in _PERMANENT_VERTEX_FAILURES:
            vertex_reason = "vertex_route_disabled_after_permanent_failure"

        openrouter_key = os.getenv(OPENROUTER_API_KEY_ENV)
        if not openrouter_key:
            raise GeminiRouteUnavailable(f"Vertex route unavailable ({vertex_reason}); OPENROUTER_API_KEY is missing")
        if self._openrouter_model in _PERMANENT_OPENROUTER_FAILURES:
            raise GeminiRouteUnavailable(
                f"Vertex route unavailable ({vertex_reason}); restricted OpenRouter route is disabled"
            )

        remaining_s = deadline - asyncio.get_running_loop().time()
        if remaining_s <= 0:
            raise GeminiRouteUnavailable(f"Vertex route unavailable ({vertex_reason}); request deadline exhausted")
        started = time.monotonic()
        try:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=f"openrouter/{self._openrouter_model}",
                    messages=messages,
                    api_key=openrouter_key,
                    timeout=remaining_s,
                    num_retries=0,
                    temperature=self.litellm_kwargs.get("temperature"),
                    max_tokens=self.litellm_kwargs.get("max_tokens"),
                    extra_body={"provider": {"only": ["google-vertex"]}},
                    metadata=llm_call_metadata(self._role, PERSONAL_KEY_ALIAS),
                ),
                timeout=remaining_s,
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Restricted OpenRouter Gemini route returned no text")
            logger.info(
                "GEMINI_ROUTE: role=%s requested_model=%s transport=openrouter configured_upstream=google-vertex "
                "observed_upstream=%s fallback_reason=%s latency_ms=%d cost_usd=%s status=fallback",
                self._role,
                self._openrouter_model,
                _observed_openrouter_provider(response),
                vertex_reason,
                round((time.monotonic() - started) * 1000),
                _observed_cost(response),
            )
            return content
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001  # convert to redacted route outcome, never log provider body/key
            reason, permanent = _openrouter_failure_reason(exc)
            if permanent:
                _PERMANENT_OPENROUTER_FAILURES.add(self._openrouter_model)
            logger.warning(
                "GEMINI_ROUTE: role=%s requested_model=%s transport=openrouter configured_upstream=google-vertex "
                "observed_upstream=unknown fallback_reason=%s latency_ms=%d cost_usd=unknown status=failed "
                "error_type=%s http_status=%s",
                self._role,
                self._openrouter_model,
                reason,
                round((time.monotonic() - started) * 1000),
                type(exc).__name__,
                _failure_status(exc) or "unknown",
            )
            raise GeminiRouteUnavailable(f"Vertex failed ({vertex_reason}); restricted OpenRouter failed ({reason})") from None

    async def _invoke_vertex(
        self,
        api_key: str,
        system_instruction: str,
        contents: list[dict[str, Any]],
        timeout_s: float,
    ) -> Any:
        client = _vertex_client(api_key, max(1, int(timeout_s * 1000)))
        config_kwargs: dict[str, Any] = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        max_tokens = self.litellm_kwargs.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            config_kwargs["max_output_tokens"] = max_tokens
        temperature = self.litellm_kwargs.get("temperature")
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        return await client.aio.models.generate_content(
            model=self._vertex_model,
            contents=contents,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )


def is_gemini_openrouter_model(model: str) -> bool:
    parts = model.split("/")
    return len(parts) >= 3 and parts[0] == "openrouter" and parts[1] == "google" and parts[2].startswith("gemini-")


def build_vertex_first_gemini(model: str, *, role: str | None = None, **kwargs: Any) -> VertexFirstGeminiLlm:
    return VertexFirstGeminiLlm(model=model, role=role, **kwargs)