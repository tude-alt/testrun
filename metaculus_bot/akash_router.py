"""AkashML first, personal OpenRouter second, without changing GeneralLlm's API.

Credentials are resolved for each request, after the application's .env loading.
Provider clients are per-call objects: neither API keys nor the successful route
are written onto a shared model singleton. Search-specific clients stay separate.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from typing import Any
from urllib.parse import urlsplit

import httpx
import litellm
from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse
from forecasting_tools.ai_models.general_llm import ModelInputType
from forecasting_tools.ai_models.resource_managers.hard_limit_manager import HardLimitExceededError
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import MonetaryCostManager
from openai import APIConnectionError, APIStatusError, APITimeoutError

logger = logging.getLogger(__name__)

AKASHML_BASE_URL = "https://api.akashml.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# These are upstream IDs, not LiteLLM provider prefixes. Verified against the
# providers' public catalogs on 2026-10-26. Override paired IDs in llm_configs.
DEFAULT_MODEL_PAIRS: tuple[tuple[str, str], ...] = (
    ("zai-org/GLM-5.3", "z-ai/glm-5.3"),
    ("Qwen/Qwen3.6-35B-A3B", "qwen/qwen3.6-35b-a3b"),
    ("openai/gpt-oss-120b", "openai/gpt-oss-120b"),
)


class ModelRoutingError(RuntimeError):
    """Both configured routes were unavailable; contains no upstream payload."""


class MissingProviderKey(RuntimeError):
    """A credential is absent; the exception names the variable, never its value."""


class EmptyModelResponseError(RuntimeError):
    """A provider returned no usable answer."""


def _model_id(value: str, prefix: str) -> str:
    value = value.strip()
    if value.startswith(prefix + "/"):
        value = value[len(prefix) + 1 :]
    if not value or "/" not in value or any(c.isspace() for c in value):
        raise ValueError(f"Expected a vendor/model identifier for {prefix}")
    return value


def _base_url(variable: str, default: str) -> str:
    value = (os.getenv(variable) or default).strip().rstrip("/")
    parts = urlsplit(value)
    if (
        parts.scheme != "https"
        or not parts.netloc
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError(f"{variable} must be an HTTPS API base URL without credentials or query parameters")
    return value


def _status_code(error: Exception) -> int | None:
    value = getattr(error, "status_code", None)
    return value if isinstance(value, int) else None


def _error_label(error: Exception) -> str:
    # Exception messages can contain keys, request bodies, or prompt excerpts.
    # Do not interpolate them into logs or the final combined error.
    status = _status_code(error)
    return type(error).__name__ + (f"(HTTP {status})" if status else "")


def _is_provider_failure(error: Exception) -> bool:
    if isinstance(error, (HardLimitExceededError, litellm.BudgetExceededError)):
        return False
    if isinstance(error, (MissingProviderKey, EmptyModelResponseError)):
        return True
    if isinstance(error, (TimeoutError, ConnectionError, APIConnectionError, APITimeoutError, httpx.TransportError)):
        return True
    if isinstance(error, (APIStatusError, litellm.exceptions.APIError)):
        status = _status_code(error)
        return status in {401, 402, 404, 408, 409, 429} or (status is not None and 500 <= status <= 599)
    # forecasting-tools 0.2.92 validates provider content with these exact errors.
    # Do not swallow unrelated RuntimeError or AssertionError programming bugs.
    if isinstance(error, RuntimeError) and str(error).startswith("LLM answer is an empty string."):
        return True
    if isinstance(error, AssertionError) and str(error).startswith("Answer is not a string and is of type:"):
        return True
    return False


class AkashPrimaryLlm(GeneralLlm):
    """A stable model slot with one bounded attempt per inference provider.

    `timeout` is the budget for the entire provider pair. Akash gets at most
    `primary_timeout_fraction` of it; OpenRouter can use the remaining budget.
    Existing caller-level forecast deadlines and cancellation still apply.

    Only personal AKASHML_API_KEY and OPENROUTER_API_KEY are read. The donated
    key, external LiteLLM gateway and default OPENAI_API_KEY are never selected.
    """

    def __init__(
        self,
        akash_model: str,
        openrouter_model: str,
        *,
        role: str = "inference",
        timeout: float = 240,
        primary_timeout_fraction: float = 0.6,
        max_tokens: int = 16_384,
        temperature: float | None = None,
        allowed_tries: int = 1,
        stream: bool = False,
        openrouter_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive, finite number")
        if not 0 < primary_timeout_fraction < 1:
            raise ValueError("primary_timeout_fraction must be between 0 and 1")
        if allowed_tries != 1:
            raise ValueError("AkashPrimaryLlm uses allowed_tries=1; outer forecast runners own retries")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if stream:
            raise ValueError("This forecasting adapter returns complete responses and requires stream=False")
        self.akash_model = _model_id(akash_model, "akashml")
        self.openrouter_model = _model_id(openrouter_model, "openrouter")
        self.role = role
        self.timeout = float(timeout)
        self.primary_timeout_fraction = primary_timeout_fraction
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._openrouter_kwargs = dict(openrouter_kwargs or {})
        allowed_overrides = {"reasoning", "reasoning_effort", "provider"}
        if set(self._openrouter_kwargs) - allowed_overrides:
            raise ValueError("Only reasoning, reasoning_effort and provider are supported OpenRouter overrides")

        # A real GeneralLlm subclass preserves verified-output parsing, framework
        # isinstance checks and token helpers. No secret is stored in its kwargs.
        super().__init__(
            model=f"openai/{self.akash_model}",
            allowed_tries=1,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            stream=False,
        )
        self.model = f"akashml/{self.akash_model}"

    def to_dict(self) -> dict[str, Any]:
        """Safe framework metadata; GeneralLlm's default serializes API kwargs."""
        return {
            "original_model": self.model,
            "model": self.model,
            "openrouter_fallback_model": self.openrouter_model,
            "role": self.role,
            "allowed_tries": 1,
            "timeout": self.timeout,
            "primary_timeout_fraction": self.primary_timeout_fraction,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

    async def _call_provider(
        self, provider: str, prompt: ModelInputType, timeout: float
    ) -> TextTokenCostResponse:
        MonetaryCostManager.raise_error_if_limit_would_be_reached()
        if provider == "akashml":
            key_name = "AKASHML_API_KEY"
            model = f"openai/{self.akash_model}"
            base_url = _base_url("AKASHML_API_BASE_URL", AKASHML_BASE_URL)
            extra: dict[str, Any] = {}
        else:
            key_name = "OPENROUTER_API_KEY"
            model = f"openrouter/{self.openrouter_model}"
            base_url = _base_url("OPENROUTER_API_BASE_URL", OPENROUTER_BASE_URL)
            extra = dict(self._openrouter_kwargs)
        key = (os.getenv(key_name) or "").strip()
        if not key:
            raise MissingProviderKey(f"{key_name} is not set")

        delegate = GeneralLlm(
            model=model,
            api_key=key,
            base_url=base_url,
            temperature=self.temperature,
            timeout=timeout,
            max_tokens=self.max_tokens,
            stream=False,
            allowed_tries=1,
            num_retries=0,
            **extra,
        )
        # No nested model retries: the outer adapter and forecast runner budget
        # the attempts. GeneralLlm still normalizes usage, citations and costs.
        response = await delegate._mockable_direct_call_to_model(prompt)
        if not response.data or not response.data.strip():
            raise EmptyModelResponseError("Provider returned an empty answer")
        logger.info(
            "LLM_ROUTE_SUCCESS role=%s slot=%s provider=%s model=%s prompt_tokens=%s completion_tokens=%s",
            self.role, self.model, provider,
            self.akash_model if provider == "akashml" else self.openrouter_model,
            response.prompt_tokens_used, response.completion_tokens_used,
        )
        if response.cost == 0:
            logger.warning(
                "LLM_COST_UNVERIFIED role=%s provider=%s model=%s: tracked cost is zero; this does not establish a free call",
                self.role, provider, self.akash_model if provider == "akashml" else self.openrouter_model,
            )
        return response

    async def _mockable_direct_call_to_model(self, prompt: ModelInputType) -> TextTokenCostResponse:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        failures: list[str] = []
        for provider in ("akashml", "openrouter"):
            remaining = deadline - loop.time()
            if remaining <= 0:
                failures.append(f"{provider}: TimeoutError")
                break
            allowance = min(remaining, self.timeout * self.primary_timeout_fraction) if provider == "akashml" else remaining
            try:
                async with asyncio.timeout(allowance):
                    return await self._call_provider(provider, prompt, allowance)
            except asyncio.CancelledError:
                # A closed question, shutdown or parent deadline must never
                # start another paid request through the fallback route.
                raise
            except Exception as error:
                if not _is_provider_failure(error):
                    raise
                label = _error_label(error)
                failures.append(f"{provider}: {label}")
                logger.warning(
                    "LLM_ROUTE_FAILURE role=%s slot=%s provider=%s reason=%s next=%s",
                    self.role, self.model, provider, label,
                    "openrouter" if provider == "akashml" else "none",
                )
        raise ModelRoutingError(f"No inference route succeeded for {self.role}: {'; '.join(failures)}") from None


def build_llm_with_openrouter_fallback(
    model: str,
    *,
    role: str = "inference",
    openrouter_model: str | None = None,
    **kwargs: Any,
) -> AkashPrimaryLlm:
    """Factory for Akash slots, including the prediction-market utility calls.

    Existing search modules keep their original provider-aware factory. This
    factory deliberately requires an Akash-prefixed model, preventing accidental
    rerouting of native web-search/grounded-search requests to a text model.
    """
    if not model.startswith("akashml/"):
        raise ValueError("Akash-first factory requires model='akashml/<vendor>/<model>'")
    akash_model = _model_id(model, "akashml")
    fallback = openrouter_model or dict(DEFAULT_MODEL_PAIRS).get(akash_model)
    if not fallback:
        raise ValueError("Set an explicit OpenRouter fallback ID for this Akash model")
    provider_options = {
        key: kwargs.pop(key) for key in ("reasoning", "reasoning_effort", "provider") if key in kwargs
    }
    supplied_options = kwargs.pop("openrouter_kwargs", {})
    return AkashPrimaryLlm(
        akash_model=akash_model,
        openrouter_model=fallback,
        role=role,
        openrouter_kwargs={**provider_options, **supplied_options},
        **kwargs,
    )
