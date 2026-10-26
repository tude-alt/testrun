"""Central model configuration: AkashML primary, personal OpenRouter fallback.

The forecasting roster and text utility slots use the same bounded adapter.
Search services retain their own provider-specific clients and credentials.
"""

from __future__ import annotations

import os
from typing import Any

from forecasting_tools import GeneralLlm

from metaculus_bot.akash_router import DEFAULT_MODEL_PAIRS, AkashPrimaryLlm, build_llm_with_openrouter_fallback

__all__ = [
    "DISAGREEMENT_ANALYZER_LLM", "FORECASTER_LLMS", "FORECASTER_MODEL_NAMES",
    "MARKET_QUERY_AUTHOR_LLM_CONFIG", "MARKET_RANKER_LLM_CONFIG", "PARSER_LLM",
    "RESEARCHER_LLM", "STACKER_FALLBACK_LLM", "STACKER_LLM", "SUMMARIZER_LLM",
]

# One routing cycle per invocation. Existing forecast/stacker deadlines remain
# authoritative. Avoid the old per-provider 3 x 480-second retry multiplication.
# Conservative configurable output caps replace the unverified 64k assumption.
REASONING_MODEL_CONFIG: dict[str, Any] = {
    "temperature": None,
    "max_tokens": int(os.getenv("AKASHML_FORECAST_MAX_TOKENS") or "16384"),
    "stream": False,
    "timeout": float(os.getenv("AKASHML_FORECAST_TIMEOUT") or "240"),
    "allowed_tries": 1,
}
UTILITY_MODEL_CONFIG: dict[str, Any] = {
    "temperature": None,
    "max_tokens": int(os.getenv("AKASHML_UTILITY_MAX_TOKENS") or "8192"),
    "stream": False,
    "timeout": float(os.getenv("AKASHML_UTILITY_TIMEOUT") or "120"),
    "allowed_tries": 1,
}
ACCEPTABLE_QUANTS = ["fp8", "fp16", "bf16", "fp32", "unknown"]


def forecaster_role(model: str) -> str:
    """Return a stable vendor role for a supported provider/vendor/model slug."""
    parts = model.split("/")
    if len(parts) == 2 and parts[0] == "openai" and parts[1]:
        # Keep the legacy gateway-label helper compatible; no gateway is built.
        return f"forecaster:{parts[1]}"
    if len(parts) >= 3 and parts[0] in {"akashml", "openrouter"} and all(parts):
        return f"forecaster:{parts[1]}"
    raise ValueError("Expected akashml/<vendor>/<model>, openrouter/<vendor>/<model> or openai/<alias>")


def _roster_pairs() -> tuple[tuple[str, str], ...]:
    primary = (os.getenv("AKASHML_FORECAST_MODELS") or "").strip()
    fallback = (os.getenv("OPENROUTER_FORECAST_MODELS") or "").strip()
    if not primary and not fallback:
        return DEFAULT_MODEL_PAIRS
    if not primary or not fallback:
        raise ValueError("Set both AKASHML_FORECAST_MODELS and OPENROUTER_FORECAST_MODELS as paired comma-separated lists")
    primary_ids = [value.strip().removeprefix("akashml/") for value in primary.split(",")]
    fallback_ids = [value.strip().removeprefix("openrouter/") for value in fallback.split(",")]
    if len(primary_ids) != len(fallback_ids) or not all(primary_ids + fallback_ids):
        raise ValueError("Primary and fallback model lists must be nonempty and have the same number of entries")
    if len(set(primary_ids)) != len(primary_ids):
        raise ValueError("Duplicate Akash forecasters are not independent ensemble members")
    return tuple(zip(primary_ids, fallback_ids, strict=True))


def _forecaster_slot(akash_model: str, openrouter_model: str) -> AkashPrimaryLlm:
    slug = akash_model if akash_model.startswith("akashml/") else f"akashml/{akash_model}"
    return build_llm_with_openrouter_fallback(
        model=slug, openrouter_model=openrouter_model,
        role=forecaster_role(slug), **REASONING_MODEL_CONFIG,
    )


FORECASTER_LLMS: list[GeneralLlm] = [
    _forecaster_slot(primary, fallback) for primary, fallback in _roster_pairs()
]


def _forecaster_display_name(llm: GeneralLlm) -> str:
    """Short label for a forecaster — strip the routing and vendor prefixes."""
    return llm.model.rsplit("/", 1)[-1]


FORECASTER_MODEL_NAMES = [_forecaster_display_name(llm) for llm in FORECASTER_LLMS]

_UTILITY_PRIMARY = (os.getenv("AKASHML_UTILITY_MODEL") or "openai/gpt-oss-120b").strip()
_UTILITY_FALLBACK = (os.getenv("OPENROUTER_UTILITY_MODEL") or "openai/gpt-oss-120b").strip()
_UTILITY_SLUG = _UTILITY_PRIMARY if _UTILITY_PRIMARY.startswith("akashml/") else f"akashml/{_UTILITY_PRIMARY}"


def _utility(role: str, **overrides: Any) -> AkashPrimaryLlm:
    return build_llm_with_openrouter_fallback(
        model=_UTILITY_SLUG, openrouter_model=_UTILITY_FALLBACK,
        role=role, **{**UTILITY_MODEL_CONFIG, **overrides},
    )


SUMMARIZER_LLM: GeneralLlm = _utility("summarizer")
PARSER_LLM: GeneralLlm = _utility("parser")
RESEARCHER_LLM = SUMMARIZER_LLM
DISAGREEMENT_ANALYZER_LLM: GeneralLlm = _utility("crux_analyzer")

# Stacking remains disabled in the provided Mantic environment. If enabled,
# each stacker has provider failover; the existing aggregation fallback remains.
STACKER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    model="akashml/" + DEFAULT_MODEL_PAIRS[0][0],
    openrouter_model=DEFAULT_MODEL_PAIRS[0][1], role="stacker",
    **{**REASONING_MODEL_CONFIG, "timeout": min(REASONING_MODEL_CONFIG["timeout"], 180)},
)
STACKER_FALLBACK_LLM: GeneralLlm = _utility("stacker_fallback", timeout=120)

# apply_fix.py redirects ONLY the prediction-market provider's factory import
# to akash_router. Native web search, Sonar, Gemini grounding and other research
# clients keep their original imports and capabilities.
MARKET_RANKER_LLM_CONFIG: dict[str, Any] = {
    "model": _UTILITY_SLUG, "openrouter_model": _UTILITY_FALLBACK,
    "role": "market_ranker", **UTILITY_MODEL_CONFIG, "timeout": 90,
}
MARKET_QUERY_AUTHOR_LLM_CONFIG: dict[str, Any] = {
    "model": _UTILITY_SLUG, "openrouter_model": _UTILITY_FALLBACK,
    "role": "market_query_author", **UTILITY_MODEL_CONFIG, "timeout": 45,
}
