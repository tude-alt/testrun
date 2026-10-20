"""Tests for the free-model forecaster lineup used by the ablation runner.

Posture pinned by these tests:

* Lineup is 4 free models. ``gpt-oss-120b:free`` was removed (task #16,
  donated-key wrapper 404'd). ``glm-4.5-air:free`` was removed
  (Phase A.3 Package 3b, 2026-05-14, after qid 43171 hallucinated
  partial-week TSA data with sigma=13K vs ensemble sigma ~965K).
* The FREE forecaster + parser are constructed as plain ``GeneralLlm`` (no
  ``api_key`` set), so litellm reads ``OPENROUTER_API_KEY`` from env at
  invoke time — ``:free`` models bypass the donated key because their
  providers aren't in ``DONATED_KEY_PROVIDERS`` (donated key 404s).
* The PROD lineup routes through the donated-key wrapper: ablation IS Metaculus
  work, and its anthropic/openai models are donated-covered, so they bill to the
  donated key with personal-key fallback.
"""

from __future__ import annotations

from forecasting_tools import GeneralLlm


def test_free_forecaster_models_count_is_two() -> None:
    """Only Qwen and DeepSeek free tiers are included in the OpenRouter free lineup."""
    from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS

    assert len(FREE_FORECASTER_MODELS) == 2


def test_free_lineup_is_qwen_and_deepseek_only() -> None:
    from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS

    assert all("qwen" in m or "deepseek" in m for m in FREE_FORECASTER_MODELS)
    assert all(model.endswith(":free") for model in FREE_FORECASTER_MODELS)
    assert all(model.startswith("openrouter/") for model in FREE_FORECASTER_MODELS)
    assert not any("gemma" in m or "minimax" in m or "nemotron" in m for m in FREE_FORECASTER_MODELS)


def test_build_free_forecaster_llms_returns_two_plain_general_llms() -> None:
    """Builder constructs plain GeneralLlm (no api_key, no FallbackOpenRouterLlm wrap)."""
    from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS, build_free_forecaster_llms
    from metaculus_bot.fallback_openrouter import FallbackOpenRouterLlm

    llms = build_free_forecaster_llms()
    assert len(llms) == 2
    for llm, expected_model in zip(llms, FREE_FORECASTER_MODELS, strict=False):
        assert isinstance(llm, GeneralLlm)
        assert not isinstance(llm, FallbackOpenRouterLlm), (
            f"{expected_model} must be plain GeneralLlm (no donated-key wrap)"
        )
        assert llm.model == expected_model


def test_build_free_parser_llm_uses_qwen_free_variant() -> None:
    """The free parser stays in the same Qwen/DeepSeek tier and the lineup is intentionally restricted."""
    from metaculus_bot.ablation.forecaster_lineup import FREE_PARSER_MODEL, build_free_parser_llm
    from metaculus_bot.fallback_openrouter import FallbackOpenRouterLlm

    parser = build_free_parser_llm()
    assert isinstance(parser, GeneralLlm)
    assert not isinstance(parser, FallbackOpenRouterLlm)
    assert parser.model == FREE_PARSER_MODEL
    assert "qwen" in FREE_PARSER_MODEL or "deepseek" in FREE_PARSER_MODEL


def test_build_prod_forecaster_llms_routes_via_donated_wrapper(monkeypatch) -> None:
    """Prod ensemble bills to the donated key: each model is wrapped in
    FallbackOpenRouterLlm (donated primary -> personal fallback). Ablation IS
    Metaculus work, and these anthropic/openai models are donated-covered.

    Requires both keys set + distinct for the wrapper to engage (matches
    ``build_llm_with_openrouter_fallback``); in CI neither is set, so we inject
    distinct placeholders.
    """
    from metaculus_bot.ablation.forecaster_lineup import PROD_FORECASTER_MODELS, build_prod_forecaster_llms
    from metaculus_bot.fallback_openrouter import FallbackOpenRouterLlm

    monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
    monkeypatch.setenv("OPENROUTER_API_KEY", "general")

    llms = build_prod_forecaster_llms()
    assert len(llms) == len(PROD_FORECASTER_MODELS) == 3
    for llm, expected_model in zip(llms, PROD_FORECASTER_MODELS, strict=False):
        assert isinstance(llm, FallbackOpenRouterLlm), (
            f"{expected_model} must route via the donated-key wrapper (Metaculus work -> donated key)"
        )
        assert llm.model == expected_model
