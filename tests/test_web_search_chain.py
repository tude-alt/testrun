from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from metaculus_bot.research import web_search_chain as chain
from metaculus_bot.research.provider_diagnostics import pop_provider_detail


class EmptySession:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *_args: object) -> None:
        return None


def _result(provider: str, source_count: int = 1) -> chain.WebSearchResult:
    return chain.WebSearchResult(
        provider=provider,
        text=f"## {provider}\n\n[Title](https://example.com/source)\nSnippet text",
        raw={"provider": provider},
        source_count=source_count,
        cost_usd="unknown",
    )


@pytest.fixture(autouse=True)
def clear_permanent_failures() -> None:
    chain._PERMANENT_FAILURES.clear()


@pytest.mark.asyncio
async def test_direct_perplexity_runs_first_and_keeps_source_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-key")
    monkeypatch.setenv("NIMBLE_API_KEY", "nimble-key")
    session_context = EmptySession()
    calls: list[str] = []

    async def call(
        _session: object, provider: str, _query: str, *, deadline: float, is_benchmarking: bool
    ) -> chain.WebSearchResult:
        calls.append(provider)
        assert deadline > 0
        assert is_benchmarking is False
        return _result(provider)

    monkeypatch.setattr(chain, "build_session", lambda **_kwargs: session_context)
    monkeypatch.setattr(chain, "_call_route", call)
    text, provider = await chain.run_web_search_chain("question", qid=7001)

    assert calls == ["perplexity"]
    assert provider == "perplexity"
    assert "Retrieved at:" not in text


@pytest.mark.asyncio
async def test_missing_perplexity_skips_to_nimble_then_you(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setenv("NIMBLE_API_KEY", "nimble-key")
    monkeypatch.setenv("YDC_API_KEY", "you-key")
    session_context = EmptySession()
    calls: list[str] = []

    async def call(
        _session: object, provider: str, _query: str, *, deadline: float, is_benchmarking: bool
    ) -> chain.WebSearchResult:
        calls.append(provider)
        if provider == "nimble":
            raise chain.WebSearchRouteError(provider, "timeout_or_transport_error", permanent=False)
        return _result(provider)

    monkeypatch.setattr(chain, "build_session", lambda **_kwargs: session_context)
    monkeypatch.setattr(chain, "_call_route", call)
    text, provider = await chain.run_web_search_chain("question", qid=7002)

    assert calls == ["nimble", "you"]
    assert provider == "you"
    assert "https://example.com/source" in text
    detail = pop_provider_detail(7002, "nimble")
    assert detail["route"]["actual_provider"] == "you"
    assert detail["sources"]["nimble"] == "timeout_or_transport_error"
    assert detail["sources"]["you"] == "ok(1)"


@pytest.mark.asyncio
async def test_permanent_perplexity_failure_is_not_retried_for_later_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-key")
    monkeypatch.setenv("NIMBLE_API_KEY", "nimble-key")
    monkeypatch.setattr(chain, "build_session", lambda **_kwargs: EmptySession())
    calls: list[str] = []

    async def call(
        _session: object, provider: str, _query: str, *, deadline: float, is_benchmarking: bool
    ) -> chain.WebSearchResult:
        calls.append(provider)
        if provider == "perplexity":
            raise chain.WebSearchRouteError(provider, "authentication_or_permission", permanent=True)
        return _result(provider)

    monkeypatch.setattr(chain, "_call_route", call)
    assert (await chain.run_web_search_chain("question one", qid=7003))[1] == "nimble"
    assert (await chain.run_web_search_chain("question two", qid=7004))[1] == "nimble"

    assert calls == ["perplexity", "nimble", "nimble"]


def test_perplexity_result_requires_real_cited_sources_and_keeps_citations() -> None:
    raw = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "A sourced answer.",
                        "annotations": [
                            {"type": "url_citation", "title": "Official report", "url": "https://example.gov/report"}
                        ],
                    }
                ],
            }
        ],
        "usage": {"cost": {"total_cost": 0.0021}},
    }
    result = chain._perplexity_result(raw)

    assert result.provider == "perplexity"
    assert result.source_count == 1
    assert "A sourced answer." in result.text
    assert "Official report" in result.text
    assert "https://example.gov/report" in result.text
    assert result.cost_usd == "0.00210000"

    with pytest.raises(chain.WebSearchRouteError, match="empty_or_unusable_results"):
        chain._perplexity_result({"output": [{"type": "message", "content": [{"type": "output_text", "text": "Unsupported answer without citations", "annotations": []}]}]})


def test_nimble_and_you_results_preserve_fields_and_ignore_bad_results() -> None:
    nimble = chain._nimble_result(
        {
            "results": [
                {
                    "title": "Nimble source",
                    "url": "https://example.com/a",
                    "description": "Nimble snippet",
                    "content": "",
                    "additional_data": {"publish_date": "2026-09-28"},
                },
                {"title": "bad source", "url": "javascript:alert(1)", "description": "not a URL"},
            ]
        }
    )
    you = chain._you_result(
        {
            "results": {
                "web": [
                    {
                        "title": "You source",
                        "url": "https://example.org/b",
                        "description": "You snippet",
                        "page_age": "2026-09-27T12:00:00Z",
                    }
                ],
                "news": [],
            }
        }
    )

    assert nimble.source_count == 1
    assert "published: 2026-09-28" in nimble.text
    assert "Nimble snippet" in nimble.text
    assert you.source_count == 1
    assert "published: 2026-09-27T12:00:00Z" in you.text
    assert "https://example.org/b" in you.text


def test_retry_after_accepts_seconds_and_http_dates() -> None:
    assert chain._retry_after({"Retry-After": "4"}) == 4.0
    assert chain._retry_after({"Retry-After": "invalid"}) is None
