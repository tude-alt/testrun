"""Direct Perplexity research with bounded Nimbleway and You.com fallbacks."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import aiohttp
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    NIMBLE_API_KEY_ENV,
    NIMBLE_SEARCH_API_URL,
    PERPLEXITY_AGENT_API_URL,
    PERPLEXITY_API_KEY_ENV,
    PERPLEXITY_RESEARCH_MODEL,
    WEB_SEARCH_CHAIN_WALL_TIMEOUT,
    WEB_SEARCH_REQUEST_TIMEOUT,
    WEB_SEARCH_RETRY_MAX_ATTEMPTS,
    YOU_SEARCH_API_URL,
    YDC_API_KEY_ENV,
)
from metaculus_bot.research.http_fetch import build_session
from metaculus_bot.prompts import OUTSIDE_VENUE_MARKET_ODDS_POLICY
from metaculus_bot.research.provider_diagnostics import record_provider_detail
from metaculus_bot.research.raw_log import record_raw_research

logger = logging.getLogger(__name__)

_PROVIDER_ORDER = ("perplexity", "nimble", "you")
_PERMANENT_FAILURES: set[str] = set()
_ZERO_QUOTA_TERMS = ("quota is 0", "zero quota", "no credits", "insufficient credits", "billing disabled")
_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})


class WebSearchUnavailable(RuntimeError):
    """All configured web-search providers failed; message contains only redacted causes."""


class WebSearchRouteError(Exception):
    def __init__(self, provider: str, reason: str, *, permanent: bool, retry_after: float | None = None) -> None:
        super().__init__(f"{provider}:{reason}")
        self.provider = provider
        self.reason = reason
        self.permanent = permanent
        self.retry_after = retry_after


@dataclass(frozen=True)
class WebSearchResult:
    provider: str
    text: str
    raw: dict[str, Any]
    source_count: int
    cost_usd: str


def configured_web_search_provider(start_at: str = "perplexity") -> str | None:
    for provider in _PROVIDER_ORDER[_PROVIDER_ORDER.index(start_at) :]:
        if _provider_key(provider):
            return provider
    return None


def _provider_key(provider: str) -> str | None:
    env_name = {
        "perplexity": PERPLEXITY_API_KEY_ENV,
        "nimble": NIMBLE_API_KEY_ENV,
        "you": YDC_API_KEY_ENV,
    }[provider]
    return os.getenv(env_name) or None


def _retry_after(headers: Any) -> float | None:
    value = headers.get("Retry-After") if headers is not None else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            parsed = parsedate_to_datetime(str(value))
            return max(0.0, (parsed - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _error_detail(body: str) -> str:
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return body[:500].lower()
    if isinstance(parsed, dict):
        for key in ("message", "detail", "error", "code"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value[:500].lower()
            if isinstance(value, dict):
                for nested in ("message", "detail", "code"):
                    if isinstance(value.get(nested), str):
                        return value[nested][:500].lower()
    return ""


def _classify_http_error(provider: str, status: int, detail: str, headers: Any) -> WebSearchRouteError:
    if any(term in detail for term in _ZERO_QUOTA_TERMS) or (status == 429 and "quota" in detail):
        return WebSearchRouteError(provider, "zero_quota", permanent=True)
    if status in (401, 403):
        return WebSearchRouteError(provider, "authentication_or_permission", permanent=True)
    if status in (400, 404, 422):
        return WebSearchRouteError(provider, "configuration_or_endpoint", permanent=True)
    if status == 402:
        return WebSearchRouteError(provider, "account_quota_or_billing", permanent=True)
    if status in _TRANSIENT_STATUSES:
        return WebSearchRouteError(provider, f"http_{status}", permanent=False, retry_after=_retry_after(headers))
    return WebSearchRouteError(provider, f"http_{status}", permanent=True)


async def _post_json(
    session: aiohttp.ClientSession,
    *,
    provider: str,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    deadline: float,
) -> dict[str, Any]:
    for attempt in range(WEB_SEARCH_RETRY_MAX_ATTEMPTS):
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise WebSearchRouteError(provider, "shared_deadline_exhausted", permanent=False)
        try:
            async with asyncio.timeout(remaining):
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=min(WEB_SEARCH_REQUEST_TIMEOUT, remaining),
                ) as response:
                    body = await response.text()
                    if response.status < 200 or response.status >= 300:
                        error = _classify_http_error(provider, response.status, _error_detail(body), response.headers)
                    else:
                        try:
                            decoded = json.loads(body)
                        except (ValueError, TypeError) as exc:
                            raise WebSearchRouteError(provider, "invalid_json_response", permanent=True) from exc
                        if not isinstance(decoded, dict):
                            raise WebSearchRouteError(provider, "invalid_response_shape", permanent=True)
                        return decoded
            if error.permanent or attempt + 1 >= WEB_SEARCH_RETRY_MAX_ATTEMPTS:
                raise error
            delay = error.retry_after if error.retry_after is not None else random.uniform(0.15, 0.6)
            if delay >= deadline - asyncio.get_running_loop().time():
                raise error
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise
        except WebSearchRouteError:
            raise
        except (TimeoutError, asyncio.TimeoutError, aiohttp.ClientError):
            if attempt + 1 >= WEB_SEARCH_RETRY_MAX_ATTEMPTS:
                raise WebSearchRouteError(provider, "timeout_or_transport_error", permanent=False) from None
            delay = random.uniform(0.15, 0.6)
            if delay >= deadline - asyncio.get_running_loop().time():
                raise WebSearchRouteError(provider, "timeout_or_transport_error", permanent=False) from None
            await asyncio.sleep(delay)
    raise WebSearchRouteError(provider, "retry_budget_exhausted", permanent=False)


def _valid_source(url: Any, title: Any, snippet: Any) -> bool:
    if not isinstance(url, str) or not isinstance(title, str) or not isinstance(snippet, str):
        return False
    parts = urlsplit(url)
    return parts.scheme in {"http", "https"} and bool(parts.netloc and title.strip() and snippet.strip())


def _dedupe_sources(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for source in sources:
        url = source["url"].strip()
        if url not in seen:
            seen.add(url)
            result.append(source)
    return result


def _render_sources(provider: str, sources: list[dict[str, str]], *, answer: str = "") -> str:
    if not sources:
        raise WebSearchRouteError(provider, "empty_or_unusable_results", permanent=False)
    lines = [answer.strip(), "", f"Retrieved at: {datetime.now(UTC).isoformat()}", ""]
    for index, source in enumerate(sources, 1):
        published = f" | published: {source['published_at']}" if source.get("published_at") else ""
        lines.extend(
            [
                f"{index}. [{source['title']}]({source['url']}){published}",
                f"   {source['snippet']}",
            ]
        )
    return "\n".join(lines).strip()


def _perplexity_result(raw: dict[str, Any]) -> WebSearchResult:
    answer_parts: list[str] = []
    source_records: list[dict[str, str]] = []
    outputs = raw.get("output")
    if isinstance(outputs, list):
        for output in outputs:
            if not isinstance(output, dict) or output.get("type") != "message":
                continue
            content = output.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "output_text":
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    answer_parts.append(text.strip())
                annotations = part.get("annotations")
                if isinstance(annotations, list):
                    for annotation in annotations:
                        if not isinstance(annotation, dict):
                            continue
                        url = annotation.get("url")
                        title = annotation.get("title") or url
                        if _valid_source(url, title, title):
                            source_records.append({"url": url, "title": title, "snippet": title})
    sources = _dedupe_sources(source_records)
    text = _render_sources("perplexity", sources, answer="\n\n".join(answer_parts))
    usage = raw.get("usage")
    usage_cost = usage.get("cost") if isinstance(usage, dict) else None
    total_cost = usage_cost.get("total_cost") if isinstance(usage_cost, dict) else None
    cost = f"{total_cost:.8f}" if isinstance(total_cost, (int, float)) else "unknown"
    return WebSearchResult("perplexity", text, raw, len(sources), cost)


def _nimble_result(raw: dict[str, Any]) -> WebSearchResult:
    records = raw.get("results")
    if not isinstance(records, list):
        raise WebSearchRouteError("nimble", "invalid_response_shape", permanent=True)
    sources: list[dict[str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        additional = record.get("additional_data") if isinstance(record.get("additional_data"), dict) else {}
        snippet = record.get("content") or record.get("description")
        title, url = record.get("title"), record.get("url")
        if _valid_source(url, title, snippet):
            source = {"url": url, "title": title, "snippet": snippet}
            published = additional.get("publish_date") or additional.get("published_at")
            if isinstance(published, str) and published.strip():
                source["published_at"] = published.strip()
            sources.append(source)
    sources = _dedupe_sources(sources)
    text = _render_sources("nimble", sources)
    return WebSearchResult("nimble", text, raw, len(sources), "unknown")


def _you_result(raw: dict[str, Any]) -> WebSearchResult:
    results = raw.get("results")
    if not isinstance(results, dict):
        raise WebSearchRouteError("you", "invalid_response_shape", permanent=True)
    sources: list[dict[str, str]] = []
    for category in ("web", "news"):
        records = results.get(category)
        if isinstance(records, dict):
            records = records.get("results")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            contents = record.get("contents") if isinstance(record.get("contents"), dict) else {}
            highlights = contents.get("highlights")
            if isinstance(highlights, list):
                snippet = " ".join(str(item) for item in highlights if isinstance(item, str) and item.strip())
            else:
                snippets = record.get("snippets")
                snippet = " ".join(str(item) for item in snippets if isinstance(item, str) and item.strip()) if isinstance(snippets, list) else None
                snippet = snippet or record.get("description") or contents.get("markdown") or contents.get("html")
            title, url = record.get("title"), record.get("url")
            if _valid_source(url, title, snippet):
                source = {"url": url, "title": title, "snippet": snippet}
                published = record.get("page_age")
                if isinstance(published, str) and published.strip():
                    source["published_at"] = published.strip()
                sources.append(source)
    sources = _dedupe_sources(sources)
    text = _render_sources("you", sources)
    return WebSearchResult("you", text, raw, len(sources), "unknown")


async def _call_route(
    session: aiohttp.ClientSession,
    provider: str,
    query: str,
    *,
    deadline: float,
    is_benchmarking: bool,
) -> WebSearchResult:
    key = _provider_key(provider)
    if not key:
        raise WebSearchRouteError(provider, "credential_missing", permanent=True)
    if provider == "perplexity":
        raw = await _post_json(
            session,
            provider=provider,
            url=PERPLEXITY_AGENT_API_URL,
            payload={
                "model": PERPLEXITY_RESEARCH_MODEL,
                "input": query,
                "instructions": (
                    "Research the question using web search. Return a concise factual summary with citations. "
                    "Do not make a forecast."
                    if is_benchmarking
                    else (
                        "Research the question using web search. Return a concise factual summary with citations. "
                        "Do not make a forecast. Cover relevant prediction markets outside the venues already "
                        f"covered by our live market snapshot: {OUTSIDE_VENUE_MARKET_ODDS_POLICY}"
                    )
                ),
                "tools": [{"type": "web_search"}],
                "max_output_tokens": 3000,
            },
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            deadline=deadline,
        )
        return _perplexity_result(raw)
    if provider == "nimble":
        raw = await _post_json(
            session,
            provider=provider,
            url=NIMBLE_SEARCH_API_URL,
            payload={"query": query, "search_depth": "lite", "max_results": 8, "output_format": "markdown"},
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            deadline=deadline,
        )
        return _nimble_result(raw)
    if provider == "you":
        raw = await _post_json(
            session,
            provider=provider,
            url=YOU_SEARCH_API_URL,
            payload={"query": query, "count": 8},
            headers={"X-API-Key": key, "Content-Type": "application/json"},
            deadline=deadline,
        )
        return _you_result(raw)
    raise ValueError(f"Unsupported web search provider {provider!r}")


async def run_web_search_chain(
    question_text: str,
    *,
    qid: int | None = None,
    start_at: str = "perplexity",
    diagnostics_name: str | None = None,
    is_benchmarking: bool = False,
) -> tuple[str, str | None]:
    """Run Perplexity → Nimbleway → You.com under one wall deadline."""
    if start_at not in _PROVIDER_ORDER:
        raise ValueError(f"Unknown web search start provider: {start_at}")
    chain = _PROVIDER_ORDER[_PROVIDER_ORDER.index(start_at) :]
    first_configured = configured_web_search_provider(start_at)
    attempted: dict[str, str] = {}
    started = time.monotonic()
    deadline = asyncio.get_running_loop().time() + WEB_SEARCH_CHAIN_WALL_TIMEOUT
    registry_name = diagnostics_name or first_configured or start_at

    async with build_session(timeout_s=WEB_SEARCH_REQUEST_TIMEOUT, headers={"Accept": "application/json"}) as session:
        for provider in chain:
            if not _provider_key(provider):
                attempted[provider] = "not_configured"
                continue
            if provider in _PERMANENT_FAILURES:
                attempted[provider] = "disabled_after_permanent_failure"
                continue
            if asyncio.get_running_loop().time() >= deadline:
                attempted[provider] = "shared_deadline_exhausted"
                break
            route_started = time.monotonic()
            try:
                result = await _call_route(
                    session,
                    provider,
                    question_text,
                    deadline=deadline,
                    is_benchmarking=is_benchmarking,
                )
            except asyncio.CancelledError:
                raise
            except WebSearchRouteError as exc:
                attempted[provider] = exc.reason
                if exc.permanent:
                    _PERMANENT_FAILURES.add(provider)
                logger.warning(
                    "WEB_RESEARCH_ROUTE: role=primary requested_model=%s transport=%s observed_provider=unknown "
                    "fallback_reason=%s latency_ms=%d cost_usd=unknown status=failed",
                    PERPLEXITY_RESEARCH_MODEL if provider == "perplexity" else "search",
                    provider,
                    exc.reason,
                    round((time.monotonic() - route_started) * 1000),
                )
                continue
            except Exception as exc:  # noqa: BLE001  # malformed upstream response only disables its current route
                attempted[provider] = "invalid_response"
                _PERMANENT_FAILURES.add(provider)
                logger.warning(
                    "WEB_RESEARCH_ROUTE: role=primary transport=%s observed_provider=unknown "
                    "fallback_reason=invalid_response latency_ms=%d cost_usd=unknown status=failed error_type=%s",
                    provider,
                    round((time.monotonic() - route_started) * 1000),
                    type(exc).__name__,
                )
                continue

            attempted[provider] = f"ok({result.source_count})"
            record_raw_research(qid=qid, provider=provider, payload=result.raw)
            fallback_reason = next(
                (
                    f"{name}:{reason}"
                    for name, reason in attempted.items()
                    if name != provider and reason not in {"not_configured"}
                ),
                "none",
            )
            logger.info(
                "WEB_RESEARCH_ROUTE: role=primary requested_model=%s transport=%s observed_provider=%s "
                "fallback_reason=%s latency_ms=%d cost_usd=%s status=%s",
                PERPLEXITY_RESEARCH_MODEL if provider == "perplexity" else "search",
                provider,
                result.provider,
                fallback_reason,
                round((time.monotonic() - started) * 1000),
                result.cost_usd,
                "ok" if provider == first_configured else "fallback",
            )
            detail = {
                "sources": attempted.copy(),
                "route": {
                    "requested_provider": first_configured or start_at,
                    "actual_provider": result.provider,
                    "fallback_reason": fallback_reason,
                    "latency_ms": round((time.monotonic() - started) * 1000),
                    "cost_usd": result.cost_usd,
                },
            }
            record_provider_detail(qid, registry_name, detail)
            return result.text, result.provider

    detail = {
        "sources": attempted.copy(),
        "route": {
            "requested_provider": first_configured or start_at,
            "actual_provider": None,
            "fallback_reason": "all_eligible_routes_unavailable",
            "latency_ms": round((time.monotonic() - started) * 1000),
            "cost_usd": "unknown",
        },
    }
    record_provider_detail(qid, registry_name, detail)
    causes = ", ".join(f"{provider}={reason}" for provider, reason in attempted.items()) or "no_credentials_configured"
    raise WebSearchUnavailable(f"all eligible web-search routes unavailable: {causes}")


def web_search_provider(
    start_at: str = "perplexity",
    *,
    diagnostics_name: str | None = None,
    is_benchmarking: bool = False,
):
    async def _fetch(question: MetaculusQuestion) -> str:
        text, _provider = await run_web_search_chain(
            question.question_text,
            qid=getattr(question, "id_of_question", None),
            start_at=start_at,
            diagnostics_name=diagnostics_name,
            is_benchmarking=is_benchmarking,
        )
        return text

    return _fetch
