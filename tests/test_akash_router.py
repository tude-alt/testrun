"""Offline contract tests for direct AkashML to personal OpenRouter failover.

Run from the repository root with::

    python -m unittest discover -s tests -p 'test_akash_router.py' -v

The installed forecasting-tools implementation is real. Every provider call is
intercepted at GeneralLlm's documented testing seam, so no keys, network access,
inference credits, or tournament submissions are needed.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
import unittest
from typing import Any
from unittest.mock import patch

# Keep the framework's import-time price-catalog lookup offline for this suite.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

import httpx
import openai
from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse
from forecasting_tools.ai_models.model_tracker import ModelTracker
from forecasting_tools.ai_models.resource_managers.hard_limit_manager import (
    HardLimitExceededError,
)
from pydantic import BaseModel

from metaculus_bot.akash_router import AkashPrimaryLlm, ModelRoutingError


AKASH_MODEL = "zai-org/offline-primary"
OPENROUTER_MODEL = "z-ai/offline-fallback"
AKASH_KEY = "offline-test-akash-key-not-a-real-secret"
OPENROUTER_KEY = "offline-test-personal-openrouter-key-not-a-real-secret"
DONATED_KEY = "offline-test-donated-key-must-never-be-used"


def response(text: str = "forecast result", *, model: str = AKASH_MODEL) -> TextTokenCostResponse:
    return TextTokenCostResponse(
        data=text,
        prompt_tokens_used=20,
        completion_tokens_used=8,
        total_tokens_used=28,
        model=model,
        cost=0.00001,
    )


def status_error(status: int, message: str = "Synthetic provider error") -> openai.APIStatusError:
    request = httpx.Request("POST", "https://offline-provider.example/v1/chat/completions")
    reply = httpx.Response(status, request=request, json={"error": {"message": message}})
    return openai.APIStatusError(message, response=reply, body=reply.json())


def record_call(delegate: GeneralLlm, prompt: Any) -> dict[str, Any]:
    return {
        **delegate.litellm_kwargs,
        "delegate_model": delegate.model,
        "allowed_tries": delegate.allowed_tries,
        "prompt": prompt,
    }


class ParsedForecast(BaseModel):
    probability: float
    rationale: str


class AkashRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.environ = patch.dict(
            os.environ,
            {
                "AKASHML_API_KEY": AKASH_KEY,
                "OPENROUTER_API_KEY": OPENROUTER_KEY,
                "OAI_ANTH_OPENROUTER_KEY": DONATED_KEY,
                "DONATED_OPENROUTER_KEY_ENABLED": "false",
                "GEMINI_USE_DONATED_OPENROUTER_KEY": "false",
            },
            clear=True,
        )
        self.environ.start()
        self.addCleanup(self.environ.stop)
        # Fake model IDs deliberately have no pricing entry. Silence that unrelated
        # framework warning; provider-routing log records remain visible to tests.
        tracking_warning = patch.object(ModelTracker, "give_cost_tracking_warning_if_needed")
        tracking_warning.start()
        self.addCleanup(tracking_warning.stop)

    def make_llm(self, **overrides: Any) -> AkashPrimaryLlm:
        return AkashPrimaryLlm(
            **{
                "akash_model": AKASH_MODEL,
                "openrouter_model": OPENROUTER_MODEL,
                "role": "forecaster:offline-test",
                "timeout": 2.0,
                "primary_timeout_fraction": 0.6,
                "max_tokens": 256,
                "temperature": None,
                "allowed_tries": 1,
                **overrides,
            }
        )

    async def test_primary_success_does_not_call_fallback(self) -> None:
        calls: list[dict[str, Any]] = []

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            return response("primary succeeded")

        llm = self.make_llm()
        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            result = await llm.invoke("forecast")

        self.assertEqual(result, "primary succeeded")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model"], f"openai/{AKASH_MODEL}")
        self.assertEqual(calls[0]["base_url"], "https://api.akashml.com/v1")
        self.assertEqual(calls[0]["api_key"], AKASH_KEY)
        self.assertEqual(calls[0]["allowed_tries"], 1)
        self.assertEqual(calls[0]["num_retries"], 0)
        self.assertIsInstance(llm, GeneralLlm)
        self.assertEqual(llm.model, f"akashml/{AKASH_MODEL}")

    async def test_each_recoverable_http_status_switches_endpoint_key_and_model(self) -> None:
        for status in (401, 402, 404, 408, 409, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                calls: list[dict[str, Any]] = []

                async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
                    calls.append(record_call(delegate, prompt))
                    if len(calls) == 1:
                        raise status_error(status)
                    return response("fallback succeeded", model=OPENROUTER_MODEL)

                llm = self.make_llm()
                with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
                    self.assertEqual(await llm.invoke("forecast"), "fallback succeeded")

                self.assertEqual(len(calls), 2, "A primary error must not multiply model attempts")
                self.assertEqual(calls[0]["api_key"], AKASH_KEY)
                self.assertEqual(calls[1]["api_key"], OPENROUTER_KEY)
                self.assertEqual(calls[1]["base_url"], "https://openrouter.ai/api/v1")
                self.assertEqual(calls[1]["model"], f"openrouter/{OPENROUTER_MODEL}")
                self.assertEqual(calls[1]["allowed_tries"], 1)
                self.assertEqual(calls[1]["num_retries"], 0)
                self.assertNotEqual(calls[1]["api_key"], DONATED_KEY)
                self.assertEqual(llm.model, f"akashml/{AKASH_MODEL}")

    async def test_primary_timeout_cancels_primary_and_leaves_time_for_fallback(self) -> None:
        calls: list[dict[str, Any]] = []
        primary_cancelled = asyncio.Event()

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            if len(calls) == 1:
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    primary_cancelled.set()
                    raise
            return response("fallback after timeout", model=OPENROUTER_MODEL)

        llm = self.make_llm(timeout=0.3, primary_timeout_fraction=0.6)
        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            result = await asyncio.wait_for(llm.invoke("forecast"), timeout=1.5)

        self.assertEqual(result, "fallback after timeout")
        self.assertTrue(primary_cancelled.is_set())
        self.assertEqual(len(calls), 2)
        self.assertGreater(calls[1]["timeout"], 0)
        self.assertLess(calls[0]["timeout"], 0.3)
        self.assertLessEqual(calls[1]["timeout"], 0.3)

    async def test_transport_failure_uses_fallback(self) -> None:
        calls: list[dict[str, Any]] = []

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            if len(calls) == 1:
                raise httpx.ConnectError("Synthetic connection failure")
            return response("connection recovered", model=OPENROUTER_MODEL)

        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            self.assertEqual(await self.make_llm().invoke("forecast"), "connection recovered")
        self.assertEqual(len(calls), 2)

    async def test_known_empty_provider_error_uses_fallback(self) -> None:
        calls: list[dict[str, Any]] = []

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            if len(calls) == 1:
                # Exact error prefix raised by forecasting-tools 0.2.92.
                raise RuntimeError("LLM answer is an empty string. The model was offline")
            return response("nonempty fallback", model=OPENROUTER_MODEL)

        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            self.assertEqual(await self.make_llm().invoke("forecast"), "nonempty fallback")
        self.assertEqual(len(calls), 2)

    async def test_missing_primary_key_uses_personal_fallback(self) -> None:
        del os.environ["AKASHML_API_KEY"]
        calls: list[dict[str, Any]] = []

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            return response("fallback only", model=OPENROUTER_MODEL)

        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            self.assertEqual(await self.make_llm().invoke("forecast"), "fallback only")

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model"], f"openrouter/{OPENROUTER_MODEL}")
        self.assertEqual(calls[0]["api_key"], OPENROUTER_KEY)

    async def test_missing_fallback_key_does_not_prevent_primary_success(self) -> None:
        del os.environ["OPENROUTER_API_KEY"]
        calls: list[dict[str, Any]] = []

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            return response("primary without fallback credentials")

        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            self.assertEqual(await self.make_llm().invoke("forecast"), "primary without fallback credentials")
        self.assertEqual(len(calls), 1)

    async def test_both_keys_missing_fail_without_contacting_provider(self) -> None:
        del os.environ["AKASHML_API_KEY"]
        del os.environ["OPENROUTER_API_KEY"]
        calls: list[dict[str, Any]] = []

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            raise AssertionError("No provider should be called without its key")

        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            with self.assertRaises(ModelRoutingError):
                await self.make_llm().invoke("forecast")
        self.assertEqual(calls, [])

    async def test_primary_failure_with_missing_fallback_key_is_loud(self) -> None:
        del os.environ["OPENROUTER_API_KEY"]
        calls: list[dict[str, Any]] = []

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            raise status_error(503)

        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            with self.assertRaises(ModelRoutingError):
                await self.make_llm().invoke("forecast")
        self.assertEqual(len(calls), 1)

    async def test_credentials_are_loaded_at_call_time_and_never_serialized(self) -> None:
        del os.environ["AKASHML_API_KEY"]
        del os.environ["OPENROUTER_API_KEY"]
        llm = self.make_llm()
        calls: list[dict[str, Any]] = []

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            return response()

        os.environ["AKASHML_API_KEY"] = AKASH_KEY
        os.environ["OPENROUTER_API_KEY"] = OPENROUTER_KEY
        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            await llm.invoke("first call")
            os.environ["AKASHML_API_KEY"] = "offline-rotated-akash-key"
            await llm.invoke("second call")

        self.assertEqual(calls[0]["api_key"], AKASH_KEY)
        self.assertEqual(calls[1]["api_key"], "offline-rotated-akash-key")
        saved_configuration = json.dumps(llm.to_dict(), default=str)
        for secret in (AKASH_KEY, OPENROUTER_KEY, DONATED_KEY, "offline-rotated-akash-key"):
            self.assertNotIn(secret, saved_configuration)
        self.assertNotIn("api_key", llm.litellm_kwargs)

    async def test_explicit_provider_base_url_overrides_remain_separate(self) -> None:
        os.environ["AKASHML_API_BASE_URL"] = "https://akash-proxy.example/v1"
        os.environ["OPENROUTER_API_BASE_URL"] = "https://openrouter-proxy.example/api/v1"
        calls: list[dict[str, Any]] = []

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            if len(calls) == 1:
                raise status_error(503)
            return response(model=OPENROUTER_MODEL)

        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            await self.make_llm().invoke("forecast")
        self.assertEqual(calls[0]["base_url"], os.environ["AKASHML_API_BASE_URL"])
        self.assertEqual(calls[1]["base_url"], os.environ["OPENROUTER_API_BASE_URL"])
        self.assertEqual(calls[0]["api_key"], AKASH_KEY)
        self.assertEqual(calls[1]["api_key"], OPENROUTER_KEY)

    async def test_both_provider_failures_raise_sanitized_error_and_logs(self) -> None:
        calls: list[dict[str, Any]] = []

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            # An upstream error can echo request credentials. They must not reach
            # a public Actions log, including an uncaught exception traceback.
            raise status_error(503, f"Upstream leaked {AKASH_KEY} and {OPENROUTER_KEY}")

        llm = self.make_llm()
        rendered_exception = ""
        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            with self.assertLogs("metaculus_bot.akash_router", level="INFO") as captured:
                try:
                    await llm.invoke("forecast")
                except ModelRoutingError as error:
                    rendered_exception = "".join(traceback.format_exception(error))
                else:
                    self.fail("Both failed providers must produce ModelRoutingError")

        self.assertEqual(len(calls), 2)
        self.assertTrue(rendered_exception)
        visible_output = rendered_exception + "\n".join(captured.output) + json.dumps(llm.to_dict(), default=str)
        for secret in (AKASH_KEY, OPENROUTER_KEY, DONATED_KEY):
            self.assertNotIn(secret, visible_output)

    async def test_unrelated_programming_errors_propagate_without_fallback(self) -> None:
        for error in (TypeError("bug in request construction"), ValueError("invalid internal state"), RuntimeError("bug")):
            with self.subTest(error=type(error).__name__):
                calls: list[dict[str, Any]] = []

                async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
                    calls.append(record_call(delegate, prompt))
                    raise error

                with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
                    with self.assertRaises(type(error)) as caught:
                        await self.make_llm().invoke("forecast")
                self.assertIs(caught.exception, error)
                self.assertEqual(len(calls), 1)

    async def test_bad_request_propagates_instead_of_spending_on_fallback(self) -> None:
        calls: list[dict[str, Any]] = []
        error = status_error(400, "Invalid request shape")

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            raise error

        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            with self.assertRaises(openai.APIStatusError) as caught:
                await self.make_llm().invoke("forecast")
        self.assertIs(caught.exception, error)
        self.assertEqual(len(calls), 1)

    async def test_monetary_budget_exception_propagates_without_fallback(self) -> None:
        calls: list[dict[str, Any]] = []
        error = HardLimitExceededError("Offline monetary budget exhausted")

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            raise error

        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            with self.assertRaises(HardLimitExceededError) as caught:
                await self.make_llm().invoke("forecast")
        self.assertIs(caught.exception, error)
        self.assertEqual(len(calls), 1)

    async def test_outer_cancellation_never_starts_fallback(self) -> None:
        calls: list[dict[str, Any]] = []
        provider_started = asyncio.Event()
        provider_cancelled = asyncio.Event()

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            provider_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                provider_cancelled.set()
                raise
            return response()

        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            task = asyncio.create_task(self.make_llm().invoke("forecast"))
            await asyncio.wait_for(provider_started.wait(), timeout=1.0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertTrue(provider_cancelled.is_set())
        self.assertEqual(len(calls), 1)

    async def test_inherited_structured_output_parser_works_after_failover(self) -> None:
        calls: list[dict[str, Any]] = []

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            if len(calls) == 1:
                raise status_error(503)
            return response('{"probability": 0.62, "rationale": "Verified offline"}', model=OPENROUTER_MODEL)

        llm = self.make_llm()
        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            parsed = await llm.invoke_and_return_verified_type(
                "Return the forecast as JSON", ParsedForecast, allowed_invoke_tries_for_failed_output=1
            )

        self.assertIsInstance(parsed, ParsedForecast)
        self.assertEqual(parsed.probability, 0.62)
        self.assertEqual(parsed.rationale, "Verified offline")
        self.assertEqual(len(calls), 2)

    async def test_message_payload_is_preserved_between_providers(self) -> None:
        calls: list[dict[str, Any]] = []
        messages = [{"role": "system", "content": "Be calibrated"}, {"role": "user", "content": "Forecast"}]

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            if len(calls) == 1:
                raise status_error(429)
            return response(model=OPENROUTER_MODEL)

        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            await self.make_llm().invoke(messages)

        self.assertEqual(calls[0]["prompt"], messages)
        self.assertEqual(calls[1]["prompt"], messages)

    async def test_shared_wrapper_keeps_identity_during_concurrent_primary_and_fallback_calls(self) -> None:
        calls: list[dict[str, Any]] = []

        async def provider(delegate: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
            calls.append(record_call(delegate, prompt))
            if delegate.model.startswith("openai/"):
                if prompt == "needs fallback":
                    raise status_error(503)
                await asyncio.sleep(0.01)
                return response("primary question result")
            return response("fallback question result", model=OPENROUTER_MODEL)

        llm = self.make_llm()
        with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new=provider):
            result = await asyncio.gather(llm.invoke("primary works"), llm.invoke("needs fallback"))

        self.assertEqual(result, ["primary question result", "fallback question result"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(llm.model, f"akashml/{AKASH_MODEL}")
        self.assertEqual(llm.akash_model, AKASH_MODEL)
        self.assertEqual(llm.openrouter_model, OPENROUTER_MODEL)


if __name__ == "__main__":
    unittest.main()
