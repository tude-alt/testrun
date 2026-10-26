"""Offline startup validation. Never makes an inference or publishes a forecast."""

from __future__ import annotations

import argparse
import ast
import importlib
import os
from pathlib import Path
import sys


def check(repo: Path, mantic: bool) -> list[str]:
    errors: list[str] = []
    sys.path.insert(0, str(repo))
    config_loader = importlib.import_module("metaculus_bot.config")
    config_loader.load_environment()

    if mantic:
        for key in ("MANTIC_TOKEN", "AKASHML_API_KEY", "OPENROUTER_API_KEY"):
            if not (os.getenv(key) or "").strip():
                errors.append(f"Missing {key}")
        for key in ("DONATED_OPENROUTER_KEY_ENABLED", "GEMINI_USE_DONATED_OPENROUTER_KEY"):
            if (os.getenv(key) or "").strip().lower() not in {"false", "0", "no", "off"}:
                errors.append(f"{key} must be explicitly false for Mantic")
    if (os.getenv("TS_ANCHOR_CHART_ENABLED") or "false").strip().lower() in {"true", "1", "yes", "on"}:
        errors.append("Keep TS_ANCHOR_CHART_ENABLED=false for this text forecasting roster")

    from metaculus_bot import constants, llm_configs
    from metaculus_bot.akash_router import AkashPrimaryLlm

    roster = llm_configs.FORECASTER_LLMS
    minimum = constants.MIN_FORECASTERS_TO_PUBLISH
    if len(roster) < minimum:
        errors.append(
            f"Configured {len(roster)} forecasters but MIN_FORECASTERS_TO_PUBLISH={minimum}; "
            "configure enough distinct paired models before running"
        )
    for model in roster:
        if not isinstance(model, AkashPrimaryLlm):
            errors.append("Every forecast slot must use AkashPrimaryLlm")
        elif model.timeout >= constants.FORECASTER_SOFT_DEADLINE:
            errors.append(
                f"AKASHML_FORECAST_TIMEOUT={model.timeout:g} must be below "
                f"FORECASTER_SOFT_DEADLINE={constants.FORECASTER_SOFT_DEADLINE:g} "
                "so the OpenRouter fallback has time to finish"
            )

    research = repo / "metaculus_bot" / "research"
    candidates = sorted(research.rglob("prediction_market.py"))
    if len(candidates) != 1:
        errors.append("Expected one prediction_market.py; use apply_fix.py to install and validate the utility routing")
    else:
        tree = ast.parse(candidates[0].read_text(encoding="utf-8"))
        routed = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "metaculus_bot.akash_router"
            and any(alias.name == "build_llm_with_openrouter_fallback" for alias in node.names)
            for node in ast.walk(tree)
        )
        if not routed:
            errors.append("Prediction-market factory still uses the old router; run apply_fix.py")

    print(f"Configured {len(roster)} forecast slots; publication minimum is {minimum}.")
    for model in roster:
        if isinstance(model, AkashPrimaryLlm):
            print(f"  AkashML {model.akash_model} -> personal OpenRouter {model.openrouter_model}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mantic", action="store_true")
    args = parser.parse_args()
    try:
        errors = check(Path(__file__).resolve().parents[1], args.mantic)
    except Exception as error:
        # Import/runtime exceptions may embed credentials. Do not echo a repr.
        print(f"Configuration check could not complete ({type(error).__name__}). Check imports and configuration locally.", file=sys.stderr)
        return 1
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print("Offline configuration check passed. No inference requests or forecasts were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
