"""Command-line entry point for the benchmark suite.

Usage::

    python -m app.benchmarks.runner            # deterministic mode
    python -m app.benchmarks.runner --llm      # with the configured provider

Exits non-zero when any case fails, so it can be used as a check in CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.ai.pipeline import QuestionPipeline
from app.ai.providers.openai_compatible import create_provider
from app.benchmarks.evaluator import evaluate, routing_coverage
from app.benchmarks.questions import BENCHMARK_CASES
from app.core.config import get_settings
from app.core.logging import configure_logging


async def _run(use_llm: bool) -> int:
    settings = get_settings()
    provider = None
    if use_llm:
        if not settings.llm_configured:
            print(
                "LLM requested but no API key is configured; running the deterministic "
                "path instead."
            )
        else:
            provider = create_provider()
            print(f"Using LLM provider at {settings.llm_base_url} ({settings.llm_model}).")

    report = await evaluate(
        BENCHMARK_CASES, provider=provider, pipeline=QuestionPipeline(provider=provider)
    )
    print(report.render())

    coverage = routing_coverage()
    if coverage.missing:
        print("\nAnalysis types with no benchmark case:")
        for item in coverage.missing:
            print(f"  - {item}")

    return 0 if not report.failed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Operations Intelligence benchmarks.")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="generate narratives with the configured LLM provider instead of the "
        "deterministic fallback",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the per-case outcomes as JSON on stdout",
    )
    args = parser.parse_args(argv)

    configure_logging(get_settings())
    exit_code = asyncio.run(_run(args.llm))

    if args.json:
        from app.benchmarks.evaluator import evaluate as run_evaluate

        async def _json() -> None:
            provider = create_provider() if args.llm and get_settings().llm_configured else None
            report = await run_evaluate(BENCHMARK_CASES, provider=provider)
            print(json.dumps([_outcome_json(item) for item in report.outcomes], indent=2))

        asyncio.run(_json())

    return exit_code


def _outcome_json(outcome: object) -> dict[str, object]:
    return {
        "case_id": getattr(outcome, "case_id", None),
        "question": getattr(outcome, "question", None),
        "passed": getattr(outcome, "passed", None),
        "failures": list(getattr(outcome, "failures", ())),
        "analysis_type": getattr(outcome, "analysis_type", None),
        "metric": getattr(outcome, "metric", None),
        "period": getattr(outcome, "period", None),
        "comparison_period": getattr(outcome, "comparison_period", None),
        "evidence_count": getattr(outcome, "evidence_count", None),
        "findings_count": getattr(outcome, "findings_count", None),
        "generated_by": getattr(outcome, "generated_by", None),
        "grounded": getattr(outcome, "grounded", None),
        "rejected": getattr(outcome, "rejected", None),
        "duration_ms": getattr(outcome, "duration_ms", None),
    }


if __name__ == "__main__":
    sys.exit(main())
