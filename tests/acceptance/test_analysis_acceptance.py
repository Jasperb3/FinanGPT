from __future__ import annotations

import time
import pytest

def _run_acceptance(orchestrator, question: str) -> float:
    start = time.perf_counter()
    result = orchestrator.analyze_question(question)
    duration = time.perf_counter() - start
    assert result.answer
    assert result.visualization_hints
    return duration


def test_acceptance_apple_vs_msft_growth(acceptance_orchestrator):
    duration = _run_acceptance(
        acceptance_orchestrator,
        "How has Apple's quarterly revenue growth rate evolved over the past five years compared to Microsoft's?",
    )
    assert duration < 1.0


def test_acceptance_portfolio_net_income_changes(acceptance_orchestrator):
    duration = _run_acceptance(
        acceptance_orchestrator,
        "For each company in my portfolio, what was the year-over-year change in net income?",
    )
    assert duration < 1.0
