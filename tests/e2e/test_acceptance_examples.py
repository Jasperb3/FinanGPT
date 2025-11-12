from __future__ import annotations

import pytest

from tests.acceptance.test_analysis_acceptance import _run_acceptance


@pytest.mark.e2e
@pytest.mark.parametrize(
    "question",
    [
        "How has Apple's quarterly revenue growth rate evolved over the past five years compared to Microsoft's?",
        "Compare the 30-day, 90-day, and 1-year price volatility of Tesla and its top three peers since 2020",
    ],
)
def test_acceptance_examples(acceptance_orchestrator, question):
    duration = _run_acceptance(acceptance_orchestrator, question)
    assert duration < 1.0
