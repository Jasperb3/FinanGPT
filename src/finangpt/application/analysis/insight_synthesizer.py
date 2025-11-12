"""Convert structured insights into natural language responses."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from finangpt.application.analysis.schemas import Insight
from finangpt.infrastructure.llm.ollama_service import LLMServiceError, OllamaLLMService

__all__ = ["InsightSynthesizer"]


class InsightSynthesizer:
    def __init__(self, llm_service: OllamaLLMService) -> None:
        self._llm = llm_service

    def synthesize(
        self,
        question: str,
        insights: list[Insight],
        data_map: Dict[int, pd.DataFrame],
    ) -> str:
        if not insights:
            return "No insights could be generated from the available data."
        prompt = self._build_prompt(question, insights)
        try:
            return self._llm.generate_text(prompt).strip()
        except LLMServiceError:
            joined = "\n".join(f"- {insight.finding}: {insight.significance}" for insight in insights)
            return f"Key findings for '{question}':\n{joined}"

    def _build_prompt(self, question: str, insights: list[Insight]) -> str:
        insight_lines = "\n".join(
            f"- {insight.finding} ({insight.category}): {insight.significance}" for insight in insights
        )
        return (
            "You are a financial analyst. Summarize the insights below into a concise answer.\n"
            f"QUESTION: {question}\n"
            f"INSIGHTS:\n{insight_lines}\n"
            "Respond with a helpful paragraph under 150 words."
        )
