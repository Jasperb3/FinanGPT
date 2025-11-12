"""Builds structured conversation context using recent history."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from finangpt.infrastructure.conversation.sqlite_repository import ConversationTurn
from finangpt.infrastructure.llm.ollama_service import OllamaLLMService

__all__ = ["ConversationContext", "ContextBuilder"]


@dataclass(frozen=True)
class ConversationContext:
    summary: str
    referenced_tickers: tuple[str, ...]
    referenced_metrics: tuple[str, ...]
    recent_questions: tuple[str, ...]

    def __str__(self) -> str:  # pragma: no cover - formatting helper
        return (
            f"Summary: {self.summary}\n"
            f"Tickers: {', '.join(self.referenced_tickers) or 'None'}\n"
            f"Metrics: {', '.join(self.referenced_metrics) or 'None'}"
        )


class ContextBuilder:
    _TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")
    _METRIC_KEYWORDS = {
        "revenue",
        "net income",
        "eps",
        "cash flow",
        "margin",
        "ebitda",
        "debt",
        "free cash flow",
        "growth",
        "opex",
        "operating income",
        "gross profit",
    }

    def __init__(self, llm_service: OllamaLLMService, summary_enabled: bool = True) -> None:
        self._llm = llm_service
        self._summary_enabled = summary_enabled

    def build_context(self, history: Sequence[ConversationTurn]) -> ConversationContext:
        if not history:
            return ConversationContext(
                summary="No prior conversation",
                referenced_tickers=(),
                referenced_metrics=(),
                recent_questions=(),
            )

        summary = self._summarize(history) if self._summary_enabled else "Context summary disabled"
        tickers = tuple(sorted(self._extract_tickers(history)))
        metrics = tuple(sorted(self._extract_metrics(history)))
        recent_questions = tuple(
            turn.content
            for turn in history
            if turn.role == "user"
        )[-5:]

        return ConversationContext(
            summary=summary,
            referenced_tickers=tickers,
            referenced_metrics=metrics,
            recent_questions=recent_questions,
        )

    def _summarize(self, history: Sequence[ConversationTurn]) -> str:
        prompt_lines = ["Summarize the following conversation in 2 sentences with bullet insights:" ]
        for turn in history:
            prompt_lines.append(f"{turn.role.upper()}: {turn.content}")
        prompt = "\n".join(prompt_lines)
        try:
            return self._llm.generate_text(prompt)
        except Exception:  # pragma: no cover - fallback path
            return "Summary unavailable"

    def _extract_tickers(self, history: Sequence[ConversationTurn]) -> set[str]:
        tickers: set[str] = set()
        for turn in history:
            tickers.update(
                match.group(0)
                for match in self._TICKER_PATTERN.finditer(turn.content.upper())
                if len(match.group(0)) <= 5 and match.group(0).isalpha()
            )
        return tickers

    def _extract_metrics(self, history: Sequence[ConversationTurn]) -> set[str]:
        metrics: set[str] = set()
        for turn in history:
            content_lower = turn.content.lower()
            for keyword in self._METRIC_KEYWORDS:
                if keyword in content_lower:
                    metrics.add(keyword)
        return metrics
