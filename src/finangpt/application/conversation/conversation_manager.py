"""Application service that manages conversation persistence and context."""

from __future__ import annotations

from typing import TYPE_CHECKING

from finangpt.application.conversation.context_builder import ContextBuilder, ConversationContext
from finangpt.infrastructure.conversation.sqlite_repository import SQLiteConversationRepository

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from finangpt.application.analysis.orchestrator import AnalysisResult

__all__ = ["ConversationManager"]


class ConversationManager:
    def __init__(
        self,
        conversation_repo: SQLiteConversationRepository,
        context_builder: ContextBuilder,
        history_limit: int = 10,
    ) -> None:
        self._repo = conversation_repo
        self._context_builder = context_builder
        self._history_limit = max(1, history_limit)

    def get_context(self, conversation_id: str) -> ConversationContext:
        history = self._repo.get_history(conversation_id, limit=self._history_limit)
        return self._context_builder.build_context(history)

    def record_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        self._repo.add_turn(conversation_id, role, content, metadata)

    def add_exchange(self, conversation_id: str, question: str, result: "AnalysisResult") -> None:
        metadata = {
            "question": question,
            "insights": [insight.finding for insight in result.insights],
            "viz_hints": [hint.chart_type for hint in result.visualization_hints],
        }
        self.record_turn(conversation_id, "assistant", result.answer, metadata)
