from __future__ import annotations

from datetime import datetime

import pandas as pd

from finangpt.application.analysis.orchestrator import AnalysisResult
from finangpt.application.analysis.schemas import QueryPlan, VisualizationHint
from finangpt.application.conversation.context_builder import ContextBuilder
from finangpt.application.conversation.conversation_manager import ConversationManager
from finangpt.infrastructure.conversation.sqlite_repository import SQLiteConversationRepository


class StubLLM:
    def __init__(self, response: str = "Summary") -> None:
        self.response = response

    def generate_text(self, prompt: str) -> str:
        return self.response


def test_conversation_roundtrip(tmp_path):
    repo = SQLiteConversationRepository(str(tmp_path / "conversations.db"))
    builder = ContextBuilder(llm_service=StubLLM())
    manager = ConversationManager(repo, builder, history_limit=5)

    conv_id = "conv-test"
    manager.record_turn(conv_id, "user", "How is AAPL revenue trending?", {"topic": "revenue"})
    manager.record_turn(conv_id, "assistant", "AAPL revenue grew 10%", None)

    context = manager.get_context(conv_id)
    assert "AAPL" in context.referenced_tickers

    plan = QueryPlan(complexity="simple", requires_visualization=False, reasoning="", steps=[])
    result = AnalysisResult(
        question="Second",
        plan=plan,
        insights=[],
        answer="Follow-up answer",
        visualization_hints=[VisualizationHint(chart_type="line", description="", fields=[])],
        data_by_step={},
    )
    manager.add_exchange(conv_id, "Second", result)

    history = repo.get_history(conv_id, limit=10)
    assert len(history) == 3
    assert any(turn.role == "assistant" and "Follow-up" in turn.content for turn in history)
