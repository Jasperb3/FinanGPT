from __future__ import annotations

from datetime import datetime

from finangpt.application.conversation.context_builder import ContextBuilder
from finangpt.infrastructure.conversation.sqlite_repository import ConversationTurn


class FakeLLM:
    def __init__(self, response: str = "Summary") -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def make_turn(role: str, content: str) -> ConversationTurn:
    return ConversationTurn(
        conversation_id="conv",
        role=role,
        content=content,
        metadata=None,
        created_at=datetime.now(),
    )


def test_context_builder_extracts_tickers_and_metrics():
    llm = FakeLLM("Conversation summary")
    builder = ContextBuilder(llm_service=llm)
    history = [
        make_turn("user", "Let's discuss AAPL revenue growth"),
        make_turn("assistant", "AAPL revenue is rising"),
        make_turn("user", "Compare MSFT EPS to TSLA margins"),
    ]

    context = builder.build_context(history)

    assert context.summary == "Conversation summary"
    assert set(context.referenced_tickers) >= {"AAPL", "MSFT", "TSLA"}
    assert "revenue" in context.referenced_metrics
    assert llm.prompts  # summary invoked
