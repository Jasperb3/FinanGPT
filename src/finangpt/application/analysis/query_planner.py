"""LLM-powered query planner for FinanGPT."""

from __future__ import annotations

from typing import Protocol

from finangpt.application.analysis.schemas import QueryPlan
from finangpt.application.conversation.context_builder import ConversationContext
from finangpt.infrastructure.llm.ollama_service import OllamaLLMService

__all__ = ["SchemaProvider", "QueryPlanner"]


class SchemaProvider(Protocol):
    def get_schema_description(self) -> str:  # pragma: no cover - protocol
        ...


class QueryPlanner:
    """Use the LLM to convert a natural language question into a query plan."""

    def __init__(self, llm_service: OllamaLLMService, schema_provider: SchemaProvider) -> None:
        self._llm = llm_service
        self._schema_provider = schema_provider

    def create_plan(self, question: str, context: ConversationContext | dict | None = None) -> QueryPlan:
        schema_description = self._schema_provider.get_schema_description()
        prompt = self._build_prompt(question=question, schema_description=schema_description, context=context)
        return self._llm.generate_structured(prompt, QueryPlan)

    def _build_prompt(
        self,
        question: str,
        schema_description: str,
        context: ConversationContext | dict | None,
    ) -> str:
        if isinstance(context, ConversationContext):
            context_summary = (
                f"Summary: {context.summary}\n"
                f"Tickers: {', '.join(context.referenced_tickers) or 'None'}\n"
                f"Metrics: {', '.join(context.referenced_metrics) or 'None'}"
            )
        elif context:
            context_summary = str(context)
        else:
            context_summary = "No prior context"
        return (
            "You are a senior financial data analyst. "
            "Break the user's question into executable SQL steps.\n"
            f"QUESTION: {question}\n"
            f"CONTEXT: {context_summary}\n"
            "DATABASE SCHEMA:\n"
            f"{schema_description}\n"
            "Respond ONLY with JSON that matches the required schema."
        )
