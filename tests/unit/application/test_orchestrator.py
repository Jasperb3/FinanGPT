from __future__ import annotations

import pandas as pd

from finangpt.application.analysis.orchestrator import AnalysisOrchestrator
from finangpt.application.analysis.schemas import Insight, QueryPlan, QueryStep, VisualizationHint
from finangpt.application.conversation.context_builder import ConversationContext


class FakePlanner:
    def __init__(self) -> None:
        self.called_with: list[str] = []

    def create_plan(self, question: str, context):
        self.called_with.append(question)
        return QueryPlan(
            complexity="simple",
            requires_visualization=False,
            reasoning="",
            steps=[
                QueryStep(
                    step_number=1,
                    description="Fetch data",
                    sql_query="SELECT 1 AS value",
                    dependencies=[],
                    expected_columns=["value"],
                )
            ],
        )


class FakeRetriever:
    def __init__(self) -> None:
        self.calls = []

    def execute_step(self, step, dependencies):
        self.calls.append(step.step_number)
        return pd.DataFrame({"value": [1, 2, 3]})


class FakeAnalyzer:
    def analyze(self, question, plan, data_map, context=None):
        return [
            Insight(
                finding="Test insight",
                significance="Average value = 2",
                data_points=["rows=3"],
                category="summary",
            )
        ]


class FakeSynthesizer:
    def synthesize(self, question, insights, data_map):
        return "Answer"


class FakeVizDetector:
    def detect(self, question, plan, insights, data_map):
        return [
            VisualizationHint(chart_type="bar", description="Compare", fields=["value"])
        ]


class FakeConversationManager:
    def __init__(self) -> None:
        self.logged: list[tuple[str, str, str]] = []
        self.context = ConversationContext(
            summary="Prev discussion",
            referenced_tickers=("AAPL",),
            referenced_metrics=("revenue",),
            recent_questions=("What about Apple?",),
        )

    def get_context(self, conversation_id: str):
        return self.context

    def record_turn(self, conversation_id: str, role: str, content: str, metadata=None):
        self.logged.append((conversation_id, role, content))

    def add_exchange(self, conversation_id: str, question: str, result):
        self.logged.append((conversation_id, "assistant", result.answer))


def test_orchestrator_runs_full_pipeline():
    fake_manager = FakeConversationManager()
    orchestrator = AnalysisOrchestrator(
        query_planner=FakePlanner(),
        data_retriever=FakeRetriever(),
        result_analyzer=FakeAnalyzer(),
        insight_synthesizer=FakeSynthesizer(),
        viz_detector=FakeVizDetector(),
        conversation_manager=fake_manager,
    )

    result = orchestrator.analyze_question("question", conversation_id="conv-1")

    assert result.answer == "Answer"
    assert result.insights and result.visualization_hints
    assert 1 in result.data_by_step
    assert any(role == "assistant" for _, role, _ in fake_manager.logged)
