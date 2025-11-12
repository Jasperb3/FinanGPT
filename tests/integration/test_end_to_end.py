from __future__ import annotations

import duckdb
import pandas as pd

from finangpt.application.analysis.data_retriever import DataRetriever
from finangpt.application.analysis.insight_synthesizer import InsightSynthesizer
from finangpt.application.analysis.orchestrator import AnalysisOrchestrator
from finangpt.application.analysis.query_planner import QueryPlanner
from finangpt.application.analysis.result_analyzer import ResultAnalyzer
from finangpt.application.analysis.schemas import QueryPlan, QueryStep
from finangpt.application.analysis.visualization_detector import VisualizationDetector
from finangpt.shared.dependency_injection import DuckDBFinancialRepository, RedisCacheRepository


class StubLLM:
    def __init__(self) -> None:
        self.plan_payload = {
            "complexity": "simple",
            "requires_visualization": True,
            "reasoning": "",
            "steps": [
                {
                    "step_number": 1,
                    "description": "Load sample data",
                    "sql_query": "SELECT ticker, revenue FROM financials",
                    "dependencies": [],
                    "expected_columns": ["ticker", "revenue"],
                }
            ],
        }

    def generate_structured(self, prompt: str, schema):
        return schema.model_validate(self.plan_payload)

    def generate_text(self, prompt: str) -> str:
        return "Synthetic summary"


class StubSchemaProvider:
    def get_schema_description(self) -> str:
        return "financials(ticker, revenue)"


def test_end_to_end_pipeline_runs_successfully(tmp_path):
    conn = duckdb.connect(database=str(tmp_path / "test.duckdb"))
    df = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "revenue": [100, 200]})
    conn.execute("CREATE TABLE financials AS SELECT * FROM df")

    repo = DuckDBFinancialRepository(conn)
    cache = RedisCacheRepository("memory://")

    llm = StubLLM()
    planner = QueryPlanner(llm_service=llm, schema_provider=StubSchemaProvider())
    retriever = DataRetriever(repository=repo, cache_repository=cache, config=type("Cfg", (), {"cache_ttl_seconds": 1})())
    analyzer = ResultAnalyzer(llm_service=llm)
    synthesizer = InsightSynthesizer(llm_service=llm)
    viz = VisualizationDetector(llm_service=llm)

    orchestrator = AnalysisOrchestrator(
        query_planner=planner,
        data_retriever=retriever,
        result_analyzer=analyzer,
        insight_synthesizer=synthesizer,
        viz_detector=viz,
    )

    result = orchestrator.analyze_question("Show sample data")

    assert result.data_by_step[1].equals(df)
    assert result.visualization_hints
    assert result.answer
