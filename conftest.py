"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("FINANGPT_DISABLE_CACHE_METRICS", "1")
os.environ.setdefault("FINANGPT_LEGACY_LIMITS", "1")
_ENV_VARS = ["MONGO_URI", "OLLAMA_URL", "MODEL_NAME"]
_BASE_ENV = {var: os.environ.get(var) for var in _ENV_VARS}

from dataclasses import dataclass

import duckdb
import pandas as pd
import pytest

from finangpt.application.analysis.data_retriever import DataRetriever
from finangpt.application.analysis.insight_synthesizer import InsightSynthesizer
from finangpt.application.analysis.orchestrator import AnalysisOrchestrator
from finangpt.application.analysis.query_planner import QueryPlanner
from finangpt.application.analysis.result_analyzer import ResultAnalyzer
from finangpt.application.analysis.visualization_detector import VisualizationDetector
from finangpt.shared.dependency_injection import DuckDBFinancialRepository, RedisCacheRepository


@pytest.fixture(autouse=True)
def _reset_env_vars():
    for var, value in _BASE_ENV.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value
    try:
        yield
    finally:
        for var, value in _BASE_ENV.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


@dataclass
class _AcceptanceConfig:
    cache_ttl_seconds: int = 30


class AcceptanceLLM:
    def __init__(self) -> None:
        self._last_question = ""

    def generate_structured(self, prompt: str, schema):
        question_line = next((line for line in prompt.splitlines() if line.startswith("QUESTION")), "")
        self._last_question = question_line
        lower_q = question_line.lower()
        metric = "net_income" if "net income" in lower_q else "revenue"
        payload = {
            "complexity": "moderate",
            "requires_visualization": True,
            "reasoning": "",
            "steps": [
                {
                    "step_number": 1,
                    "description": f"Aggregate {metric}",
                    "sql_query": f"SELECT ticker, period, {metric} FROM financials",
                    "dependencies": [],
                    "expected_columns": ["ticker", "period", metric],
                }
            ],
        }
        return schema.model_validate(payload)

    def generate_text(self, prompt: str) -> str:
        return f"Insightful summary for {self._last_question or 'question'}."


class AcceptanceSchemaProvider:
    def get_schema_description(self) -> str:
        return "financials(ticker, period, revenue, net_income)"


@pytest.fixture(scope="module")
def acceptance_orchestrator(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("acceptance")
    conn = duckdb.connect(database=str(tmp_dir / "acceptance.duckdb"))
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "TSLA", "AAPL", "MSFT"],
            "period": ["2020", "2020", "2020", "2021", "2021"],
            "revenue": [100, 90, 70, 130, 110],
            "net_income": [20, 15, 5, 30, 18],
        }
    )
    conn.execute("CREATE TABLE financials AS SELECT * FROM df")

    repo = DuckDBFinancialRepository(conn)
    cache = RedisCacheRepository("memory://")
    llm = AcceptanceLLM()
    planner = QueryPlanner(llm_service=llm, schema_provider=AcceptanceSchemaProvider())
    retriever = DataRetriever(repository=repo, cache_repository=cache, config=_AcceptanceConfig())
    analyzer = ResultAnalyzer(llm_service=llm)
    synthesizer = InsightSynthesizer(llm_service=llm)
    viz = VisualizationDetector(llm_service=llm)

    return AnalysisOrchestrator(
        query_planner=planner,
        data_retriever=retriever,
        result_analyzer=analyzer,
        insight_synthesizer=synthesizer,
        viz_detector=viz,
    )
