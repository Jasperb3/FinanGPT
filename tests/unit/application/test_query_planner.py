from __future__ import annotations

from finangpt.application.analysis.query_planner import QueryPlanner
from finangpt.application.analysis.schemas import QueryPlan


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompts: list[str] = []

    def generate_structured(self, prompt: str, schema):
        self.prompts.append(prompt)
        return schema.model_validate(self.payload)


class FakeSchemaProvider:
    def get_schema_description(self) -> str:
        return "table financials (ticker, revenue)"


def test_query_planner_returns_plan_from_llm():
    payload = {
        "complexity": "simple",
        "requires_visualization": False,
        "reasoning": "direct",
        "steps": [
            {
                "step_number": 1,
                "description": "Fetch revenue",
                "sql_query": "SELECT * FROM financials",
                "dependencies": [],
                "expected_columns": ["ticker", "revenue"],
            }
        ],
    }
    planner = QueryPlanner(llm_service=FakeLLM(payload), schema_provider=FakeSchemaProvider())

    plan = planner.create_plan("Show revenue for AAPL")

    assert isinstance(plan, QueryPlan)
    assert plan.steps[0].sql_query.startswith("SELECT")
    assert "AAPL" not in plan.steps[0].sql_query  # from payload
