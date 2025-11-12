"""Shared Pydantic schemas for analysis prompts and responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "QueryStep",
    "QueryPlan",
    "Insight",
    "VisualizationHint",
    "ResultAnalysisConfig",
]


class QueryStep(BaseModel):
    step_number: int
    description: str
    sql_query: str
    dependencies: list[int] = Field(default_factory=list)
    expected_columns: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    complexity: Literal["simple", "moderate", "complex"]
    requires_visualization: bool = False
    reasoning: str | None = None
    steps: list[QueryStep]


class Insight(BaseModel):
    finding: str
    significance: str
    data_points: list[str]
    category: Literal["trend", "comparison", "anomaly", "correlation", "summary"]


class VisualizationHint(BaseModel):
    chart_type: Literal["line", "bar", "scatter", "table", "heatmap", "map", "pie"]
    description: str
    fields: list[str] = Field(default_factory=list)


class ResultAnalysisConfig(BaseModel):
    max_insights: int = Field(gt=0)
    min_confidence_threshold: float = Field(ge=0, le=1)
