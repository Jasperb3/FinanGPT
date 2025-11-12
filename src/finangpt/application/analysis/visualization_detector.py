"""Simple heuristics to suggest relevant visualizations."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from finangpt.application.analysis.schemas import Insight, QueryPlan, VisualizationHint

__all__ = ["VisualizationDetector"]


class VisualizationDetector:
    def __init__(self, llm_service) -> None:
        self._llm = llm_service

    def detect(
        self,
        question: str,
        plan: QueryPlan,
        insights: list[Insight],
        data_map: Dict[int, pd.DataFrame],
    ) -> list[VisualizationHint]:
        hints: list[VisualizationHint] = []
        if plan.requires_visualization or "trend" in question.lower():
            hints.append(
                VisualizationHint(
                    chart_type="line",
                    description="Plot key metrics over time",
                    fields=self._first_numeric_columns(data_map),
                )
            )
        if "compare" in question.lower() and len(data_map) > 1:
            hints.append(
                VisualizationHint(
                    chart_type="bar",
                    description="Compare grouped metrics across entities",
                    fields=self._first_numeric_columns(data_map),
                )
            )
        return hints

    def _first_numeric_columns(self, data_map: Dict[int, pd.DataFrame]) -> list[str]:
        for frame in data_map.values():
            numeric = frame.select_dtypes(include=["number"]).columns.tolist()
            if numeric:
                return numeric[:3]
        return []
