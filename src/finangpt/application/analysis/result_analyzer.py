"""Transform retrieved data into structured insights."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from finangpt.application.analysis.schemas import Insight, QueryPlan, ResultAnalysisConfig
from finangpt.application.conversation.context_builder import ConversationContext

__all__ = ["ResultAnalyzer"]


class ResultAnalyzer:
    def __init__(self, llm_service, config: ResultAnalysisConfig | dict | None = None) -> None:
        self._llm = llm_service
        self._config = self._coerce_config(config)

    def analyze(
        self,
        question: str,
        plan: QueryPlan,
        data_map: Dict[int, pd.DataFrame],
        context: ConversationContext | None = None,
    ) -> list[Insight]:
        insights: list[Insight] = []
        for step in plan.steps:
            frame = data_map.get(step.step_number)
            if frame is None:
                continue
            summary = self._summarize_frame(frame)
            if context:
                summary = f"{summary} | Context: {context.summary}"
            category = "trend" if "date" in (col.lower() for col in frame.columns) else "summary"
            insight = Insight(
                finding=f"Step {step.step_number}: {step.description}",
                significance=summary,
                data_points=[f"rows={len(frame)}", f"columns={', '.join(frame.columns)}"],
                category=category,
            )
            insights.append(insight)
            if len(insights) >= self._config.max_insights:
                break
        return insights

    def _summarize_frame(self, frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No rows returned"
        numeric_cols = frame.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            return "Contains categorical data"
        col = numeric_cols[0]
        return f"Average {col} = {frame[col].mean():.2f}"

    def _coerce_config(self, config: ResultAnalysisConfig | dict | None) -> ResultAnalysisConfig:
        if config is None:
            return ResultAnalysisConfig(max_insights=5, min_confidence_threshold=0.5)
        if isinstance(config, ResultAnalysisConfig):
            return config
        if hasattr(config, "model_dump"):
            data = config.model_dump()
        else:
            data = dict(config)
        return ResultAnalysisConfig(**data)
