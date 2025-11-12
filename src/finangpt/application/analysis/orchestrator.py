"""High-level orchestrator for the FinanGPT analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from finangpt.application.analysis.schemas import Insight, QueryPlan, VisualizationHint
from finangpt.application.conversation.context_builder import ConversationContext

__all__ = ["AnalysisResult", "AnalysisOrchestrator"]


@dataclass(slots=True)
class AnalysisResult:
    question: str
    plan: QueryPlan
    insights: list[Insight]
    answer: str
    visualization_hints: list[VisualizationHint]
    data_by_step: Dict[int, pd.DataFrame]


class AnalysisOrchestrator:
    def __init__(
        self,
        query_planner,
        data_retriever,
        result_analyzer,
        insight_synthesizer,
        viz_detector,
        conversation_manager=None,
    ) -> None:
        self._query_planner = query_planner
        self._data_retriever = data_retriever
        self._result_analyzer = result_analyzer
        self._insight_synthesizer = insight_synthesizer
        self._viz_detector = viz_detector
        self._conversation_manager = conversation_manager

    def analyze_question(self, question: str, conversation_id: str | None = None) -> AnalysisResult:
        context: ConversationContext | None = None
        if conversation_id and self._conversation_manager:
            try:
                context = self._conversation_manager.get_context(conversation_id)
            except Exception:
                context = None

        plan = self._query_planner.create_plan(question, context)
        data_by_step: Dict[int, pd.DataFrame] = {}
        for step in plan.steps:
            dependencies = {dep: data_by_step[dep] for dep in step.dependencies if dep in data_by_step}
            data_by_step[step.step_number] = self._data_retriever.execute_step(step, dependencies)

        if conversation_id and self._conversation_manager:
            question_metadata = {"step_count": len(plan.steps), "complexity": plan.complexity}
            self._conversation_manager.record_turn(conversation_id, "user", question, question_metadata)

        insights = self._result_analyzer.analyze(question, plan, data_by_step, context=context)
        answer = self._insight_synthesizer.synthesize(question, insights, data_by_step)
        viz_hints = self._viz_detector.detect(question, plan, insights, data_by_step)

        analysis_result = AnalysisResult(
            question=question,
            plan=plan,
            insights=insights,
            answer=answer,
            visualization_hints=viz_hints,
            data_by_step=data_by_step,
        )

        if conversation_id and self._conversation_manager:
            try:
                self._conversation_manager.add_exchange(conversation_id, question, analysis_result)
            except Exception:
                pass

        return analysis_result
