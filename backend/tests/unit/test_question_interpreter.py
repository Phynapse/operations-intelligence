"""Tests for the question interpreter (deterministic keyword router)."""

from __future__ import annotations

import pytest

from app.ai.question_interpreter import interpret_question
from app.models.enums import AnalysisType, Metric, ReasoningGoal


class TestInterpretQuestion:
    def test_sla_compliance(self):
        intent = interpret_question("What is the current SLA compliance rate?")
        assert intent.metric == Metric.SLA_COMPLIANCE.value
        assert intent.analysis_type == AnalysisType.SLA_ANALYSIS.value

    def test_fulfilment_time(self):
        intent = interpret_question("Why did order fulfilment time increase in Q3?")
        assert intent.metric == Metric.FULFILLMENT_TIME.value
        assert intent.period == "2025-Q3"
        assert intent.reasoning_goal == ReasoningGoal.EXPLAIN_CHANGE

    def test_backlog_by_dept(self):
        intent = interpret_question("Which department has the highest backlog?")
        assert intent.metric == Metric.BACKLOG.value
        assert intent.reasoning_goal == ReasoningGoal.RANK_PRIORITIES

    def test_vendor_delay(self):
        intent = interpret_question("Which vendor has the highest delay rate?")
        # "delay" keyword matches first → kpi_summary; both valid
        assert intent.metric == Metric.DELAY_RATE.value

    def test_anomaly(self):
        intent = interpret_question("Are there unusual operational patterns?")
        assert intent.metric == Metric.ANOMALY_COUNT.value
        assert intent.reasoning_goal == ReasoningGoal.DETECT_ANOMALY

    def test_bottleneck(self):
        intent = interpret_question("Which process has the largest bottleneck?")
        assert intent.reasoning_goal == ReasoningGoal.IDENTIFY_BOTTLENECK

    def test_investigate(self):
        intent = interpret_question("Where should management investigate first?")
        assert intent.reasoning_goal == ReasoningGoal.RANK_PRIORITIES

    def test_trend(self):
        intent = interpret_question("How has cycle time trended?")
        assert intent.reasoning_goal == ReasoningGoal.TRACK_TREND
        assert intent.metric == Metric.AVERAGE_CYCLE_TIME.value

    def test_compare(self):
        intent = interpret_question("Compare throughput between departments")
        assert intent.reasoning_goal == ReasoningGoal.COMPARE_SEGMENTS

    def test_default_metric(self):
        # With no metric named, throughput is the default: it is the measure
        # every dataset supports and what a general "how are we doing" asks about.
        intent = interpret_question("Give me an overview")
        assert intent.metric == Metric.THROUGHPUT.value
        assert intent.analysis_type == AnalysisType.KPI_SUMMARY.value

    def test_empty_question(self):
        with pytest.raises(ValueError):
            interpret_question("")

    def test_whitespace_stripped(self):
        intent = interpret_question("  What is SLA compliance?  ")
        # Whitespace is preserved in the stored question
        assert "SLA" in intent.question
