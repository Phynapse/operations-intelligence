"""Analytics registry: the single mapping from analysis type to analyzer.

The registry is the mechanism that enforces the "no arbitrary execution" rule.
A plan can only reference an :class:`~app.models.enums.AnalysisType`, and only
registered analyzers are ever instantiated. There is no dynamic import, no
plugin discovery and no eval path from user input to code execution.

Adding an analysis type therefore requires three deliberate edits: add the enum
member, implement an analyzer, register it here.
"""

from __future__ import annotations

from functools import lru_cache

from app.analytics.anomaly_analyzer import AnomalyAnalyzer
from app.analytics.backlog_analyzer import BacklogAnalyzer
from app.analytics.base import BaseAnalyzer
from app.analytics.bottleneck_analyzer import BottleneckAnalyzer
from app.analytics.contribution_analyzer import ContributionAnalyzer
from app.analytics.cycle_time_analyzer import CycleTimeAnalyzer
from app.analytics.department_analyzer import DepartmentAnalyzer
from app.analytics.kpi_analyzer import KPIAnalyzer
from app.analytics.period_comparison_analyzer import PeriodComparisonAnalyzer
from app.analytics.sla_analyzer import SLAAnalyzer
from app.analytics.throughput_analyzer import ThroughputAnalyzer
from app.analytics.trend_analyzer import TrendAnalyzer
from app.analytics.vendor_analyzer import VendorAnalyzer
from app.models.enums import AnalysisType, DatasetId, ReasoningGoal
from app.utils.errors import UnsupportedAnalysisError


@lru_cache(maxsize=1)
def get_registry() -> dict[AnalysisType, BaseAnalyzer]:
    """The process-wide analysis type -> analyzer mapping.

    Analyzers are stateless and cheap to construct, so one instance per type is
    shared for the lifetime of the process.
    """
    instances: tuple[BaseAnalyzer, ...] = (
        KPIAnalyzer(),
        TrendAnalyzer(),
        CycleTimeAnalyzer(),
        ThroughputAnalyzer(),
        BacklogAnalyzer(),
        SLAAnalyzer(),
        BottleneckAnalyzer(),
        VendorAnalyzer(),
        DepartmentAnalyzer(),
        AnomalyAnalyzer(),
        PeriodComparisonAnalyzer(),
        ContributionAnalyzer(),
    )
    registry: dict[AnalysisType, BaseAnalyzer] = {}
    for analyzer in instances:
        for analysis_type in analyzer.supported_types:
            registry[analysis_type] = analyzer
    return registry


@lru_cache(maxsize=1)
def get_analyzer(analysis_type: AnalysisType) -> BaseAnalyzer:
    """Resolve the analyzer that handles an analysis type."""
    analyzer = get_registry().get(analysis_type)
    if analyzer is None:  # pragma: no cover - enum and registry are kept in sync
        supported = ", ".join(sorted(item.value for item in get_registry()))
        raise UnsupportedAnalysisError(
            f"analysis type '{analysis_type.value}' is not implemented; supported: {supported}"
        )
    return analyzer


def supported_analysis_types() -> list[str]:
    """Every analysis type the platform can execute."""
    return sorted(item.value for item in get_registry())


def required_datasets_for(analysis_type: AnalysisType) -> tuple[DatasetId, ...]:
    return get_analyzer(analysis_type).required_datasets


def default_reasoning_goal_for(analysis_type: AnalysisType) -> ReasoningGoal:
    """A sensible reasoning goal when the planner does not supply one."""
    mapping = {
        AnalysisType.KPI_SUMMARY: ReasoningGoal.SUMMARIZE_PERFORMANCE,
        AnalysisType.TREND_ANALYSIS: ReasoningGoal.TRACK_TREND,
        AnalysisType.PERIOD_COMPARISON: ReasoningGoal.EXPLAIN_CHANGE,
        AnalysisType.CYCLE_TIME_ANALYSIS: ReasoningGoal.EXPLAIN_CHANGE,
        AnalysisType.THROUGHPUT_ANALYSIS: ReasoningGoal.TRACK_TREND,
        AnalysisType.BACKLOG_ANALYSIS: ReasoningGoal.RANK_PRIORITIES,
        AnalysisType.SLA_ANALYSIS: ReasoningGoal.ASSESS_SLA,
        AnalysisType.BOTTLENECK_ANALYSIS: ReasoningGoal.IDENTIFY_BOTTLENECK,
        AnalysisType.VENDOR_ANALYSIS: ReasoningGoal.IDENTIFY_BOTTLENECK,
        AnalysisType.DEPARTMENT_ANALYSIS: ReasoningGoal.COMPARE_SEGMENTS,
        AnalysisType.ANOMALY_ANALYSIS: ReasoningGoal.DETECT_ANOMALY,
        AnalysisType.PROCESS_ANALYSIS: ReasoningGoal.IDENTIFY_BOTTLENECK,
        AnalysisType.CONTRIBUTION_ANALYSIS: ReasoningGoal.EXPLAIN_CHANGE,
    }
    return mapping[analysis_type]
