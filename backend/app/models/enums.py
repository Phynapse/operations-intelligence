"""Domain enumerations and value objects shared across the application.

These types define the *controlled vocabulary* of the system. The analysis
planner is only ever allowed to produce values from these enumerations, which is
what prevents arbitrary or unsupported analysis from being executed.
"""

from __future__ import annotations

from enum import StrEnum


class AnalysisType(StrEnum):
    """The complete set of deterministic operations the platform supports.

    There is deliberately no "generic" or "custom" member: an unknown analysis
    request is rejected at validation time rather than being approximated.
    """

    KPI_SUMMARY = "kpi_summary"
    TREND_ANALYSIS = "trend_analysis"
    PERIOD_COMPARISON = "period_comparison"
    CYCLE_TIME_ANALYSIS = "cycle_time_analysis"
    THROUGHPUT_ANALYSIS = "throughput_analysis"
    BACKLOG_ANALYSIS = "backlog_analysis"
    SLA_ANALYSIS = "sla_analysis"
    BOTTLENECK_ANALYSIS = "bottleneck_analysis"
    VENDOR_ANALYSIS = "vendor_analysis"
    DEPARTMENT_ANALYSIS = "department_analysis"
    ANOMALY_ANALYSIS = "anomaly_analysis"
    PROCESS_ANALYSIS = "process_analysis"
    CONTRIBUTION_ANALYSIS = "contribution_analysis"


class Metric(StrEnum):
    """Canonical operational metrics.

    Metric names are the join key between an LLM-produced plan and the
    deterministic analytics layer. If a metric is not in this list, no
    calculation is performed.
    """

    THROUGHPUT = "throughput"
    AVERAGE_CYCLE_TIME = "average_cycle_time"
    MEDIAN_CYCLE_TIME = "median_cycle_time"
    PROCESSING_TIME = "processing_time"
    QUEUE_TIME = "queue_time"
    QUEUE_CONTRIBUTION = "queue_contribution"
    BACKLOG = "backlog"
    BACKLOG_CHANGE = "backlog_change"
    SLA_COMPLIANCE = "sla_compliance"
    SLA_BREACH_RATE = "sla_breach_rate"
    DELAY_RATE = "delay_rate"
    FULFILLMENT_TIME = "fulfillment_time"
    ON_TIME_DELIVERY_RATE = "on_time_delivery_rate"
    ORDER_VALUE = "order_value"
    PRODUCTIVITY_PER_HEAD = "productivity_per_head"
    ANOMALY_COUNT = "anomaly_count"


class Dimension(StrEnum):
    """Supported analytical dimensions for slicing metrics."""

    DEPARTMENT = "department"
    PROCESS = "process"
    OPERATION_TYPE = "operation_type"
    VENDOR = "vendor"
    PRODUCT_CATEGORY = "product_category"
    PRIORITY = "priority"
    STATUS = "status"
    PERIOD = "period"
    QUARTER = "quarter"


class ReasoningGoal(StrEnum):
    """What the explanation layer is being asked to accomplish.

    This influences the emphasis of the generated narrative, never the numbers.
    """

    EXPLAIN_CHANGE = "explain_change"
    IDENTIFY_BOTTLENECK = "identify_bottleneck"
    DETECT_ANOMALY = "detect_anomaly"
    ASSESS_SLA = "assess_sla"
    RANK_PRIORITIES = "rank_priorities"
    SUMMARIZE_PERFORMANCE = "summarize_performance"
    COMPARE_SEGMENTS = "compare_segments"
    TRACK_TREND = "track_trend"


class DatasetId(StrEnum):
    """Registered dataset identifiers. Used for safe, validated file access."""

    OPERATIONS = "operations"
    ORDERS = "orders"
    SLA_METRICS = "sla_metrics"
    VENDORS = "vendors"
    DEPARTMENTS = "departments"


class Unit(StrEnum):
    """Measurement units used by KPIs and evidence."""

    HOURS = "hours"
    OPERATIONS = "operations"
    ORDERS = "orders"
    ITEMS = "items"
    PERCENT = "percent"
    CURRENCY = "currency"
    COUNT = "count"


class Severity(StrEnum):
    """Severity levels for findings and anomalies."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


#: Analysis types that inherently require two periods of data.
COMPARISON_ANALYSIS_TYPES: frozenset[AnalysisType] = frozenset(
    {AnalysisType.PERIOD_COMPARISON, AnalysisType.CONTRIBUTION_ANALYSIS}
)

#: Analysis types that read from the SLA rollup dataset.
SLA_ANALYSIS_TYPES: frozenset[AnalysisType] = frozenset({AnalysisType.SLA_ANALYSIS})

#: Analysis types that read from the orders dataset.
ORDER_ANALYSIS_TYPES: frozenset[AnalysisType] = frozenset({AnalysisType.VENDOR_ANALYSIS})
