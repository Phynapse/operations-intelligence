"""Question interpreter: turns a natural-language question into a structured intent.

In phase 1 this is a **deterministic keyword router**. It does not call an LLM.
It uses keyword matching, period detection and heuristic dimension extraction to
produce the same structured intent that an LLM will produce in phase 2. This
means the rest of the pipeline can be built and tested *today* without an LLM.

The interface is deliberately the same as what the future LLM interpreter will
have so the swap is a one-line change.
"""

from __future__ import annotations

import re

from app.ai.analysis_planner import DEFAULT_METRIC_BY_TYPE
from app.ai.plan_validator import ALLOWED_DIMENSIONS
from app.models.enums import (
    AnalysisType,
    Dimension,
    Metric,
    ReasoningGoal,
)
from app.schemas.ai import QuestionIntent
from app.utils.periods import period_sort_key

# ---------------------------------------------------------------------------
# Static lookup tables
# ---------------------------------------------------------------------------

_METRIC_KEYWORDS: dict[str, tuple[Metric, str]] = {
    "how long": (Metric.AVERAGE_CYCLE_TIME, "cycle_time_analysis"),
    "end to end": (Metric.AVERAGE_CYCLE_TIME, "cycle_time_analysis"),
    "cycle time": (Metric.AVERAGE_CYCLE_TIME, "cycle_time_analysis"),
    "cycle_time": (Metric.AVERAGE_CYCLE_TIME, "cycle_time_analysis"),
    "processing time": (Metric.PROCESSING_TIME, "cycle_time_analysis"),
    "queue time": (Metric.QUEUE_TIME, "cycle_time_analysis"),
    "throughput": (Metric.THROUGHPUT, "throughput_analysis"),
    "completed operations": (Metric.THROUGHPUT, "throughput_analysis"),
    "backlog": (Metric.BACKLOG, "backlog_analysis"),
    "open items": (Metric.BACKLOG, "backlog_analysis"),
    "promise": (Metric.ON_TIME_DELIVERY_RATE, "vendor_analysis"),
    "promised": (Metric.ON_TIME_DELIVERY_RATE, "vendor_analysis"),
    "sla": (Metric.SLA_COMPLIANCE, "sla_analysis"),
    "compliance": (Metric.SLA_COMPLIANCE, "sla_analysis"),
    "breach": (Metric.SLA_BREACH_RATE, "sla_analysis"),
    "fulfilment time": (Metric.FULFILLMENT_TIME, "period_comparison"),
    "fulfillment time": (Metric.FULFILLMENT_TIME, "period_comparison"),
    "on-time": (Metric.ON_TIME_DELIVERY_RATE, "vendor_analysis"),
    "on time": (Metric.ON_TIME_DELIVERY_RATE, "vendor_analysis"),
    "delay": (Metric.DELAY_RATE, "kpi_summary"),
    "delayed": (Metric.DELAY_RATE, "kpi_summary"),
    "vendor": (Metric.DELAY_RATE, "vendor_analysis"),
    "anomal": (Metric.ANOMALY_COUNT, "anomaly_analysis"),
    "unusual": (Metric.ANOMALY_COUNT, "anomaly_analysis"),
    "order value": (Metric.ORDER_VALUE, "period_comparison"),
    "bottleneck": (Metric.AVERAGE_CYCLE_TIME, "bottleneck_analysis"),
    "constraint": (Metric.AVERAGE_CYCLE_TIME, "bottleneck_analysis"),
}

#: Phrasing that makes the question about movement over time. This overrides the
#: analysis type but never the metric: "how has cycle time trended" is a trend
#: question about cycle time, not a generic trend question.
_TREND_KEYWORDS: tuple[str, ...] = (
    "trend",
    "trended",
    "over time",
    "month by month",
    "quarter by quarter",
    "week by week",
    "year over year",
    "progression",
    "trajectory",
)

#: Metric and analysis type used when the question names neither. Throughput is
#: the default because it is the measure every operational dataset supports and
#: the one a general "how are we doing" question is asking about.
DEFAULT_METRIC: tuple[Metric, str] = (Metric.THROUGHPUT, "kpi_summary")

#: Question shapes that mean "decompose a change across a dimension". These map
#: to the contribution analyzer rather than to a plain period comparison.
_CONTRIBUTION_KEYWORDS: tuple[str, ...] = (
    "contribut",
    "driver",
    "drove",
    "driving",
    "what caused the change",
    "decompose",
    "break down the change",
    "breakdown",
    "accounted for",
    "responsible for the change",
)

_DIMENSION_KEYWORDS: dict[str, Dimension] = {
    "department": Dimension.DEPARTMENT,
    "process": Dimension.PROCESS,
    "operation type": Dimension.OPERATION_TYPE,
    "vendor": Dimension.VENDOR,
    "product": Dimension.PRODUCT_CATEGORY,
    "category": Dimension.PRODUCT_CATEGORY,
    "priority": Dimension.PRIORITY,
    "status": Dimension.STATUS,
}

#: Analysis type implied when a dimension is the subject of the question and the
#: metric keyword alone would not slice by it.
_DIMENSION_ANALYSIS: dict[Dimension, str] = {
    Dimension.DEPARTMENT: "department_analysis",
    Dimension.VENDOR: "vendor_analysis",
    Dimension.PROCESS: "process_analysis",
}

#: Patterns that mark a dimension as the *subject* of the question rather than a
#: passing mention: "which vendor ...", "by department", "per process".
_SUBJECT_DIMENSION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:which|what|who)\s+(?:\w+\s+){0,2}?"
        r"(department|process|vendor|supplier|product|category|priority|status)(?:s|es)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:by|per|across|for each|split by|broken down by)\s+"
        r"(department|process|vendor|supplier|product|category|priority|status)(?:s|es)?\b",
        re.IGNORECASE,
    ),
)

#: Aliases accepted in subject patterns, mapped to canonical dimension keywords.
_SUBJECT_ALIASES: dict[str, str] = {
    "supplier": "vendor",
    "operation type": "operation type",
}

_REASONING_KEYWORDS: dict[str, ReasoningGoal] = {
    "why": ReasoningGoal.EXPLAIN_CHANGE,
    "reason": ReasoningGoal.EXPLAIN_CHANGE,
    "caus": ReasoningGoal.EXPLAIN_CHANGE,
    "explain": ReasoningGoal.EXPLAIN_CHANGE,
    "bottleneck": ReasoningGoal.IDENTIFY_BOTTLENECK,
    "constraint": ReasoningGoal.IDENTIFY_BOTTLENECK,
    "anomal": ReasoningGoal.DETECT_ANOMALY,
    "unusual": ReasoningGoal.DETECT_ANOMALY,
    "investigate": ReasoningGoal.RANK_PRIORITIES,
    "where should": ReasoningGoal.RANK_PRIORITIES,
    "sla": ReasoningGoal.ASSESS_SLA,
    "compliance": ReasoningGoal.ASSESS_SLA,
    "compare": ReasoningGoal.COMPARE_SEGMENTS,
    "versus": ReasoningGoal.COMPARE_SEGMENTS,
    "trend": ReasoningGoal.TRACK_TREND,
    "how": ReasoningGoal.TRACK_TREND,
}

_PERIOD_PATTERNS = [
    (re.compile(r"\bq([1-4])\b", re.IGNORECASE), lambda m: f"2025-Q{m.group(1)}"),
    (
        re.compile(r"\b(\d{4})[-\s]?q([1-4])\b", re.IGNORECASE),
        lambda m: f"{m.group(1)}-Q{m.group(2)}",
    ),
    (re.compile(r"\b(\d{4})[-](\d{2})\b"), lambda m: f"{m.group(1)}-{m.group(2)}"),
]

#: A bare year, used only when the question named no quarter or month. Matching
#: "2025" in "Q4 2025" as well would offer a whole-year window alongside the
#: quarter the question actually asked about.
_YEAR_PATTERN = re.compile(r"\b(\d{4})\b")

#: Requests the platform cannot answer, with the reason shown to the user. These
#: are detected before any analysis runs so the caller learns why, rather than
#: receiving a confident answer built from data that cannot support it.
UNSUPPORTED_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"\b(forecast|predict|prediction|projection|project)\w*\b",
        "the dataset is historical only, so it cannot produce a forecast",
    ),
    (
        r"\b(next|upcoming|future|coming)\s+(month|quarter|year|week)\b",
        "no data exists beyond the loaded period",
    ),
    (
        r"\bwhat will\b",
        "the dataset is historical only, so it cannot describe future outcomes",
    ),
    (
        r"\bshould we (hire|buy|invest|outsourc)\w*\b",
        "this platform reports measured operations, not investment decisions",
    ),
    (
        r"\b(revenue|profit|margin|ebitda|p&l|balance sheet"
        r"|invoice to our customer)\b",
        "the datasets cover operational throughput and time, not financials",
    ),
    (
        r"\b(hiring|headcount plan|salary|payroll|employee satisfaction"
        r"|engagement)\b",
        "the datasets contain allocated headcount, not people data",
    ),
    (
        r"\b(weather|market share|competitor|competitors|industry benchmark)\b",
        "the datasets are internal to this operation",
    ),
    (
        r"\b(send|email|notify|schedule|book|ticket|raise a ticket"
        r"|create a report for)\b",
        "this platform answers questions and does not take actions",
    ),
    (
        r"\b(delete|drop|truncate|update|modify|write|insert|upload"
        r"|overwrite|append)\s+(?:the\s+)?(?:\w+\s+){0,2}?"
        r"(data|row|rows|record|records|file|files|table|dataset"
        r"|figure|figures|value|values|number|numbers)\b",
        "the datasets are read-only",
    ),
    (
        r"\b(sql|query the database|raw sql|select .* from)\b",
        "analysis runs through fixed registered operations, not ad-hoc SQL",
    ),
    (
        r"\b(explain the code|show me the source|print your (prompt|instructions)"
        r"|system prompt)\b",
        "the internal prompt is not exposed",
    ),
    (
        r"\bhttps?://",
        "the question is not an operational question",
    ),
)

#: Subjects that suggest the question is about the assistant rather than the
#: operation. Kept separate so the reason can be specific.
_META_SUBJECT_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"\byou are\b|\byou are a\b",
        "the question addresses the assistant rather than the operation",
    ),
    (
        r"\bwhat are your (instructions|tools|rules|guidelines)\b",
        "internal configuration is not exposed",
    ),
)


def detect_unsupported_terms(question: str) -> list[str]:
    """Return human-readable reasons a question cannot be answered.

    Every returned string explains *why* the platform cannot serve the request,
    so the API can reject it with an actionable message instead of a generic
    error. Detection is deliberately conservative: a phrase only blocks a
    question when it clearly asks for something outside historical operational
    analytics.
    """
    text = question.lower()
    reasons: list[str] = []
    for pattern, reason in UNSUPPORTED_PATTERNS + _META_SUBJECT_PATTERNS:
        if re.search(pattern, text):
            reasons.append(reason)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(reasons))


def interpret_question(question: str) -> QuestionIntent:
    """Parse a natural-language operational question into a structured intent.

    This is the deterministic Phase 1 implementation. In Phase 2 this will be
    replaced by an LLM call with the same input/output contract.
    """
    text = question.lower()

    unsupported = detect_unsupported_terms(question)

    metric, analysis_type = _detect_metric(text)
    named_metric = analysis_type != DEFAULT_METRIC[1] or any(
        keyword in text for keyword in _METRIC_KEYWORDS
    )

    if any(keyword in text for keyword in _TREND_KEYWORDS) and analysis_type != "trend_analysis":
        analysis_type = "trend_analysis"

    department = _detect_department(text)
    process = _detect_process(text)
    vendor_id = _detect_vendor(text)

    # The dimension a question is *about*: named as its subject, or named
    # explicitly as an entity to analyse. "Which vendor ..." and "the delay rate
    # for VND-101" both make the vendor the subject even though only the second
    # contains the word "vendor".
    subject_dimension = _detect_subject_dimension(text)
    if subject_dimension is None:
        subject_dimension = _subject_from_named_entity(vendor_id, department)

    if subject_dimension is not None:
        if not named_metric:
            # Nothing named a metric, so the dimension is the whole question:
            # "which departments need attention" is a department analysis, not a
            # throughput summary that happens to mention departments.
            analysis_type = _DIMENSION_ANALYSIS.get(subject_dimension, analysis_type)
            # The default metric must be one this analysis type can measure,
            # otherwise the plan is rejected for a question that named no metric.
            metric = DEFAULT_METRIC_BY_TYPE.get(AnalysisType(analysis_type), metric)
        elif not _type_can_slice(analysis_type, subject_dimension):
            analysis_type = _DIMENSION_ANALYSIS.get(subject_dimension, analysis_type)

    if _wants_contribution(text):
        analysis_type = _contribution_type_for(analysis_type)

    dimensions = _detect_dimensions(text)
    if subject_dimension is not None and subject_dimension not in dimensions:
        dimensions.insert(0, subject_dimension)

    period, comparison_period = _detect_periods(text, analysis_type)
    reasoning_goal = _detect_reasoning(text, analysis_type)

    return QuestionIntent(
        question=question,
        analysis_type=analysis_type,
        metric=metric.value,
        dimensions=tuple(d.value for d in dimensions),
        period=period,
        comparison_period=comparison_period,
        department=department,
        process=process,
        vendor_id=vendor_id,
        reasoning_goal=reasoning_goal,
        unsupported_terms=tuple(unsupported),
    )


def _subject_from_named_entity(vendor_id: str | None, department: str | None) -> Dimension | None:
    """An explicitly named entity makes that dimension the question's subject."""
    if vendor_id:
        return Dimension.VENDOR
    if department:
        return Dimension.DEPARTMENT
    return None


def _wants_contribution(text: str) -> bool:
    """True when the question asks what drove a change."""
    return any(keyword in text for keyword in _CONTRIBUTION_KEYWORDS)


def _detect_subject_dimension(text: str) -> Dimension | None:
    """The dimension the question is *about*, rather than merely mentioning."""
    for pattern in _SUBJECT_DIMENSION_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        token = _SUBJECT_ALIASES.get(match.group(1), match.group(1))
        for keyword, dimension in _DIMENSION_KEYWORDS.items():
            if keyword == token:
                return dimension
    return None


def _type_can_slice(analysis_type: str, dimension: Dimension) -> bool:
    """True when an analysis type is allowed to slice by ``dimension``."""
    try:
        resolved = AnalysisType(analysis_type)
    except ValueError:
        return False
    return dimension in ALLOWED_DIMENSIONS.get(resolved, frozenset())


def _contribution_type_for(analysis_type: str) -> str:
    """Choose between contribution decomposition and a plain period comparison.

    Contribution analysis needs a metric it can measure independently in each
    window for every dimension member. SLA compliance does not qualify: its
    rollup is already a single rate per department-process-month, so slicing it
    further adds nothing the SLA analyzer does not already report. Such questions
    stay period comparisons rather than being rejected.
    """
    return "period_comparison" if analysis_type == "sla_analysis" else "contribution_analysis"


def _detect_metric(text: str) -> tuple[Metric, str]:
    for keyword, (metric, analysis_type) in _METRIC_KEYWORDS.items():
        if keyword in text:
            return metric, analysis_type
    return DEFAULT_METRIC


def _detect_dimensions(text: str) -> list[Dimension]:
    dims: list[Dimension] = []
    for keyword, dimension in _DIMENSION_KEYWORDS.items():
        if keyword in text and dimension not in dims:
            dims.append(dimension)
    return dims


def _detect_periods(text: str, analysis_type: str) -> tuple[str | None, str | None]:
    """Find the periods the question names.

    For a comparison question the periods are ordered chronologically, not by
    position in the sentence: "between Q2 and Q3" asks about Q3 against Q2, so
    the later window becomes the current period. Taking them in text order would
    silently analyse the wrong window and report the wrong direction of change.
    """
    periods: list[str] = []
    for pattern, formatter in _PERIOD_PATTERNS:
        for match in pattern.finditer(text):
            period = formatter(match)
            if period not in periods:
                periods.append(period)

    if not periods:
        year_match = _YEAR_PATTERN.search(text)
        if year_match is not None:
            return year_match.group(1), None
        return None, None

    if analysis_type in ("period_comparison", "contribution_analysis") and len(periods) >= 2:
        ordered = sorted(periods, key=period_sort_key)
        return ordered[-1], ordered[-2]

    return periods[0], None


def _detect_reasoning(text: str, default_type: str) -> ReasoningGoal:
    for keyword, goal in _REASONING_KEYWORDS.items():
        if keyword in text:
            return goal
    goal_map: dict[str, ReasoningGoal] = {
        "kpi_summary": ReasoningGoal.SUMMARIZE_PERFORMANCE,
        "trend_analysis": ReasoningGoal.TRACK_TREND,
        "period_comparison": ReasoningGoal.EXPLAIN_CHANGE,
        "cycle_time_analysis": ReasoningGoal.EXPLAIN_CHANGE,
        "throughput_analysis": ReasoningGoal.TRACK_TREND,
        "backlog_analysis": ReasoningGoal.RANK_PRIORITIES,
        "sla_analysis": ReasoningGoal.ASSESS_SLA,
        "bottleneck_analysis": ReasoningGoal.IDENTIFY_BOTTLENECK,
        "vendor_analysis": ReasoningGoal.IDENTIFY_BOTTLENECK,
        "department_analysis": ReasoningGoal.COMPARE_SEGMENTS,
        "anomaly_analysis": ReasoningGoal.DETECT_ANOMALY,
        "process_analysis": ReasoningGoal.IDENTIFY_BOTTLENECK,
    }
    return goal_map.get(default_type, ReasoningGoal.SUMMARIZE_PERFORMANCE)


def _detect_department(text: str) -> str | None:
    """Extract a department id such as ``DEP-102``, normalised to upper case."""
    match = re.search(r"\bdep-?(\d{3})\b", text, re.IGNORECASE)
    return f"DEP-{match.group(1)}" if match else None


def _detect_process(text: str) -> str | None:
    process_names = [
        "order intake",
        "customer support",
        "returns processing",
        "receiving",
        "picking and packing",
        "outbound shipping",
        "invoice processing",
        "accounts payable",
        "period reconciliation",
        "incident resolution",
        "service request fulfilment",
        "system administration",
        "vendor onboarding",
        "purchase order management",
        "contract administration",
    ]
    for name in process_names:
        if name.lower() in text:
            return name
    return None


def _detect_vendor(text: str) -> str | None:
    """Extract a vendor id such as ``VND-101``, normalised to upper case."""
    match = re.search(r"\bvnd-?(\d{3})\b", text, re.IGNORECASE)
    return f"VND-{match.group(1)}" if match else None


__all__ = [
    "UNSUPPORTED_PATTERNS",
    "QuestionIntent",
    "detect_unsupported_terms",
    "interpret_question",
]
