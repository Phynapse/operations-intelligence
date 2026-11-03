"""Anomaly detection: unusual operational months, with a stated method.

Two complementary detectors are used, each applied to a monthly series so that
weekday and month-end noise is already averaged out:

**Robust z-score (modified z)**
    ``0.6745 * (x - median) / MAD``. The median and MAD are used instead of the
    mean and standard deviation because a single extreme month would otherwise
    inflate the standard deviation and hide itself. A point is flagged when its
    modified z-score exceeds 3.5, the conventional threshold for this estimator.

**Trailing baseline deviation**
    current month versus the trailing three-month mean. This catches sustained
    level shifts that a robust z-score against the whole window can miss when
    the shift is recent.

Both detectors are reported with the numbers that produced them, so a reader can
see exactly why a month was called unusual. Neither detector asserts a cause.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import pandas as pd

from app.analytics.base import (
    AnalysisContext,
    AnalysisOutcome,
    BaseAnalyzer,
    build_evidence,
    build_finding,
    build_kpi,
)
from app.analytics.trend_analyzer import METRIC_SPECS, monthly_series
from app.models.enums import AnalysisType, DatasetId, Metric, Severity, Unit
from app.schemas.analysis import EvidenceItem, Finding
from app.utils.calculations import percentage_change, round_or_none
from app.utils.errors import InsufficientDataError
from app.utils.periods import period_sort_key

#: Conventional modified z-score threshold.
MODIFIED_Z_THRESHOLD = 3.5
#: Number of trailing months used by the baseline detector.
BASELINE_WINDOW = 3
#: Minimum absolute percentage deviation from the trailing baseline to flag.
BASELINE_DEVIATION_PERCENT = 15.0
#: Minimum observations before either detector is trusted.
MIN_OBSERVATIONS = 5

#: Metrics where a rise is adverse; used only for severity, never for detection.
LOWER_IS_BETTER: frozenset[Metric] = frozenset(
    {
        Metric.AVERAGE_CYCLE_TIME,
        Metric.PROCESSING_TIME,
        Metric.QUEUE_TIME,
        Metric.BACKLOG,
        Metric.DELAY_RATE,
        Metric.SLA_BREACH_RATE,
    }
)

DIMENSION_COLUMNS: dict[str, str] = {
    "department": "department",
    "process": "process",
    "operation_type": "operation_type",
    "vendor": "vendor_id",
}


@dataclass(frozen=True, slots=True)
class Anomaly:
    """One flagged observation."""

    period: str
    value: float
    detector: str
    statistic: str
    #: Numeric form of the detector output, used for ranking.
    statistic_value: float
    threshold: float
    baseline: float | None
    baseline_deviation_percent: float | None
    metric: Metric
    dimension: str | None = None

    @property
    def direction(self) -> str:
        if self.baseline is None:
            return "spike" if self.statistic_value > 0 else "dip"
        return "spike" if self.value >= self.baseline else "dip"


class AnomalyAnalyzer(BaseAnalyzer):
    """Detect and describe unusual periods in operational metrics."""

    name = "anomaly_analyzer"
    supported_types = frozenset({AnalysisType.ANOMALY_ANALYSIS})
    required_datasets = (DatasetId.OPERATIONS, DatasetId.SLA_METRICS, DatasetId.DEPARTMENTS)

    def analyze(self, context: AnalysisContext) -> AnalysisOutcome:
        frame = context.primary_frame()
        if frame.empty:
            raise InsufficientDataError(
                f"no records found for {context.plan.period or 'the requested period'}"
            )

        metrics = self._metrics(context)
        outcome = AnalysisOutcome()
        anomalies: list[Anomaly] = []

        for metric in metrics:
            series = [
                row for row in monthly_series(frame, metric, context) if row["value"] is not None
            ]
            if len(series) < MIN_OBSERVATIONS:
                outcome.extend_caveats(
                    f"{metric.value} has only {len(series)} monthly observation(s); anomaly "
                    f"detection needs at least {MIN_OBSERVATIONS} and was skipped for this metric"
                )
                continue
            anomalies.extend(detect(series, metric))

        column = self._dimension_column(context)
        dimension_anomalies: list[Anomaly] = []
        if column:
            names = context.repository.department_names
            for key, group in frame.groupby(column, dropna=True):
                label = names.get(str(key), str(key))
                for metric in metrics:
                    series = [
                        row
                        for row in monthly_series(group, metric, context)
                        if row["value"] is not None
                    ]
                    if len(series) < MIN_OBSERVATIONS:
                        continue
                    for found in detect(series, metric):
                        dimension_anomalies.append(
                            Anomaly(
                                period=found.period,
                                value=found.value,
                                detector=found.detector,
                                statistic=found.statistic,
                                statistic_value=found.statistic_value,
                                threshold=found.threshold,
                                baseline=found.baseline,
                                baseline_deviation_percent=found.baseline_deviation_percent,
                                metric=found.metric,
                                dimension=label,
                            )
                        )

        anomalies.sort(key=lambda item: (item.period, -abs(item.statistic_value)))
        ranked = _rank(anomalies)[: context.top_n]

        outcome.kpis = [
            build_kpi(
                name="Anomalies detected",
                value=float(len(anomalies)),
                unit=Unit.COUNT,
                source_dataset=DatasetId.OPERATIONS,
                period=context.plan.period or "all periods",
                calculation=(
                    f"Count of monthly observations flagged by either detector across "
                    f"{len(metrics)} metric(s)"
                ),
            ),
            build_kpi(
                name="Detection window",
                value=float(len(monthly_series(frame, metrics[0], context))),
                unit=Unit.COUNT,
                source_dataset=DatasetId.OPERATIONS,
                period=context.plan.period or "all periods",
                calculation="Number of monthly observations in the detection window",
            ),
        ]

        outcome.evidence = self._evidence(ranked, context)
        outcome.findings.extend(self._findings(anomalies, dimension_anomalies, metrics, context))
        outcome.extend_caveats(*self._caveats(frame, metrics, context, bool(anomalies)))
        outcome.row_count_examined = len(frame)
        return outcome

    # -- internals -----------------------------------------------------
    @staticmethod
    def _metrics(context: AnalysisContext) -> list[Metric]:
        requested = context.plan.metric
        if requested is not None and requested in METRIC_SPECS:
            return [requested]
        return [
            Metric.AVERAGE_CYCLE_TIME,
            Metric.SLA_COMPLIANCE,
            Metric.THROUGHPUT,
            Metric.BACKLOG,
        ]

    @staticmethod
    def _dimension_column(context: AnalysisContext) -> str | None:
        for dimension in context.plan.dimensions:
            column = DIMENSION_COLUMNS.get(dimension.value)
            if column:
                return column
        return None

    @staticmethod
    def _evidence(ranked: list[Anomaly], context: AnalysisContext) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for anomaly in ranked:
            spec = METRIC_SPECS[anomaly.metric]
            items.append(
                build_evidence(
                    metric=f"{anomaly.metric.value} anomaly ({anomaly.detector})",
                    value=round_or_none(anomaly.value),
                    comparison_value=round_or_none(anomaly.baseline),
                    unit=spec["unit"],
                    source_dataset=DatasetId.SLA_METRICS
                    if anomaly.metric is Metric.SLA_COMPLIANCE
                    else DatasetId.OPERATIONS,
                    dimension=anomaly.dimension,
                    period=anomaly.period,
                    calculation_description=anomaly.statistic,
                    severity=_severity(anomaly),
                )
            )
        if not items:
            items.append(
                build_evidence(
                    metric="Anomaly scan",
                    value=0.0,
                    unit=Unit.COUNT,
                    source_dataset=DatasetId.OPERATIONS,
                    period=context.plan.period or "all periods",
                    calculation_description=(
                        f"No monthly observation exceeded a modified z-score of "
                        f"{MODIFIED_Z_THRESHOLD} or deviated more than "
                        f"{BASELINE_DEVIATION_PERCENT:.0f}% from its trailing "
                        f"{BASELINE_WINDOW}-month baseline"
                    ),
                )
            )
        return items

    def _findings(
        self,
        anomalies: list[Anomaly],
        dimension_anomalies: list[Anomaly],
        metrics: list[Metric],
        context: AnalysisContext,
    ) -> list[Finding]:
        findings: list[Finding] = []
        period_label = context.plan.period or "all periods"

        if not anomalies and not dimension_anomalies:
            findings.append(
                build_finding(
                    title="No unusual operational patterns detected",
                    detail=(
                        f"No monthly observation of {', '.join(m.value for m in metrics)} exceeded "
                        f"a modified z-score of {MODIFIED_Z_THRESHOLD} or deviated more than "
                        f"{BASELINE_DEVIATION_PERCENT:.0f}% from its trailing "
                        f"{BASELINE_WINDOW}-month baseline in {period_label}."
                    ),
                    severity=Severity.INFO,
                    metric=context.plan.metric.value if context.plan.metric else None,
                    period=period_label,
                    is_statistical=True,
                    statistic="0 of the monthly observations crossed either threshold",
                )
            )
            return findings

        for anomaly in _rank(anomalies)[: context.top_n]:
            direction = anomaly.direction
            spec = METRIC_SPECS[anomaly.metric]
            unit_label = (
                "hours"
                if spec["unit"] is Unit.HOURS
                else "points"
                if spec["unit"] is Unit.PERCENT
                else "items"
                if spec["unit"] is Unit.ITEMS
                else "operations"
            )
            comparison = (
                f" against a trailing baseline of {anomaly.baseline:,.2f}"
                f" ({anomaly.baseline_deviation_percent:+.1f}%)"
                if anomaly.baseline is not None and anomaly.baseline_deviation_percent is not None
                else ""
            )
            findings.append(
                build_finding(
                    title=f"Unusual {anomaly.metric.value.replace('_', ' ')} in {anomaly.period}",
                    detail=(
                        f"{anomaly.metric.value.replace('_', ' ')} was {anomaly.value:,.2f} "
                        f"{unit_label} in {anomaly.period}{comparison}. This is "
                        f"{'above' if direction == 'spike' else 'below'} the level the rest of "
                        "the series would predict."
                    ),
                    severity=_severity(anomaly),
                    metric=anomaly.metric.value,
                    period=anomaly.period,
                    is_statistical=True,
                    statistic=anomaly.statistic,
                )
            )

        clustered = _cluster_by_dimension(dimension_anomalies)
        for label, count, worst in clustered[: context.top_n]:
            findings.append(
                build_finding(
                    title=f"Recurring anomalies in {label}",
                    detail=(
                        f"{label} recorded {count} anomalous monthly observation(s) across the "
                        f"analysed metrics, the most recent or largest being {worst.period} at "
                        f"{worst.value:,.2f}. Repeated anomalies in one area are more likely to be "
                        "a structural issue than a one-off event."
                    ),
                    severity=_severity(worst),
                    metric=worst.metric.value,
                    dimension=label,
                    period=worst.period,
                    is_statistical=True,
                    statistic=f"{count} flagged observation(s) attributed to this dimension value",
                )
            )
        return findings

    def _caveats(
        self, frame: pd.DataFrame, metrics: list[Metric], context: AnalysisContext, found: bool
    ) -> list[str]:
        caveats = [
            "anomalies are identified on monthly aggregates, so a shift confined to a few days "
            "inside one month is not detectable here"
        ]
        if not found:
            caveats.append(
                "with 12 monthly observations these detectors have limited power; an absence of "
                "flags is weak evidence that operations are stable"
            )
        months = frame["period"].nunique()
        if months < MIN_OBSERVATIONS:
            caveats.append(
                f"the analysis window contains {months} monthly bucket(s), below the "
                f"{MIN_OBSERVATIONS} needed for a meaningful detection power"
            )
        return caveats


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def modified_z_scores(values: list[float]) -> list[float | None]:
    """Median/MAD based z-scores. Returns ``None`` when MAD is zero."""
    if len(values) < 2:
        return [None] * len(values)
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    if mad <= 0:
        return [None] * len(values)
    return [round(0.6745 * (value - median) / mad, 4) for value in values]


def detect(series: list[dict], metric: Metric) -> list[Anomaly]:
    """Run both detectors over a monthly series."""
    values = [float(row["value"]) for row in series]
    periods = [str(row["period"]) for row in series]
    anomalies: list[Anomaly] = []

    scores = modified_z_scores(values)
    for period, value, score in zip(periods, values, scores, strict=True):
        if score is None or abs(score) <= MODIFIED_Z_THRESHOLD:
            continue
        baseline = statistics.median(values)
        anomalies.append(
            Anomaly(
                period=period,
                value=round_or_none(value) or 0.0,
                detector="robust_z_score",
                statistic=(
                    f"modified z-score = {score:+.2f} against the series median, threshold "
                    f"+/-{MODIFIED_Z_THRESHOLD}"
                ),
                statistic_value=score,
                threshold=MODIFIED_Z_THRESHOLD,
                baseline=round_or_none(baseline),
                baseline_deviation_percent=percentage_change(value, baseline),
                metric=metric,
            )
        )

    for index in range(BASELINE_WINDOW, len(values)):
        window = values[index - BASELINE_WINDOW : index]
        baseline = statistics.fmean(window)
        deviation = percentage_change(values[index], baseline)
        if deviation is None or abs(deviation) < BASELINE_DEVIATION_PERCENT:
            continue
        # Skip points already reported by the robust z-score detector.
        if any(a.period == periods[index] and a.detector == "robust_z_score" for a in anomalies):
            continue
        anomalies.append(
            Anomaly(
                period=periods[index],
                value=round_or_none(values[index]) or 0.0,
                detector="trailing_baseline",
                statistic=(
                    f"{deviation:+.1f}% versus the mean of the preceding {BASELINE_WINDOW} "
                    f"months ({baseline:,.2f}); threshold +/-{BASELINE_DEVIATION_PERCENT:.0f}%"
                ),
                statistic_value=deviation,
                threshold=BASELINE_DEVIATION_PERCENT,
                baseline=round_or_none(baseline),
                baseline_deviation_percent=deviation,
                metric=metric,
            )
        )
    return anomalies


def _rank(anomalies: list[Anomaly]) -> list[Anomaly]:
    """Order anomalies by how far they sit from their own baseline."""
    return sorted(
        anomalies,
        key=lambda item: (
            abs(item.baseline_deviation_percent or 0.0),
            abs(item.statistic_value or 0.0),
        ),
        reverse=True,
    )


def _cluster_by_dimension(anomalies: list[Anomaly]) -> list[tuple[str, int, Anomaly]]:
    grouped: dict[str, list[Anomaly]] = {}
    for anomaly in anomalies:
        if anomaly.dimension:
            grouped.setdefault(anomaly.dimension, []).append(anomaly)
    return sorted(
        ((label, len(items), _rank(items)[0]) for label, items in grouped.items()),
        key=lambda item: (item[1], abs(item[2].baseline_deviation_percent or 0.0)),
        reverse=True,
    )


def _severity(anomaly: Anomaly) -> Severity:
    deviation = abs(anomaly.baseline_deviation_percent or 0.0)
    adverse = (anomaly.direction == "spike") == (anomaly.metric not in LOWER_IS_BETTER)
    if deviation >= 60.0:
        return Severity.CRITICAL if adverse else Severity.MEDIUM
    if deviation >= 30.0:
        return Severity.HIGH if adverse else Severity.LOW
    if deviation >= 15.0:
        return Severity.MEDIUM if adverse else Severity.LOW
    return Severity.LOW


def anomaly_month_series(
    frame: pd.DataFrame, context: AnalysisContext, metric: Metric
) -> list[dict]:
    """Public helper used by the dashboard service."""
    return sorted(
        (row for row in monthly_series(frame, metric, context) if row["value"] is not None),
        key=lambda row: period_sort_key(row["period"]),
    )


__all__ = [
    "MODIFIED_Z_THRESHOLD",
    "Anomaly",
    "AnomalyAnalyzer",
    "detect",
    "modified_z_scores",
]
