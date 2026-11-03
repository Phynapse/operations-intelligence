"""Bottleneck analysis: rank operational areas by constrained capacity.

A bottleneck is not simply "the slowest process". A process is a bottleneck when
its own demand outruns the capacity it can serve, so this analyzer combines
three deterministic signals into a single explainable score:

1. **Cycle time** - how long work takes once it is picked up (weight 0.45).
2. **Queue share** - how much of that time is spent waiting (weight 0.35).
3. **Backlog pressure** - open items relative to recent daily throughput (weight 0.20).

Each signal is min-max normalised across the compared set before weighting, so
the score is scale-free and no single signal with a large natural range dominates.
The component values that produced the score are always emitted alongside it, so
a reader can disagree with the ranking without re-deriving it.
"""

from __future__ import annotations

import pandas as pd

from app.analytics.base import (
    AnalysisContext,
    AnalysisOutcome,
    BaseAnalyzer,
    build_evidence,
    build_finding,
    build_kpi,
)
from app.models.enums import AnalysisType, DatasetId, Severity, Unit
from app.schemas.analysis import EvidenceItem, Finding
from app.utils.calculations import normalise_series, round_or_none, safe_float
from app.utils.errors import InsufficientDataError

DIMENSION_COLUMNS: dict[str, str] = {
    "department": "department",
    "process": "process",
    "operation_type": "operation_type",
}

#: Signal weights. They sum to 1.0.
WEIGHTS: dict[str, float] = {"cycle": 0.45, "queue_share": 0.35, "backlog_pressure": 0.20}

#: A segment scoring at or above this on the 0-1 normalised scale is flagged.
BOTTLENECK_SCORE_THRESHOLD = 0.60


class BottleneckAnalyzer(BaseAnalyzer):
    """Rank departments or processes by an explainable constraint score."""

    name = "bottleneck_analyzer"
    supported_types = frozenset({AnalysisType.BOTTLENECK_ANALYSIS})
    required_datasets = (DatasetId.OPERATIONS, DatasetId.DEPARTMENTS)

    def analyze(self, context: AnalysisContext) -> AnalysisOutcome:
        primary = context.primary_frame()
        if primary.empty:
            raise InsufficientDataError(
                f"no operational records found for {context.plan.period or 'the requested period'}"
            )

        outcome = AnalysisOutcome()
        period_label = context.plan.period or "all periods"
        column = self._dimension_column(context)
        comparison = context.comparison_frame()

        table = bottleneck_table(primary, column)
        if table.empty:
            raise InsufficientDataError(f"not enough records to rank bottlenecks by {column}")

        outcome.kpis = [
            build_kpi(
                name="Top bottleneck",
                value=round_or_none(table.iloc[0]["score"], 4),
                unit=Unit.PERCENT,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                dimension=_label(table.iloc[0], column, context.repository.department_names),
                calculation=(
                    "weighted 0-1 constraint score: "
                    f"{WEIGHTS['cycle']}*cycle + {WEIGHTS['queue_share']}*queue share + "
                    f"{WEIGHTS['backlog_pressure']}*backlog pressure, reported as a percentage"
                ),
            ),
            build_kpi(
                name="Segments above bottleneck threshold",
                value=float(int((table["score"] >= BOTTLENECK_SCORE_THRESHOLD).sum())),
                unit=Unit.COUNT,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                calculation=(
                    f"count of {column} values with a constraint score at or above "
                    f"{BOTTLENECK_SCORE_THRESHOLD:.2f}"
                ),
            ),
        ]

        leader = table.iloc[0]
        leader_label = _label(leader, column, context.repository.department_names)
        outcome.evidence = [
            build_evidence(
                metric=f"Bottleneck constraint score by {column}",
                value=round_or_none(leader["score"], 4),
                unit=Unit.PERCENT,
                source_dataset=DatasetId.OPERATIONS,
                dimension=leader_label,
                period=period_label,
                calculation_description=(
                    "Min-max normalised weighted score of average cycle time (0.45), queue share "
                    "(0.35) and backlog pressure (0.20); the highest-scoring segment is reported"
                ),
                sample_size=int(leader["volume"]),
                severity=_score_severity(leader["score"]),
            ),
            build_evidence(
                metric=f"Average cycle time of the leading bottleneck by {column}",
                value=round_or_none(leader["cycle"]),
                unit=Unit.HOURS,
                source_dataset=DatasetId.OPERATIONS,
                dimension=leader_label,
                period=period_label,
                calculation_description=(
                    "completed_operations-weighted mean of total_cycle_time_hours for the "
                    "highest-scoring segment"
                ),
                sample_size=int(leader["volume"]),
            ),
            build_evidence(
                metric=f"Queue share of the leading bottleneck by {column}",
                value=round_or_none(leader["queue_share"]),
                unit=Unit.PERCENT,
                source_dataset=DatasetId.OPERATIONS,
                dimension=leader_label,
                period=period_label,
                calculation_description=(
                    "queue_time_hours / total_cycle_time_hours * 100 for the highest-scoring "
                    "segment, weighted by completed_operations"
                ),
                sample_size=int(leader["volume"]),
                severity=_queue_severity(leader["queue_share"]),
            ),
            build_evidence(
                metric=f"Backlog pressure of the leading bottleneck by {column}",
                value=round_or_none(leader["backlog_pressure"]),
                unit=Unit.PERCENT,
                source_dataset=DatasetId.OPERATIONS,
                dimension=leader_label,
                period=period_label,
                calculation_description=(
                    "Closing backlog_open_items divided by average daily completed_operations, "
                    "expressed as the number of days of outstanding work"
                ),
            ),
        ]

        flagged = table.loc[table["score"] >= BOTTLENECK_SCORE_THRESHOLD].head(context.top_n)
        department_names = context.repository.department_names
        outcome.evidence.extend(
            _runner_up_evidence(table.head(context.top_n), column, department_names)
        )

        outcome.findings.extend(
            _bottleneck_findings(table, column, department_names, period_label, flagged)
        )

        if comparison is not None and not comparison.empty:
            comparison_table = bottleneck_table(comparison, column)
            outcome.evidence.extend(
                _shift_evidence(table, comparison_table, column, department_names)
            )
            outcome.findings.extend(
                _escalation_findings(table, comparison_table, column, department_names)
            )

        if len(table) < 3:
            outcome.extend_caveats(
                f"only {len(table)} {column} value(s) were compared; min-max normalisation is "
                "unreliable with very few segments"
            )
        outcome.row_count_examined = len(primary)
        return outcome

    @staticmethod
    def _dimension_column(context: AnalysisContext) -> str:
        for dimension in context.plan.dimensions:
            column = DIMENSION_COLUMNS.get(dimension.value)
            if column:
                return column
        return "process"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def bottleneck_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Compute the constraint score for each value of ``column``.

    Returns an empty frame when there are no usable rows. The score is only
    meaningful relative to the other rows in the same table, so the normalising
    range is derived here rather than persisted anywhere.
    """
    if frame is None or frame.empty or column not in frame.columns:
        return pd.DataFrame(
            columns=[column, "cycle", "queue_share", "backlog_pressure", "score", "volume"]
        )

    ordered = frame.sort_values("date")
    last_date = ordered["date"].max()
    daily_volume = float(ordered["completed_operations"].sum()) / max(
        1, int(ordered["date"].nunique())
    )
    closing = (
        ordered.loc[ordered["date"] == last_date]
        .groupby(column, dropna=True)["backlog_open_items"]
        .sum()
    )

    rows: list[dict[str, object]] = []
    for key, group in ordered.groupby(column, dropna=True):
        weights = group["completed_operations"].fillna(0).astype("float64")
        if float(weights.sum()) <= 0:
            weights = pd.Series(1.0, index=group.index)
        cycle = _weighted(group["total_cycle_time_hours"], weights)
        queue = _weighted(group["queue_time_hours"], weights)
        processing = _weighted(group["processing_time_hours"], weights)
        total = (queue or 0.0) + (processing or 0.0)
        queue_share = round_or_none((queue or 0.0) / total * 100.0) if total > 0 else 0.0
        backlog_value = safe_float(closing.get(key)) or 0.0
        rows.append(
            {
                column: key,
                "cycle": round_or_none(cycle),
                "queue": round_or_none(queue),
                "queue_share": queue_share or 0.0,
                "backlog": round_or_none(backlog_value, 1),
                "backlog_pressure": round_or_none(backlog_value / daily_volume)
                if daily_volume > 0
                else 0.0,
                "volume": int(group["completed_operations"].sum()),
                "days": int(group["date"].nunique()),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    cycle_norm = normalise_series(table["cycle"].tolist())
    queue_norm = normalise_series(table["queue_share"].tolist())
    pressure_norm = normalise_series(table["backlog_pressure"].tolist())
    table["cycle_norm"] = [v or 0.0 for v in cycle_norm]
    table["queue_norm"] = [v or 0.0 for v in queue_norm]
    table["pressure_norm"] = [v or 0.0 for v in pressure_norm]
    table["score"] = [
        round(
            WEIGHTS["cycle"] * (c or 0.0)
            + WEIGHTS["queue_share"] * (q or 0.0)
            + WEIGHTS["backlog_pressure"] * (p or 0.0),
            4,
        )
        for c, q, p in zip(
            table["cycle_norm"], table["queue_norm"], table["pressure_norm"], strict=True
        )
    ]
    return table.sort_values("score", ascending=False).reset_index(drop=True)


def _weighted(values: pd.Series, weights: pd.Series) -> float | None:
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return round_or_none(values.mean())
    return round_or_none(float((values * weights).sum() / total_weight))


def _runner_up_evidence(
    table: pd.DataFrame, column: str, names: dict[str, str]
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for _, row in table.iloc[1:].iterrows():
        items.append(
            build_evidence(
                metric=f"Bottleneck constraint score by {column}",
                value=round_or_none(row["score"], 4),
                unit=Unit.PERCENT,
                source_dataset=DatasetId.OPERATIONS,
                dimension=_label(row, column, names),
                calculation_description=(
                    "Weighted 0-1 constraint score (cycle time 0.45, queue share 0.35, "
                    "backlog pressure 0.20) for this segment"
                ),
                sample_size=int(row["volume"]),
                severity=_score_severity(row["score"]),
            )
        )
    return items


def _shift_evidence(
    current: pd.DataFrame, comparison: pd.DataFrame, column: str, names: dict[str, str]
) -> list[EvidenceItem]:
    if current.empty or comparison.empty:
        return []
    joined = current[[column, "score", "cycle"]].merge(
        comparison[[column, "score", "cycle"]],
        on=column,
        how="inner",
        suffixes=("", "_comparison"),
    )
    if joined.empty:
        return []
    joined["score_change"] = joined["score"] - joined["score_comparison"]
    ordered = joined.reindex(joined["score_change"].abs().sort_values(ascending=False).index)
    row = ordered.iloc[0].to_dict()
    previous_score = row.pop("score_comparison")
    previous_cycle = row.pop("cycle_comparison")
    return [
        build_evidence(
            metric=f"Largest bottleneck score movement by {column}",
            value=round_or_none(row["score"], 4),
            comparison_value=round_or_none(previous_score, 4),
            unit=Unit.PERCENT,
            source_dataset=DatasetId.OPERATIONS,
            dimension=_label(row, column, names),
            calculation_description=(
                f"Weighted constraint score for each {column} value in both periods, ranked by "
                "absolute difference between periods"
            ),
            severity=_score_severity(row["score"]),
        ),
        build_evidence(
            metric=f"Average cycle time of the largest bottleneck movement by {column}",
            value=round_or_none(row["cycle"]),
            comparison_value=round_or_none(previous_cycle),
            unit=Unit.HOURS,
            source_dataset=DatasetId.OPERATIONS,
            dimension=_label(row, column, names),
            calculation_description=(
                "completed_operations-weighted mean of total_cycle_time_hours for the segment "
                "with the largest constraint score movement"
            ),
        ),
    ]


def _bottleneck_findings(
    table: pd.DataFrame,
    column: str,
    names: dict[str, str],
    period_label: str,
    flagged: pd.DataFrame,
) -> list[Finding]:
    findings: list[Finding] = []
    leader = table.iloc[0]
    leader_label = _label(leader, column, names)
    findings.append(
        build_finding(
            title=f"Primary constraint: {leader_label}",
            detail=(
                f"{leader_label} scores {leader['score']:.2f} on the constraint scale, the highest "
                f"of {len(table)} {column} values. Average cycle time is "
                f"{leader['cycle']:.2f} hours with {leader['queue_share']:.0f}% of that spent "
                f"waiting, and it closed {leader['backlog']:,.0f} backlog items - about "
                f"{leader['backlog_pressure']:.1f} days of its own recent throughput."
            ),
            severity=_score_severity(leader["score"]),
            metric="bottleneck_score",
            dimension=leader_label,
            period=period_label,
            statistic=(
                f"score {leader['score']:.4f} = {WEIGHTS['cycle']}*{leader['cycle_norm']:.2f} + "
                f"{WEIGHTS['queue_share']}*{leader['queue_norm']:.2f} + "
                f"{WEIGHTS['backlog_pressure']}*{leader['pressure_norm']:.2f}"
            ),
        )
    )
    for _, row in flagged.iloc[1:].iterrows():
        label = _label(row, column, names)
        findings.append(
            build_finding(
                title=f"Secondary constraint: {label}",
                detail=(
                    f"{label} also exceeds the bottleneck threshold at {row['score']:.2f}, with a "
                    f"{row['cycle']:.2f} hour average cycle time and {row['queue_share']:.0f}% "
                    "queue share."
                ),
                severity=Severity.MEDIUM,
                metric="bottleneck_score",
                dimension=label,
                period=period_label,
                statistic=f"score {row['score']:.4f}",
            )
        )
    if flagged.empty:
        findings.append(
            build_finding(
                title="No segment exceeds the bottleneck threshold",
                detail=(
                    f"No {column} value reached the constraint score of "
                    f"{BOTTLENECK_SCORE_THRESHOLD:.2f} in {period_label}."
                ),
                severity=Severity.INFO,
                metric="bottleneck_score",
                period=period_label,
            )
        )
    return findings


def _escalation_findings(
    current: pd.DataFrame, comparison: pd.DataFrame, column: str, names: dict[str, str]
) -> list[Finding]:
    if current.empty or comparison.empty:
        return []
    joined = current[[column, "score", "cycle", "queue_share"]].merge(
        comparison[[column, "score", "cycle"]], on=column, how="inner", suffixes=("", "_comparison")
    )
    if joined.empty:
        return []
    joined["score_change"] = joined["score"] - joined["score_comparison"]
    rising = joined.loc[joined["score_change"] >= 0.10].sort_values("score_change", ascending=False)
    if rising.empty:
        return []
    row = rising.iloc[0]
    label = _label(row, column, names)
    return [
        build_finding(
            title=f"Constraint worsening: {label}",
            detail=(
                f"{label}'s constraint score rose from {row['score_comparison']:.2f} to "
                f"{row['score']:.2f} between the two periods. Average cycle time moved from "
                f"{row['cycle_comparison']:.2f} to {row['cycle']:.2f} hours and queue share is "
                f"now {row['queue_share']:.0f}%."
            ),
            severity=_score_severity(row["score"]),
            metric="bottleneck_score",
            dimension=label,
            statistic=f"constraint score change = {row['score_change']:+.4f}",
        )
    ]


def _score_severity(score: float | None) -> Severity:
    value = safe_float(score)
    if value is None:
        return Severity.INFO
    if value >= 0.85:
        return Severity.CRITICAL
    if value >= BOTTLENECK_SCORE_THRESHOLD:
        return Severity.HIGH
    if value >= 0.35:
        return Severity.MEDIUM
    return Severity.LOW


def _queue_severity(queue_share: float | None) -> Severity:
    value = safe_float(queue_share)
    if value is None:
        return Severity.INFO
    if value >= 60.0:
        return Severity.HIGH
    if value >= 40.0:
        return Severity.MEDIUM
    return Severity.LOW


def _label(row: dict | pd.Series, column: str, department_names: dict[str, str]) -> str:
    value = str(row[column])
    return department_names.get(value, value)
