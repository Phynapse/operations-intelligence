"""Vendor analysis: which third parties are contributing to operational delay.

Two sources are combined deliberately:

* ``orders.csv`` gives **order-level** reliability: on-time rate against the
  customer promise, and average transit time, per vendor;
* ``operations.csv`` gives **daily operational** reliability: the share of a
  vendor's work that ran late against the internal process promise.

Reporting both keeps the analysis honest. A vendor can look good on orders and
poor on operations (or the reverse) when it only affects part of the flow, and
that distinction is exactly what a manager needs before acting.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.analytics.base import (
    AnalysisContext,
    AnalysisOutcome,
    BaseAnalyzer,
    build_evidence,
    build_finding,
    build_kpi,
)
from app.analytics.cycle_time_analyzer import cycle_components
from app.models.enums import AnalysisType, DatasetId, Severity, Unit
from app.schemas.analysis import EvidenceItem, Finding
from app.utils.calculations import round_or_none, safe_float
from app.utils.errors import InsufficientDataError

#: On-time rate at or below this puts a vendor on the watchlist.
WATCHLIST_ON_TIME_THRESHOLD = 85.0
#: Operational delay rate at or above this puts a vendor on the watchlist.
WATCHLIST_DELAY_RATE_THRESHOLD = 10.0
#: Orders below this count make a rate too noisy to act on.
MIN_ORDERS_FOR_RATE = 20


class VendorAnalyzer(BaseAnalyzer):
    """Rank vendors by delay contribution and explain each ranking."""

    name = "vendor_analyzer"
    supported_types = frozenset({AnalysisType.VENDOR_ANALYSIS})
    required_datasets = (
        DatasetId.ORDERS,
        DatasetId.OPERATIONS,
        DatasetId.VENDORS,
        DatasetId.DEPARTMENTS,
    )

    def analyze(self, context: AnalysisContext) -> AnalysisOutcome:
        orders = context.frame(DatasetId.ORDERS)
        operations = context.primary_frame()
        if orders.empty and operations.empty:
            raise InsufficientDataError(
                f"no records found for {context.plan.period or 'the requested period'}"
            )

        outcome = AnalysisOutcome()
        period_label = context.plan.period or "all periods"
        comparison_orders = context.frame(DatasetId.ORDERS, window="comparison")
        comparison_operations = context.frame(DatasetId.OPERATIONS, window="comparison")

        vendor_names = _vendor_names(context)
        profiles = vendor_table(orders, operations, vendor_names)
        if profiles.empty:
            raise InsufficientDataError("no vendor-linked records were found in this period")

        watchlist = profiles.loc[
            (profiles["on_time_rate"] <= WATCHLIST_ON_TIME_THRESHOLD)
            | (profiles["delay_rate"] >= WATCHLIST_DELAY_RATE_THRESHOLD)
        ].sort_values("late_orders", ascending=False)

        leader = profiles.sort_values("late_orders", ascending=False).iloc[0]
        outcome.kpis = [
            build_kpi(
                name="Vendors with late orders",
                value=float(len(watchlist)),
                unit=Unit.COUNT,
                source_dataset=DatasetId.ORDERS,
                period=period_label,
                calculation=(
                    f"count of vendors with an on-time rate at or below "
                    f"{WATCHLIST_ON_TIME_THRESHOLD:.0f}% or an operational delay rate at or "
                    f"above {WATCHLIST_DELAY_RATE_THRESHOLD:.0f}%"
                ),
            ),
            build_kpi(
                name="Largest delay contributor",
                value=safe_float(leader["late_orders"]),
                unit=Unit.ORDERS,
                source_dataset=DatasetId.ORDERS,
                period=period_label,
                dimension=str(leader["vendor_name"]),
                calculation="count of late or cancelled orders attributed to the vendor",
            ),
            build_kpi(
                name="Overall on-time delivery rate",
                value=on_time_rate(
                    float(profiles["late_orders"].sum()),
                    float(profiles["counted_orders"].sum()),
                ),
                unit=Unit.PERCENT,
                source_dataset=DatasetId.ORDERS,
                period=period_label,
                calculation="on-time orders / non-cancelled orders * 100 across all vendors",
            ),
        ]

        outcome.evidence = self._evidence(profiles, watchlist, context, period_label)

        if not comparison_orders.empty or not comparison_operations.empty:
            outcome.evidence.extend(
                self._comparison_evidence(
                    profiles,
                    vendor_table(comparison_orders, comparison_operations, vendor_names),
                    context,
                    period_label,
                )
            )

        outcome.findings.extend(self._findings(profiles, watchlist, context, period_label))
        outcome.extend_caveats(*self._caveats(profiles, period_label))
        outcome.row_count_examined = len(orders) + len(operations)
        return outcome

    # -- internals -----------------------------------------------------
    def _evidence(
        self,
        profiles: pd.DataFrame,
        watchlist: pd.DataFrame,
        context: AnalysisContext,
        period_label: str,
    ) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        if not watchlist.empty:
            worst = watchlist.iloc[0]
            items.append(
                build_evidence(
                    metric="Vendor on-time delivery rate",
                    value=round_or_none(worst["on_time_rate"]),
                    unit=Unit.PERCENT,
                    source_dataset=DatasetId.ORDERS,
                    dimension=str(worst["vendor_name"]),
                    period=period_label,
                    calculation_description=(
                        "On-time orders / non-cancelled orders * 100 per vendor, ranked lowest "
                        "first; the weakest vendor is reported"
                    ),
                    sample_size=int(worst["counted_orders"]),
                    severity=_vendor_severity(worst),
                )
            )
            items.append(
                build_evidence(
                    metric="Vendor operational delay rate",
                    value=round_or_none(worst["delay_rate"]),
                    unit=Unit.PERCENT,
                    source_dataset=DatasetId.OPERATIONS,
                    dimension=str(worst["vendor_name"]),
                    period=period_label,
                    calculation_description=(
                        "Delayed operations / completed operations * 100 per vendor, from "
                        "vendor-linked operations.csv records, ranked highest first"
                    ),
                    sample_size=int(worst["operations_completed"]),
                    severity=_vendor_severity(worst),
                )
            )
            items.append(
                build_evidence(
                    metric="Vendor average transit time",
                    value=round_or_none(worst["avg_transit"]),
                    unit=Unit.HOURS,
                    source_dataset=DatasetId.ORDERS,
                    dimension=str(worst["vendor_name"]),
                    period=period_label,
                    calculation_description=(
                        "Mean of transit_time_hours over non-cancelled orders routed to the "
                        "leading vendor"
                    ),
                    sample_size=int(worst["counted_orders"]),
                )
            )
        items.extend(_reference_evidence(context, period_label))
        return items

    def _comparison_evidence(
        self,
        current: pd.DataFrame,
        comparison: pd.DataFrame,
        context: AnalysisContext,
        period_label: str,
    ) -> list[EvidenceItem]:
        if comparison.empty:
            return []
        joined = current[["vendor_name", "on_time_rate"]].merge(
            comparison[["vendor_name", "on_time_rate"]],
            on="vendor_name",
            how="inner",
            suffixes=("", "_comparison"),
        )
        if joined.empty:
            return []
        joined["change"] = joined["on_time_rate"] - joined["on_time_rate_comparison"]
        ordered = joined.reindex(joined["change"].abs().sort_values(ascending=False).index)
        row = ordered.iloc[0].to_dict()
        previous = row.pop("on_time_rate_comparison")
        return [
            build_evidence(
                metric="Largest vendor reliability movement",
                value=round_or_none(row["on_time_rate"]),
                comparison_value=round_or_none(previous),
                unit=Unit.PERCENT,
                source_dataset=DatasetId.ORDERS,
                dimension=str(row["vendor_name"]),
                calculation_description=(
                    "On-time rate per vendor in both periods, ranked by absolute difference "
                    "between periods"
                ),
                severity=_delta_severity(row["change"]),
            )
        ]

    def _findings(
        self,
        profiles: pd.DataFrame,
        watchlist: pd.DataFrame,
        context: AnalysisContext,
        period_label: str,
    ) -> list[Finding]:
        findings: list[Finding] = []
        if watchlist.empty:
            findings.append(
                build_finding(
                    title="No vendor is breaching its reliability thresholds",
                    detail=(
                        f"Every vendor in {period_label} held an on-time rate above "
                        f"{WATCHLIST_ON_TIME_THRESHOLD:.0f}% and an operational delay rate below "
                        f"{WATCHLIST_DELAY_RATE_THRESHOLD:.0f}%."
                    ),
                    severity=Severity.INFO,
                    metric="on_time_delivery_rate",
                    period=period_label,
                )
            )
            return findings

        for _, row in watchlist.head(context.top_n).iterrows():
            name = str(row["vendor_name"])
            parts: list[str] = []
            if row["counted_orders"] >= MIN_ORDERS_FOR_RATE:
                parts.append(
                    f"delivered {row['late_orders']:,.0f} of {row['counted_orders']:,.0f} "
                    f"orders late ({row['on_time_rate']:.1f}% on time)"
                )
            if row["operations_completed"] > 0:
                parts.append(
                    f"an operational delay rate of {row['delay_rate']:.1f}% across "
                    f"{row['operations_completed']:,.0f} operations"
                )
            if safe_float(row["avg_transit"]) is not None:
                parts.append(
                    f"an average transit time of {row['avg_transit']:.2f} hours against a "
                    f"portfolio average of {row['avg_transit_portfolio']:.2f} hours"
                )
            findings.append(
                build_finding(
                    title=f"Delay contributor: {name}",
                    detail="; ".join(parts).capitalize() + f" in {period_label}.",
                    severity=_vendor_severity(row),
                    metric="on_time_delivery_rate",
                    dimension=name,
                    period=period_label,
                    statistic=_rate_pair_statistic(row),
                )
            )

        divergent = profiles.loc[
            (profiles["on_time_rate"] <= WATCHLIST_ON_TIME_THRESHOLD)
            & (profiles["delay_rate"] < WATCHLIST_DELAY_RATE_THRESHOLD)
        ]
        if not divergent.empty:
            row = divergent.iloc[0]
            findings.append(
                build_finding(
                    title=f"Delivery-only problem at {row['vendor_name']}",
                    detail=(
                        f"{row['vendor_name']} misses customer promises "
                        f"({row['on_time_rate']:.1f}% on time) while its internal operations stay "
                        f"within tolerance ({row['delay_rate']:.1f}% delayed). The delay is "
                        "therefore in transit or handover rather than in our own handling."
                    ),
                    severity=Severity.MEDIUM,
                    metric="on_time_delivery_rate",
                    dimension=str(row["vendor_name"]),
                    period=period_label,
                    statistic=_rate_pair_statistic(row),
                )
            )
        return findings

    def _caveats(self, profiles: pd.DataFrame, period_label: str) -> list[str]:
        caveats = [
            "vendor attribution uses the primary vendor recorded against each operational record; "
            "work split between vendors is attributed to that single vendor"
        ]
        thin = profiles.loc[profiles["counted_orders"] < MIN_ORDERS_FOR_RATE]
        if not thin.empty:
            caveats.append(
                f"{len(thin)} vendor(s) have fewer than {MIN_ORDERS_FOR_RATE} orders in "
                f"{period_label}; their rates are reported but are too noisy to act on"
            )
        absent = [
            name
            for name, group in profiles.groupby("vendor_name")
            if group["counted_orders"].sum() == 0
        ]
        if absent:
            caveats.append(
                f"{len(absent)} vendor(s) appear only in operations.csv and have no order-level "
                "reliability data for this period"
            )
        return caveats


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def vendor_table(
    orders: pd.DataFrame | None,
    operations: pd.DataFrame | None,
    vendor_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Per-vendor order-level and operational reliability metrics.

    ``vendor_names`` maps ``vendor_id`` to a display name. It is supplied by the
    caller from ``vendors.csv`` so this module holds no vendor reference data of
    its own.
    """
    columns = [
        "vendor_id",
        "vendor_name",
        "counted_orders",
        "late_orders",
        "on_time_rate",
        "avg_transit",
        "avg_promised",
        "operations_completed",
        "operations_delayed",
        "delay_rate",
        "avg_cycle_time",
    ]
    frames: list[pd.DataFrame] = []

    if orders is not None and not orders.empty:
        working = orders.loc[orders["vendor_id"].notna()].copy()
        if not working.empty:
            cancelled = working["status"] == "cancelled"
            counted = working.loc[~cancelled]
            grouped = pd.DataFrame(
                {
                    "vendor_id": counted["vendor_id"],
                    "counted_orders": 1,
                    "late_orders": (counted["status"] == "delivered_late").astype(int),
                    "transit_total": counted["transit_time_hours"].fillna(0.0),
                    "promised_total": counted["promised_time_hours"].fillna(0.0),
                }
            )
            order_agg = (
                grouped.groupby("vendor_id")
                .agg(
                    counted_orders=("counted_orders", "sum"),
                    late_orders=("late_orders", "sum"),
                    transit_total=("transit_total", "sum"),
                    promised_total=("promised_total", "sum"),
                )
                .reset_index()
            )
            order_agg["on_time_rate"] = [
                round_or_none((total - late) / total * 100.0)
                for total, late in zip(
                    order_agg["counted_orders"], order_agg["late_orders"], strict=True
                )
            ]
            order_agg["avg_transit"] = [
                round_or_none(total / count) if count else None
                for total, count in zip(
                    order_agg["transit_total"], order_agg["counted_orders"], strict=True
                )
            ]
            order_agg["avg_promised"] = [
                round_or_none(total / count) if count else None
                for total, count in zip(
                    order_agg["promised_total"], order_agg["counted_orders"], strict=True
                )
            ]
            _ = cancelled.sum()
            frames.append(order_agg.drop(columns=["transit_total", "promised_total"]))

    if operations is not None and not operations.empty:
        working = operations.loc[operations["vendor_id"].notna()].copy()
        if not working.empty:
            working["cycle_total"] = working["total_cycle_time_hours"].fillna(0.0) * working[
                "completed_operations"
            ].fillna(0)
            operation_agg = (
                working.groupby("vendor_id")
                .agg(
                    operations_completed=("completed_operations", "sum"),
                    operations_delayed=("delayed_operations", "sum"),
                    cycle_total=("cycle_total", "sum"),
                )
                .reset_index()
            )
            operation_agg["delay_rate"] = [
                round_or_none(delayed / completed * 100.0) if completed else None
                for completed, delayed in zip(
                    operation_agg["operations_completed"],
                    operation_agg["operations_delayed"],
                    strict=True,
                )
            ]
            operation_agg["avg_cycle_time"] = [
                round_or_none(total / completed) if completed else None
                for total, completed in zip(
                    operation_agg["cycle_total"], operation_agg["operations_completed"], strict=True
                )
            ]
            frames.append(operation_agg.drop(columns=["cycle_total"]))

    if not frames:
        return pd.DataFrame(columns=columns)

    merged = frames[0]
    for extra in frames[1:]:
        merged = merged.merge(extra, on="vendor_id", how="outer")

    for column in columns:
        if column not in merged.columns:
            merged[column] = pd.NA
    merged = merged[columns].copy()
    merged["counted_orders"] = merged["counted_orders"].fillna(0).astype(int)
    merged["late_orders"] = merged["late_orders"].fillna(0).astype(int)
    merged["operations_completed"] = merged["operations_completed"].fillna(0).astype(int)
    merged["operations_delayed"] = merged["operations_delayed"].fillna(0).astype(int)
    lookup = dict(vendor_names or {})
    merged["vendor_name"] = merged["vendor_id"].map(lookup).fillna(merged["vendor_id"])
    merged["avg_transit_portfolio"] = _portfolio_transit(merged)
    return merged.sort_values("late_orders", ascending=False).reset_index(drop=True)


def _portfolio_transit(table: pd.DataFrame) -> float:
    values = table.loc[table["counted_orders"] > 0, "avg_transit"].dropna()
    if values.empty:
        return 0.0
    return round_or_none(values.mean()) or 0.0


def _vendor_names(context: AnalysisContext) -> dict[str, str]:
    """Map vendor ids to display names, sourced from vendors.csv."""
    frame = vendor_reference(context)
    if frame.empty:
        return {}
    return {str(row["vendor_id"]): str(row["vendor_name"]) for _, row in frame.iterrows()}


def vendor_reference(context: AnalysisContext) -> pd.DataFrame:
    """The static vendor profile dataset, used as corroborating evidence."""
    try:
        return context.frame(DatasetId.VENDORS)
    except Exception:  # pragma: no cover - reference data is optional
        return pd.DataFrame()


def _reference_evidence(context: AnalysisContext, period_label: str) -> list[EvidenceItem]:
    reference = vendor_reference(context)
    if reference.empty:
        return []
    worst = reference.sort_values("delay_rate", ascending=False).iloc[0]
    return [
        build_evidence(
            metric="Vendor reference delay rate",
            value=round_or_none(worst["delay_rate"]),
            unit=Unit.PERCENT,
            source_dataset=DatasetId.VENDORS,
            dimension=str(worst["vendor_name"]),
            period=period_label,
            calculation_description=(
                "delay_rate as published in vendors.csv; this is a full-period reference figure, "
                "not a value computed for the current period"
            ),
            sample_size=None,
            is_estimate=True,
            severity=_rate_severity(worst["delay_rate"]),
        )
    ]


def _vendor_severity(row: pd.Series) -> Severity:
    on_time = safe_float(row.get("on_time_rate")) or 100.0
    delay = safe_float(row.get("delay_rate")) or 0.0
    if on_time <= 60.0 or delay >= 25.0:
        return Severity.CRITICAL
    if on_time <= 75.0 or delay >= 18.0:
        return Severity.HIGH
    if on_time <= WATCHLIST_ON_TIME_THRESHOLD or delay >= WATCHLIST_DELAY_RATE_THRESHOLD:
        return Severity.MEDIUM
    return Severity.LOW


def _rate_severity(rate: float | None) -> Severity:
    value = safe_float(rate)
    if value is None:
        return Severity.INFO
    if value >= 25.0:
        return Severity.HIGH
    if value >= WATCHLIST_DELAY_RATE_THRESHOLD:
        return Severity.MEDIUM
    return Severity.LOW


def _delta_severity(change: float | None) -> Severity:
    magnitude = abs(safe_float(change) or 0.0)
    if magnitude >= 20.0:
        return Severity.HIGH
    if magnitude >= 8.0:
        return Severity.MEDIUM
    return Severity.LOW


def cycle_breakdown(frame: pd.DataFrame) -> dict[str, Any]:
    """Cycle-time decomposition for a vendor-linked frame (helper for services)."""
    return cycle_components(frame)


def on_time_rate(late: float | None, total: float | None) -> float | None:
    """``(total - late) / total * 100`` with a zero-total guard."""
    total_value = safe_float(total)
    late_value = safe_float(late)
    if total_value is None or total_value <= 0 or late_value is None:
        return None
    return round_or_none((total_value - late_value) / total_value * 100.0)


def _rate_pair_statistic(row: Mapping[str, object] | pd.Series) -> str:
    """Render a vendor's two rates, omitting any that could not be measured.

    A vendor with no orders in the window has an undefined on-time rate.
    Formatting that as ``nan%`` would put a meaningless figure in front of a
    reader, so an unmeasured rate is named as unavailable instead.
    """
    parts: list[str] = []
    on_time = safe_float(row["on_time_rate"])
    delay = safe_float(row["delay_rate"])
    parts.append(
        f"on-time rate {on_time:.1f}%" if on_time is not None else "on-time rate unavailable"
    )
    parts.append(f"delay rate {delay:.1f}%" if delay is not None else "delay rate unavailable")
    return ", ".join(parts)


__all__ = [
    "WATCHLIST_DELAY_RATE_THRESHOLD",
    "WATCHLIST_ON_TIME_THRESHOLD",
    "VendorAnalyzer",
    "on_time_rate",
    "vendor_table",
]
