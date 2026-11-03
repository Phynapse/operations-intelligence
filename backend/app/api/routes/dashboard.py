"""Dashboard endpoint: pre-computed operational overview.

Returns KPIs, monthly trends for key metrics, and top bottleneck areas in a
single request so the frontend can render the dashboard without multiple round
trips.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from app.data.repository import get_repository
from app.models.enums import DatasetId
from app.utils.calculations import round_or_none, safe_float, safe_ratio
from app.utils.periods import period_sort_key

router = APIRouter()


# -- response models --------------------------------------------------------


class DashboardKPI(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    value: float | None
    unit: str
    period: str | None = None


class TrendPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    period: str
    value: float | None


class BottleneckEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    cycle_time: float | None
    queue_share: float | None
    backlog: float | None


class DepartmentOverview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    department: str
    throughput: float | None
    cycle_time: float | None
    sla_compliance: float | None
    backlog: float | None


class DashboardData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    generated_at: dt.datetime
    period_start: str | None = None
    period_end: str | None = None
    kpis: list[DashboardKPI] = Field(default_factory=list)
    throughput_trend: list[TrendPoint] = Field(default_factory=list)
    cycle_time_trend: list[TrendPoint] = Field(default_factory=list)
    sla_trend: list[TrendPoint] = Field(default_factory=list)
    backlog_trend: list[TrendPoint] = Field(default_factory=list)
    bottlenecks: list[BottleneckEntry] = Field(default_factory=list)
    departments: list[DepartmentOverview] = Field(default_factory=list)


# -- endpoint ---------------------------------------------------------------


@router.get("/dashboard", response_model=DashboardData)
async def dashboard() -> DashboardData:
    repo = get_repository()
    ops = repo.frame(DatasetId.OPERATIONS)
    sla = repo.frame(DatasetId.SLA_METRICS)

    periods = sorted(ops["period"].dropna().unique(), key=period_sort_key)
    period_start = periods[0] if periods else None
    period_end = periods[-1] if periods else None

    # --- headline KPIs (full period) ---
    total_completed = safe_float(ops["completed_operations"].sum())
    total_delayed = safe_float(ops["delayed_operations"].sum())
    n_days = ops["date"].nunique() if "date" in ops.columns else 1

    throughput_per_day = round_or_none(total_completed / n_days if n_days else None, 1)
    avg_cycle = _weighted_mean(ops, "total_cycle_time_hours", "completed_operations")
    queue_share = _queue_share(ops)
    delay_rate = round_or_none(safe_ratio(total_delayed, total_completed) * 100 if total_completed else None, 2)

    sla_met = safe_float(sla["sla_met_cases"].sum()) if "sla_met_cases" in sla.columns else 0
    sla_total = safe_float(sla["total_cases"].sum()) if "total_cases" in sla.columns else 0
    sla_compliance = round_or_none(safe_ratio(sla_met, sla_total) * 100 if sla_total else None, 2)

    last_day = ops.loc[ops["date"] == ops["date"].max()] if "date" in ops.columns else ops
    backlog = round_or_none(safe_float(last_day["backlog_open_items"].sum()), 0)

    kpis = [
        DashboardKPI(name="Throughput / day", value=throughput_per_day, unit="operations", period=f"{period_start} to {period_end}"),
        DashboardKPI(name="Avg cycle time", value=avg_cycle, unit="hours", period=f"{period_start} to {period_end}"),
        DashboardKPI(name="SLA compliance", value=sla_compliance, unit="percent", period=f"{period_start} to {period_end}"),
        DashboardKPI(name="Closing backlog", value=backlog, unit="items", period=f"{period_start} to {period_end}"),
        DashboardKPI(name="Delay rate", value=delay_rate, unit="percent", period=f"{period_start} to {period_end}"),
        DashboardKPI(name="Queue share", value=queue_share, unit="percent", period=f"{period_start} to {period_end}"),
    ]

    # --- monthly trends ---
    throughput_trend = _monthly_throughput(ops, periods)
    cycle_time_trend = _monthly_cycle_time(ops, periods)
    sla_trend_data = _monthly_sla(sla, periods)
    backlog_trend = _monthly_backlog(ops, periods)

    # --- bottlenecks by process ---
    bottlenecks = _top_bottlenecks(ops, top_n=5)

    # --- department overview ---
    departments = _department_overview(ops, sla)

    return DashboardData(
        generated_at=dt.datetime.now(dt.UTC),
        period_start=period_start,
        period_end=period_end,
        kpis=kpis,
        throughput_trend=throughput_trend,
        cycle_time_trend=cycle_time_trend,
        sla_trend=sla_trend_data,
        backlog_trend=backlog_trend,
        bottlenecks=bottlenecks,
        departments=departments,
    )


# -- helpers ----------------------------------------------------------------


def _weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str) -> float | None:
    mask = df[value_col].notna() & df[weight_col].notna() & (df[weight_col] > 0)
    subset = df.loc[mask]
    if subset.empty:
        return None
    total_weight = subset[weight_col].sum()
    if total_weight == 0:
        return None
    return round_or_none(float((subset[value_col] * subset[weight_col]).sum() / total_weight), 2)


def _queue_share(df: pd.DataFrame) -> float | None:
    mask = df["total_cycle_time_hours"].notna() & (df["total_cycle_time_hours"] > 0)
    subset = df.loc[mask]
    if subset.empty:
        return None
    weights = subset["completed_operations"].fillna(1)
    q = (subset["queue_time_hours"] / subset["total_cycle_time_hours"] * weights).sum()
    w = weights.sum()
    return round_or_none(float(q / w * 100) if w else None, 1)


def _monthly_throughput(ops: pd.DataFrame, periods: list[str]) -> list[TrendPoint]:
    result = []
    for period in periods:
        chunk = ops.loc[ops["period"] == period]
        completed = safe_float(chunk["completed_operations"].sum())
        n_days = chunk["date"].nunique() if "date" in chunk.columns else 1
        val = round_or_none(completed / n_days if n_days else None, 1)
        result.append(TrendPoint(period=period, value=val))
    return result


def _monthly_cycle_time(ops: pd.DataFrame, periods: list[str]) -> list[TrendPoint]:
    result = []
    for period in periods:
        chunk = ops.loc[ops["period"] == period]
        val = _weighted_mean(chunk, "total_cycle_time_hours", "completed_operations")
        result.append(TrendPoint(period=period, value=val))
    return result


def _monthly_sla(sla: pd.DataFrame, periods: list[str]) -> list[TrendPoint]:
    result = []
    for period in periods:
        chunk = sla.loc[sla["period"] == period]
        met = safe_float(chunk["sla_met_cases"].sum()) if "sla_met_cases" in chunk.columns else 0
        total = safe_float(chunk["total_cases"].sum()) if "total_cases" in chunk.columns else 0
        val = round_or_none(met / total * 100 if total > 0 else None, 2)
        result.append(TrendPoint(period=period, value=val))
    return result


def _monthly_backlog(ops: pd.DataFrame, periods: list[str]) -> list[TrendPoint]:
    result = []
    for period in periods:
        chunk = ops.loc[ops["period"] == period]
        if chunk.empty:
            result.append(TrendPoint(period=period, value=None))
            continue
        last_day = chunk.loc[chunk["date"] == chunk["date"].max()]
        val = round_or_none(safe_float(last_day["backlog_open_items"].sum()), 0)
        result.append(TrendPoint(period=period, value=val))
    return result


def _top_bottlenecks(ops: pd.DataFrame, top_n: int = 5) -> list[BottleneckEntry]:
    """Rank processes by cycle time (descending)."""
    if "process" not in ops.columns:
        return []
    groups = ops.groupby("process", sort=False)
    entries = []
    for process, group in groups:
        ct = _weighted_mean(group, "total_cycle_time_hours", "completed_operations")
        qs = _queue_share(group)
        last_day = group.loc[group["date"] == group["date"].max()] if "date" in group.columns else group
        bl = round_or_none(safe_float(last_day["backlog_open_items"].sum()), 0)
        entries.append(BottleneckEntry(name=str(process), cycle_time=ct, queue_share=qs, backlog=bl))
    entries.sort(key=lambda e: e.cycle_time or 0, reverse=True)
    return entries[:top_n]


def _department_overview(ops: pd.DataFrame, sla: pd.DataFrame) -> list[DepartmentOverview]:
    if "department" not in ops.columns:
        return []

    # Build department ID → name mapping from the departments dataset
    repo = get_repository()
    try:
        deps_df = repo.frame(DatasetId.DEPARTMENTS)
        id_to_name: dict[str, str] = dict(
            zip(deps_df["department_id"], deps_df["department_name"])
        )
    except Exception:
        id_to_name = {}

    result = []
    for dept in sorted(ops["department"].unique()):
        dept_ops = ops.loc[ops["department"] == dept]

        # SLA uses department_name (e.g. "Fulfillment"), ops uses department_id (e.g. "DEP-102")
        dept_name = id_to_name.get(str(dept), str(dept))
        if "department" in sla.columns:
            dept_sla = sla.loc[sla["department"] == dept_name]
            if dept_sla.empty:
                dept_sla = sla.loc[sla["department"] == dept]
        else:
            dept_sla = pd.DataFrame()

        completed = safe_float(dept_ops["completed_operations"].sum())
        n_days = dept_ops["date"].nunique() if "date" in dept_ops.columns else 1
        tp = round_or_none(completed / n_days if n_days else None, 1)
        ct = _weighted_mean(dept_ops, "total_cycle_time_hours", "completed_operations")

        met = safe_float(dept_sla["sla_met_cases"].sum()) if not dept_sla.empty and "sla_met_cases" in dept_sla.columns else 0
        total = safe_float(dept_sla["total_cases"].sum()) if not dept_sla.empty and "total_cases" in dept_sla.columns else 0
        sla_comp = round_or_none(met / total * 100 if total > 0 else None, 2)

        last_day = dept_ops.loc[dept_ops["date"] == dept_ops["date"].max()] if "date" in dept_ops.columns else dept_ops
        bl = round_or_none(safe_float(last_day["backlog_open_items"].sum()), 0)

        result.append(DepartmentOverview(
            department=dept_name if dept_name != str(dept) else str(dept),
            throughput=tp,
            cycle_time=ct,
            sla_compliance=sla_comp,
            backlog=bl,
        ))
    return result
