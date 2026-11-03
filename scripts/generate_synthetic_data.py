"""Deterministic generator for the Operations Intelligence synthetic dataset.

The dataset is not random noise. It is produced by a small, explicit
*operating model*:

    daily arrivals  ->  daily service capacity (with backlog-driven flex)
    ->  closing backlog ->  queue time (queue model)
    ->  processing time (drifting per-process baseline)
    ->  total cycle time = queue + processing
    ->  delayed operations (promise-relative, blended with vendor reliability)
    ->  SLA compliance (SLA-target-relative, aggregated monthly)

Because every derived dataset (``sla_metrics.csv``, ``vendors.csv``,
``orders.csv``) is aggregated *from* ``operations.csv`` or driven *by* it, the
numbers are internally consistent by construction rather than by luck.

Seven deliberate but realistic patterns are embedded so the analytics layer has
something real to find. They are documented in ``docs/data-model.md`` and are
tuned to be discoverable without being obvious:

  P1  DEP-102 outbound shipping   - carrier processing time degrades ~+25%
  P2  DEP-105 contract admin      - demand grows without added capacity -> backlog
  P3  VND-102 RapidRoute Freight  - persistently high carrier delay rate
  P4  DEP-105 Procurement         - SLA compliance deteriorates through the year
  P5  DEP-104 incident resolution - queue time climbs, processing time flat
  P6  DEP-103 invoice processing  - cycle time and SLA compliance improve
  P7  VND-310 NetGuard Support    - delay rate degrades sharply in H2

Usage::

    python scripts/generate_synthetic_data.py [--out data] [--seed 20250101]

The output is byte-stable for a fixed ``--seed``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

GENERATOR_VERSION = "1.0.0"
DEFAULT_SEED = 20250101
DEFAULT_START = "2025-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_GENERATED_AT = "2026-01-01T00:00:00Z"

# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------

DEPARTMENTS: list[dict[str, object]] = [
    {
        "department_id": "DEP-101",
        "department_name": "Customer Operations",
        "operational_area": "Front Office",
        "headcount": 34,
    },
    {
        "department_id": "DEP-102",
        "department_name": "Fulfillment",
        "operational_area": "Warehouse & Logistics",
        "headcount": 48,
    },
    {
        "department_id": "DEP-103",
        "department_name": "Finance Operations",
        "operational_area": "Back Office",
        "headcount": 22,
    },
    {
        "department_id": "DEP-104",
        "department_name": "IT Support",
        "operational_area": "Technical Services",
        "headcount": 18,
    },
    {
        "department_id": "DEP-105",
        "department_name": "Procurement",
        "operational_area": "Supply Chain",
        "headcount": 16,
    },
]

DEPARTMENT_NAME_BY_ID = {str(d["department_id"]): str(d["department_name"]) for d in DEPARTMENTS}


@dataclass(frozen=True)
class VendorSpec:
    vendor_id: str
    vendor_name: str
    category: str
    department_id: str
    #: intrinsic share of cases that miss the operational promise
    delay_floor: float
    #: multiplier on the hosting process's processing hours
    processing_multiplier: float
    #: multiplier on outward freight transit hours for orders routed to this vendor
    transit_multiplier: float = 1.0
    #: linear daily ramp added to ``delay_floor`` across the window (pattern P7)
    delay_drift_per_day: float = 0.0


VENDORS: list[VendorSpec] = [
    VendorSpec("VND-101", "Northline Freight", "Logistics", "DEP-102", 0.045, 0.97, 1.00),
    VendorSpec("VND-102", "RapidRoute Freight", "Logistics", "DEP-102", 0.260, 1.14, 1.38),
    VendorSpec("VND-103", "BlueHarbor Logistics", "Logistics", "DEP-102", 0.035, 0.94, 0.96),
    VendorSpec("VND-104", "Meridian Components", "Manufacturing", "DEP-102", 0.120, 1.05, 1.05),
    VendorSpec("VND-105", "Atlas Packaging Supply", "Packaging", "DEP-102", 0.040, 0.95, 0.99),
    VendorSpec("VND-110", "Clearwater Print Services", "Outsourced Services", "DEP-101", 0.060, 1.00, 1.00),
    VendorSpec("VND-210", "LedgerWorks Advisory", "Professional Services", "DEP-103", 0.050, 1.00, 1.00),
    VendorSpec("VND-206", "Summit Advisory Group", "Professional Services", "DEP-105", 0.140, 1.06, 1.08),
    VendorSpec("VND-207", "Peak Procurement Partners", "Professional Services", "DEP-105", 0.045, 0.96, 0.97),
    VendorSpec("VND-208", "Sterling Office Systems", "Office Services", "DEP-105", 0.130, 1.03, 1.05),
    VendorSpec("VND-309", "CoreTech Services", "IT Services", "DEP-104", 0.050, 0.98, 1.00),
    VendorSpec("VND-310", "NetGuard Support", "IT Services", "DEP-104", 0.055, 1.02, 1.05, 0.00052),
    VendorSpec("VND-401", "Redline Industrial Supply", "Industrial Supplies", "DEP-105", 0.150, 1.07, 1.10),
    VendorSpec("VND-402", "Halcyon Safety Equipment", "Safety", "DEP-105", 0.040, 0.96, 0.98),
]
VENDOR_BY_ID = {v.vendor_id: v for v in VENDORS}


@dataclass(frozen=True)
class ProcessSpec:
    """Static parameters plus the deliberate behavioural patterns."""

    department_id: str
    process: str
    operation_type: str
    base_capacity: float
    base_demand: float
    #: internal / external customer promise in hours
    promise_hours: float
    #: contractual SLA target in hours (tighter than the promise)
    sla_target_hours: float
    demand_slope: float = 0.0
    demand_drift: float = 0.0
    base_processing_hours: float = 2.0
    processing_drift: float = 0.0
    base_queue_hours: float = 1.0
    queue_sensitivity: float = 2.0
    flex_coefficient: float = 0.30
    headcount_share: float = 1.0
    vendor_mix: tuple[tuple[str, float], ...] = ()
    vendor_coverage: float = 0.0
    #: this process's daily throughput defines the order volume
    order_source: bool = False
    #: arrivals are derived from the order volume rather than the demand model
    order_linked: bool = False
    order_scale: float = 1.0
    internal_daily_arrivals: float = 0.0
    month_end_factor: float = 1.0
    pattern_note: str = ""

    @property
    def key(self) -> str:
        return f"{self.department_id}|{self.process}"


PROCESSES: list[ProcessSpec] = [
    ProcessSpec(
        department_id="DEP-101",
        process="order intake",
        operation_type="Order Management",
        base_capacity=26.0,
        base_demand=0.0,
        promise_hours=12.0,
        sla_target_hours=3.6,
        base_processing_hours=1.55,
        base_queue_hours=0.6,
        queue_sensitivity=1.6,
        headcount_share=0.34,
        vendor_mix=(("VND-110", 1.0),),
        vendor_coverage=0.30,
        order_linked=True,
        internal_daily_arrivals=5.5,
        pattern_note="arrivals driven by order volume plus internal intake; stable SLA",
    ),
    ProcessSpec(
        department_id="DEP-101",
        process="customer support",
        operation_type="Customer Service",
        base_capacity=62.0,
        base_demand=56.0,
        promise_hours=24.0,
        sla_target_hours=4.6,
        demand_drift=0.00012,
        base_processing_hours=2.40,
        base_queue_hours=0.8,
        queue_sensitivity=1.4,
        headcount_share=0.46,
    ),
    ProcessSpec(
        department_id="DEP-101",
        process="returns processing",
        operation_type="Reverse Logistics",
        base_capacity=14.0,
        base_demand=12.2,
        promise_hours=48.0,
        sla_target_hours=6.5,
        base_processing_hours=3.10,
        base_queue_hours=1.1,
        queue_sensitivity=1.5,
        headcount_share=0.20,
    ),
    ProcessSpec(
        department_id="DEP-102",
        process="receiving",
        operation_type="Inbound Logistics",
        base_capacity=42.0,
        base_demand=39.0,
        promise_hours=24.0,
        sla_target_hours=5.2,
        demand_drift=0.00008,
        base_processing_hours=2.80,
        base_queue_hours=1.0,
        queue_sensitivity=1.6,
        headcount_share=0.30,
        vendor_mix=(("VND-104", 0.6), ("VND-105", 0.4)),
        vendor_coverage=0.75,
    ),
    ProcessSpec(
        department_id="DEP-102",
        process="picking and packing",
        operation_type="Outbound Operations",
        base_capacity=19.0,
        base_demand=18.5,
        promise_hours=12.0,
        sla_target_hours=7.0,
        demand_drift=0.00018,
        base_processing_hours=3.55,
        base_queue_hours=1.2,
        queue_sensitivity=1.8,
        headcount_share=0.40,
        order_source=True,
        pattern_note="source of order volume; steady throughput with mild growth",
    ),
    ProcessSpec(
        department_id="DEP-102",
        process="outbound shipping",
        operation_type="Outbound Logistics",
        base_capacity=19.5,
        base_demand=0.0,
        promise_hours=24.0,
        sla_target_hours=7.0,
        base_processing_hours=4.10,
        processing_drift=0.00058,
        base_queue_hours=0.9,
        queue_sensitivity=1.7,
        headcount_share=0.30,
        vendor_mix=(("VND-101", 0.45), ("VND-102", 0.35), ("VND-103", 0.20)),
        vendor_coverage=0.85,
        order_linked=True,
        order_scale=0.985,
        pattern_note="P1 - carrier processing time degrades ~+25% across the year",
    ),
    ProcessSpec(
        department_id="DEP-103",
        process="invoice processing",
        operation_type="Billing",
        base_capacity=68.0,
        base_demand=63.0,
        promise_hours=24.0,
        sla_target_hours=3.0,
        base_processing_hours=2.00,
        processing_drift=-0.00120,
        base_queue_hours=0.5,
        queue_sensitivity=1.3,
        headcount_share=0.46,
        vendor_mix=(("VND-210", 1.0),),
        vendor_coverage=0.25,
        pattern_note="P6 - processing time and SLA compliance improve over the year",
    ),
    ProcessSpec(
        department_id="DEP-103",
        process="accounts payable",
        operation_type="Payments",
        base_capacity=30.0,
        base_demand=28.0,
        promise_hours=48.0,
        sla_target_hours=4.6,
        base_processing_hours=2.60,
        base_queue_hours=0.9,
        queue_sensitivity=1.4,
        headcount_share=0.34,
    ),
    ProcessSpec(
        department_id="DEP-103",
        process="period reconciliation",
        operation_type="Reporting",
        base_capacity=18.0,
        base_demand=15.5,
        promise_hours=72.0,
        sla_target_hours=8.5,
        base_processing_hours=5.00,
        base_queue_hours=1.6,
        queue_sensitivity=1.5,
        headcount_share=0.20,
        month_end_factor=1.9,
    ),
    ProcessSpec(
        department_id="DEP-104",
        process="incident resolution",
        operation_type="Technical Support",
        base_capacity=22.0,
        base_demand=22.0,
        promise_hours=12.0,
        sla_target_hours=9.0,
        demand_slope=0.017,
        base_processing_hours=5.40,
        base_queue_hours=1.2,
        queue_sensitivity=2.5,
        flex_coefficient=0.34,
        headcount_share=0.46,
        pattern_note="P5 - arrivals climb while processing time stays flat; queue explodes",
    ),
    ProcessSpec(
        department_id="DEP-104",
        process="service request fulfilment",
        operation_type="Technical Support",
        base_capacity=30.0,
        base_demand=28.0,
        promise_hours=48.0,
        sla_target_hours=6.5,
        base_processing_hours=3.40,
        base_queue_hours=0.9,
        queue_sensitivity=1.5,
        headcount_share=0.34,
        vendor_mix=(("VND-309", 0.45), ("VND-310", 0.55)),
        vendor_coverage=0.55,
    ),
    ProcessSpec(
        department_id="DEP-104",
        process="system administration",
        operation_type="Infrastructure",
        base_capacity=38.0,
        base_demand=26.0,
        promise_hours=72.0,
        sla_target_hours=5.0,
        demand_drift=0.00010,
        base_processing_hours=2.20,
        base_queue_hours=0.6,
        queue_sensitivity=1.2,
        headcount_share=0.20,
        vendor_mix=(("VND-309", 1.0),),
        vendor_coverage=0.20,
    ),
    ProcessSpec(
        department_id="DEP-105",
        process="vendor onboarding",
        operation_type="Supplier Management",
        base_capacity=6.2,
        base_demand=5.6,
        promise_hours=120.0,
        sla_target_hours=11.0,
        base_processing_hours=6.40,
        base_queue_hours=2.2,
        queue_sensitivity=1.6,
        headcount_share=0.28,
        vendor_mix=(("VND-206", 0.5), ("VND-207", 0.3), ("VND-208", 0.2)),
        vendor_coverage=0.60,
    ),
    ProcessSpec(
        department_id="DEP-105",
        process="purchase order management",
        operation_type="Sourcing",
        base_capacity=25.0,
        base_demand=24.0,
        promise_hours=24.0,
        sla_target_hours=4.6,
        demand_slope=0.006,
        base_processing_hours=2.40,
        base_queue_hours=0.9,
        queue_sensitivity=1.7,
        headcount_share=0.40,
        vendor_mix=(("VND-401", 0.45), ("VND-402", 0.35), ("VND-207", 0.20)),
        vendor_coverage=0.70,
    ),
    ProcessSpec(
        department_id="DEP-105",
        process="contract administration",
        operation_type="Governance",
        base_capacity=8.0,
        base_demand=7.4,
        promise_hours=48.0,
        sla_target_hours=16.0,
        demand_slope=0.014,
        base_processing_hours=4.00,
        base_queue_hours=1.8,
        queue_sensitivity=2.0,
        flex_coefficient=0.28,
        headcount_share=0.32,
        pattern_note="P2/P4 - contract portfolio grows without added capacity -> backlog and SLA decay",
    ),
]


@dataclass(frozen=True)
class ProductCategorySpec:
    category: str
    share: float
    promise_hours: float
    avg_order_value: float
    base_handling_hours: float
    transit_hours: float
    handling_drift: float = 0.0
    transit_spread: float = 0.22
    pattern_note: str = ""


PRODUCT_CATEGORIES: list[ProductCategorySpec] = [
    ProductCategorySpec(
        category="Standard Supplies",
        share=0.31,
        promise_hours=34.0,
        avg_order_value=940.0,
        base_handling_hours=9.2,
        transit_hours=20.0,
        handling_drift=-0.00030,
        pattern_note="P6b - internal handling time improves steadily through the year",
    ),
    ProductCategorySpec(
        category="Technical Equipment",
        share=0.22,
        promise_hours=47.0,
        avg_order_value=2650.0,
        base_handling_hours=13.5,
        transit_hours=26.0,
        handling_drift=0.00012,
    ),
    ProductCategorySpec(
        category="Furniture & Fixtures",
        share=0.16,
        promise_hours=62.0,
        avg_order_value=3180.0,
        base_handling_hours=16.0,
        transit_hours=32.0,
        handling_drift=0.00034,
        transit_spread=0.32,
        pattern_note="P1b - internal handling time drifts upward through the year",
    ),
    ProductCategorySpec(
        category="Digital Subscriptions",
        share=0.19,
        promise_hours=2.75,
        avg_order_value=410.0,
        base_handling_hours=2.4,
        transit_hours=0.0,
        handling_drift=-0.00018,
        transit_spread=0.10,
    ),
    ProductCategorySpec(
        category="Safety & Compliance",
        share=0.12,
        promise_hours=38.0,
        avg_order_value=1580.0,
        base_handling_hours=10.5,
        transit_hours=21.0,
    ),
]

#: originating department mix for orders (which unit owns the customer relationship)
ORDER_ORIGIN_MIX: tuple[tuple[str, float], ...] = (
    ("DEP-101", 0.54),
    ("DEP-102", 0.30),
    ("DEP-104", 0.09),
    ("DEP-105", 0.07),
)

#: originating department -> freight/fulfilment partner used for that order
ORDER_CARRIER_MIX: dict[str, tuple[tuple[str, float], ...]] = {
    "DEP-101": (("VND-101", 0.50), ("VND-103", 0.50)),
    "DEP-102": (("VND-101", 0.40), ("VND-102", 0.25), ("VND-103", 0.35)),
    "DEP-104": (("VND-309", 0.70), ("VND-310", 0.30)),
    "DEP-105": (("VND-401", 0.60), ("VND-402", 0.40)),
}

WEEKDAY_DEMAND_FACTOR = (1.00, 1.02, 1.00, 1.01, 1.06, 0.46, 0.26)  # Mon..Sun
WEEKEND_CAPACITY_FACTOR = 0.88
MONTH_SEASONALITY = (0.93, 0.94, 1.02, 1.00, 1.03, 1.02, 0.95, 0.94, 1.05, 1.07, 1.19, 1.23)
PRIORITY_MIX: tuple[tuple[str, float], ...] = (
    ("standard", 0.68),
    ("expedited", 0.24),
    ("critical", 0.08),
)

ORDER_CANCEL_RATE = 0.015
RETURN_RATE = 0.038

#: knee of the promise-relative delay curve (delay pressure is zero below this)
DELAY_KNEE_RATIO = 0.50
#: exponent of the SLA-relative compliance curve
SLA_CURVATURE = 8.0


# --------------------------------------------------------------------------
# Small deterministic helpers
# --------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def round2(value: float) -> float:
    return round(value + 0.0, 2)


def round1(value: float) -> float:
    return round(value + 0.0, 1)


def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def seeded_rng(seed: int, *parts: object) -> random.Random:
    """Stable per-entity RNG so row order never changes the output."""
    return random.Random(f"{seed}:{'|'.join(str(p) for p in parts)}")


def pick_weighted(rng: random.Random, options: tuple[tuple[str, float], ...]) -> str:
    roll = rng.random()
    cumulative = 0.0
    for label, weight in options:
        cumulative += weight
        if roll <= cumulative:
            return label
    return options[-1][0]


def quarter_of(day: dt.date) -> str:
    return f"{day.year}-Q{(day.month - 1) // 3 + 1}"


def period_of(day: dt.date) -> str:
    return f"{day.year}-{day.month:02d}"


def days_to_month_end(day: dt.date) -> int:
    next_month_first = day.replace(day=1) + dt.timedelta(days=32)
    last_day = next_month_first.replace(day=1) - dt.timedelta(days=1)
    return (last_day - day).days


def sla_met_fraction(ratio: float) -> float:
    """Share of cases meeting an SLA target, as a function of average/target.

    Monotone, smooth and bounded in (0, 1). Calibrated so that a process
    running at half its target is comfortably compliant, a process running at
    its target is roughly a coin flip, and a process running at 1.5x its target
    is close to fully breaching.
    """
    pressure = max(0.0, ratio) ** SLA_CURVATURE
    return clamp(1.0 / (1.0 + pressure), 0.03, 0.985)


def promise_delay_fraction(ratio: float) -> float:
    """Share of operations late relative to a (looser) operational promise."""
    pressure = DELAY_KNEE_RATIO
    knee_excess = max(0.0, ratio - pressure)
    return clamp(0.030 + 0.70 * (knee_excess**1.6), 0.008, 0.95)


# --------------------------------------------------------------------------
# Core simulation
# --------------------------------------------------------------------------


@dataclass
class DayRecord:
    date: dt.date
    department_id: str
    department: str
    process: str
    operation_type: str
    status: str
    priority: str
    period: str
    quarter: str
    completed_operations: int
    delayed_operations: int
    processing_time_hours: float
    queue_time_hours: float
    total_cycle_time_hours: float
    employee_count: int
    vendor_id: str
    backlog_open_items: float


@dataclass
class SimulatedProcess:
    spec: ProcessSpec
    records: list[DayRecord] = field(default_factory=list)


def simulate_process(
    spec: ProcessSpec,
    start: dt.date,
    end: dt.date,
    seed: int,
    order_volume: Mapping[dt.date, float],
) -> SimulatedProcess:
    """Run the daily capacity/backlog simulation for a single process."""
    sim = SimulatedProcess(spec=spec)
    backlog = 0.0
    headcount = int(DEPARTMENTS_BY_ID[spec.department_id]["headcount"])
    employee_count = max(1, int(round(headcount * spec.headcount_share)))

    for index, day in enumerate(daterange(start, end)):
        rng = seeded_rng(seed, spec.department_id, spec.process, day.isoformat())

        capacity = spec.base_capacity
        if day.weekday() >= 5:
            capacity *= WEEKEND_CAPACITY_FACTOR

        if spec.order_linked:
            linked = order_volume.get(day, 0.0)
            arrivals = linked * spec.order_scale
            arrivals += spec.internal_daily_arrivals * WEEKDAY_DEMAND_FACTOR[day.weekday()]
        else:
            arrivals = spec.base_demand + spec.demand_slope * index
            arrivals *= 1.0 + spec.demand_drift * index
            arrivals *= WEEKDAY_DEMAND_FACTOR[day.weekday()]
            arrivals *= MONTH_SEASONALITY[day.month - 1]
            if spec.month_end_factor > 1.0 and days_to_month_end(day) <= 2:
                arrivals *= spec.month_end_factor
            arrivals *= 1.0 + rng.uniform(-0.11, 0.11)
        arrivals = max(0.0, round(arrivals))

        # Service capacity flexes logarithmically with the closing backlog,
        # which is what stops a slow deterioration from diverging to infinity.
        flex = 1.0 + spec.flex_coefficient * math.log1p(backlog / max(capacity, 1.0))
        served = min(backlog + arrivals, capacity * flex)
        completed = max(0, min(int(round(served)), int(round(backlog + arrivals))))
        backlog = max(0.0, backlog + arrivals - completed)

        vendor_id = ""
        vendor_proc_mult = 1.0
        vendor_delay_floor = 0.0
        vendor_ramp = 0.0
        if spec.vendor_mix and rng.random() < spec.vendor_coverage:
            vendor_id = pick_weighted(rng, spec.vendor_mix)
            vendor = VENDOR_BY_ID[vendor_id]
            vendor_proc_mult = vendor.processing_multiplier
            vendor_delay_floor = vendor.delay_floor + vendor.delay_drift_per_day * index
            vendor_ramp = 1.0 + vendor.delay_drift_per_day * 1.6 * index

        processing = spec.base_processing_hours * ((1.0 + spec.processing_drift) ** index)
        processing *= vendor_proc_mult
        # A vendor whose reliability decays also slows down physically across the
        # window. The multiplier is applied to the components so that
        # total_cycle_time_hours always equals queue_time_hours + processing_time_hours.
        processing *= vendor_ramp
        processing *= 1.0 + rng.uniform(-0.09, 0.09)

        queue = spec.base_queue_hours + spec.queue_sensitivity * (backlog / max(capacity, 1.0))
        queue *= 1.0 + rng.uniform(-0.08, 0.08)

        processing = round2(max(0.05, processing))
        queue = round2(max(0.0, queue))
        total = round2(processing + queue)

        delay_fraction = promise_delay_fraction(total / spec.promise_hours)
        delay_fraction = max(delay_fraction, vendor_delay_floor)
        delay_fraction = clamp(delay_fraction + rng.uniform(-0.015, 0.015), 0.0, 0.95)
        delayed = max(0, min(int(round(completed * delay_fraction)), completed))

        observed_delay = (delayed / completed) if completed else 0.0
        if observed_delay >= 0.20:
            status = "delayed"
        elif backlog >= capacity or observed_delay >= 0.10:
            status = "at_risk"
        else:
            status = "completed"

        sim.records.append(
            DayRecord(
                date=day,
                department_id=spec.department_id,
                department=DEPARTMENT_NAME_BY_ID[spec.department_id],
                process=spec.process,
                operation_type=spec.operation_type,
                status=status,
                priority=pick_weighted(rng, PRIORITY_MIX),
                period=period_of(day),
                quarter=quarter_of(day),
                completed_operations=completed,
                delayed_operations=delayed,
                processing_time_hours=processing,
                queue_time_hours=queue,
                total_cycle_time_hours=total,
                employee_count=employee_count,
                vendor_id=vendor_id,
                backlog_open_items=round1(backlog),
            )
        )
    return sim


def simulate_all(
    start: dt.date,
    end: dt.date,
    seed: int,
) -> tuple[list[SimulatedProcess], dict[dt.date, float]]:
    """Two-stage simulation: order volume first, then order-linked processes."""
    volume_specs = [s for s in PROCESSES if s.order_source]
    consumer_specs = [s for s in PROCESSES if not s.order_source]

    simulated: dict[str, SimulatedProcess] = {}
    for spec in volume_specs:
        simulated[spec.key] = simulate_process(spec, start, end, seed, {})

    order_volume: dict[dt.date, float] = {}
    for spec in volume_specs:
        for record in simulated[spec.key].records:
            order_volume[record.date] = order_volume.get(record.date, 0.0) + float(
                record.completed_operations
            )

    for spec in consumer_specs:
        simulated[spec.key] = simulate_process(spec, start, end, seed, order_volume)

    return [simulated[spec.key] for spec in PROCESSES], order_volume


# --------------------------------------------------------------------------
# Orders (derived from the picking-and-packing simulation)
# --------------------------------------------------------------------------

PICK_PACK_BASE_PROCESSING = 3.55


def weighted_category(rng: random.Random, day: dt.date) -> ProductCategorySpec:
    """Categories have seasonal mix shifts (Q4 supplies surge)."""
    weights: list[float] = []
    for spec in PRODUCT_CATEGORIES:
        weight = spec.share
        if day.month in (11, 12) and spec.category == "Standard Supplies":
            weight *= 1.35
        if day.month in (1, 2) and spec.category == "Digital Subscriptions":
            weight *= 1.20
        weights.append(weight)
    total = sum(weights)
    roll = rng.random() * total
    cumulative = 0.0
    for spec, weight in zip(PRODUCT_CATEGORIES, weights, strict=True):
        cumulative += weight
        if roll <= cumulative:
            return spec
    return PRODUCT_CATEGORIES[-1]


def generate_orders(
    pick_pack: SimulatedProcess,
    start: dt.date,
    seed: int,
) -> list[dict[str, object]]:
    """One order per completed pick-and-pack operation."""
    rows: list[dict[str, object]] = []
    sequence = 0
    for record in pick_pack.records:
        count = record.completed_operations
        if count <= 0:
            continue
        rng = seeded_rng(seed, "orders", record.date.isoformat())
        day_index = (record.date - start).days
        handling_baseline = max(0.05, record.processing_time_hours)
        for _ in range(count):
            sequence += 1
            origin = pick_weighted(rng, ORDER_ORIGIN_MIX)
            category = weighted_category(rng, record.date)
            carrier = pick_weighted(rng, ORDER_CARRIER_MIX[origin])
            carrier_spec = VENDOR_BY_ID[carrier]

            handling = category.base_handling_hours * (
                (1.0 + category.handling_drift) ** day_index
            )
            handling *= handling_baseline / PICK_PACK_BASE_PROCESSING
            handling = max(0.25, handling * (1.0 + rng.uniform(-0.16, 0.16)))

            if category.transit_hours <= 0.0:
                transit = handling * rng.uniform(-0.05, 0.05)
            else:
                transit = max(
                    0.0,
                    category.transit_hours * (1.0 + rng.gauss(0.0, category.transit_spread)),
                )
                transit *= carrier_spec.transit_multiplier
                # reliability ramp vendors also slow down physically over the year
                transit *= 1.0 + carrier_spec.delay_drift_per_day * 3.0 * day_index

            promised = float(category.promise_hours)
            actual = round2(max(0.25, handling + transit))
            cancelled = rng.random() < ORDER_CANCEL_RATE
            returned = (not cancelled) and rng.random() < RETURN_RATE

            if cancelled:
                status = "cancelled"
                delay_hours = 0.0
            elif actual > promised:
                status = "delivered_late"
                delay_hours = round2(actual - promised)
            else:
                status = "returned" if returned else "delivered"
                delay_hours = 0.0

            rows.append(
                {
                    "order_id": f"ORD-{record.date.year}-{sequence:06d}",
                    "order_date": record.date.isoformat(),
                    "period": record.period,
                    "quarter": record.quarter,
                    "department": origin,
                    "product_category": category.category,
                    "order_value": round2(
                        max(
                            25.0,
                            rng.gauss(
                                category.avg_order_value,
                                0.28 * category.avg_order_value,
                            ),
                        )
                    ),
                    "promised_time_hours": promised,
                    "actual_time_hours": actual,
                    "fulfillment_time_hours": round2(handling),
                    "transit_time_hours": round2(transit),
                    "status": status,
                    "vendor_id": carrier,
                    "delay_hours": delay_hours,
                }
            )
    return rows


# --------------------------------------------------------------------------
# Aggregations
# --------------------------------------------------------------------------


def generate_sla_metrics(simulated: list[SimulatedProcess], seed: int) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, str, str], list[DayRecord]] = {}
    spec_by_key = {s.key: s for s in PROCESSES}
    for sim in simulated:
        for record in sim.records:
            key = (record.period, record.quarter, record.department, record.process)
            buckets.setdefault(key, []).append(record)

    rows: list[dict[str, object]] = []
    for period, quarter, department, process in sorted(buckets):
        records = buckets[(period, quarter, department, process)]
        spec = spec_by_key[f"{_department_id(department)}|{process}"]
        total_cases = sum(r.completed_operations for r in records)
        rng = seeded_rng(seed, "sla", period, department, process)

        if total_cases <= 0:
            average_cycle = 0.0
            average_queue = 0.0
            met = 0
            breached = 0
            compliance = 0.0
        else:
            weight = float(total_cases)
            average_cycle = (
                sum(r.total_cycle_time_hours * r.completed_operations for r in records) / weight
            )
            average_queue = (
                sum(r.queue_time_hours * r.completed_operations for r in records) / weight
            )
            fraction = sla_met_fraction(average_cycle / spec.sla_target_hours)
            fraction = clamp(fraction + rng.uniform(-0.012, 0.012), 0.03, 0.985)
            met = int(round(total_cases * fraction))
            breached = total_cases - met
            compliance = round2(met / total_cases * 100.0)

        rows.append(
            {
                "period": period,
                "quarter": quarter,
                "department": department,
                "process": process,
                "total_cases": total_cases,
                "sla_target_hours": spec.sla_target_hours,
                "average_cycle_time_hours": round2(average_cycle),
                "average_queue_time_hours": round2(average_queue),
                "sla_met_cases": met,
                "sla_breached_cases": breached,
                "sla_compliance_percentage": compliance,
                "backlog_open_items": round1(max(r.backlog_open_items for r in records)),
            }
        )
    return rows


def generate_vendor_profiles(
    simulated: list[SimulatedProcess],
    months: int,
    seed: int,
) -> list[dict[str, object]]:
    buckets: dict[str, dict[str, float]] = {
        v.vendor_id: {"completed": 0.0, "delayed": 0.0, "proc_weighted": 0.0, "on_target": 0.0}
        for v in VENDORS
    }
    spec_by_key = {s.key: s for s in PROCESSES}

    for sim in simulated:
        spec = spec_by_key[sim.spec.key]
        for record in sim.records:
            if not record.vendor_id:
                continue
            bucket = buckets[record.vendor_id]
            completed = float(record.completed_operations)
            bucket["completed"] += completed
            bucket["delayed"] += float(record.delayed_operations)
            bucket["proc_weighted"] += record.processing_time_hours * completed
            fraction = sla_met_fraction(record.total_cycle_time_hours / spec.sla_target_hours)
            bucket["on_target"] += fraction * completed

    rows: list[dict[str, object]] = []
    for vendor in sorted(VENDORS, key=lambda v: v.vendor_id):
        bucket = buckets[vendor.vendor_id]
        completed = bucket["completed"]
        rng = seeded_rng(seed, "vendor", vendor.vendor_id)
        rows.append(
            {
                "vendor_id": vendor.vendor_id,
                "vendor_name": vendor.vendor_name,
                "category": vendor.category,
                "department": DEPARTMENT_NAME_BY_ID[vendor.department_id],
                "monthly_volume": round1(completed / months) if months else 0.0,
                "average_processing_time": round2(bucket["proc_weighted"] / completed)
                if completed
                else 0.0,
                "delay_rate": round2(bucket["delayed"] / completed * 100.0) if completed else 0.0,
                "sla_compliance": round2(bucket["on_target"] / completed * 100.0)
                if completed
                else 0.0,
            }
        )
    return rows


def _department_id(department_name: str) -> str:
    for dept_id, name in DEPARTMENT_NAME_BY_ID.items():
        if name == department_name:
            return dept_id
    raise KeyError(department_name)


DEPARTMENTS_BY_ID = {str(d["department_id"]): d for d in DEPARTMENTS}


# --------------------------------------------------------------------------
# Dataset manifests and writing
# --------------------------------------------------------------------------

OPERATIONS_HEADER = [
    "operation_id",
    "date",
    "period",
    "quarter",
    "department",
    "process",
    "operation_type",
    "status",
    "priority",
    "completed_operations",
    "delayed_operations",
    "processing_time_hours",
    "queue_time_hours",
    "total_cycle_time_hours",
    "employee_count",
    "vendor_id",
    "backlog_open_items",
]

ORDERS_HEADER = [
    "order_id",
    "order_date",
    "period",
    "quarter",
    "department",
    "product_category",
    "order_value",
    "promised_time_hours",
    "actual_time_hours",
    "fulfillment_time_hours",
    "transit_time_hours",
    "status",
    "vendor_id",
    "delay_hours",
]

SLA_HEADER = [
    "period",
    "quarter",
    "department",
    "process",
    "total_cases",
    "sla_target_hours",
    "average_cycle_time_hours",
    "average_queue_time_hours",
    "sla_met_cases",
    "sla_breached_cases",
    "sla_compliance_percentage",
    "backlog_open_items",
]

VENDORS_HEADER = [
    "vendor_id",
    "vendor_name",
    "category",
    "department",
    "monthly_volume",
    "average_processing_time",
    "delay_rate",
    "sla_compliance",
]

DEPARTMENTS_HEADER = [
    "department_id",
    "department_name",
    "operational_area",
    "headcount",
]

MANIFEST_HEADER = [
    "dataset_id",
    "file_name",
    "description",
    "grain",
    "row_count",
    "primary_key",
    "time_column",
    "dimensions",
    "measures",
    "generator_version",
    "seed",
    "generated_at",
]

DATASET_MANIFEST: list[dict[str, object]] = [
    {
        "dataset_id": "operations",
        "file_name": "operations.csv",
        "description": (
            "Daily operational throughput, cycle time, queue and backlog by department and process."
        ),
        "grain": "one row per department-process-day",
        "primary_key": "operation_id",
        "time_column": "date",
        "dimensions": (
            "department, process, operation_type, status, priority, vendor_id, period, quarter"
        ),
        "measures": (
            "completed_operations, delayed_operations, processing_time_hours, queue_time_hours, "
            "total_cycle_time_hours, employee_count, backlog_open_items"
        ),
    },
    {
        "dataset_id": "orders",
        "file_name": "orders.csv",
        "description": (
            "Order-level fulfilment detail for fulfilment-time, promise and vendor transit analysis."
        ),
        "grain": "one row per order",
        "primary_key": "order_id",
        "time_column": "order_date",
        "dimensions": "department, product_category, status, vendor_id, period, quarter",
        "measures": (
            "order_value, promised_time_hours, actual_time_hours, fulfillment_time_hours, "
            "delay_hours"
        ),
    },
    {
        "dataset_id": "sla_metrics",
        "file_name": "sla_metrics.csv",
        "description": "Monthly SLA rollup aggregated from operations.csv.",
        "grain": "one row per department-process-month",
        "primary_key": "department + process + period",
        "time_column": "period",
        "dimensions": "department, process, period, quarter",
        "measures": (
            "total_cases, sla_met_cases, sla_breached_cases, sla_compliance_percentage, "
            "average_cycle_time_hours, average_queue_time_hours, backlog_open_items"
        ),
    },
    {
        "dataset_id": "vendors",
        "file_name": "vendors.csv",
        "description": "Vendor reference profile aggregated from vendor-linked operational records.",
        "grain": "one row per vendor",
        "primary_key": "vendor_id",
        "time_column": "",
        "dimensions": "vendor_id, vendor_name, category, department",
        "measures": "monthly_volume, average_processing_time, delay_rate, sla_compliance",
    },
    {
        "dataset_id": "departments",
        "file_name": "departments.csv",
        "description": "Department reference data and allocated headcount.",
        "grain": "one row per department",
        "primary_key": "department_id",
        "time_column": "",
        "dimensions": "department_id, department_name, operational_area",
        "measures": "headcount",
    },
]


def write_csv(path: Path, header: list[str], rows: list[dict[str, object]]) -> int:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_operation_rows(simulated: list[SimulatedProcess]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sequence = 0
    for sim in simulated:
        for record in sim.records:
            sequence += 1
            rows.append(
                {
                    "operation_id": f"OP-{record.date.year}-{sequence:06d}",
                    "date": record.date.isoformat(),
                    "period": record.period,
                    "quarter": record.quarter,
                    "department": record.department_id,
                    "process": record.process,
                    "operation_type": record.operation_type,
                    "status": record.status,
                    "priority": record.priority,
                    "completed_operations": record.completed_operations,
                    "delayed_operations": record.delayed_operations,
                    "processing_time_hours": record.processing_time_hours,
                    "queue_time_hours": record.queue_time_hours,
                    "total_cycle_time_hours": record.total_cycle_time_hours,
                    "employee_count": record.employee_count,
                    "vendor_id": record.vendor_id,
                    "backlog_open_items": record.backlog_open_items,
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic operations dataset.")
    parser.add_argument("--out", default="data", help="output directory (default: data)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--generated-at", default=DEFAULT_GENERATED_AT)
    args = parser.parse_args(argv)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    simulated, _order_volume = simulate_all(start, end, args.seed)
    pick_pack = next(s for s in simulated if s.spec.order_source)

    operation_rows = build_operation_rows(simulated)
    order_rows = generate_orders(pick_pack, start, args.seed)
    months = len({str(r["period"]) for r in operation_rows})
    sla_rows = generate_sla_metrics(simulated, args.seed)
    vendor_rows = generate_vendor_profiles(simulated, months, args.seed)
    department_rows = [dict(d) for d in DEPARTMENTS]

    counts = {
        "operations": write_csv(out_dir / "operations.csv", OPERATIONS_HEADER, operation_rows),
        "orders": write_csv(out_dir / "orders.csv", ORDERS_HEADER, order_rows),
        "sla_metrics": write_csv(out_dir / "sla_metrics.csv", SLA_HEADER, sla_rows),
        "vendors": write_csv(out_dir / "vendors.csv", VENDORS_HEADER, vendor_rows),
        "departments": write_csv(out_dir / "departments.csv", DEPARTMENTS_HEADER, department_rows),
    }

    manifest_rows = [
        {
            **spec,
            "row_count": counts[str(spec["dataset_id"])],
            "generator_version": GENERATOR_VERSION,
            "seed": args.seed,
            "generated_at": args.generated_at,
        }
        for spec in DATASET_MANIFEST
    ]
    write_csv(out_dir / "manifest.csv", MANIFEST_HEADER, manifest_rows)

    for dataset_id, count in counts.items():
        print(f"{dataset_id:>12}: {count:>6} rows -> {out_dir}")
    print(f"{'manifest':>12}: {len(manifest_rows):>6} rows -> {out_dir}")
    print(f"patterns: {sum(1 for s in PROCESSES if s.pattern_note)} annotated processes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
