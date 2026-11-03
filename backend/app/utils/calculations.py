"""Deterministic numeric helpers.

Every operational calculation in the platform goes through this module. The
contract enforced here is:

* **never return NaN or infinity** - an undefined calculation returns ``None``
  and the caller records a caveat;
* a zero (or missing) denominator is an explicit, handled condition, never a
  division error;
* a previous value of exactly zero yields ``percentage_change = None`` rather
  than an unbounded or misleading percentage.

These functions are pure. They contain no I/O, no logging of user text, and no
LLM interaction, which makes them straightforward to unit test and to audit.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import cast

Number = float | int | None

#: Number of decimal places used for reported percentages.
PERCENT_PRECISION = 2


#: A value that may carry a number. The coercers below accept ``object`` rather
#: than ``Number`` because they are also the guard that stops a non-numeric value
#: read from a dataset row from reaching a calculation.
Numeric = float | int | str


def is_finite(value: object) -> bool:
    """True when ``value`` is a real, finite number (``None`` is not)."""
    if value is None or isinstance(value, bool):
        return False
    try:
        number = float(cast("Numeric", value))
    except (TypeError, ValueError):
        return False
    return not math.isnan(number) and not math.isinf(number)


def safe_float(value: object, default: float | None = None) -> float | None:
    """Coerce to a finite float, or return ``default``.

    The finiteness test is repeated here rather than delegated to
    :func:`is_finite` so that the ``None`` case is narrowed for the caller.
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(cast("Numeric", value))
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def round_or_none(value: object, precision: int = PERCENT_PRECISION) -> float | None:
    """Round a finite number, or return ``None`` for undefined input."""
    coerced = safe_float(value)
    if coerced is None:
        return None
    return round(coerced, precision)


def percentage_change(current: Number, previous: Number) -> float | None:
    """``(current - previous) / previous * 100``.

    Returns ``None`` when the change is undefined: either value missing, or the
    previous value is zero (a percentage change from zero is not interpretable).
    """
    current_value = safe_float(current)
    previous_value = safe_float(previous)
    if current_value is None or previous_value is None:
        return None
    if previous_value == 0:
        return None
    result = (current_value - previous_value) / abs(previous_value) * 100.0
    if not math.isfinite(result):
        return None
    return round(result, PERCENT_PRECISION)


def absolute_change(current: Number, previous: Number) -> float | None:
    """``current - previous``, or ``None`` when either value is missing."""
    current_value = safe_float(current)
    previous_value = safe_float(previous)
    if current_value is None or previous_value is None:
        return None
    result = current_value - previous_value
    if not math.isfinite(result):
        return None
    return round(result, PERCENT_PRECISION)


def safe_ratio(numerator: Number, denominator: Number) -> float | None:
    """``numerator / denominator``, or ``None`` when undefined.

    Unlike :func:`percentage_change` this does not special-case a zero
    denominator beyond returning ``None``; it never raises.
    """
    numerator_value = safe_float(numerator)
    denominator_value = safe_float(denominator)
    if numerator_value is None or denominator_value is None:
        return None
    if denominator_value == 0:
        return None
    result = numerator_value / denominator_value
    if not math.isfinite(result):
        return None
    return result


def percentage_of_total(numerator: Number, denominator: Number) -> float | None:
    """``numerator / denominator * 100`` with a zero-denominator guard."""
    ratio = safe_ratio(numerator, denominator)
    if ratio is None:
        return None
    return round(ratio * 100.0, PERCENT_PRECISION)


def sla_compliance(met_cases: Number, total_cases: Number) -> float | None:
    """``sla_met_cases / total_cases * 100``. ``None`` when total is zero."""
    return percentage_of_total(met_cases, total_cases)


def sla_breach_rate(breached_cases: Number, total_cases: Number) -> float | None:
    """``sla_breached_cases / total_cases * 100``. ``None`` when total is zero."""
    return percentage_of_total(breached_cases, total_cases)


def delay_rate(delayed_operations: Number, total_operations: Number) -> float | None:
    """``delayed_operations / total_operations * 100``. ``None`` when total is zero."""
    return percentage_of_total(delayed_operations, total_operations)


def contribution(numerator: Number, total: Number) -> float | None:
    """Share of a whole, in percent, with a zero-total guard.

    Used for queue contribution and processing contribution of cycle time.
    """
    return percentage_of_total(numerator, total)


def backlog_change(current_backlog: Number, previous_backlog: Number) -> float | None:
    """``current_backlog - previous_backlog``. ``None`` when either is missing."""
    return absolute_change(current_backlog, previous_backlog)


def throughput(completed_operations: Number, period_days: Number) -> float | None:
    """``completed_operations / period_days``. ``None`` for a zero-length period."""
    rate = safe_ratio(completed_operations, period_days)
    if rate is None:
        return None
    return round(rate, PERCENT_PRECISION)


def mean(values: Iterable[Number]) -> float | None:
    """Arithmetic mean that ignores ``None`` and non-finite entries."""
    cleaned = [v for v in (safe_float(x) for x in values) if v is not None]
    if not cleaned:
        return None
    return round(math.fsum(cleaned) / len(cleaned), PERCENT_PRECISION)


def weighted_mean(values: Sequence[Number], weights: Sequence[Number]) -> float | None:
    """Weighted mean of ``values`` by ``weights``; entries with zero weight are skipped."""
    if len(values) != len(weights):
        raise ValueError("values and weights must be the same length")
    total_weight = 0.0
    accumulator = 0.0
    for value, weight in zip(values, weights, strict=True):
        value_number = safe_float(value)
        weight_number = safe_float(weight)
        if value_number is None or weight_number is None or weight_number <= 0:
            continue
        accumulator += value_number * weight_number
        total_weight += weight_number
    if total_weight <= 0:
        return None
    return round(accumulator / total_weight, PERCENT_PRECISION)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_non_negative(value: Number) -> float:
    """Floor a value at zero. Cycle time, backlog and delay counts cannot be negative."""
    coerced = safe_float(value)
    if coerced is None:
        return 0.0
    return max(0.0, coerced)


def z_scores(values: Sequence[Number]) -> list[float | None]:
    """Population z-scores. Degenerate input yields ``None`` for every entry."""
    cleaned = [safe_float(v) for v in values]
    present = [v for v in cleaned if v is not None]
    if len(present) < 2:
        return [None] * len(cleaned)
    centre = math.fsum(present) / len(present)
    variance = math.fsum((v - centre) ** 2 for v in present) / len(present)
    if variance <= 0:
        return [None] * len(cleaned)
    stdev = math.sqrt(variance)
    return [None if v is None else round((v - centre) / stdev, 4) for v in cleaned]


def rolling_baseline(values: Sequence[Number], window: int = 3) -> list[float | None]:
    """Trailing rolling mean, excluding the current point.

    The final entry of each window is the most recent observation, so the result
    at index ``i`` is the mean of ``values[i-window:i]`` (or fewer at the start).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    result: list[float | None] = []
    for index in range(len(values)):
        start = max(0, index - window)
        baseline_values = values[start:index]
        result.append(mean(baseline_values) if baseline_values else None)
    return result


def linear_trend(values: Sequence[Number]) -> dict[str, float | None] | None:
    """Least-squares slope/intercept and R^2 for a numeric series.

    Returns ``None`` when fewer than two usable points exist or when every point
    is identical (no slope to report). ``slope`` is expressed in units per step.
    """
    points = [(index, v) for index, v in enumerate(safe_float(x) for x in values) if v is not None]
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(points)
    mean_x = math.fsum(xs) / n
    mean_y = math.fsum(ys) / n
    sxx = math.fsum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = math.fsum((x - mean_x) * (y - mean_y) for x, y in points)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = math.fsum((y - mean_y) ** 2 for y in ys)
    ss_res = math.fsum((y - (slope * x + intercept)) ** 2 for x, y in points)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None
    return {
        "slope": round(slope, PERCENT_PRECISION),
        "intercept": round(intercept, PERCENT_PRECISION),
        "r_squared": round(r_squared, 4) if r_squared is not None else None,
        "observations": float(n),
    }


def clamp_percentage(value: Number) -> float | None:
    """Coerce a ratio into a valid percentage, rejecting out-of-range input."""
    coerced = safe_float(value)
    if coerced is None:
        return None
    return round(clamp(coerced, 0.0, 100.0), PERCENT_PRECISION)


def normalise_series(values: Sequence[Number]) -> list[float | None]:
    """Min-max normalise a series to the 0-1 range, preserving ``None``."""
    coerced = [safe_float(value) for value in values]
    present = [value for value in coerced if value is not None]
    if not present:
        return [None] * len(values)
    low = min(present)
    high = max(present)
    span = high - low
    if span <= 0:
        return [None if value is None else 0.0 for value in coerced]
    return [None if value is None else round((value - low) / span, 4) for value in coerced]
