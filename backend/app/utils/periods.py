"""Period parsing and comparison helpers.

The system speaks in three period granularities: ``YYYY-MM`` months,
``YYYY`` years and ``YYYY-Qn`` quarters. All of them are parsed and compared
here so that "Q3 versus Q2" behaves identically whether the planner expressed
it as a quarter string, two month ranges, or a single month.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from app.utils.calculations import safe_float

_MONTH_PATTERN = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})$")
_QUARTER_PATTERN = re.compile(r"^(?P<year>\d{4})-Q(?P<quarter>[1-4])$")
_YEAR_PATTERN = re.compile(r"^(?P<year>\d{4})$")
_RANGE_PATTERN = re.compile(
    r"^(?P<start>\d{4}(?:-\d{2})?(?:-Q[1-4])?)\s+to\s+(?P<end>\d{4}(?:-\d{2})?(?:-Q[1-4])?)$"
)

_QUARTER_MONTHS: dict[int, tuple[int, int]] = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}


@dataclass(frozen=True, slots=True)
class PeriodRange:
    """An inclusive span of monthly buckets."""

    start: str
    end: str
    label: str

    @property
    def months(self) -> tuple[str, ...]:
        return tuple(expand_period_range(self.start, self.end))

    @property
    def month_count(self) -> int:
        return len(self.months)

    def contains(self, month: str) -> bool:
        return self.start <= month <= self.end


def parse_period(period: str) -> tuple[int, int] | None:
    """Return ``(year, month)`` for any supported period string, else ``None``."""
    if not period:
        return None
    value = period.strip()

    month_match = _MONTH_PATTERN.match(value)
    if month_match:
        year = int(month_match.group("year"))
        month = int(month_match.group("month"))
        return (year, month) if 1 <= month <= 12 else None

    quarter_match = _QUARTER_PATTERN.match(value)
    if quarter_match:
        year = int(quarter_match.group("year"))
        quarter = int(quarter_match.group("quarter"))
        _first_month, last_month = _QUARTER_MONTHS[quarter]
        return year, last_month

    year_match = _YEAR_PATTERN.match(value)
    if year_match:
        return int(year_match.group("year")), 12

    return None


def period_sort_key(period: str) -> tuple[int, int]:
    """Chronological sort key; unparseable periods sort last."""
    parsed = parse_period(period)
    return parsed if parsed is not None else (9999, 12)


def months_covered_by(value: str) -> list[str] | None:
    """Every ``YYYY-MM`` bucket a period expression covers, or ``None``.

    Accepts a year (``2025``), a month (``2025-03``), a quarter (``2025-Q3``),
    an explicit range (``2025-03 to 2025-06``) or a single month.
    """
    text = (value or "").strip()
    if not text:
        return None

    range_match = _RANGE_PATTERN.match(text)
    if range_match:
        start = months_covered_by(range_match.group("start"))
        end = months_covered_by(range_match.group("end"))
        if not start or not end:
            return None
        return list(expand_period_range(start[-1], end[-1]))

    parsed = parse_period(text)
    if parsed is None:
        return None
    year, last_month = parsed

    # A bare month is that month alone. Only a bare year means "year to date",
    # and only a quarter spans three months.
    first_month = last_month if _MONTH_PATTERN.match(text) else 1
    quarter_match = _QUARTER_PATTERN.match(text)
    if quarter_match:
        first_month = _QUARTER_MONTHS[int(quarter_match.group("quarter"))][0]
    return [f"{year}-{month:02d}" for month in range(first_month, last_month + 1)]


def is_complete_period(period: str, available_months: set[str]) -> bool:
    """True when every month of ``period`` exists in the loaded dataset."""
    months = months_covered_by(period)
    if months is None:
        return False
    return set(months).issubset(available_months)


def expand_period_range(start: str, end: str) -> Iterator[str]:
    """Yield every ``YYYY-MM`` bucket between ``start`` and ``end`` inclusive."""
    start_parsed = parse_period(start)
    end_parsed = parse_period(end)
    if start_parsed is None or end_parsed is None:
        return
    start_year, start_month = start_parsed
    end_year, end_month = end_parsed
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield f"{year}-{month:02d}"
        month += 1
        if month > 12:
            month = 1
            year += 1


def resolve_period_range(
    period: str | None,
    available_months: AbstractSet[str],
    comparison_period: str | None = None,
) -> tuple[PeriodRange | None, PeriodRange | None, list[str]]:
    """Resolve requested periods against the months the dataset actually has.

    Returns ``(primary, comparison, caveats)``. Missing periods are reported as
    caveats rather than silently substituted, because silently analysing the
    wrong period is worse than refusing to analyse one.
    """
    caveats: list[str] = []
    ordered = sorted(available_months, key=period_sort_key)
    if not ordered:
        return None, None, ["no periods are available in the loaded dataset"]

    primary = _period_range(period, available_months, ordered, caveats, label="period")
    comparison = _period_range(
        comparison_period, available_months, ordered, caveats, label="comparison period"
    )
    if primary is None and period is None and comparison is None:
        primary = PeriodRange(start=ordered[0], end=ordered[-1], label="all available periods")
    return primary, comparison, caveats


def _period_range(
    requested: str | None,
    available: AbstractSet[str],
    ordered: list[str],
    caveats: list[str],
    label: str,
) -> PeriodRange | None:
    if requested is None:
        return None
    months = months_covered_by(requested)
    if months is None:
        caveats.append(f"unrecognised {label} '{requested}'; it was ignored")
        return None

    present = [month for month in months if month in available]
    if not present:
        caveats.append(
            f"the {label} '{requested}' contains no data in this dataset; "
            "the most recent available period was used instead"
        )
        return PeriodRange(start=ordered[-1], end=ordered[-1], label=requested)

    if len(present) < len(months):
        missing = [month for month in months if month not in available]
        caveats.append(
            f"the {label} '{requested}' is incomplete: {', '.join(missing)} "
            "not present in this dataset"
        )

    return PeriodRange(start=present[0], end=present[-1], label=requested)


def preceding_period(
    period_range: PeriodRange, available_months: AbstractSet[str]
) -> PeriodRange | None:
    """The equally-sized period immediately before ``period_range``.

    Used as the default comparison window for period-over-period questions.
    """
    span = period_range.month_count
    all_months = sorted(available_months, key=period_sort_key)
    index = all_months.index(period_range.start) if period_range.start in all_months else -1
    if index <= 0:
        return None
    end_index = index - 1
    start_index = max(0, end_index - span + 1)
    window = all_months[start_index : end_index + 1]
    if not window:
        return None
    label = f"{window[0]} to {window[-1]}"
    if len(window) == 1:
        label = window[0]
    return PeriodRange(start=window[0], end=window[-1], label=label)


def days_in_month(month: str) -> int | None:
    """Number of days in a ``YYYY-MM`` bucket, accounting for leap years."""
    if not _MONTH_PATTERN.match((month or "").strip()):
        return None
    parsed = parse_period(month)
    if parsed is None:
        return None
    year, month_number = parsed
    lengths = (31, 29 if _is_leap(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    return lengths[month_number - 1]


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def month_day_fraction(month: str, elapsed_days: float | None = None) -> float | None:
    """Fraction of a month elapsed. ``elapsed_days`` allows partial-month handling.

    Used to distinguish a genuinely flat throughput trend from one that only
    looks flat because the final month is incomplete.
    """
    total_days = days_in_month(month)
    if total_days is None:
        return None
    if elapsed_days is None:
        return 1.0
    ratio = safe_float(elapsed_days)
    if ratio is None:
        return None
    return round(min(1.0, max(0.0, ratio / total_days)), 4)
