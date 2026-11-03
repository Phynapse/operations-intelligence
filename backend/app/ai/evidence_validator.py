"""Evidence validation: the numeric grounding gate.

The evidence package is the only thing the LLM is allowed to know, so the only
question that matters after it responds is: **does every number it wrote appear in
that package?**

This module answers that mechanically. It never asks a model whether it
fabricated something; it extracts numeric tokens from the model's own text and
looks each one up in the evidence. A token with no matching value is an
unsupported claim. This is deliberately dumb and deterministic, because a
check that depends on another LLM judgement is not a check.

Three classes of token are handled differently:

``matched``
    The number equals an evidence value, comparison value, absolute change or
    percentage change, within a rounding tolerance. This is the normal case.

``rounded``
    The number is a rounded or re-scaled form of an evidence value (``12.3``
    for ``12.34``, ``3400`` for ``3,421.7``). Allowed, because prose
    legitimately rounds, but recorded so the caller knows it happened.

``unsupported``
    The number matches nothing. Rejected. The offending text is returned so the
    explanation layer can be discarded or repaired rather than shown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.schemas.analysis import EvidenceItem

logger = get_logger(__name__)

#: Numbers closer than this to an evidence value count as equal. Evidence is
#: stored rounded to 4 decimals, so 1e-3 covers rounding without being loose
#: enough to let a fabricated number through.
ABSOLUTE_TOLERANCE = 1e-3

#: Relative tolerance, applied alongside the absolute one so large values are
#: not held to an unreasonably tight absolute standard.
RELATIVE_TOLERANCE = 1e-4

#: Percentage points a prose figure may differ from evidence and still count as
#: a rounding of the same number.
ROUNDING_TOLERANCE = 0.005

#: How far a rounded figure may sit from the value it rounds, as a fraction of
#: that value. Without this, any number would be a valid "rounding" of some
#: evidence value at a coarse enough precision: 55.38 rounds to 100 at the
#: hundreds place, which would let a fabricated "100% compliance" pass. A
#: rounding must stay within this band of the truth.
MAX_ROUNDING_RELATIVE_ERROR = 0.05

#: Reference codes and period labels that contain digits but are not
#: measurements. These are masked out before extraction so ``DEP-102`` is never
#: read as "one hundred and two" and ``2025-07`` never as the number seven.
#: Period labels matter most: nearly every narrative names its window, so
#: without this mask every grounded answer would fail its own check.
_IDENTIFIER_PATTERN = re.compile(
    r"(?:dep|vnd|ord|op)[-_]?\d+"
    r"|\d{4}-q[1-4]"
    r"|\d{4}-\d{2}"
    r"|\bq[1-4]\b",
    re.IGNORECASE,
)

#: A number token: optional sign, digits with optional thousands separators,
#: optional decimal part. Anchored so digits inside an id are not extracted.
_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])"  # 1,234.56
    r"|(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w])"  # 1234.56
)

#: Tokens that are never measurements: years, counts that are part of a
#: reference code, and ordinals used in prose.
_NON_MEASUREMENT_TOKENS: frozenset[str] = frozenset(
    {
        # The dataset covers a single calendar year; a bare year is context.
        "2024",
        "2025",
        "2026",
        # Small ordinals: "top 3 processes", "the 2 drivers".
        "1",
        "2",
        "3",
        "4",
        "5",
    }
)


@dataclass(frozen=True, slots=True)
class GroundingIssue:
    """One number in generated text that the evidence does not support."""

    value: float
    raw_token: str
    field_name: str
    excerpt: str

    def __str__(self) -> str:
        return (
            f"{self.raw_token!r} in {self.field_name} has no matching evidence value "
            f"(nearest context: {self.excerpt!r})"
        )


@dataclass(frozen=True, slots=True)
class GroundingReport:
    """Outcome of validating generated text against the evidence package."""

    #: Numbers found in the text and their match status.
    checked: int = 0
    matched: int = 0
    rounded: int = 0
    unsupported: tuple[GroundingIssue, ...] = ()

    @property
    def is_grounded(self) -> bool:
        return not self.unsupported

    @property
    def unsupported_count(self) -> int:
        return len(self.unsupported)

    def summary(self) -> str:
        if self.checked == 0:
            return "no numbers were cited"
        parts = [f"{self.matched} of {self.checked} cited numbers matched evidence exactly"]
        if self.rounded:
            parts.append(f"{self.rounded} matched after rounding")
        if self.unsupported:
            parts.append(f"{self.unsupported_count} had no evidence")
        return "; ".join(parts)

    def describe(self) -> dict[str, object]:
        return {
            "numbers_checked": self.checked,
            "matched": self.matched,
            "matched_after_rounding": self.rounded,
            "unsupported": self.unsupported_count,
            "grounded": self.is_grounded,
        }


@dataclass(slots=True)
class EvidenceIndex:
    """Fast lookup of every number the response body makes available.

    Built once per explanation so validating a long answer stays cheap.

    The index deliberately spans everything the *caller can see* - evidence
    values, KPI values and the figures quoted inside deterministic findings -
    because the invariant being enforced is not "the model may only repeat
    evidence rows" but "every number in the narrative appears in the response the
    reader receives". Findings are published alongside the evidence, so a figure
    quoted there is available to the reader even when it was not promoted to an
    evidence row.

    Nothing here widens what the *model* may say: a model that invents a figure
    appearing in no finding, KPI or evidence item is still caught.
    """

    #: Exact-ish values, used for the primary comparison.
    values: tuple[float, ...] = ()
    #: Absolute magnitudes, used to recognise re-scaled prose ("3,421" for 3421.7).
    magnitudes: tuple[float, ...] = ()
    #: The set of evidence ids that can legitimately be cited.
    evidence_ids: frozenset[str] = field(default_factory=frozenset)
    #: Evidence item count, used to report coverage.
    item_count: int = 0

    @classmethod
    def from_items(cls, items: tuple[EvidenceItem, ...], id_prefix: str = "E") -> EvidenceIndex:
        return cls._build(
            values=_evidence_values(items),
            ids=frozenset(f"{id_prefix}{index + 1:02d}" for index in range(len(items))),
            item_count=len(items),
        )

    @classmethod
    def from_published(
        cls,
        items: tuple[EvidenceItem, ...],
        kpis: tuple[object, ...] = (),
        findings: tuple[object, ...] = (),
        caveats: tuple[str, ...] = (),
        id_prefix: str = "E",
    ) -> EvidenceIndex:
        """Index every number the response publishes, not just the evidence rows."""
        values = list(_evidence_values(items))

        for kpi in kpis:
            candidate = getattr(kpi, "value", None)
            if candidate is not None:
                values.append(float(candidate))

        for finding in findings:
            for text in (
                getattr(finding, "detail", None),
                getattr(finding, "title", None),
                getattr(finding, "statistic", None),
            ):
                if isinstance(text, str) and text:
                    values.extend(value for value, _ in extract_numbers(text))

        # Caveats are returned to the reader alongside the answer, and they
        # legitimately carry figures such as "7 vendors have fewer than 20
        # orders", so they count as published too.
        for caveat in caveats:
            if caveat:
                values.extend(value for value, _ in extract_numbers(caveat))

        return cls._build(
            values=tuple(values),
            ids=frozenset(f"{id_prefix}{index + 1:02d}" for index in range(len(items))),
            item_count=len(items),
        )

    @classmethod
    def _build(
        cls, values: tuple[float, ...], ids: frozenset[str], item_count: int
    ) -> EvidenceIndex:
        magnitudes = tuple(abs(value) for value in values)
        return cls(values=values, magnitudes=magnitudes, evidence_ids=ids, item_count=item_count)

    def contains(self, value: float) -> bool:
        """True when ``value`` is an evidence value or an obvious rounding of one."""
        if any(_close(value, candidate) for candidate in self.values):
            return True
        magnitude = abs(value)
        return any(_close(magnitude, candidate) for candidate in self.magnitudes)


def _evidence_values(items: tuple[EvidenceItem, ...]) -> tuple[float, ...]:
    """Every numeric field an evidence item publishes."""
    values: list[float] = []
    for item in items:
        for candidate in (
            item.value,
            item.comparison_value,
            item.absolute_change,
            item.percentage_change,
            item.sample_size,
        ):
            if candidate is not None:
                values.append(float(candidate))
    return tuple(values)


def _close(left: float, right: float) -> bool:
    if left != left or right != right:  # NaN
        return False
    if abs(left - right) <= ABSOLUTE_TOLERANCE:
        return True
    scale = max(abs(left), abs(right))
    return scale > 0 and abs(left - right) <= scale * RELATIVE_TOLERANCE


#: Decimal places tried when testing whether a written number is a rounding of an
#: evidence value. Negative places cover rounding to tens, hundreds and thousands.
_ROUNDING_PLACES: tuple[int, ...] = (3, 2, 1, 0, -1, -2, -3)


def is_rounding_of(value: float, index: EvidenceIndex) -> bool:
    """True when ``value`` is a coarser or rounded presentation of evidence.

    Prose legitimately writes "about 3,400" for an evidence value of 3421.7, or
    "3.3 hours" for 3.27. Both are roundings, not fabrications, so they are
    accepted and counted separately from exact matches.

    The test is: does the written number equal the evidence value rounded to
    *some* decimal place, and does it stay within
    :data:`MAX_ROUNDING_RELATIVE_ERROR` of the truth? For 3421.7 that admits
    3421.7, 3422 (nearest unit), 3420 and 3400 — and rejects 3900, which is a
    different number, and 3000, which is too coarse to be a faithful statement
    of the value. The relative guard is what stops any figure from passing as
    the rounding of some sufficiently coarse presentation of another figure.
    """
    magnitude = abs(value)
    if magnitude == 0.0:
        return False
    for candidate in (*index.values, *index.magnitudes):
        target = abs(candidate)
        if target == 0.0:
            continue
        if abs(magnitude - target) / target > MAX_ROUNDING_RELATIVE_ERROR:
            continue
        for places in _ROUNDING_PLACES:
            try:
                rounded = round(target, places)
            except (OverflowError, ValueError):  # pragma: no cover - non-finite
                continue
            if abs(magnitude - rounded) <= ROUNDING_TOLERANCE:
                return True
    return False


def extract_numbers(text: str) -> list[tuple[float, str]]:
    """Extract measurement-like numbers from free text.

    Reference codes and period labels are masked first, so an id never becomes a
    measurement. Returns ``(value, raw_token)`` pairs in document order.
    Integers written without a decimal point come back as floats so they compare
    cleanly with evidence values.
    """
    if not text:
        return []
    masked = _IDENTIFIER_PATTERN.sub(lambda match: " " * len(match.group(0)), text)

    found: list[tuple[float, str]] = []
    for match in _NUMBER_PATTERN.finditer(masked):
        raw = match.group(0)
        token = raw.replace(",", "").lstrip("+")
        try:
            value = float(token)
        except ValueError:  # pragma: no cover - pattern only matches numerics
            continue
        if raw in _NON_MEASUREMENT_TOKENS:
            continue
        found.append((value, raw))
    return found


def validate_grounding(
    text: str,
    index: EvidenceIndex,
    *,
    field_name: str = "answer",
) -> GroundingReport:
    """Check that every number in ``text`` is supported by the evidence package.

    Returns a :class:`GroundingReport`. The caller decides what to do with
    unsupported values; this function only reports them.
    """
    numbers = extract_numbers(text)
    if not numbers:
        return GroundingReport()

    matched = 0
    rounded = 0
    unsupported: list[GroundingIssue] = []

    for value, raw in numbers:
        if index.contains(value):
            matched += 1
        elif is_rounding_of(value, index):
            rounded += 1
        else:
            unsupported.append(
                GroundingIssue(
                    value=value,
                    raw_token=raw,
                    field_name=field_name,
                    excerpt=_excerpt(text, raw),
                )
            )

    report = GroundingReport(
        checked=len(numbers),
        matched=matched,
        rounded=rounded,
        unsupported=tuple(unsupported),
    )
    logger.info(
        "grounding_checked",
        extra={"field_name": field_name, **report.describe()},
    )
    return report


def validate_evidence_ids(
    cited: tuple[str, ...],
    index: EvidenceIndex,
    *,
    field_name: str = "key_findings",
) -> tuple[str, ...]:
    """Return only the cited evidence ids that actually exist.

    A finding that cites ``E99`` when the package holds six items is making a
    claim about something it was never shown, so the id is dropped rather than
    passed through to the UI.
    """
    if not index.evidence_ids:
        return ()
    return tuple(item for item in cited if item in index.evidence_ids)


def _excerpt(text: str, token: str, width: int = 60) -> str:
    """A short window of text around ``token``, for error messages."""
    index = text.find(token)
    if index < 0:
        return text[:width].strip()
    start = max(0, index - width // 2)
    end = min(len(text), index + len(token) + width // 2)
    return text[start:end].replace("\n", " ").strip()


__all__ = [
    "ABSOLUTE_TOLERANCE",
    "RELATIVE_TOLERANCE",
    "EvidenceIndex",
    "GroundingIssue",
    "GroundingReport",
    "extract_numbers",
    "is_rounding_of",
    "validate_evidence_ids",
    "validate_grounding",
]
