"""Evidence package construction.

The evidence package is the contract between the deterministic analytics layer
and the LLM explanation layer. It is deliberately narrow:

* it contains **only** numbers that a deterministic analyzer produced;
* every item states its unit, source dataset and calculation;
* it is capped and ordered by severity so the LLM cannot be handed an unbounded
  pile of rows to anchor on.

The LLM receives this package, not the DataFrames. That is the mechanism that
prevents the model from inventing, re-deriving or rounding numbers.
"""

from __future__ import annotations

import uuid

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.enums import Severity
from app.schemas.ai import EvidenceSummary
from app.schemas.analysis import KPI, AnalysisResult, EvidenceItem, Finding

logger = get_logger(__name__)

#: Severity ordering used to prioritise the package.
SEVERITY_RANK: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}

#: Hard cap on how many evidence items reach the LLM.
MAX_ITEMS_FOR_LLM = 24


class EvidencePackage:
    """An ordered, capped view of an analysis result's evidence."""

    __slots__ = ("caveats", "dropped_count", "findings", "items", "kpis")

    def __init__(
        self,
        items: tuple[EvidenceItem, ...],
        findings: tuple[Finding, ...],
        kpis: tuple[KPI, ...],
        caveats: tuple[str, ...],
        dropped_count: int = 0,
    ) -> None:
        self.items = items
        self.findings = findings
        self.kpis = kpis
        self.caveats = caveats
        self.dropped_count = dropped_count

    def __len__(self) -> int:
        return len(self.items)

    def summaries(self) -> tuple[EvidenceSummary, ...]:
        """Render the package as the payload handed to the explanation LLM."""
        return tuple(_to_summary(item, index) for index, item in enumerate(self.items))

    def finding_statements(self) -> tuple[str, ...]:
        """Deterministic findings phrased as plain sentences for the prompt."""
        return tuple(
            f"[{finding.severity.value}] {finding.title}: {finding.detail}"
            for finding in self.findings
        )

    def describe(self) -> dict[str, object]:
        """A compact description used in logs and the API envelope."""
        return {
            "evidence_items": len(self.items),
            "dropped_items": self.dropped_count,
            "findings": len(self.findings),
            "kpis": len(self.kpis),
            "caveats": len(self.caveats),
        }


class EvidenceService:
    """Builds :class:`EvidencePackage` objects from analysis results."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def build(self, result: AnalysisResult) -> EvidencePackage:
        """Select, order and cap the evidence for a result."""
        limit = max(self._settings.evidence_top_n * 4, MAX_ITEMS_FOR_LLM)
        ordered = self._order(result.evidence)
        kept = tuple(ordered[:limit])
        dropped = len(ordered) - len(kept)

        withheld_note = (
            (
                f"{dropped} lower-priority evidence item(s) were withheld from the narrative "
                "layer to keep it focused; the full set remains in the analysis result",
            )
            if dropped
            else ()
        )

        package = EvidencePackage(
            items=kept,
            findings=result.findings,
            kpis=result.kpis,
            caveats=(*result.caveats, *withheld_note),
            dropped_count=dropped,
        )
        logger.info(
            "evidence_package_built",
            extra={"analysis_id": result.analysis_id, **package.describe()},
        )
        return package

    @staticmethod
    def _order(evidence: tuple[EvidenceItem, ...]) -> list[EvidenceItem]:
        """Severity first, then by how much the item actually moved.

        Items that carry a comparison are more useful for explanation than
        standalone levels, so they win ties within the same severity.
        """

        def sort_key(item: EvidenceItem) -> tuple[int, int, int, float, str]:
            severity = SEVERITY_RANK.get(item.severity, 9)
            has_value = 0 if item.value is not None else 1
            has_comparison = 0 if item.comparison_value is not None else 1
            movement = abs(item.percentage_change or item.absolute_change or 0.0)
            return (severity, has_value, has_comparison, -movement, item.metric)

        return sorted(evidence, key=sort_key)


def _to_summary(item: EvidenceItem, index: int) -> EvidenceSummary:
    """Convert an evidence item into its LLM-facing summary form."""
    return EvidenceSummary(
        evidence_id=f"E{index + 1:02d}",
        metric=item.metric,
        value=item.value,
        comparison_value=item.comparison_value,
        absolute_change=item.absolute_change,
        percentage_change=item.percentage_change,
        unit=item.unit.value,
        dimension=item.dimension,
        period=item.period,
        source_dataset=item.source_dataset.value,
        calculation_description=item.calculation_description,
    )


def evidence_id_map(result: AnalysisResult) -> dict[str, EvidenceItem]:
    """Stable ``evidence_id -> EvidenceItem`` map for one result."""
    ordered = EvidenceService._order(result.evidence)
    return {f"E{index + 1:02d}": item for index, item in enumerate(ordered)}


def build_trace_id() -> str:
    return uuid.uuid4().hex[:12]


_service: EvidenceService | None = None


def get_evidence_service() -> EvidenceService:
    global _service
    if _service is None:
        _service = EvidenceService()
    return _service


def reset_evidence_service() -> None:
    global _service
    _service = None


__all__ = [
    "SEVERITY_RANK",
    "EvidencePackage",
    "EvidenceService",
    "evidence_id_map",
    "get_evidence_service",
    "reset_evidence_service",
]
