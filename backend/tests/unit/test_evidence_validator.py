"""Tests for the evidence validator: the numeric grounding gate."""

from __future__ import annotations

import pytest

from app.ai.evidence_validator import (
    ABSOLUTE_TOLERANCE,
    EvidenceIndex,
    GroundingReport,
    extract_numbers,
    is_rounding_of,
    validate_evidence_ids,
    validate_grounding,
)
from app.models.enums import DatasetId, Severity, Unit
from app.schemas.analysis import EvidenceItem

pytestmark = pytest.mark.unit


def _item(**overrides) -> EvidenceItem:
    defaults = {
        "metric": "Average cycle time",
        "value": 3.27,
        "comparison_value": 3.13,
        "absolute_change": 0.14,
        "percentage_change": 4.47,
        "unit": Unit.HOURS,
        "dimension": "Procurement",
        "period": "2025-07 to 2025-09",
        "comparison_period": "2025-04 to 2025-06",
        "source_dataset": DatasetId.OPERATIONS,
        "calculation_description": "completed_operations-weighted mean",
        "severity": Severity.MEDIUM,
        "sample_size": 9000,
    }
    defaults.update(overrides)
    return EvidenceItem(**defaults)


class TestExtractNumbers:
    def test_extracts_decimals_and_thousands(self):
        found = extract_numbers("Rose to 3.27 hours from 3.13, across 1,388 operations.")
        assert [value for value, _ in found] == [3.27, 3.13, 1388.0]

    def test_negative_and_signed_values(self):
        found = extract_numbers("change of -0.14 and +2.5 points")
        assert [value for value, _ in found] == [-0.14, 2.5]

    def test_identifier_codes_are_not_measurements(self):
        found = extract_numbers("DEP-102 and VND-207 in 2025-Q3")
        assert found == []

    def test_bare_year_is_not_a_measurement(self):
        assert extract_numbers("across 2025 the average rose") == []

    def test_small_ordinals_are_not_measurements(self):
        assert extract_numbers("the top 3 processes and the 2 vendors") == []

    def test_empty_text(self):
        assert extract_numbers("") == []

    def test_none_text_is_tolerated(self):
        assert extract_numbers(None) == []


class TestGrounding:
    def test_matching_numbers_are_accepted(self):
        index = EvidenceIndex.from_items((_item(),))
        report = validate_grounding(
            "Cycle time rose to 3.27 hours from 3.13, a 4.47% increase.", index
        )
        assert report.is_grounded
        assert report.checked == 3
        assert report.matched == 3
        assert report.unsupported == ()

    def test_fabricated_number_is_rejected(self):
        index = EvidenceIndex.from_items((_item(),))
        report = validate_grounding("Throughput reached 98000 units per day.", index)
        assert not report.is_grounded
        assert report.unsupported_count == 1
        issue = report.unsupported[0]
        assert issue.value == 98000.0
        assert "98000" in str(issue)

    def test_comparison_value_is_accepted(self):
        index = EvidenceIndex.from_items((_item(),))
        assert validate_grounding("previously 3.13", index).is_grounded

    def test_change_fields_are_accepted(self):
        index = EvidenceIndex.from_items((_item(),))
        assert validate_grounding("up 0.14 hours, 4.47%", index).is_grounded

    def test_sample_size_is_accepted(self):
        index = EvidenceIndex.from_items((_item(),))
        assert validate_grounding("across 9000 operations", index).is_grounded

    def test_text_with_no_numbers_is_grounded(self):
        index = EvidenceIndex.from_items((_item(),))
        report = validate_grounding("Procurement carries the largest share.", index)
        assert report.is_grounded
        assert report.checked == 0
        assert "no numbers were cited" in report.summary()

    def test_issue_excerpt_is_bounded(self):
        index = EvidenceIndex.from_items((_item(),))
        report = validate_grounding("x" * 300 + " 777777", index)
        assert not report.is_grounded
        assert len(report.unsupported[0].excerpt) <= 70

    def test_field_name_is_recorded(self):
        index = EvidenceIndex.from_items((_item(),))
        report = validate_grounding("424242", index, field_name="recommendations")
        assert report.unsupported[0].field_name == "recommendations"

    def test_exact_tolerance_boundary(self):
        index = EvidenceIndex.from_items((_item(),))
        just_inside = validate_grounding(f"{3.27 + ABSOLUTE_TOLERANCE / 2:.6f}", index)
        just_outside = validate_grounding(f"{3.27 + ABSOLUTE_TOLERANCE * 100:.6f}", index)
        assert just_inside.is_grounded
        assert not just_outside.is_grounded


class TestRounding:
    @pytest.mark.parametrize("value", [3422.0, 3420.0, 3400.0])
    def test_faithful_presentations_are_rounding(self, value):
        index = EvidenceIndex.from_items(
            (
                _item(
                    value=3421.7,
                    comparison_value=None,
                    absolute_change=None,
                    percentage_change=None,
                    sample_size=None,
                ),
            )
        )
        assert is_rounding_of(value, index)

    @pytest.mark.parametrize("value", [3900.0, 3902.0, 999.0, 3000.0])
    def test_different_or_too_coarse_numbers_are_not_rounding(self, value):
        index = EvidenceIndex.from_items(
            (
                _item(
                    value=3421.7,
                    comparison_value=None,
                    absolute_change=None,
                    percentage_change=None,
                    sample_size=None,
                ),
            )
        )
        assert not is_rounding_of(value, index)

    def test_relative_guard_blocks_a_fabricated_headline_figure(self):
        """A far-off number must not pass as a coarse rounding of another value."""
        index = EvidenceIndex.from_items((_item(percentage_change=55.38, value=1.86),))
        # 55.38 would round to 100 at the hundreds place without the guard.
        assert not is_rounding_of(100.0, index)

    def test_one_decimal_of_two_decimal_evidence(self):
        index = EvidenceIndex.from_items((_item(),))
        assert is_rounding_of(3.3, index)

    def test_rounding_is_reported_separately_from_exact(self):
        index = EvidenceIndex.from_items((_item(),))
        report = validate_grounding("Cycle time is about 3.3 hours versus 3.27.", index)
        assert report.is_grounded
        assert report.matched == 1
        assert report.rounded == 1

    def test_rounding_does_not_rescue_a_different_number(self):
        index = EvidenceIndex.from_items((_item(),))
        report = validate_grounding("Cycle time is 9.9 hours.", index)
        assert not report.is_grounded


class TestPublishedIndex:
    def test_includes_kpi_values(self):
        class _Kpi:
            value = 3421.7

        index = EvidenceIndex.from_published((_item(),), kpis=(_Kpi(),))
        assert validate_grounding("throughput was 3421.7 per day", index).is_grounded

    def test_includes_finding_numbers(self):
        class _Finding:
            title = "Queue share shifted"
            detail = "Queue share moved 6.4 percentage points."
            statistic = "queue share delta = +6.40 pp"

        index = EvidenceIndex.from_published((_item(),), findings=(_Finding(),))
        assert validate_grounding("Queue share moved 6.4 points.", index).is_grounded

    def test_includes_caveat_numbers(self):
        index = EvidenceIndex.from_published(
            (_item(),), caveats=("7 vendors have fewer than 20 orders in the window",)
        )
        assert validate_grounding("Only 7 vendors are thin enough to ignore.", index).is_grounded

    def test_still_rejects_a_figure_from_nowhere(self):
        index = EvidenceIndex.from_published((_item(),), caveats=("7 vendors",))
        assert not validate_grounding("12 vendors are thin.", index).is_grounded


class TestEvidenceIds:
    def test_known_ids_are_kept(self):
        index = EvidenceIndex.from_items((_item(), _item(metric="Throughput")))
        assert validate_evidence_ids(("E01", "E02"), index) == ("E01", "E02")

    def test_unknown_ids_are_dropped(self):
        index = EvidenceIndex.from_items((_item(),))
        assert validate_evidence_ids(("E01", "E99"), index) == ("E01",)

    def test_no_evidence_means_no_ids_survive(self):
        index = EvidenceIndex.from_items(())
        assert validate_evidence_ids(("E01",), index) == ()

    def test_non_list_input_is_empty(self):
        index = EvidenceIndex.from_items((_item(),))
        assert validate_evidence_ids(("E01",), index) == ("E01",)


class TestReport:
    def test_summary_describes_mixed_outcomes(self):
        report = GroundingReport(checked=5, matched=3, rounded=1, unsupported=())
        text = report.summary()
        assert "5" in text and "3" in text and "1" in text

    def test_summary_counts_unsupported(self):
        index = EvidenceIndex.from_items((_item(),))
        report = validate_grounding("value 555555", index)
        assert "no evidence" in report.summary()

    def test_describe_is_json_friendly(self):
        report = GroundingReport(checked=2, matched=2)
        described = report.describe()
        assert described["numbers_checked"] == 2
        assert described["grounded"] is True
        assert all(isinstance(value, (int, bool, str)) for value in described.values())
