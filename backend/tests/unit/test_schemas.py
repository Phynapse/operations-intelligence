"""Tests for schema validation and data loading."""

from __future__ import annotations

import datetime as dt

import pytest

from app.schemas.operational import (
    DepartmentRecord,
    OperationalRecord,
    OrderRecord,
    SlaMetricRecord,
    VendorRecord,
)
from app.utils.periods import (
    days_in_month,
    expand_period_range,
    is_complete_period,
    month_day_fraction,
    parse_period,
    period_sort_key,
    preceding_period,
    resolve_period_range,
)

# -- operational record ----------------------------------------------


class TestOperationalRecord:
    def test_valid(self):
        rec = OperationalRecord(
            operation_id="OP-001",
            date=dt.date(2025, 3, 15),
            period="2025-03",
            quarter="2025-Q1",
            department="DEP-101",
            process="order intake",
            operation_type="Order Management",
            status="completed",
            priority="standard",
            completed_operations=50,
            delayed_operations=3,
            processing_time_hours=2.5,
            queue_time_hours=1.2,
            total_cycle_time_hours=3.7,
            employee_count=12,
            vendor_id=None,
            backlog_open_items=5.0,
        )
        assert rec.delay_rate == pytest.approx(6.0)
        assert rec.queue_contribution == pytest.approx(32.43, abs=0.02)

    def test_blank_vendor_becomes_none(self):
        rec = OperationalRecord(
            operation_id="OP-001",
            date=dt.date(2025, 1, 1),
            period="2025-01",
            quarter="2025-Q1",
            department="DEP-101",
            process="order intake",
            operation_type="Order Management",
            status="completed",
            priority="standard",
            completed_operations=10,
            delayed_operations=0,
            processing_time_hours=1.0,
            queue_time_hours=0.5,
            total_cycle_time_hours=1.5,
            employee_count=10,
            vendor_id="",
            backlog_open_items=0.0,
        )
        assert rec.vendor_id is None

    def test_delayed_exceeds_completed(self):
        with pytest.raises(ValueError):
            OperationalRecord(
                operation_id="OP-001",
                date=dt.date(2025, 1, 1),
                period="2025-01",
                quarter="2025-Q1",
                department="DEP-101",
                process="order intake",
                operation_type="Order Management",
                status="completed",
                priority="standard",
                completed_operations=5,
                delayed_operations=10,
                processing_time_hours=1.0,
                queue_time_hours=0.5,
                total_cycle_time_hours=1.5,
                employee_count=10,
                vendor_id=None,
                backlog_open_items=0.0,
            )

    def test_delay_rate_zero_volume(self):
        rec = OperationalRecord(
            operation_id="OP-001",
            date=dt.date(2025, 1, 1),
            period="2025-01",
            quarter="2025-Q1",
            department="DEP-101",
            process="order intake",
            operation_type="Order Management",
            status="completed",
            priority="standard",
            completed_operations=0,
            delayed_operations=0,
            processing_time_hours=1.0,
            queue_time_hours=0.5,
            total_cycle_time_hours=1.5,
            employee_count=10,
            vendor_id=None,
            backlog_open_items=0.0,
        )
        assert rec.delay_rate == 0.0


# -- order record ----------------------------------------------------


class TestOrderRecord:
    def test_valid(self):
        rec = OrderRecord(
            order_id="ORD-001",
            order_date=dt.date(2025, 6, 1),
            period="2025-06",
            quarter="2025-Q2",
            department="DEP-101",
            product_category="Standard Supplies",
            order_value=1250.0,
            promised_time_hours=48.0,
            actual_time_hours=42.0,
            fulfillment_time_hours=18.0,
            transit_time_hours=24.0,
            status="delivered",
            vendor_id="VND-101",
            delay_hours=0.0,
        )
        assert rec.is_late is False

    def test_late(self):
        rec = OrderRecord(
            order_id="ORD-002",
            order_date=dt.date(2025, 6, 1),
            period="2025-06",
            quarter="2025-Q2",
            department="DEP-101",
            product_category="Standard Supplies",
            order_value=500.0,
            promised_time_hours=48.0,
            actual_time_hours=55.0,
            fulfillment_time_hours=18.0,
            transit_time_hours=37.0,
            status="delivered_late",
            vendor_id="VND-102",
            delay_hours=7.0,
        )
        assert rec.is_late is True


# -- SLA record ------------------------------------------------------


class TestSLAMetricRecord:
    def test_valid(self):
        rec = SlaMetricRecord(
            period="2025-06",
            quarter="2025-Q2",
            department="DEP-102",
            process="picking and packing",
            total_cases=500,
            sla_target_hours=8.0,
            average_cycle_time_hours=6.5,
            average_queue_time_hours=2.0,
            sla_met_cases=475,
            sla_breached_cases=25,
            sla_compliance_percentage=95.0,
            backlog_open_items=10.0,
        )
        assert rec.sla_compliance_percentage == pytest.approx(95.0)

    def test_cases_mismatch(self):
        with pytest.raises(ValueError, match="sla_breached_cases must equal"):
            SlaMetricRecord(
                period="2025-06",
                quarter="2025-Q2",
                department="DEP-102",
                process="picking and packing",
                total_cases=500,
                sla_target_hours=8.0,
                average_cycle_time_hours=6.5,
                average_queue_time_hours=2.0,
                sla_met_cases=475,
                sla_breached_cases=30,  # 500 - 475 = 25, not 30
                sla_compliance_percentage=95.0,
                backlog_open_items=10.0,
            )


# -- vendor & department ---------------------------------------------


class TestVendorRecord:
    def test_valid(self):
        rec = VendorRecord(
            vendor_id="VND-101",
            vendor_name="Northline Freight",
            category="Logistics",
            department="Fulfillment",
            monthly_volume=200.0,
            average_processing_time=4.5,
            delay_rate=5.0,
            sla_compliance=92.0,
        )
        assert rec.vendor_id == "VND-101"


class TestDepartmentRecord:
    def test_valid(self):
        rec = DepartmentRecord(
            department_id="DEP-101",
            department_name="Customer Operations",
            operational_area="Front Office",
            headcount=34,
        )
        assert rec.headcount == 34


# -- period helpers --------------------------------------------------


class TestParsePeriod:
    def test_month(self):
        assert parse_period("2025-06") == (2025, 6)

    def test_quarter(self):
        assert parse_period("2025-Q3") == (2025, 9)

    def test_year(self):
        assert parse_period("2025") == (2025, 12)

    def test_invalid(self):
        assert parse_period("abc") is None
        assert parse_period("") is None


class TestExpandPeriodRange:
    def test_months(self):
        result = list(expand_period_range("2025-01", "2025-03"))
        assert result == ["2025-01", "2025-02", "2025-03"]

    def test_year_cross(self):
        result = list(expand_period_range("2024-11", "2025-02"))
        assert result == ["2024-11", "2024-12", "2025-01", "2025-02"]


class TestIsCompletePeriod:
    def test_complete(self):
        months = {"2025-01", "2025-02", "2025-03"}
        assert is_complete_period("2025-Q1", months) is True

    def test_incomplete(self):
        months = {"2025-01", "2025-03"}
        assert is_complete_period("2025-Q1", months) is False

    def test_empty(self):
        assert is_complete_period("2025-01", set()) is False


class TestPrecedingPeriod:
    def test_month(self):
        pr = type(
            "PeriodRange",
            (),
            {"start": "2025-06", "end": "2025-06", "month_count": 1, "months": ["2025-06"]},
        )()
        result = preceding_period(
            pr, {"2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"}
        )
        assert result is not None
        assert result.start == "2025-05"

    def test_no_preceding(self):
        pr = type(
            "PeriodRange",
            (),
            {"start": "2025-01", "end": "2025-01", "month_count": 1, "months": ["2025-01"]},
        )()
        result = preceding_period(pr, {"2025-01"})
        assert result is None


class TestPeriodSortKey:
    def test_order(self):
        items = sorted(["2025-Q2", "2025-01", "2025-Q1", "2025-12"], key=period_sort_key)
        assert items == ["2025-01", "2025-Q1", "2025-Q2", "2025-12"]


class TestDaysInMonth:
    def test_leap(self):
        assert days_in_month("2024-02") == 29

    def test_normal(self):
        assert days_in_month("2025-02") == 28

    def test_31(self):
        assert days_in_month("2025-01") == 31

    def test_30(self):
        assert days_in_month("2025-04") == 30


class TestResolvePeriodRange:
    def test_quarter(self):
        months = {"2025-04", "2025-05", "2025-06", "2025-07"}
        primary, _comp, _caveats = resolve_period_range("2025-Q2", months)
        assert primary is not None
        assert primary.start == "2025-04"
        assert primary.end == "2025-06"

    def test_no_data(self):
        primary, _comp, caveats = resolve_period_range("2025-Q2", set())
        assert primary is None
        assert len(caveats) >= 1

    def test_default_all(self):
        months = {"2025-01", "2025-02"}
        primary, _comp, _caveats = resolve_period_range(None, months)
        assert primary is not None
        assert primary.start == "2025-01"


class TestMonthDayFraction:
    def test_full(self):
        assert month_day_fraction("2025-06") == 1.0

    def test_partial(self):
        assert month_day_fraction("2025-06", 15.0) == pytest.approx(0.5)
