"""Tests for deterministic numeric helpers.

Every calculation is verified against hand-computed expectations so that
a future refactor cannot silently change the behaviour of the analytics
layer.
"""

from __future__ import annotations

import pytest

from app.utils.calculations import (
    absolute_change,
    backlog_change,
    clamp,
    clamp_non_negative,
    clamp_percentage,
    contribution,
    delay_rate,
    is_finite,
    linear_trend,
    mean,
    normalise_series,
    percentage_change,
    percentage_of_total,
    rolling_baseline,
    round_or_none,
    safe_float,
    safe_ratio,
    sla_breach_rate,
    sla_compliance,
    throughput,
    weighted_mean,
    z_scores,
)

# -- basic helpers ---------------------------------------------------


class TestIsFinite:
    @pytest.mark.parametrize("value", [0, 1.5, -3.2, 0.0])
    def test_finite(self, value):
        assert is_finite(value)

    @pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf"), True])
    def test_not_finite(self, value):
        assert not is_finite(value)

    def test_string_rejected(self):
        assert not is_finite("hello")


class TestSafeFloat:
    def test_valid(self):
        assert safe_float(3.14) == pytest.approx(3.14)

    def test_none(self):
        assert safe_float(None) is None

    def test_nan(self):
        assert safe_float(float("nan")) is None

    def test_default(self):
        assert safe_float(None, default=-1.0) == pytest.approx(-1.0)


class TestRoundOrNone:
    def test_rounds(self):
        assert round_or_none(3.14159, 2) == pytest.approx(3.14)

    def test_none_input(self):
        assert round_or_none(None) is None

    def test_nan(self):
        assert round_or_none(float("nan")) is None


# -- change functions ------------------------------------------------


class TestPercentageChange:
    def test_basic(self):
        assert percentage_change(120, 100) == pytest.approx(20.0)

    def test_negative(self):
        assert percentage_change(80, 100) == pytest.approx(-20.0)

    def test_zero_previous(self):
        assert percentage_change(10, 0) is None

    def test_missing(self):
        assert percentage_change(None, 100) is None
        assert percentage_change(10, None) is None


class TestAbsoluteChange:
    def test_basic(self):
        assert absolute_change(120, 100) == pytest.approx(20.0)

    def test_negative(self):
        assert absolute_change(80, 100) == pytest.approx(-20.0)

    def test_missing(self):
        assert absolute_change(None, 100) is None


# -- ratio / percentage-of-total -------------------------------------


class TestSafeRatio:
    def test_basic(self):
        assert safe_ratio(3, 4) == pytest.approx(0.75)

    def test_zero_denominator(self):
        assert safe_ratio(5, 0) is None

    def test_missing(self):
        assert safe_ratio(None, 4) is None
        assert safe_ratio(5, None) is None


class TestPercentageOfTotal:
    def test_basic(self):
        assert percentage_of_total(25, 100) == pytest.approx(25.0)

    def test_zero_denominator(self):
        assert percentage_of_total(5, 0) is None


# -- domain-specific ------------------------------------------------


class TestSLACompliance:
    def test_compliant(self):
        assert sla_compliance(95, 100) == pytest.approx(95.0)

    def test_zero(self):
        assert sla_compliance(0, 100) == pytest.approx(0.0)

    def test_zero_total(self):
        assert sla_compliance(0, 0) is None


class TestSLABreachRate:
    def test_basic(self):
        assert sla_breach_rate(5, 100) == pytest.approx(5.0)

    def test_zero_total(self):
        assert sla_breach_rate(0, 0) is None


class TestDelayRate:
    def test_basic(self):
        assert delay_rate(15, 100) == pytest.approx(15.0)

    def test_zero_total(self):
        assert delay_rate(5, 0) is None


class TestContribution:
    def test_basic(self):
        assert contribution(30, 100) == pytest.approx(30.0)

    def test_zero_total(self):
        assert contribution(30, 0) is None


class TestBacklogChange:
    def test_basic(self):
        assert backlog_change(150, 100) == pytest.approx(50.0)

    def test_missing(self):
        assert backlog_change(None, 100) is None


class TestThroughput:
    def test_basic(self):
        assert throughput(1000, 30) == pytest.approx(33.33)

    def test_zero_days(self):
        assert throughput(100, 0) is None

    def test_missing(self):
        assert throughput(None, 30) is None


# -- statistics ------------------------------------------------------


class TestMean:
    def test_basic(self):
        assert mean([1, 2, 3]) == pytest.approx(2.0)

    def test_with_none(self):
        assert mean([1, None, 3]) == pytest.approx(2.0)

    def test_empty(self):
        assert mean([]) is None

    def test_with_nan(self):
        assert mean([1.0, float("nan"), 3.0]) == pytest.approx(2.0)


class TestWeightedMean:
    def test_basic(self):
        assert weighted_mean([10.0, 20.0], [1.0, 3.0]) == pytest.approx(17.5)

    def test_mismatch(self):
        with pytest.raises(ValueError):
            weighted_mean([1.0], [2.0, 3.0])

    def test_empty(self):
        assert weighted_mean([], []) is None

    def test_zero_weights(self):
        assert weighted_mean([10.0, 20.0], [0.0, 0.0]) is None


class TestZScores:
    def test_basic(self):
        scores = z_scores([10, 20, 30, 40, 50])
        assert scores is not None
        assert len(scores) == 5
        assert all(s is not None for s in scores)

    def test_single(self):
        assert z_scores([10]) == [None]

    def test_flat(self):
        assert z_scores([5, 5, 5]) == [None, None, None]


class TestRollingBaseline:
    def test_basic(self):
        values = [10, 20, 30, 40]
        baseline = rolling_baseline(values, window=2)
        assert baseline[0] is None  # no preceding values
        assert baseline[1] == pytest.approx(10.0)  # mean of [10]
        assert baseline[2] == pytest.approx(15.0)  # mean of [10, 20]
        assert baseline[3] == pytest.approx(25.0)  # mean of [20, 30]


class TestLinearTrend:
    def test_straight_line(self):
        result = linear_trend([10, 20, 30, 40])
        assert result is not None
        assert result["slope"] == pytest.approx(10.0)
        assert result["r_squared"] == pytest.approx(1.0)

    def test_flat(self):
        result = linear_trend([5, 5, 5])
        assert result is not None
        assert result["slope"] == pytest.approx(0.0)
        assert result["r_squared"] is None

    def test_insufficient(self):
        assert linear_trend([5]) is None


class TestNormaliseSeries:
    def test_basic(self):
        result = normalise_series([0, 50, 100])
        assert result == [0.0, 0.5, 1.0]

    def test_flat(self):
        assert normalise_series([5, 5, 5]) == [0.0, 0.0, 0.0]

    def test_empty(self):
        assert normalise_series([]) == []


class TestClamp:
    def test_within(self):
        assert clamp(5, 0, 10) == 5

    def test_below(self):
        assert clamp(-5, 0, 10) == 0

    def test_above(self):
        assert clamp(15, 0, 10) == 10


class TestClampNonNegative:
    def test_negative(self):
        assert clamp_non_negative(-5.0) == 0.0

    def test_positive(self):
        assert clamp_non_negative(5.0) == 5.0

    def test_none(self):
        assert clamp_non_negative(None) == 0.0


class TestClampPercentage:
    def test_valid(self):
        assert clamp_percentage(45.5) == pytest.approx(45.5)

    def test_below(self):
        assert clamp_percentage(-10.0) == pytest.approx(0.0)

    def test_above(self):
        assert clamp_percentage(150.0) == pytest.approx(100.0)

    def test_none(self):
        assert clamp_percentage(None) is None
