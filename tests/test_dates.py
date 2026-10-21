"""Year fractions, moving dates and the TARGET calendar."""

import datetime as dt

import pytest

from eurocurve.dates import add_months, add_years, easter_sunday, is_target_day, target_days, target_holidays, year_fraction


def test_a_year_of_365_days_is_one():
    assert year_fraction(dt.date(2026, 9, 30), dt.date(2027, 9, 30)) == 1.0


def test_a_year_with_february_29_is_a_little_longer():
    assert year_fraction(dt.date(2027, 9, 30), dt.date(2028, 9, 30)) == pytest.approx(366 / 365)


def test_year_fraction_is_negative_backwards():
    assert year_fraction(dt.date(2026, 9, 30), dt.date(2026, 9, 1)) == pytest.approx(-29 / 365)


def test_add_years_keeps_day_and_month():
    assert add_years(dt.date(2035, 8, 15), -9) == dt.date(2026, 8, 15)


def test_add_years_from_february_29():
    assert add_years(dt.date(2028, 2, 29), -1) == dt.date(2027, 2, 28)
    assert add_years(dt.date(2028, 2, 29), 4) == dt.date(2032, 2, 29)


def test_add_months_to_a_shorter_month():
    assert add_months(dt.date(2026, 3, 31), -1) == dt.date(2026, 2, 28)
    assert add_months(dt.date(2027, 12, 31), 2) == dt.date(2028, 2, 29)


def test_add_months_across_a_year_end():
    assert add_months(dt.date(2026, 1, 15), -3) == dt.date(2025, 10, 15)


@pytest.mark.parametrize("year, day", [(2025, dt.date(2025, 4, 20)), (2026, dt.date(2026, 4, 5)), (2027, dt.date(2027, 3, 28))])
def test_easter_sunday(year, day):
    assert easter_sunday(year) == day


def test_target_holidays_in_2026():
    assert target_holidays(2026) == {
        dt.date(2026, 1, 1),
        dt.date(2026, 4, 3),
        dt.date(2026, 4, 6),
        dt.date(2026, 5, 1),
        dt.date(2026, 12, 25),
        dt.date(2026, 12, 26),
    }


def test_weekends_and_holidays_are_not_target_days():
    assert not is_target_day(dt.date(2026, 10, 3))  # Saturday
    assert not is_target_day(dt.date(2026, 4, 3))  # Good Friday
    assert is_target_day(dt.date(2026, 4, 7))


def test_target_days_over_easter():
    assert target_days(dt.date(2026, 4, 2), dt.date(2026, 4, 8)) == [dt.date(2026, 4, 2), dt.date(2026, 4, 7), dt.date(2026, 4, 8)]
