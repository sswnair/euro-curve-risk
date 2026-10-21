"""Dates: the curve's time axis, moving a date by years or months, and the TARGET calendar.

The ECB publishes a curve for every TARGET business day: every weekday except
New Year's Day, Good Friday, Easter Monday, May 1 and December 25 and 26.
"""

from __future__ import annotations

import datetime as dt

DAYS_IN_YEAR = 365.0


def year_fraction(start: dt.date, end: dt.date) -> float:
    """Actual days from start to end over 365. This is the time axis the curve is read on."""
    return (end - start).days / DAYS_IN_YEAR


def add_years(day: dt.date, years: int) -> dt.date:
    """The same day and month `years` later, or earlier when years is negative.

    February 29 becomes February 28 in a year that doesn't have one.
    """
    try:
        return day.replace(year=day.year + years)
    except ValueError:
        return day.replace(year=day.year + years, day=28)


def add_months(day: dt.date, months: int) -> dt.date:
    """The same day `months` later or earlier, pulled back to the month's last day when needed."""
    index = day.year * 12 + day.month - 1 + months
    year, month = divmod(index, 12)
    month += 1
    for d in (day.day, 30, 29, 28):
        try:
            return dt.date(year, month, d)
        except ValueError:
            continue
    raise ValueError(f"can't move {day} by {months} months")


def easter_sunday(year: int) -> dt.date:
    """Gregorian Easter (the anonymous algorithm in Meeus, Astronomical Algorithms)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return dt.date(year, month, day + 1)


def target_holidays(year: int) -> set[dt.date]:
    easter = easter_sunday(year)
    return {
        dt.date(year, 1, 1),
        easter - dt.timedelta(days=2),
        easter + dt.timedelta(days=1),
        dt.date(year, 5, 1),
        dt.date(year, 12, 25),
        dt.date(year, 12, 26),
    }


def is_target_day(day: dt.date) -> bool:
    return day.weekday() < 5 and day not in target_holidays(day.year)


def target_days(start: dt.date, end: dt.date) -> list[dt.date]:
    """TARGET business days from start to end, both included."""
    days, day = [], start
    while day <= end:
        if is_target_day(day):
            days.append(day)
        day += dt.timedelta(days=1)
    return days
