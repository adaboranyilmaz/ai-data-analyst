"""The Czech bank data's original birth number, and the rule that encodes sex in it.

The PKDD'99 data stored a client's date of birth and sex in one six-digit number: YYMMDD for
men, and YYMM+50DD for women (50 added to the month). BIRD's copy of the data has already
decoded it into `client.gender` and `client.birth_date`; these functions state the rule so it
can be checked against those columns, and applied to data that still carries the number.
Two-digit years are 19YY: every client in the data was born between 1911 and 1987.
"""

from __future__ import annotations

import datetime as dt

WOMEN_MONTH_OFFSET = 50


def decode_birth_number(number: int | str) -> tuple[str, dt.date]:
    """(sex, date of birth) from a birth number: "F" or "M", and the date."""
    digits = f"{int(number):06d}"
    if len(digits) != 6:
        raise ValueError(f"a birth number has six digits: {number!r}")
    yy, mm, dd = int(digits[:2]), int(digits[2:4]), int(digits[4:])
    sex = "F" if mm > WOMEN_MONTH_OFFSET else "M"
    if sex == "F":
        mm -= WOMEN_MONTH_OFFSET
    return sex, dt.date(1900 + yy, mm, dd)  # raises ValueError for an impossible date


def encode_birth_number(sex: str, birth_date: dt.date) -> str:
    if sex not in ("F", "M"):
        raise ValueError(f"sex is F or M: {sex!r}")
    if not 1900 <= birth_date.year <= 1999:
        raise ValueError(f"a birth number holds only 19YY years: {birth_date}")
    month = birth_date.month + (WOMEN_MONTH_OFFSET if sex == "F" else 0)
    return f"{birth_date.year % 100:02d}{month:02d}{birth_date.day:02d}"
