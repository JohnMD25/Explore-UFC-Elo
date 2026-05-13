"""Tests for data.cleaning.event_details_cleaner."""

from __future__ import annotations

import pandas as pd
import pytest

from data.cleaning.event_details_cleaner import (
    OUTPUT_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    clean_event_details,
)


def _make_raw(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=REQUIRED_INPUT_COLUMNS)


def test_happy_path() -> None:
    """Two real-world rows clean to expected schema and values."""
    raw = _make_raw([
        {
            "EVENT": "UFC 328: Chimaev vs. Strickland",
            "URL": "http://ufcstats.com/event-details/9eedac48b497de5a",
            "DATE": "May 09, 2026",
            "LOCATION": "Newark, New Jersey, USA",
        },
        {
            "EVENT": "UFC Fight Night: Edwards vs. Brady",
            "URL": "http://ufcstats.com/event-details/cc2ad11b1f9d818b",
            "DATE": "March 22, 2025",
            "LOCATION": "London, England, United Kingdom",
        },
    ])
    cleaned = clean_event_details(raw)

    assert list(cleaned.columns) == OUTPUT_COLUMNS
    assert len(cleaned) == 2
    assert cleaned.loc[0, "event_name"] == "UFC 328: Chimaev vs. Strickland"
    assert cleaned.loc[0, "event_date"] == "2026-05-09"
    assert cleaned.loc[1, "event_date"] == "2025-03-22"


def test_date_parsing_across_months() -> None:
    """ISO-8601 conversion works across multiple months."""
    raw = _make_raw([
        {"EVENT": f"E{i}", "URL": f"u{i}", "DATE": d, "LOCATION": "L"}
        for i, d in enumerate([
            "January 01, 2020",
            "July 04, 2021",
            "December 31, 2022",
            "September 14, 2023",
        ])
    ])
    cleaned = clean_event_details(raw)
    assert cleaned["event_date"].tolist() == [
        "2020-01-01", "2021-07-04", "2022-12-31", "2023-09-14",
    ]


def test_location_verbatim() -> None:
    """LOCATION preserved verbatim (no splitting; whitespace stripped only)."""
    raw = _make_raw([
        {"EVENT": "A", "URL": "u1", "DATE": "May 09, 2026",
         "LOCATION": "Newark, New Jersey, USA"},
        {"EVENT": "B", "URL": "u2", "DATE": "May 02, 2026",
         "LOCATION": "Perth, Western Australia, Australia"},
    ])
    cleaned = clean_event_details(raw)
    assert cleaned.loc[0, "event_location"] == "Newark, New Jersey, USA"
    assert cleaned.loc[1, "event_location"] == "Perth, Western Australia, Australia"


def test_url_verbatim() -> None:
    """URL preserved verbatim (no slug extraction)."""
    raw = _make_raw([
        {"EVENT": "A", "URL": "http://ufcstats.com/event-details/9eedac48b497de5a",
         "DATE": "May 09, 2026", "LOCATION": "L"},
    ])
    cleaned = clean_event_details(raw)
    assert (
        cleaned.loc[0, "event_url"]
        == "http://ufcstats.com/event-details/9eedac48b497de5a"
    )


def test_missing_column_raises() -> None:
    """Missing a required column fails loud."""
    raw = pd.DataFrame({
        "EVENT": ["A"], "URL": ["u"], "DATE": ["May 09, 2026"],
        # LOCATION missing
    })
    with pytest.raises(ValueError, match="missing required columns"):
        clean_event_details(raw)


def test_malformed_date_raises() -> None:
    """A date in the wrong format fails loud after parsing."""
    raw = _make_raw([
        # ISO format — won't match "%B %d, %Y"
        {"EVENT": "A", "URL": "u", "DATE": "2026-05-09", "LOCATION": "L"},
    ])
    with pytest.raises(ValueError, match="failed to parse"):
        clean_event_details(raw)


def test_duplicate_event_name_raises() -> None:
    """Two rows with the same EVENT name fail the primary-key assertion."""
    raw = _make_raw([
        {"EVENT": "UFC 1", "URL": "u1", "DATE": "May 09, 2026", "LOCATION": "L"},
        {"EVENT": "UFC 1", "URL": "u2", "DATE": "May 10, 2026", "LOCATION": "L"},
    ])
    with pytest.raises(ValueError, match="Duplicate event_name"):
        clean_event_details(raw)


def test_whitespace_stripped() -> None:
    """Leading/trailing whitespace in text columns is stripped."""
    raw = _make_raw([
        {"EVENT": "  UFC 1  ", "URL": "  http://x  ",
         "DATE": "May 09, 2026", "LOCATION": "  London  "},
    ])
    cleaned = clean_event_details(raw)
    assert cleaned.loc[0, "event_name"] == "UFC 1"
    assert cleaned.loc[0, "event_url"] == "http://x"
    assert cleaned.loc[0, "event_location"] == "London"