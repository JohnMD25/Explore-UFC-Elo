"""Clean ufc_event_details.csv: normalise event date + location, expose event_url.

Output schema (data/cleaned/ufc/event_details_cleaned.csv):
    event_name      str     - verbatim from upstream EVENT (primary key, unique)
    event_date      str     - ISO-8601 date string (YYYY-MM-DD)
    event_location  string  - verbatim from upstream LOCATION (nullable)
    event_url       str     - verbatim full URL to ufcstats.com event page

This cleaner is the dependency for fight_results_cleaner.py v0.2.0, which joins
on event_name to attach event_date to each fight row.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ufc_stats" / "ufc_event_details.csv"
CLEANED_PATH = PROJECT_ROOT / "data" / "cleaned" / "ufc" / "event_details_cleaned.csv"

DATE_FORMAT = "%B %d, %Y"  # e.g. "March 22, 2025"

REQUIRED_INPUT_COLUMNS = ["EVENT", "URL", "DATE", "LOCATION"]
OUTPUT_COLUMNS = ["event_name", "event_date", "event_location", "event_url"]


def _parse_event_date(date_str: object) -> pd.Timestamp:
    """Parse a single DATE cell. Returns NaT on failure (logged)."""
    if pd.isna(date_str):
        return pd.NaT
    try:
        return pd.to_datetime(date_str, format=DATE_FORMAT, errors="raise")
    except (ValueError, TypeError) as exc:
        logger.warning("Failed to parse event_date=%r: %s", date_str, exc)
        return pd.NaT


def clean_event_details(event_details: pd.DataFrame) -> pd.DataFrame:
    """Transform upstream event details into the cleaned canonical frame.

    Args:
        event_details: DataFrame with columns EVENT, URL, DATE, LOCATION
            (as exported by Greco1899/scrape_ufc_stats).

    Returns:
        DataFrame with columns [event_name, event_date, event_location, event_url].
        event_date is ISO-8601 string. event_name is unique. event_location
        preserves upstream NAs.

    Raises:
        ValueError: If a required input column is missing, if EVENT or URL
            contain NA values, if any date fails to parse, or if event_name
            is not unique.
    """
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in event_details.columns]
    if missing:
        raise ValueError(f"event_details is missing required columns: {missing}")

    # EVENT and URL are required non-null fields.
    if event_details["EVENT"].isna().any():
        raise ValueError("EVENT column contains NA values; event_name must be non-null")
    if event_details["URL"].isna().any():
        raise ValueError("URL column contains NA values; event_url must be non-null")

    out = pd.DataFrame()
    out["event_name"] = event_details["EVENT"].astype(str).str.strip()
    out["event_date"] = event_details["DATE"].map(_parse_event_date)
    out["event_location"] = event_details["LOCATION"].astype("string").str.strip()
    out["event_url"] = event_details["URL"].astype(str).str.strip()

    # Fail loudly on any unparseable date.
    nat_count = int(out["event_date"].isna().sum())
    if nat_count:
        bad_rows = event_details.loc[out["event_date"].isna(), "DATE"].tolist()
        raise ValueError(
            f"{nat_count} event_date values failed to parse with format "
            f"{DATE_FORMAT!r}: {bad_rows[:5]}{'...' if nat_count > 5 else ''}"
        )

    # Convert to ISO-8601 string after the NaT assertion.
    out["event_date"] = out["event_date"].dt.strftime("%Y-%m-%d")

    # Assert event_name uniqueness (primary key).
    dupes = out["event_name"].value_counts()
    dupes = dupes[dupes > 1]
    if not dupes.empty:
        raise ValueError(
            f"Duplicate event_name values found: {dupes.head(5).to_dict()}"
        )

    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def main() -> None:
    """CLI entry point: read raw, clean, write cleaned."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw input not found: {RAW_PATH}")

    logger.info("Reading %s", RAW_PATH)
    raw = pd.read_csv(RAW_PATH)
    logger.info("Read %d rows", len(raw))

    cleaned = clean_event_details(raw)

    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEANED_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(cleaned), CLEANED_PATH)


if __name__ == "__main__":
    main()