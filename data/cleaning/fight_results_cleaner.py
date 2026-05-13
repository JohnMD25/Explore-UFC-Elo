"""
fight_results_cleaner.py
=========================================================================
Clean ufc_fight_results.csv into the canonical per-fight frame consumed
by the Elo engine and analytics layer (v0.2.0).

v0.2.0 reshape:
- New canonical PK: fight_id = slugify(EVENT) + "__" + slugify(BOUT) + "__" + fight_seq
  where fight_seq is 1-indexed per (EVENT, BOUT) group; handles tournament-era
  same-night rematches (Sakuraba vs. Silveira at UFC Ultimate Japan).
- event_date joined from event_details_cleaned.csv (left-merge on EVENT)
- bout_order = reverse row-index within each event (main event = max)
- outcome enum:  {a_win, b_win, draw, nc}  (unknown -> nc)
- method enum:   {KO/TKO, Submission, Decision, DQ, Draw, NC}
                 (other -> Decision; draw outcome forces Draw)
- round -> round_no rename
- total_rounds = bout_length_minutes // 5
- bout_type derived from weight-class title_type

Inputs (all DataFrames, all required):
    fight_results    Raw ufc_fight_results.csv
    event_details    Cleaned event_details_cleaned.csv
    weightclass      Cleaned weightclass_cleaned.csv
    lookup           fighter_name_to_id lookup

Output: data/cleaned/ufc/fight_results_cleaned.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path so package imports work regardless of
# how this module is invoked (script, `python -m`, or pytest).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.cleaning.fighter_names_cleaner import (  # noqa: E402
    attach_bout_fighter_ids,
    load_name_lookup,
)
from stats_engine.utils import slugify  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT      = _PROJECT_ROOT
LOCAL_RAW         = PROJECT_ROOT / "data" / "raw" / "ufc_stats" / "ufc_fight_results.csv"
EVENT_DETAILS_CSV = PROJECT_ROOT / "data" / "cleaned" / "ufc" / "event_details_cleaned.csv"
WEIGHTCLASS_CSV   = PROJECT_ROOT / "data" / "cleaned" / "ufc" / "weightclass_cleaned.csv"
DEFAULT_OUT       = PROJECT_ROOT / "data" / "cleaned" / "ufc" / "fight_results_cleaned.csv"

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
REQUIRED_FIGHT_RESULTS_COLUMNS = [
    "EVENT", "BOUT", "OUTCOME", "METHOD", "ROUND",
    "TIME", "TIME FORMAT", "REFEREE", "DETAILS",
]
REQUIRED_EVENT_DETAILS_COLUMNS = ["event_name", "event_date"]
REQUIRED_WEIGHTCLASS_COLUMNS = [
    "EVENT", "BOUT", "weight_class", "title_type", "is_womens", "is_tournament",
]

OUTPUT_COLUMNS = [
    # Canonical (v0.2.0)
    "fight_id", "fight_seq", "event_date", "bout_order",
    "fighter_a_id", "fighter_b_id",
    "outcome", "method",
    "round_no", "total_rounds",
    "weight_class", "bout_type",
    # Trace (preserved)
    "EVENT", "BOUT",
    "fighter_a", "fighter_b",
    "winner_id", "loser_id",
    "method_bucket", "method_detail",
    "time_seconds", "bout_length_minutes",
    "referee", "details",
]

# ---------------------------------------------------------------------------
# Method bucketing (coarse, 6 buckets). First matching pattern wins.
# ---------------------------------------------------------------------------
METHOD_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"doctor",                          "ko_tko"),
    (r"could not continue",              "ko_tko"),
    (r"^ko\b",                           "ko_tko"),
    (r"^tko\b",                          "ko_tko"),
    (r"^ko/tko\b",                       "ko_tko"),
    (r"^submission",                     "submission"),
    (r"^decision",                       "decision"),
    (r"^dq\b|disqualification",          "dq"),
    (r"^nc\b|no contest|overturn",       "no_contest"),
)

METHOD_BUCKET_TO_CANONICAL: dict[str, str] = {
    "ko_tko":     "KO/TKO",
    "submission": "Submission",
    "decision":   "Decision",
    "dq":         "DQ",
    "no_contest": "NC",
    "other":      "Decision",   # locked fallback
}

# Maps lowercase title_type from weightclass_cleaned.csv to canonical v0.2
# bout_type. Real upstream values observed in the cleaned CSV:
#   "regular" (8,209), "title" (450), "interim" (29).
# We lowercase + strip the input before lookup, and include defensive entries
# for capitalised / underscored variants in case upstream changes shape.
TITLE_TYPE_TO_BOUT_TYPE: dict[str, str] = {
    "title":         "title",
    "interim":       "interim_title",
    "interim title": "interim_title",
    "interim_title": "interim_title",
    "regular":       "non_title",
}

# ---------------------------------------------------------------------------
# Low-level parsers (preserved from v0.1)
# ---------------------------------------------------------------------------
def bucket_method(raw: object) -> str:
    if pd.isna(raw):
        return "other"
    s = str(raw).strip().lower()
    for pattern, bucket in METHOD_PATTERNS:
        if re.search(pattern, s):
            return bucket
    return "other"


_TIME_RE = re.compile(r"^(\d+):(\d{2})$")
def parse_time_to_seconds(t: object) -> object:
    if pd.isna(t):
        return pd.NA
    m = _TIME_RE.match(str(t).strip())
    if not m:
        return pd.NA
    return int(m.group(1)) * 60 + int(m.group(2))


_PAREN_RE = re.compile(r"\(([^)]+)\)")
def parse_bout_length_minutes(tf: object) -> object:
    if pd.isna(tf):
        return pd.NA
    s = str(tf).strip()
    m = _PAREN_RE.search(s)
    if not m:
        return pd.NA
    try:
        return sum(int(x.strip()) for x in m.group(1).split("-"))
    except (ValueError, AttributeError):
        return pd.NA


def derive_result(outcome: object) -> str:
    if pd.isna(outcome):
        return "unknown"
    s = str(outcome).strip().upper()
    if s in ("W/L", "L/W"):
        return "win"
    if s == "D/D":
        return "draw"
    if s.startswith("NC"):
        return "nc"
    return "unknown"


def derive_winner_loser(
    outcome: object, a_id: object, b_id: object
) -> tuple[object, object]:
    if pd.isna(outcome):
        return (pd.NA, pd.NA)
    s = str(outcome).strip().upper()
    if s == "W/L":
        return (a_id, b_id)
    if s == "L/W":
        return (b_id, a_id)
    return (pd.NA, pd.NA)


# ---------------------------------------------------------------------------
# v0.2.0 canonical derivations
# ---------------------------------------------------------------------------
def _build_fight_id(event: object, bout: object) -> str:
    return f"{slugify(event)}__{slugify(bout)}"


def _map_outcome(outcome_raw: object) -> str:
    """Map raw OUTCOME -> v0.2 outcome enum. Unknown/garbage coerce to nc."""
    if pd.isna(outcome_raw):
        return "nc"
    s = str(outcome_raw).strip().upper()
    if s == "W/L":
        return "a_win"
    if s == "L/W":
        return "b_win"
    if s == "D/D":
        return "draw"
    if s.startswith("NC"):
        return "nc"
    return "nc"


def _map_method(method_bucket: str, outcome: str) -> str:
    if outcome == "draw":
        return "Draw"
    return METHOD_BUCKET_TO_CANONICAL[method_bucket]


def _map_bout_type(title_type: object) -> str:
    """Map weightclass.title_type to canonical bout_type enum. Case-insensitive."""
    if pd.isna(title_type):
        return "non_title"
    key = str(title_type).strip().lower()
    return TITLE_TYPE_TO_BOUT_TYPE.get(key, "non_title")


def _assert_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Main cleaning pipeline (v0.2.0)
# ---------------------------------------------------------------------------
def clean_fight_results(
    fight_results: pd.DataFrame,
    event_details: pd.DataFrame,
    weightclass: pd.DataFrame,
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    """v0.2.0 fight-results cleaning pipeline. All four inputs required."""
    _assert_columns(fight_results, REQUIRED_FIGHT_RESULTS_COLUMNS, "fight_results")
    _assert_columns(event_details, REQUIRED_EVENT_DETAILS_COLUMNS, "event_details")
    _assert_columns(weightclass, REQUIRED_WEIGHTCLASS_COLUMNS, "weightclass")

    df = fight_results.copy()

    # 0. Defensive whitespace strip on merge keys. Upstream ufc_fight_results.csv
    #    ships EVENT/BOUT values with trailing whitespace that don't match the
    #    stripped values in event_details_cleaned.csv / weightclass_cleaned.csv,
    #    silently breaking exact-match joins for every row. Strip on both sides.
    df["EVENT"] = df["EVENT"].astype(str).str.strip()
    df["BOUT"]  = df["BOUT"].astype(str).str.strip()
    event_details = event_details.copy()
    event_details["event_name"] = event_details["event_name"].astype(str).str.strip()
    weightclass = weightclass.copy()
    weightclass["EVENT"] = weightclass["EVENT"].astype(str).str.strip()
    weightclass["BOUT"]  = weightclass["BOUT"].astype(str).str.strip()

    # 1. Attach fighter_a, fighter_b, fighter_a_id, fighter_b_id
    df = attach_bout_fighter_ids(df, bout_col="BOUT", lookup=lookup)

    # 2. v0.1 trace: result + winner/loser ids
    df["result"] = df["OUTCOME"].apply(derive_result)
    wl = df.apply(
        lambda r: derive_winner_loser(r["OUTCOME"], r["fighter_a_id"], r["fighter_b_id"]),
        axis=1, result_type="expand",
    )
    df["winner_id"] = wl[0].astype("Int64")
    df["loser_id"]  = wl[1].astype("Int64")

    # 3. Method bucket + detail (trace)
    df["method_bucket"] = df["METHOD"].apply(bucket_method)
    df["method_detail"] = df["METHOD"]

    # 4. Round / time / bout length
    df["round_no"]            = pd.to_numeric(df["ROUND"], errors="coerce").astype("Int64")
    df["time_seconds"]        = df["TIME"].apply(parse_time_to_seconds).astype("Int64")
    df["bout_length_minutes"] = df["TIME FORMAT"].apply(parse_bout_length_minutes).astype("Int64")

    # 5. Passthrough trace
    df["referee"] = df["REFEREE"]
    df["details"] = df["DETAILS"]

    # 6. Merge weight-class on (EVENT, BOUT)
    wc = weightclass[REQUIRED_WEIGHTCLASS_COLUMNS].drop_duplicates(["EVENT", "BOUT"])
    df = df.merge(wc, on=["EVENT", "BOUT"], how="left")

    # 7. Merge event_date on EVENT == event_name
    ed = event_details[REQUIRED_EVENT_DETAILS_COLUMNS].drop_duplicates("event_name")
    df = df.merge(ed, left_on="EVENT", right_on="event_name", how="left")
    missing_dates = int(df["event_date"].isna().sum())
    if missing_dates:
        bad = df.loc[df["event_date"].isna(), "EVENT"].unique().tolist()
        raise ValueError(
            f"{missing_dates} rows have no event_date after merge; "
            f"missing events in event_details_cleaned.csv: {bad[:5]}"
            f"{'...' if len(bad) > 5 else ''}"
        )
    df = df.drop(columns=["event_name"])

    # 8. v0.2.0 canonical derivations.
    #    fight_id = slug(EVENT)__slug(BOUT)__fight_seq, where fight_seq is
    #    1-indexed per (EVENT, BOUT) group. Locked decision (see Decision Log):
    #    ALWAYS append __N for consistency, even when N=1. Surfaced by
    #    Sakuraba vs. Silveira at UFC Ultimate Japan (same-night rematch in
    #    the 8-man tournament).
    fight_id_base   = df.apply(lambda r: _build_fight_id(r["EVENT"], r["BOUT"]), axis=1)
    df["fight_seq"] = (df.groupby(["EVENT", "BOUT"]).cumcount() + 1).astype("Int64")
    df["fight_id"]  = fight_id_base + "__" + df["fight_seq"].astype(str)
    df["outcome"]   = df["OUTCOME"].apply(_map_outcome)
    df["method"]    = df.apply(lambda r: _map_method(r["method_bucket"], r["outcome"]), axis=1)
    df["bout_type"] = df["title_type"].apply(_map_bout_type)

    # 9. total_rounds = bout_length_minutes // 5
    df["total_rounds"] = (df["bout_length_minutes"] // 5).astype("Int64")

    # 10. bout_order: reverse row-index within each EVENT group, 1-indexed.
    #     Upstream lists main event first; we want main event = max bout_order.
    df["bout_order"] = (
        df.groupby("EVENT").cumcount(ascending=False) + 1
    ).astype("Int64")

    # 11. fight_id uniqueness sanity check. Structural uniqueness is
    #     guaranteed by the (EVENT, BOUT, fight_seq) construction; this
    #     defensive assertion should never fire.
    dupes = df["fight_id"].value_counts()
    dupes = dupes[dupes > 1]
    if not dupes.empty:
        raise ValueError(
            f"Duplicate fight_id values found despite sequence suffix "
            f"({len(dupes)} keys): {dupes.head(5).to_dict()}"
        )

    # 12. Final column selection + order
    return df[OUTPUT_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Clean ufc_fight_results.csv (v0.2.0)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    for path, label in (
        (LOCAL_RAW,         "ufc_fight_results.csv"),
        (EVENT_DETAILS_CSV, "event_details_cleaned.csv"),
        (WEIGHTCLASS_CSV,   "weightclass_cleaned.csv"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path} ({label})")

    logger.info("Reading raw fight_results from %s", LOCAL_RAW)
    fr = pd.read_csv(LOCAL_RAW)
    logger.info("Reading event_details from %s", EVENT_DETAILS_CSV)
    ed = pd.read_csv(EVENT_DETAILS_CSV)
    logger.info("Reading weightclass from %s", WEIGHTCLASS_CSV)
    wc = pd.read_csv(WEIGHTCLASS_CSV)
    logger.info("Loading fighter_name_to_id lookup")
    lookup = load_name_lookup()

    logger.info("Cleaning %d raw rows...", len(fr))
    cleaned = clean_fight_results(fr, ed, wc, lookup)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.out, index=False)
    logger.info("Wrote %d rows to %s", len(cleaned), args.out)

    logger.info("outcome distribution:\n%s",  cleaned["outcome"].value_counts().to_string())
    logger.info("method distribution:\n%s",   cleaned["method"].value_counts().to_string())
    logger.info("bout_type distribution:\n%s", cleaned["bout_type"].value_counts().to_string())


if __name__ == "__main__":
    main()