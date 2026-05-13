"""
fight_results_cleaner.py
=========================================================================
Clean ufc_fight_results.csv into a per-fight frame ready for Elo and
analytics. Combines:
    - Raw fight_results.csv      (BOUT, OUTCOME, METHOD, ROUND, TIME, ...)
    - weightclass_cleaned.csv    (weight_class, title_type, is_womens, is_tournament)
    - fighter_name_to_id.csv     (via fighter_names_cleaner.attach_bout_fighter_ids)

Output: data/cleaned/ufc/fight_results_cleaned.csv

Output schema (one row per fight):
    EVENT, BOUT                                          -- trace
    fighter_a, fighter_b                                 -- name strings (left/right of " vs. ")
    fighter_a_id, fighter_b_id                           -- Int64 (nullable)
    result                                               -- "win" | "draw" | "nc" | "unknown"
    winner_id, loser_id                                  -- Int64 (NA for draw/nc)
    method_bucket                                        -- ko_tko | submission | decision | dq | no_contest | other
    method_detail                                        -- raw METHOD string (trace)
    round                                                -- Int64 (1-5)
    time_seconds                                         -- Int64
    bout_length_minutes                                  -- Int64 (scheduled total)
    weight_class, title_type, is_womens, is_tournament   -- from weight_class merge
    referee                                              -- str
    details                                              -- str (judge scorecards etc.)

Usage (run from project root):
    python data/fight_results_cleaner.py --source local
    python data/fight_results_cleaner.py --source url
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Literal

import pandas as pd

# Sibling-module import. Python adds the script's directory to sys.path[0]
# when run as `python data/fight_results_cleaner.py`, so this resolves.
from fighter_names_cleaner import (
    attach_bout_fighter_ids,
    load_name_lookup,
    report_unmatched,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"
    "ufc_fight_results.csv"
)

PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent
LOCAL_RAW       = PROJECT_ROOT / "data" / "raw" / "ufc_stats" / "ufc_fight_results.csv"
WEIGHTCLASS_CSV = PROJECT_ROOT / "data" / "cleaned" / "ufc" / "weightclass_cleaned.csv"
DEFAULT_OUT     = PROJECT_ROOT / "data" / "cleaned" / "ufc" / "fight_results_cleaned.csv"

# Method bucketing (coarse, 6 buckets). First matching pattern wins.
METHOD_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"doctor",                          "ko_tko"),       # "TKO - Doctor's Stoppage"
    (r"could not continue",              "ko_tko"),       # corner stoppage between rounds
    (r"^ko\b",                           "ko_tko"),       # "KO"
    (r"^tko\b",                          "ko_tko"),       # "TKO" / "TKO - ..."
    (r"^ko/tko\b",                       "ko_tko"),       # "KO/TKO"
    (r"^submission",                     "submission"),
    (r"^decision",                       "decision"),     # "Decision - Unanimous" etc.
    (r"^dq\b|disqualification",          "dq"),
    (r"^nc\b|no contest|overturn",       "no_contest"),
)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def bucket_method(raw: object) -> str:
    """Map a raw METHOD string to one of 6 coarse buckets."""
    if pd.isna(raw):
        return "other"
    s = str(raw).strip().lower()
    for pattern, bucket in METHOD_PATTERNS:
        if re.search(pattern, s):
            return bucket
    return "other"


_TIME_RE = re.compile(r"^(\d+):(\d{2})$")
def parse_time_to_seconds(t: object) -> object:
    """`4:47` -> 287. Returns pd.NA if unparseable."""
    if pd.isna(t):
        return pd.NA
    m = _TIME_RE.match(str(t).strip())
    if not m:
        return pd.NA
    return int(m.group(1)) * 60 + int(m.group(2))


_PAREN_RE = re.compile(r"\(([^)]+)\)")
def parse_bout_length_minutes(tf: object) -> object:
    """
    `"3 Rnd (5-5-5)"`     -> 15
    `"5 Rnd (5-5-5-5-5)"` -> 25
    `"No Time Limit"`     -> pd.NA
    """
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
    """Map raw OUTCOME (W/L, L/W, D/D, NC/NC) to win | draw | nc | unknown."""
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
    """Return (winner_id, loser_id) given OUTCOME and the two fighter ids."""
    if pd.isna(outcome):
        return (pd.NA, pd.NA)
    s = str(outcome).strip().upper()
    if s == "W/L":
        return (a_id, b_id)
    if s == "L/W":
        return (b_id, a_id)
    return (pd.NA, pd.NA)


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------
def clean_fight_results(
    fight_results: pd.DataFrame,
    weightclass: pd.DataFrame | None = None,
    lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Full fight-results cleaning pipeline."""
    df = fight_results.copy()

    # 1. Attach fighter_a, fighter_b, fighter_a_id, fighter_b_id
    df = attach_bout_fighter_ids(df, bout_col="BOUT", lookup=lookup)

    # 2. Derive result + winner/loser ids
    df["result"] = df["OUTCOME"].apply(derive_result)
    winner_loser = df.apply(
        lambda r: derive_winner_loser(
            r["OUTCOME"], r["fighter_a_id"], r["fighter_b_id"]
        ),
        axis=1, result_type="expand",
    )
    df["winner_id"] = winner_loser[0].astype("Int64")
    df["loser_id"]  = winner_loser[1].astype("Int64")

    # 3. Bucket method
    df["method_bucket"] = df["METHOD"].apply(bucket_method)
    df["method_detail"] = df["METHOD"]

    # 4. Parse round / time / bout length
    df["round"]               = pd.to_numeric(df["ROUND"], errors="coerce").astype("Int64")
    df["time_seconds"]        = df["TIME"].apply(parse_time_to_seconds).astype("Int64")
    df["bout_length_minutes"] = df["TIME FORMAT"].apply(parse_bout_length_minutes).astype("Int64")

    # 5. Passthrough columns
    df["referee"] = df["REFEREE"]
    df["details"] = df["DETAILS"]

    # 6. Merge weight-class parsed cols on (EVENT, BOUT)
    if weightclass is not None:
        wc_cols = ["EVENT", "BOUT", "weight_class", "title_type", "is_womens", "is_tournament"]
        wc = weightclass[wc_cols].drop_duplicates(["EVENT", "BOUT"])
        df = df.merge(wc, on=["EVENT", "BOUT"], how="left")

    # 7. Select + order output columns
    keep_cols = [
        "EVENT", "BOUT",
        "fighter_a", "fighter_b",
        "fighter_a_id", "fighter_b_id",
        "result", "winner_id", "loser_id",
        "method_bucket", "method_detail",
        "round", "time_seconds", "bout_length_minutes",
    ]
    if weightclass is not None:
        keep_cols += ["weight_class", "title_type", "is_womens", "is_tournament"]
    keep_cols += ["referee", "details"]
    return df[keep_cols]


# ---------------------------------------------------------------------------
# Standalone CLI for inspection
# ---------------------------------------------------------------------------
def _load_raw(source: Literal["url", "local"]) -> pd.DataFrame:
    if source == "url":
        return pd.read_csv(UPSTREAM_URL)
    if not LOCAL_RAW.exists():
        raise FileNotFoundError(
            f"Local raw file missing: {LOCAL_RAW}\nUse --source url, or curl it."
        )
    return pd.read_csv(LOCAL_RAW)


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean ufc_fight_results.csv")
    ap.add_argument("--source", choices=["url", "local"], default="url")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading inputs from {args.source}...")
    fr = _load_raw(args.source)
    print(f"      ufc_fight_results:   {len(fr):,} rows")

    wc = None
    if WEIGHTCLASS_CSV.exists():
        wc = pd.read_csv(WEIGHTCLASS_CSV)
        print(f"      weightclass_cleaned: {len(wc):,} rows")
    else:
        print(f"      WARNING: {WEIGHTCLASS_CSV} not found.")
        print("      Weight-class cols will be omitted from output.")
        print("      Run `python data/weight_class_cleaner.py --source local` first.")

    lookup = load_name_lookup()
    print(f"      fighter_name_to_id:  {len(lookup):,} entries")

    print("[2/4] Cleaning...")
    cleaned = clean_fight_results(fr, weightclass=wc, lookup=lookup)

    print("[3/4] Sanity checks:")
    report_unmatched(cleaned, ["fighter_a_id", "fighter_b_id"], label="ids")

    print("      result distribution:")
    for v, n in cleaned["result"].value_counts(dropna=False).items():
        print(f"        {n:5,}  {v}")

    print("      method_bucket distribution:")
    for v, n in cleaned["method_bucket"].value_counts(dropna=False).items():
        print(f"        {n:5,}  {v}")

    print("      bout_length_minutes distribution:")
    for v, n in cleaned["bout_length_minutes"].value_counts(dropna=False).head(10).items():
        print(f"        {n:5,}  {v}")

    if wc is not None:
        wc_missing = int(cleaned["weight_class"].isna().sum())
        print(f"      weight_class merge: {wc_missing:,} rows missing weight_class")

    print(f"[4/4] Writing -> {args.out}")
    cleaned.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()