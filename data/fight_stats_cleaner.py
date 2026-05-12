"""
fight_stats_cleaner.py
=========================================================================
Clean ufc_fight_stats.csv into a per-fighter-per-round-per-fight frame
ready for Cardio Score, output-rate metrics, and any other round-grained
analytics. Combines:
    - Raw ufc_fight_stats.csv    (EVENT, BOUT, ROUND, FIGHTER, KD, ...)
    - fighter_name_to_id.csv     (via fighter_names_cleaner.attach_fighter_ids)

Output: data/cleaned/ufc/fight_stats_cleaned.csv

Grain: one row per fighter per round per fight (matches the source).
       Two rows per round per bout (one for each fighter).

Output schema:
    EVENT, BOUT                                  -- trace + join keys to fight_results
    round_no                                     -- Int64 (1-5)
    fighter                                      -- str (upstream canonical name)
    fighter_id                                   -- Int64 (nullable)
    knockdowns                                   -- Int64
    sig_str_landed,   sig_str_attempted          -- Int64
    total_str_landed, total_str_attempted        -- Int64
    td_landed,        td_attempted               -- Int64
    sub_attempts                                 -- Int64
    reversals                                    -- Int64
    control_time_seconds                         -- Int64
    head_landed,     head_attempted              -- Int64  (target breakdown)
    body_landed,     body_attempted              -- Int64
    leg_landed,      leg_attempted               -- Int64
    distance_landed, distance_attempted          -- Int64  (position breakdown)
    clinch_landed,   clinch_attempted            -- Int64
    ground_landed,   ground_attempted            -- Int64

Usage (run from project root):
    python data/fight_stats_cleaner.py --source local
    python data/fight_stats_cleaner.py --source url
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Literal

import pandas as pd

# Sibling-module import. Python adds the script's directory to sys.path[0]
# when run as `python data/fight_stats_cleaner.py`, so this resolves.
from fighter_names_cleaner import (
    attach_fighter_ids,
    load_name_lookup,
    report_unmatched,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"
    "ufc_fight_stats.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_RAW    = PROJECT_ROOT / "data" / "raw" / "ufc_stats" / "ufc_fight_stats.csv"
DEFAULT_OUT  = PROJECT_ROOT / "data" / "cleaned" / "ufc" / "fight_stats_cleaned.csv"

# (raw_column, landed_name, attempted_name) for all nine "X of Y" pairs.
# Order is the canonical output column order for these pairs.
X_OF_Y_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("SIG.STR.",   "sig_str_landed",   "sig_str_attempted"),
    ("TOTAL STR.", "total_str_landed", "total_str_attempted"),
    ("TD",         "td_landed",        "td_attempted"),
    ("HEAD",       "head_landed",      "head_attempted"),
    ("BODY",       "body_landed",      "body_attempted"),
    ("LEG",        "leg_landed",       "leg_attempted"),
    ("DISTANCE",   "distance_landed",  "distance_attempted"),
    ("CLINCH",     "clinch_landed",    "clinch_attempted"),
    ("GROUND",     "ground_landed",    "ground_attempted"),
)


# ---------------------------------------------------------------------------
# Parsers (scalar; mirror the house style from fight_results_cleaner.py)
# ---------------------------------------------------------------------------
_X_OF_Y_RE = re.compile(r"^\s*(\d+)\s+of\s+(\d+)\s*$")
def parse_x_of_y(raw: object) -> tuple[object, object]:
    """`"17 of 23"` -> (17, 23). Returns (pd.NA, pd.NA) if unparseable."""
    if pd.isna(raw):
        return (pd.NA, pd.NA)
    m = _X_OF_Y_RE.match(str(raw))
    if not m:
        return (pd.NA, pd.NA)
    return (int(m.group(1)), int(m.group(2)))


_MMSS_RE = re.compile(r"^\s*(\d+):(\d{2})\s*$")
def parse_mmss_to_seconds(t: object) -> object:
    """`"4:47"` -> 287. Returns pd.NA if unparseable."""
    if pd.isna(t):
        return pd.NA
    m = _MMSS_RE.match(str(t))
    if not m:
        return pd.NA
    return int(m.group(1)) * 60 + int(m.group(2))


_ROUND_RE = re.compile(r"^\s*Round\s+(\d+)\s*$", re.IGNORECASE)
def parse_round(r: object) -> object:
    """`"Round 1"` -> 1. Returns pd.NA if unparseable."""
    if pd.isna(r):
        return pd.NA
    m = _ROUND_RE.match(str(r))
    if not m:
        return pd.NA
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------
def clean_fight_stats(
    fight_stats: pd.DataFrame,
    lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Full fight-stats cleaning pipeline."""
    df = fight_stats.copy()

    # 1. Rename single-token raw columns to snake_case keepers.
    df = df.rename(columns={
        "FIGHTER": "fighter",
        "KD":      "knockdowns",
        "SUB.ATT": "sub_attempts",
        "REV.":    "reversals",
    })

    # 2. Parse round_no and control time.
    df["round_no"]             = df["ROUND"].apply(parse_round).astype("Int64")
    df["control_time_seconds"] = df["CTRL"].apply(parse_mmss_to_seconds).astype("Int64")

    # 3. Attach fighter_id via the canonical name lookup.
    df = attach_fighter_ids(df, "fighter", lookup=lookup, id_col="fighter_id")

    # 4. Split every "X of Y" column into a landed/attempted Int64 pair.
    for raw_col, landed_col, attempted_col in X_OF_Y_COLUMNS:
        parsed = pd.DataFrame(
            df[raw_col].map(parse_x_of_y).tolist(),
            index=df.index,
            columns=[landed_col, attempted_col],
        )
        df[landed_col]    = parsed[landed_col].astype("Int64")
        df[attempted_col] = parsed[attempted_col].astype("Int64")

    # 5. Cast scalar-int keepers.
    for col in ("knockdowns", "sub_attempts", "reversals"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # 6. Select + order output columns.
    keep_cols = [
        "EVENT", "BOUT",
        "round_no",
        "fighter", "fighter_id",
        "knockdowns",
        "sig_str_landed",   "sig_str_attempted",
        "total_str_landed", "total_str_attempted",
        "td_landed",        "td_attempted",
        "sub_attempts",
        "reversals",
        "control_time_seconds",
        "head_landed",     "head_attempted",
        "body_landed",     "body_attempted",
        "leg_landed",      "leg_attempted",
        "distance_landed", "distance_attempted",
        "clinch_landed",   "clinch_attempted",
        "ground_landed",   "ground_attempted",
    ]
    return df[keep_cols]


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def _check_breakdown_sums(df: pd.DataFrame) -> None:
    """
    Print mismatch counts for the two independent decompositions of
    significant strikes:
        head + body + leg            should == sig_str   (target breakdown)
        distance + clinch + ground   should == sig_str   (position breakdown)
    Plus a check that sig_str_landed never exceeds total_str_landed.
    All NAs are excluded from mismatch counts (Int64 NA-propagation).
    """
    def _mismatch(a: pd.Series, b: pd.Series, c: pd.Series, ref: pd.Series) -> int:
        diff = (a + b + c) - ref
        return int(diff.dropna().ne(0).sum())

    print("      breakdown sanity checks:")
    print(f"        target   landed   mismatch: "
          f"{_mismatch(df['head_landed'],   df['body_landed'],   df['leg_landed'],    df['sig_str_landed']):,}")
    print(f"        target   attempt  mismatch: "
          f"{_mismatch(df['head_attempted'],df['body_attempted'],df['leg_attempted'], df['sig_str_attempted']):,}")
    print(f"        position landed   mismatch: "
          f"{_mismatch(df['distance_landed'],   df['clinch_landed'],   df['ground_landed'],    df['sig_str_landed']):,}")
    print(f"        position attempt  mismatch: "
          f"{_mismatch(df['distance_attempted'],df['clinch_attempted'],df['ground_attempted'], df['sig_str_attempted']):,}")

    over = int(
        (df["sig_str_landed"] > df["total_str_landed"]).dropna().sum()
    )
    print(f"        sig_str_landed > total_str_landed: {over:,}")


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
    ap = argparse.ArgumentParser(description="Clean ufc_fight_stats.csv")
    ap.add_argument("--source", choices=["url", "local"], default="url")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading inputs from {args.source}...")
    fs = _load_raw(args.source)
    print(f"      ufc_fight_stats:    {len(fs):,} rows")

    lookup = load_name_lookup()
    print(f"      fighter_name_to_id: {len(lookup):,} entries")

    print("[2/4] Cleaning...")
    cleaned = clean_fight_stats(fs, lookup=lookup)

    print("[3/4] Sanity checks:")
    report_unmatched(cleaned, ["fighter_id"], label="ids")

    print("      round_no distribution:")
    for v, n in cleaned["round_no"].value_counts(dropna=False).sort_index().items():
        print(f"        {n:5,}  Round {v}")

    _check_breakdown_sums(cleaned)

    print(f"[4/4] Writing -> {args.out}")
    cleaned.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()