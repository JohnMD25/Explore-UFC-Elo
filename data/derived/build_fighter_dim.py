"""
build_fighter_dim.py
=========================================================================
Builds the canonical fighter dimension for the UFC Data Explorer.

Inputs (raw upstream CSVs from Greco1899/scrape_ufc_stats):
    - ufc_fighter_details.csv  (FIRST, LAST, NICKNAME, URL)
    - ufc_fighter_tott.csv     (FIGHTER, HEIGHT, WEIGHT, REACH, STANCE, DOB, URL)

Optional inputs (used to pre-resolve fight-data names -> fighter_id):
    - ufc_fight_results.csv    (BOUT)
    - ufc_fight_stats.csv      (FIGHTER)

Outputs (written to data/derived/):
    - fighter_dim.csv            canonical dim, one row per fighter
    - fighter_name_to_id.csv     every name string seen -> fighter_id
                                 (exact + fuzzy-accepted, used by cleaners)
    - fighter_match_review.csv   borderline RapidFuzz matches for review

Usage (run from project root):
    python data/build_fighter_dim.py                       # fetch from URLs
    python data/build_fighter_dim.py --source local        # read data/raw/ufc_stats/
    python data/build_fighter_dim.py --no-name-resolve     # skip step 3
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UPSTREAM_BASE = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main"
RAW_FILES = {
    "details":       "ufc_fighter_details.csv",
    "tott":          "ufc_fighter_tott.csv",
    "fight_results": "ufc_fight_results.csv",
    "fight_stats":   "ufc_fight_stats.csv",
}

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
LOCAL_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ufc_stats"
DEFAULT_OUT   = PROJECT_ROOT / "data" / "derived"

# Locked in Decision Log 2026-05-11 (D8/9)
FUZZ_AUTO_ACCEPT  = 92
FUZZ_REVIEW_FLOOR = 85

# Locked in Decision Log 2026-05-12 (Pre-1999 era data: full backfill).
# Fighters appearing in ufc_fight_stats.csv (UFC 4-17, Ultimate Ultimate
# '95/'96, Ultimate Brazil; 1994-1998) but absent from upstream
# ufc_fighter_details.csv / ufc_fighter_tott.csv. Without these manual
# entries they would have NA fighter_id in fight_stats_cleaned.csv.
#
# Names MUST match the upstream FIGHTER column verbatim (no normalisation).
# Source the list by running fight_stats_cleaner.py once, then:
#   cleaned[cleaned["fighter_id"].isna()].groupby("fighter").size()
# DOB / height / weight / reach are NA-allowed: many early fighters have
# no recorded bio. Stance is NA unless you have a reliable source.
PRE_MODERN_FIGHTERS: list[dict] = [
    # TODO (one-off curation): paste in ~30-40 entries, one per unique
    # name from the NA-id groupby. Example shape:
    # {
    #     "canonical_name": "Joel Sutton",
    #     "first_name":     "Joel",
    #     "last_name":      "Sutton",
    #     "nickname":       pd.NA,
    #     "dob":            pd.NaT,
    #     "height_cm":      float("nan"),
    #     "weight_lbs":     float("nan"),
    #     "reach_in":       float("nan"),
    #     "stance":         pd.NA,
    #     "ufc_url_slug":   pd.NA,
    # },
]


# ---------------------------------------------------------------------------
# Raw-file loader
# ---------------------------------------------------------------------------
def load_raw(key: str, source: Literal["url", "local"] = "url") -> pd.DataFrame:
    """Load a raw upstream CSV from Greco's URLs or a local mirror."""
    filename = RAW_FILES[key]
    if source == "url":
        return pd.read_csv(f"{UPSTREAM_BASE}/{filename}")
    path = LOCAL_RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Local raw file missing: {path}\n"
            f"Either re-run with --source url, or curl it from "
            f"{UPSTREAM_BASE}/{filename}"
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Field parsers
# ---------------------------------------------------------------------------
def parse_dob(s: object):
    """`Jul 13, 1978` -> Timestamp. Blank/NaN/'--' -> NaT."""
    if pd.isna(s) or str(s).strip() in ("", "--"):
        return pd.NaT
    return pd.to_datetime(str(s).strip(), format="%b %d, %Y", errors="coerce")


_HEIGHT_RE = re.compile(r"^(\d+)'\s*(\d+)\"$")
def parse_height_to_cm(s: object) -> float:
    """`5' 10"` -> 177.8. `--`/blank -> NaN."""
    if pd.isna(s) or str(s).strip() in ("", "--"):
        return float("nan")
    m = _HEIGHT_RE.match(str(s).strip())
    if not m:
        return float("nan")
    feet, inches = int(m.group(1)), int(m.group(2))
    return round((feet * 12 + inches) * 2.54, 1)


def parse_weight_lbs(s: object) -> float:
    """`155 lbs.` -> 155.0. `--`/blank -> NaN."""
    if pd.isna(s) or str(s).strip() in ("", "--"):
        return float("nan")
    m = re.match(r"^(\d+)\s*lbs?\.?$", str(s).strip())
    return float(m.group(1)) if m else float("nan")


def parse_reach_in(s: object) -> float:
    """`70"` -> 70.0. `--`/blank -> NaN."""
    if pd.isna(s) or str(s).strip() in ("", "--"):
        return float("nan")
    m = re.match(r"^(\d+(?:\.\d+)?)\"$", str(s).strip())
    return float(m.group(1)) if m else float("nan")


def normalise_stance(s: object):
    """Title-case stance; blank -> NA."""
    if pd.isna(s) or str(s).strip() == "":
        return pd.NA
    return str(s).strip().title()


_SLUG_RE = re.compile(r"/fighter-details/([0-9a-f]+)/?$")
def extract_url_slug(url: object):
    """`http://ufcstats.com/fighter-details/93fe7332d16c6ad9` -> `93fe7332d16c6ad9`."""
    if pd.isna(url):
        return pd.NA
    m = _SLUG_RE.search(str(url).strip())
    return m.group(1) if m else pd.NA


# ---------------------------------------------------------------------------
# Dim build
# ---------------------------------------------------------------------------
def build_dim(
    details: pd.DataFrame,
    tott: pd.DataFrame,
    pre_modern: list[dict] | None = None,
) -> pd.DataFrame:
    """Join details + tott on URL slug, parse fields, mint sequential fighter_id.

    If `pre_modern` is provided, those manually-curated fighter rows are
    concatenated into the dim BEFORE sequential `fighter_id` assignment.
    Used for fighters absent from upstream details/tott (currently the
    pre-1999 UFC tournament era). See PRE_MODERN_FIGHTERS docstring.
    """
    d = details.copy()
    t = tott.copy()
    d["ufc_url_slug"] = d["URL"].map(extract_url_slug)
    t["ufc_url_slug"] = t["URL"].map(extract_url_slug)

    for name, df in (("details", d), ("tott", t)):
        dup = df["ufc_url_slug"].duplicated(keep=False)
        if dup.any():
            raise ValueError(
                f"{name}: {dup.sum()} duplicate URL slugs (expected unique)"
            )

    merged = d.merge(t, on="ufc_url_slug", how="outer", suffixes=("_d", "_t"))

    # Canonical name: prefer FIRST + LAST (structured); fall back to FIGHTER.
    first = merged["FIRST"].fillna("").astype(str).str.strip()
    last  = merged["LAST"].fillna("").astype(str).str.strip()
    constructed = (first + " " + last).str.strip()
    merged["canonical_name"] = constructed.mask(constructed == "", merged["FIGHTER"])
    merged["canonical_name"] = (
        merged["canonical_name"].astype(str).str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )

    # Drop rows with no resolvable name (shouldn't happen but be defensive)
    missing = merged["canonical_name"].isna()
    if missing.any():
        print(f"  WARNING: dropping {missing.sum()} rows with no name")
        merged = merged[~missing].reset_index(drop=True)

    merged["first_name"] = first.where(first != "", pd.NA)
    merged["last_name"]  = last.where(last != "",  pd.NA)
    merged["nickname"]   = (
        merged["NICKNAME"].astype(str).str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )

    merged["dob"]        = merged["DOB"].map(parse_dob)
    merged["height_cm"]  = merged["HEIGHT"].map(parse_height_to_cm)
    merged["weight_lbs"] = merged["WEIGHT"].map(parse_weight_lbs)
    merged["reach_in"]   = merged["REACH"].map(parse_reach_in)
    merged["stance"]     = merged["STANCE"].map(normalise_stance)

    cols = [
        "canonical_name",
        "first_name", "last_name", "nickname",
        "dob", "height_cm", "weight_lbs", "reach_in", "stance",
        "ufc_url_slug",
    ]
    dim = merged[cols].copy()

    # Concatenate pre-modern manual entries (if any) BEFORE id mint, so they
    # get sequential fighter_ids alongside the upstream-sourced rows.
    if pre_modern:
        pm_df = pd.DataFrame(pre_modern)
        for col in cols:
            if col not in pm_df.columns:
                pm_df[col] = pd.NA
        dim = pd.concat([dim, pm_df[cols]], ignore_index=True)
        print(f"  + appended {len(pm_df):,} pre-modern fighters")

    # Deterministic ordering before id mint.
    dim = dim.sort_values(
        ["canonical_name", "ufc_url_slug"], na_position="last"
    ).reset_index(drop=True)
    dim["fighter_id"] = dim.index + 1

    return dim[["fighter_id"] + cols]


# ---------------------------------------------------------------------------
# Name -> fighter_id resolution (RapidFuzz)
# ---------------------------------------------------------------------------
def collect_fight_data_names(
    fight_results: pd.DataFrame, fight_stats: pd.DataFrame
) -> list[str]:
    """Every distinct fighter-name string appearing in BOUT or FIGHTER columns."""
    names: set[str] = set()
    for bout in fight_results["BOUT"].dropna().unique():
        for half in str(bout).split(" vs. "):
            half = half.strip()
            if half:
                names.add(half)
    for name in fight_stats["FIGHTER"].dropna().unique():
        n = str(name).strip()
        if n:
            names.add(n)
    return sorted(names)


def resolve_names_to_ids(
    names: list[str], dim: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Resolve each name to a fighter_id via:
      1. Exact match on canonical_name
      2. RapidFuzz token_set_ratio with thresholds:
            score >= 92        -> auto-accept
            85 <= score < 92   -> flag for review (NOT auto-attached)
            score < 85         -> reject

    Returns (lookup_df, review_df, rejected_count).
    """
    from rapidfuzz import fuzz, process  # lazy import

    name_to_id = dict(zip(dim["canonical_name"], dim["fighter_id"]))
    choices    = dim["canonical_name"].tolist()

    lookup_rows, review_rows, rejected = [], [], 0
    for name in names:
        if name in name_to_id:
            lookup_rows.append({
                "name": name, "fighter_id": int(name_to_id[name]),
                "match_type": "exact", "score": 100,
            })
            continue

        result = process.extractOne(name, choices, scorer=fuzz.token_set_ratio)
        if result is None:
            rejected += 1
            continue
        candidate, score = result[0], result[1]
        if score >= FUZZ_AUTO_ACCEPT:
            lookup_rows.append({
                "name": name, "fighter_id": int(name_to_id[candidate]),
                "match_type": "fuzzy_accept", "score": int(score),
            })
        elif score >= FUZZ_REVIEW_FLOOR:
            review_rows.append({
                "name": name,
                "candidate_fighter_id": int(name_to_id[candidate]),
                "candidate_name": candidate, "score": int(score),
            })
        else:
            rejected += 1

    lookup_df = pd.DataFrame(
        lookup_rows, columns=["name", "fighter_id", "match_type", "score"]
    )
    review_df = pd.DataFrame(
        review_rows, columns=["name", "candidate_fighter_id", "candidate_name", "score"]
    )
    return lookup_df, review_df, rejected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build canonical fighter_dim.csv")
    ap.add_argument("--source", choices=["url", "local"], default="url",
                    help="Where to read raw CSVs (default: Greco1899 raw URLs).")
    ap.add_argument("--no-name-resolve", action="store_true",
                    help="Skip name->id resolution against fight data.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"Output directory (default: {DEFAULT_OUT}).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading details + tott from {args.source}...")
    details = load_raw("details", args.source)
    tott    = load_raw("tott",    args.source)
    print(f"      details: {len(details):,} rows  |  tott: {len(tott):,} rows")

    print("[2/3] Building fighter_dim...")
    dim = build_dim(details, tott, pre_modern=PRE_MODERN_FIGHTERS)
    dim_path = args.out / "fighter_dim.csv"
    dim.to_csv(dim_path, index=False)
    print(f"      wrote {len(dim):,} fighters -> {dim_path}")

    if args.no_name_resolve:
        print("[3/3] Skipped name resolution (--no-name-resolve).")
        return

    print("[3/3] Resolving fight-data names -> fighter_id...")
    fr = load_raw("fight_results", args.source)
    fs = load_raw("fight_stats",   args.source)
    names = collect_fight_data_names(fr, fs)
    print(f"      {len(names):,} distinct fighter-name strings in fight data")

    lookup, review, rejected = resolve_names_to_ids(names, dim)
    lookup_path = args.out / "fighter_name_to_id.csv"
    review_path = args.out / "fighter_match_review.csv"
    lookup.to_csv(lookup_path, index=False)
    review.to_csv(review_path, index=False)

    exact = int((lookup["match_type"] == "exact").sum())
    fuzzy = int((lookup["match_type"] == "fuzzy_accept").sum())
    print(f"      exact:         {exact:,}")
    print(f"      fuzzy >= 92:   {fuzzy:,}")
    print(f"      review 85-91:  {len(review):,}  -> {review_path}")
    print(f"      rejected <85:  {rejected:,}")
    print(f"      lookup -> {lookup_path}")


if __name__ == "__main__":
    main()