"""data/stats_loader.py — single read boundary for cleaned + derived UFC data.

All file IO for the project lives in this module. App code never opens a
CSV or constructs a `raw.githubusercontent.com` URL directly — it imports
a loader function from here. Each loader is wrapped with
`@st.cache_data(ttl=3600)` so a Streamlit session hits GitHub at most
once per hour per file. Pinned to `main`; swap the branch segment for a
commit SHA if deterministic deploys are ever needed.

Validation policy (v1): column presence only. Missing required columns
raise `ValueError`; extra columns log a warning and pass through. Dtype
coercion is deliberately deferred to consumers.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# --- URL configuration ----------------------------------------------------

GITHUB_OWNER = "JohnMD25"
GITHUB_REPO = "Explore-UFC-Elo"
GITHUB_BRANCH = "main"
_RAW_BASE = (
    f"https://raw.githubusercontent.com/"
    f"{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}"
)

URL_FIGHT_RESULTS_CLEANED = f"{_RAW_BASE}/data/cleaned/ufc/fight_results_cleaned.csv"
URL_FIGHT_STATS_CLEANED = f"{_RAW_BASE}/data/cleaned/ufc/fight_stats_cleaned.csv"
URL_WEIGHTCLASS_CLEANED = f"{_RAW_BASE}/data/cleaned/ufc/weightclass_cleaned.csv"
URL_FIGHTER_DIM = f"{_RAW_BASE}/data/derived/fighter_dim.csv"


# --- column contracts (from Data Contracts Notion page) -------------------

FIGHT_RESULTS_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "fight_id", "event_date", "bout_order",
    "fighter_a_id", "fighter_b_id",
    "outcome", "method", "round_no", "total_rounds",
    "weight_class", "bout_type",
})

# fight_stats_cleaned.csv full schema TBD on its Notion subpage. For v1 we
# enforce only the two guarantees explicitly documented in Data Contracts:
# fighter_id is attached (with 42 known NAs upstream), and round_no is
# populated post-v1.1 cleaning.
FIGHT_STATS_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "fighter_id", "round_no",
})

WEIGHTCLASS_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "EVENT", "BOUT", "WEIGHTCLASS",
    "weight_class", "title_type", "is_womens", "is_tournament",
})

FIGHTER_DIM_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "fighter_id", "canonical_name",
    "aliases", "ufc_url_slug", "wiki_url_slug",
})


# --- private helpers ------------------------------------------------------

def _assert_required_columns(
    df: pd.DataFrame,
    required: frozenset[str],
    file_label: str,
) -> None:
    """Raise on missing required columns; log a warning on extras."""
    columns = set(df.columns)
    missing = required - columns
    if missing:
        raise ValueError(
            f"{file_label}: missing required columns "
            f"{sorted(missing)}. Found: {sorted(columns)}."
        )
    extras = columns - required
    if extras:
        logger.warning(
            "%s: extra columns not in contract: %s",
            file_label, sorted(extras),
        )


def _read_csv(url: str, file_label: str) -> pd.DataFrame:
    """Single read primitive. Patched in tests; never called directly by
    app code."""
    logger.info("%s: reading %s", file_label, url)
    return pd.read_csv(url)


# --- public loaders -------------------------------------------------------

@st.cache_data(ttl=3600)
def load_fight_results_cleaned() -> pd.DataFrame:
    """Per-bout cleaned fight results. Matches the `FightRow` contract on
    the elo_engine.py Notion page; this is the engine's only input file."""
    df = _read_csv(URL_FIGHT_RESULTS_CLEANED, "fight_results_cleaned.csv")
    _assert_required_columns(
        df, FIGHT_RESULTS_REQUIRED_COLUMNS, "fight_results_cleaned.csv",
    )
    return df


@st.cache_data(ttl=3600)
def load_fight_stats_cleaned() -> pd.DataFrame:
    """Per-round fight statistics with `fighter_id` attached. Min-column
    contract for v1; tighten when the schema subpage lands."""
    df = _read_csv(URL_FIGHT_STATS_CLEANED, "fight_stats_cleaned.csv")
    _assert_required_columns(
        df, FIGHT_STATS_REQUIRED_COLUMNS, "fight_stats_cleaned.csv",
    )
    return df


@st.cache_data(ttl=3600)
def load_weightclass_cleaned() -> pd.DataFrame:
    """Parsed weight-class fields per bout. Schema per the
    weightclass_cleaned.csv Notion subpage."""
    df = _read_csv(URL_WEIGHTCLASS_CLEANED, "weightclass_cleaned.csv")
    _assert_required_columns(
        df, WEIGHTCLASS_REQUIRED_COLUMNS, "weightclass_cleaned.csv",
    )
    return df


@st.cache_data(ttl=3600)
def load_fighter_dim() -> pd.DataFrame:
    """Canonical fighter list. The bridge table every other file joins on
    via `fighter_id`."""
    df = _read_csv(URL_FIGHTER_DIM, "fighter_dim.csv")
    _assert_required_columns(
        df, FIGHTER_DIM_REQUIRED_COLUMNS, "fighter_dim.csv",
    )
    return df


# --- module-level convenience ---------------------------------------------

def clear_caches() -> None:
    """Clear every loader's Streamlit cache. Useful after a manual refresh
    push when you don't want to wait for the hourly TTL."""
    load_fight_results_cleaned.clear()
    load_fight_stats_cleaned.clear()
    load_weightclass_cleaned.clear()
    load_fighter_dim.clear()