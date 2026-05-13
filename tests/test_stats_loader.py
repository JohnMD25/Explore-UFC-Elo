"""Smoke tests for data/stats_loader.py.

Validation policy under test: required columns must be present (raise on
missing), extras log a warning and pass through. Each loader must be
wrapped by `@st.cache_data`. Every URL must be pinned to the `main`
branch. Network is fully mocked via `_read_csv` monkeypatching.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from data import stats_loader as sl


# --- helpers --------------------------------------------------------------

def _empty_df_with_columns(columns) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


@pytest.fixture(autouse=True)
def _reset_caches():
    sl.clear_caches()
    yield
    sl.clear_caches()


# --- happy path: required columns present --------------------------------

@pytest.mark.parametrize("loader, required", [
    (sl.load_fight_results_cleaned, sl.FIGHT_RESULTS_REQUIRED_COLUMNS),
    (sl.load_fight_stats_cleaned, sl.FIGHT_STATS_REQUIRED_COLUMNS),
    (sl.load_weightclass_cleaned, sl.WEIGHTCLASS_REQUIRED_COLUMNS),
    (sl.load_fighter_dim, sl.FIGHTER_DIM_REQUIRED_COLUMNS),
])
def test_loader_returns_dataframe_when_columns_present(
    monkeypatch, loader, required,
):
    monkeypatch.setattr(
        sl, "_read_csv",
        lambda url, label: _empty_df_with_columns(required),
    )
    df = loader()
    assert isinstance(df, pd.DataFrame)
    assert set(required).issubset(df.columns)


# --- missing-column policy ------------------------------------------------

def test_missing_required_column_raises(monkeypatch):
    cols = sl.FIGHT_RESULTS_REQUIRED_COLUMNS - {"outcome"}
    monkeypatch.setattr(
        sl, "_read_csv",
        lambda url, label: _empty_df_with_columns(cols),
    )
    with pytest.raises(ValueError, match="missing required columns"):
        sl.load_fight_results_cleaned()


# --- extra-columns policy -------------------------------------------------

def test_extra_columns_log_warning_but_do_not_raise(monkeypatch, caplog):
    cols = list(sl.FIGHTER_DIM_REQUIRED_COLUMNS) + ["nickname", "birth_date"]
    monkeypatch.setattr(
        sl, "_read_csv",
        lambda url, label: _empty_df_with_columns(cols),
    )
    with caplog.at_level(logging.WARNING, logger="data.stats_loader"):
        sl.load_fighter_dim()
    assert any("extra columns" in record.message for record in caplog.records)


# --- cache_data wiring ----------------------------------------------------

@pytest.mark.parametrize("loader", [
    sl.load_fight_results_cleaned,
    sl.load_fight_stats_cleaned,
    sl.load_weightclass_cleaned,
    sl.load_fighter_dim,
])
def test_loader_is_wrapped_by_streamlit_cache(loader):
    # st.cache_data wraps the function so it exposes a .clear() method.
    assert hasattr(loader, "clear"), (
        f"{loader.__name__} is missing the Streamlit cache .clear() method"
    )


# --- URLs pinned to main --------------------------------------------------

def test_all_raw_urls_pinned_to_main_branch():
    urls = [
        sl.URL_FIGHT_RESULTS_CLEANED,
        sl.URL_FIGHT_STATS_CLEANED,
        sl.URL_WEIGHTCLASS_CLEANED,
        sl.URL_FIGHTER_DIM,
    ]
    for url in urls:
        assert url.startswith(
            "https://raw.githubusercontent.com/"
            "JohnMD25/Explore-UFC-Elo/main/"
        ), f"URL not pinned to main: {url}"
