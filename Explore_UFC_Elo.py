"""Explore_UFC_Elo — Streamlit entry point.

Minimal bootstrap so the app launches with:
    streamlit run Explore_UFC_Elo.py

This is a stub. Data loaders, the Elo engine, and page modules will
be wired in here as they come online.
"""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="Explore UFC Elo",
        page_icon="🥊",
        layout="wide",
    )

    st.title("Explore UFC Elo")
    st.caption("Streamlit entry point — scaffolding only.")

    with st.sidebar:
        st.header("Navigation")
        view = st.radio(
            "View",
            options=["Home"],
            index=0,
        )

    if view == "Home":
        st.subheader("Welcome")
        st.write(
            "This is the placeholder home view. Data loaders, the Elo engine, "
            "and page modules will be wired in here."
        )
        st.info("App is running. Replace this stub as real modules come online.")


if __name__ == "__main__":
    main()