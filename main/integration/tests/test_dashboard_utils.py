"""Tests for dashboard helper functions used by the Streamlit app."""

from __future__ import annotations

from dashboard_utils import apply_filters, record_to_detail_df


def test_record_to_detail_df_has_value_column(master_df) -> None:
    record = master_df.iloc[0]
    detail_df = record_to_detail_df(record)
    assert "Field" in detail_df.columns
    assert "Value" in detail_df.columns


def test_apply_filters_by_source(master_df) -> None:
    filtered = apply_filters(
        master_df,
        selected_sources=["Audio"],
        selected_severities=["High", "Medium", "Low"],
        event_query="",
        location_query="",
        free_text="",
    )
    assert set(filtered["Source"].unique()) == {"Audio"}


def test_apply_filters_free_text(master_df) -> None:
    filtered = apply_filters(
        master_df,
        selected_sources=sorted(master_df["Source"].unique()),
        selected_severities=["High", "Medium", "Low"],
        event_query="",
        location_query="",
        free_text="fire",
    )
    assert not filtered.empty
