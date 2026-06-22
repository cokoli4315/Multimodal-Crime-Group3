"""Smoke tests ensuring dashboard logic works on real generated data."""

from __future__ import annotations

from dashboard_utils import apply_filters, record_to_detail_df, validate_master_schema


def test_incident_lookup_flow(master_df) -> None:
    validate_master_schema(master_df)
    filtered = apply_filters(
        master_df,
        selected_sources=sorted(master_df["Source"].unique()),
        selected_severities=["High", "Medium", "Low"],
        event_query="",
        location_query="",
        free_text="",
    )
    for incident_id in filtered["Incident_ID"].head(20):
        record = master_df[master_df["Incident_ID"] == incident_id].iloc[0]
        detail_df = record_to_detail_df(record)
        assert "Value" in detail_df.columns


def test_filtered_incident_lookup(master_df) -> None:
    target_id = "AUD-006"
    filtered = apply_filters(
        master_df,
        selected_sources=["Audio"],
        selected_severities=["High", "Medium", "Low"],
        event_query="",
        location_query="",
        free_text="",
    )
    assert target_id in filtered["Incident_ID"].values
    record = master_df[master_df["Incident_ID"] == target_id].iloc[0]
    assert record["Source"] == "Audio"
