"""Pure helper functions for the integration dashboard (testable without Streamlit)."""

from __future__ import annotations

import pandas as pd

REQUIRED_MASTER_COLUMNS = [
    "Incident_ID",
    "Source",
    "Event",
    "Location",
    "Time",
    "Severity",
    "Source_Record_ID",
    "Detail",
]

VALID_SEVERITIES = {"Low", "Medium", "High"}
VALID_SOURCES = {"Audio", "PDF", "Image", "Text"}


def severity_badge(severity: str) -> str:
    css = severity.lower()
    return f'<span class="severity-{css}">{severity}</span>'


def apply_filters(
    df: pd.DataFrame,
    selected_sources: list[str],
    selected_severities: list[str],
    event_query: str,
    location_query: str,
    free_text: str,
) -> pd.DataFrame:
    filtered = df.copy()

    if selected_sources:
        filtered = filtered[filtered["Source"].isin(selected_sources)]

    if selected_severities:
        filtered = filtered[filtered["Severity"].isin(selected_severities)]

    if event_query:
        filtered = filtered[
            filtered["Event"].str.contains(event_query, case=False, na=False)
        ]

    if location_query:
        filtered = filtered[
            filtered["Location"].str.contains(location_query, case=False, na=False)
        ]

    if free_text:
        mask = filtered.apply(
            lambda row: free_text.lower() in " ".join(row.astype(str)).lower(),
            axis=1,
        )
        filtered = filtered[mask]

    return filtered.sort_values(["Source", "Incident_ID"])


def record_to_detail_df(record: pd.Series) -> pd.DataFrame:
    """Convert a master incident row to a two-column Field/Value table."""
    return pd.DataFrame(
        {
            "Field": record.index.astype(str),
            "Value": record.map(lambda v: str(v) if pd.notna(v) else "Unknown").values,
        }
    )


def validate_master_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_MASTER_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"master_incidents.csv missing columns: {missing}")

    invalid_severity = set(df["Severity"].dropna().unique()) - VALID_SEVERITIES
    if invalid_severity:
        raise ValueError(f"Invalid severity values: {sorted(invalid_severity)}")

    invalid_sources = set(df["Source"].dropna().unique()) - VALID_SOURCES
    if invalid_sources:
        raise ValueError(f"Invalid source values: {sorted(invalid_sources)}")
