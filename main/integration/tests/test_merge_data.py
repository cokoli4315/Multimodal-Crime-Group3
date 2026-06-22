"""Tests for merge_data.py pipeline and output files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dashboard_utils import validate_master_schema
from merge_data import (
    AUDIO_CSV,
    DOCUMENT_CSV,
    IMAGE_CSV,
    MASTER_CSV,
    TEXT_CSV,
    export_master_tables,
)


@pytest.mark.parametrize(
    "path",
    [AUDIO_CSV, DOCUMENT_CSV, IMAGE_CSV, TEXT_CSV],
    ids=["audio", "document", "image", "text"],
)
def test_source_csv_files_exist(path: Path) -> None:
    assert path.exists(), f"Missing analyst CSV: {path}"


def test_export_master_tables_creates_outputs() -> None:
    long_master = export_master_tables()
    assert MASTER_CSV.exists()
    assert len(long_master) > 0


def test_master_csv_schema(master_df: pd.DataFrame) -> None:
    validate_master_schema(master_df)


def test_master_has_all_sources(master_df: pd.DataFrame) -> None:
    assert set(master_df["Source"].unique()) == {"Audio", "PDF", "Image", "Text"}


def test_incident_id_prefixes(master_df: pd.DataFrame) -> None:
    prefixes = master_df.groupby("Source")["Incident_ID"].first().to_dict()
    assert prefixes["Audio"].startswith("AUD-")
    assert prefixes["PDF"].startswith("DOC-")
    assert prefixes["Image"].startswith("IMG-")
    assert prefixes["Text"].startswith("TXT-")
