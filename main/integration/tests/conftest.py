"""Shared pytest fixtures for integration tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

INTEGRATION_DIR = Path(__file__).resolve().parents[1]
if str(INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_DIR))


@pytest.fixture(scope="session")
def integration_dir() -> Path:
    return INTEGRATION_DIR


@pytest.fixture(scope="session")
def master_df(integration_dir: Path) -> pd.DataFrame:
    path = integration_dir / "master_incidents.csv"
    if not path.exists():
        pytest.skip("master_incidents.csv not generated yet")
    return pd.read_csv(path)
