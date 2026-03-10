"""
Build SAMPLE_TYPES from an Excel matrix.

Expected Excel format:
- Sheet "Concentrations":
  - First column: sample name (one row per sample)
  - Remaining columns: element concentrations
- Sheet "Uncertainties":
  - Same row/column structure as "Concentrations"
  - Values are absolute uncertainties for each concentration
- Missing/blank concentration cells are interpreted as 0
- Missing/blank uncertainty cells are set to 1% of concentration
"""

from __future__ import annotations

from pathlib import Path
import re
import sqlite3
from typing import Any

import pandas as pd


def _resolve_excel_path() -> Path:
    """Resolve ./Source/Samples_Fe_matrix.xlsx relative to this file."""
    base_dir = Path(__file__).resolve().parent
    pattern = "Samples_Fe_matrix.xlsx"
    matches = sorted((base_dir / "Source").glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No Excel file found matching ./Source/{pattern}"
        )
    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple Excel files matched ./Source/{pattern}: "
            f"{', '.join(str(p.name) for p in matches)}"
        )
    return matches[0]


def _normalize_sample_id(sample_name: str, row_number: int) -> str:
    """Create a deterministic sample_id compatible with existing config style."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", sample_name).strip("_").upper()
    if not normalized:
        normalized = f"SAMPLE_{row_number:03d}"
    return normalized


def _db_has_quantparam(db_path: Path) -> bool:
    """Return True if db exists and contains QuantParam table."""
    if not db_path.exists() or db_path.stat().st_size == 0:
        return False
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='QuantParam'"
            )
            return cursor.fetchone() is not None
    except sqlite3.Error:
        return False


def _resolve_db_path() -> Path:
    """Resolve LIBS database path from known project locations."""
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "Source" / "LIBS_data_vacuum.db",
        base_dir / "LIBS_data_vacuum.db",
    ]
    for candidate in candidates:
        if _db_has_quantparam(candidate):
            return candidate
    raise FileNotFoundError(
        "No valid LIBS_data_vacuum.db found with QuantParam table in "
        "./Source/ or project root."
    )


def _load_db_elements() -> set[str]:
    """Load valid element names from QuantParam table."""
    db_path = _resolve_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT Elem_name FROM QuantParam")
        elements = {
            str(row[0]).strip()
            for row in cursor.fetchall()
            if row[0] is not None and str(row[0]).strip()
        }
    return elements


def load_sample_types_from_excel() -> list[dict[str, Any]]:
    """Load SAMPLE_TYPES entries from the configured Excel matrix file."""
    excel_path = _resolve_excel_path()
    concentrations_df = pd.read_excel(excel_path, sheet_name="Concentrations")
    uncertainties_df = pd.read_excel(excel_path, sheet_name="Uncertainties")

    if concentrations_df.empty:
        return []

    concentration_columns = [str(c).strip() for c in concentrations_df.columns]
    concentrations_df.columns = concentration_columns
    uncertainties_df.columns = [str(c).strip() for c in uncertainties_df.columns]

    sample_name_col = concentration_columns[0]
    element_cols = [
        c for c in concentration_columns[1:] if c and not c.lower().startswith("unnamed:")
    ]
    valid_db_elements = _load_db_elements()
    element_cols = [element for element in element_cols if element in valid_db_elements]

    uncertainty_name_col = uncertainties_df.columns[0]
    uncertainties_df = uncertainties_df.set_index(uncertainty_name_col)

    sample_types: list[dict[str, Any]] = []
    for row_idx, (_, row) in enumerate(concentrations_df.iterrows(), start=1):
        raw_name = row.get(sample_name_col, "")
        if pd.isna(raw_name) or str(raw_name).strip() == "":
            continue

        sample_name = str(raw_name).strip()
        uncertainty_row = (
            uncertainties_df.loc[sample_name]
            if sample_name in uncertainties_df.index
            else pd.Series(dtype=float)
        )
        if isinstance(uncertainty_row, pd.DataFrame):
            # If duplicate sample names exist, use the first matching uncertainty row.
            uncertainty_row = uncertainty_row.iloc[0]

        concentration_ranges: dict[str, tuple[float, float]] = {}

        for element in element_cols:
            value = pd.to_numeric(row.get(element, 0), errors="coerce")
            concentration = 0.0 if pd.isna(value) else float(value)
            uncertainty_value = pd.to_numeric(uncertainty_row.get(element, None), errors="coerce")
            if pd.isna(uncertainty_value):
                uncertainty = 0.01 * abs(concentration)
            else:
                uncertainty = float(uncertainty_value)

            c_min = concentration - uncertainty
            c_max = concentration + uncertainty
            concentration_ranges[element] = (c_min, c_max)

        sample_types.append(
            {
                "sample_id": _normalize_sample_id(sample_name, row_idx),
                "sample_name": sample_name,
                "n_samples": 1,
                "concentration_ranges": concentration_ranges,
            }
        )

    return sample_types


SAMPLE_TYPES = load_sample_types_from_excel()

