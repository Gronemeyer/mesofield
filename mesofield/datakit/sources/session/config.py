"""Session configuration data source."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from mesofield.datakit.sources.register import SourceContext, TableSource
from mesofield.datakit.config import settings


def _clean_config_value(value: Any) -> Any:
    """Normalize configuration values for metadata storage."""

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        lowered = stripped.lower()
        if lowered in {"nan", "na", "none", "null"}:
            return None
        return stripped

    if value is None:
        return None

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, float) and np.isnan(value):
        return None

    return value



class SessionConfigSource(TableSource):
    """Load configuration CSVs and surface subject/session attributes."""

    tag = "session_config"
    patterns = ("**/*_configuration.csv",)
    camera_tag = None
    flatten_payload = True

    DEFAULT_VARIABLE_MAPPING: ClassVar[Dict[str, tuple[str, ...]]] = {
        "subject": ("sex", "genotype", "DOS", "DOB"),
        "session": ("weight", "XYZ", "PT"),
    }
    variable_mapping: ClassVar[Dict[str, tuple[str, ...]]] = DEFAULT_VARIABLE_MAPPING.copy()


    def build_table(
        self,
        path: Path,
        *,
        context: SourceContext | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame, dict]:
        raw_df = pd.read_csv(path)

        required_columns = {"Parameter", "Value"}
        if not required_columns.issubset(raw_df.columns):
            missing = ", ".join(sorted(required_columns - set(raw_df.columns)))
            raise ValueError(f"Configuration CSV missing required columns: {missing}")

        df = raw_df.set_index("Parameter").T

        # Config is static - single timepoint
        t = np.arange(len(df), dtype=float)

        # Normalize parameter values for metadata usage
        values_row = df.iloc[0] if not df.empty else pd.Series(dtype=object)
        parameter_values = {
            str(col).strip(): _clean_config_value(values_row[col])
            for col in values_row.index
        }

        mapping = self.variable_mapping or {}
        subject_keys = mapping.get("subject", ())
        session_keys = mapping.get("session", ())

        subject_variables = {
            key: parameter_values[key]
            for key in subject_keys
            if key in parameter_values and parameter_values[key] is not None
        }
        session_variables = {
            key: parameter_values[key]
            for key in session_keys
            if key in parameter_values and parameter_values[key] is not None
        }

        dedupe_keys = set(subject_variables) | set(session_variables) | {"subject", "session"}
        parameter_values = {
            key: value for key, value in parameter_values.items() if key not in dedupe_keys
        }

        mapped_keys = set(subject_variables) | set(session_variables)
        remaining_variables = {
            key: value
            for key, value in parameter_values.items()
            if key not in mapped_keys and value is not None
        }

        flattened_scope = {
            **subject_variables,
            **session_variables,
            **remaining_variables,
        }

        meta = {
            "source_file": str(path),
            "n_params": len(parameter_values),
            "parameter_values": parameter_values,
            "subject_variables": subject_variables,
            "session_variables": session_variables,
            "unmapped_variables": remaining_variables,
            "variable_mapping": {scope: list(keys) for scope, keys in mapping.items()},
            "scope": settings.dataset.session_scope,
            **flattened_scope,
        }

        return t, df, meta