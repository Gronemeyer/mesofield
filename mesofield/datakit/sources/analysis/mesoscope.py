"""Mesoscope mean fluorescence analysis data source.

This module translates CSV exports from the mesoscope processing pipeline into
time-indexed :class:`~datakit.datamodel.LoadedStream` objects.  Each record in the
table represents one slice/ROI with derived dF/F traces, ready for alignment
against other experiment timelines.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from mesofield.datakit.sources.register import SourceContext, TimeseriesSource


class MesoMeanSource(TimeseriesSource):
    """Load mean fluorescence CSV and compute dF/F traces.

    The CSV is expected to contain at least ``Slice`` and ``Mean`` columns.  The
    loader derives a monotonically increasing ``time_elapsed_s`` column by
    dividing the sample index by :attr:`assumed_frame_rate_hz`, then appends a
    normalized ``dF_F`` column for quick plotting downstream.
    """
    tag = "meso_mean"
    patterns = ("**/*_meso-mean-trace.csv",)
    camera_tag = "meso_metadata"  # Bind to meso camera
    required_columns = ("Slice", "Mean")
    assumed_frame_rate_hz = 50.0
    normalization_baseline = "min"
    
    def build_timeseries(
        self,
        path: Path,
        *,
        context: SourceContext | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame, dict]:
        """Read a mesoscope CSV and return a normalized table."""
        df = pd.read_csv(path)
        
        if not set(self.required_columns).issubset(df.columns):
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            missing = ", ".join(sorted(missing_cols))
            raise ValueError(
                f"Expected columns {self.required_columns} not found in {path}: missing {missing}"
            )
        
        n_frames = len(df)
        t = np.arange(n_frames, dtype=np.float64) / float(self.assumed_frame_rate_hz)
        df = df.copy()
        df["time_elapsed_s"] = t
        
        #normalize the Mean column using dF/F
        df['dF_F'] = self._dff_normalize(df[self.required_columns[1]])
        
        return t, df, {"source_file": str(path), "n_slices": len(df)}

    def _dff_normalize(self, series: pd.Series) -> pd.Series:
        """Compute a simple dF/F normalization for the ``Mean`` column."""
        if self.normalization_baseline == "min":
            F0 = series.min()
        elif self.normalization_baseline == "median":
            F0 = series.median()
        else:
            raise ValueError(f"Unsupported normalization_baseline: {self.normalization_baseline}")

        if F0 == 0:
            return series - F0
        return (series - F0) / F0

    
