"""Mesoscope mean fluorescence analysis data source.

This module translates CSV exports from the mesoscope processing pipeline into
time-indexed :class:`~datakit.datamodel.LoadedStream` objects.  Each record in the
table represents one slice/ROI with derived dF/F traces, ready for alignment
against other experiment timelines.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from mesofield.datakit.sources.register import LoadContext, TimeseriesSource, document


class MesoMeanSource(TimeseriesSource):
    """Load mean fluorescence CSV.

    The CSV is expected to contain at least ``Slice`` and ``Mean`` columns.  
    """
    tag = "meso"
    patterns = ("**/*_meso-mean-trace.csv",)
    camera_tag = "meso_metadata"  # Bind to meso camera
    required_columns = ("Mean")
    assumed_frame_rate_hz = 50.0
    normalization_baseline = "min"
    
    def build_timeseries(
        self,
        path: Path,
        *,
        context: LoadContext | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame, dict]:
        """Read a mesoscope CSV and return a normalized table."""
        df = pd.read_csv(path)
        #remove the "Slice" column
        df = df.drop(columns=["Slice"])
        
        n_frames = len(df)
        t = np.arange(n_frames, dtype=np.float64) / float(self.assumed_frame_rate_hz)
        df = df.copy()
                
        return t, df, {"source_file": str(path), "n_slices": len(df)}

    
class MesoDFFSource(TimeseriesSource):
    """Load DFF fluorescence CSV.

    The CSV is expected to contain at least ``Slice`` and ``Mean`` columns.
    """
    tag = "meso"
    patterns = ("**/*_meso-collapse_first-trace.csv",)
    camera_tag = "meso_metadata"  # Bind to meso camera
    required_columns = ("Mean")
    assumed_frame_rate_hz = 50.0
    normalization_baseline = "min"
    
    def build_timeseries(
        self,
        path: Path,
        *,
        context: LoadContext | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame, dict]:
        """Read a mesoscope CSV and return a normalized table."""
        df = pd.read_csv(path)
        #remove the "Slice" column
        df = df.drop(columns=["Slice"])
        #rename the "Mean" column to "dF_F"
        df = df.rename(columns={"Mean": "dF_F"})
        
        n_frames = len(df)
        t = np.arange(n_frames, dtype=np.float64) / float(self.assumed_frame_rate_hz)
        df = df.copy()
                
        return t, df, {"source_file": str(path), "n_slices": len(df)}


