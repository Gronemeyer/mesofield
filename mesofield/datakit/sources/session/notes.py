"""Session notes data source.

Parses free-form ``*_notes.txt`` files where each line begins with a timestamp
followed by a colon and a short message.  The result is a simple dataframe with
an index-aligned ``time_elapsed_s`` axis so that manual observations can be
plotted alongside other timeseries.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime

from mesofield.datakit.sources.register import LoadContext, TimeseriesSource


class SessionNotesSource(TimeseriesSource):
    """Load timestamped notes recorded by the experimenter."""
    tag = "notes"
    patterns = ("**/*_notes.txt",)
    camera_tag = None
    flatten_payload = False
    line_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):\s*(.*)"
    timestamp_format = "%Y-%m-%d %H:%M:%S"
    empty_time_value = 0.0
    
    def build_timeseries(
        self,
        path: Path,
        *,
        context: LoadContext | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame, dict]:
        """Parse ``*_notes.txt`` into a timeline-aware dataframe."""
        with open(path, 'r') as f:
            notes = f.readlines()
        
        timestamps = []
        note_texts = []
        
        for line in notes:
            if not line.strip():
                continue
            
            match = re.match(self.line_pattern, line.strip())
            if match:
                timestamp_str, note_text = match.groups()
                try:
                    timestamp = datetime.strptime(timestamp_str, self.timestamp_format)
                    timestamps.append(timestamp)
                    note_texts.append(note_text)
                except ValueError:
                    continue
        
        if timestamps:
            # Convert to seconds relative to first note
            t = np.array([(ts - timestamps[0]).total_seconds() for ts in timestamps])
            df = pd.DataFrame({'timestamp': timestamps, 'note': note_texts})
        else:
            t = np.array([self.empty_time_value])
            df = pd.DataFrame({'timestamp': [], 'note': []})
        
        return t.astype(np.float64), df, {"source_file": str(path), "n_notes": len(note_texts)}