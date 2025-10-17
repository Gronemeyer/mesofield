"""Custom OME.TIFF writer for MDASequences.

Borrowed from the pattern shared by Christoph:
https://forum.image.sc/t/how-to-create-an-image-series-ome-tiff-from-python/42730/7

Sublcassed from pymmcore_plus.mda.handlers._5d_writer_base._5DWriterBase

These are the valid axis keys tifffile:
Supported by OME-XML
    X : width** (image width)
    Y : height** (image length)
    Z : depth** (image depth)
    T : time** (time series)
    C : channel** (acquisition path or emission wavelength)
    Modulo axes:
    S : sample** (color space and extra samples)
    A : angle** (OME)
    P : phase** (OME. In LSM, **P** maps to **position**)
    R : tile** (OME. Region, position, or mosaic)
    H : lifetime** (OME. Histogram)
    E : lambda** (OME. Excitation wavelength)
    Q : other** (OME)
Not Supported by OME-XML
    I : sequence** (generic sequence of images, frames, planes, pages)
    L : exposure** (FluoView)
    V : event** (FluoView)
    M : mosaic** (LSM 6)

Rules:
- all axes must be one of TZCYXSAPRHEQ
- len(axes) must equal len(shape)
- dimensions (order) must end with YX or YXS
- no axis can be repeated
- no more than 8 dimensions (or 9 if 'S' is included)

Non-OME (ImageJ) hyperstack axes MUST be in TZCYXS order
"""

from datetime import timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pymmcore_plus.mda.metadata import SummaryMetaV1  # type: ignore

from pymmcore_plus.mda.handlers._5d_writer_base import _5DWriterBase
from pymmcore_plus.mda.handlers import OMETiffWriter, ImageSequenceWriter
from useq import MDAEvent

import numpy as np
from pathlib import Path
import json

IMAGEJ_AXIS_ORDER = "tzcyxs"
FRAME_MD_FILENAME = "_frame_metadata.json"

class CustomWriter(_5DWriterBase[np.memmap]):
    """Custom Override of Pymmcore-Plus MDA handler that writes to a 5D OME-TIFF file.

    Data is memory-mapped to disk using numpy.memmap via tifffile.  Tifffile handles
    the OME-TIFF format. Tifffile handler customize to use BigTIFF format
    
    Frame metadata is saved to a JSON file.

    Parameters
    ----------
    filename : Path | str
        The filename to write to.  Must end with '.ome.tiff' or '.ome.tif'.
    """

    def __init__(self, filename: Path | str) -> None:
        try:
            import tifffile  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "tifffile is required to use this handler. "
                "Please `pip install tifffile`."
            ) from e

        self._filename = str(filename)
        if not self._filename.endswith((".tiff", ".tif")):  # pragma: no cover
            raise ValueError("filename must end with '.tiff' or '.tif'")
        self._is_ome = ".ome.tif" in self._filename
        
        # Custom attribute: Create a filename for the frame metadata jgronemeyer24
        self._frame_metadata_filename = self._filename + FRAME_MD_FILENAME

        super().__init__()

    def write_frame(
        self, ary: np.memmap, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """Write a frame to the file."""
        ary[index] = frame
        # print(f"Writing frame {index}")

    def new_array(
        self, position_key: str, dtype: np.dtype, sizes: dict[str, int]
    ) -> np.memmap:
        """Create a new tifffile file and memmap for this position."""
        from tifffile import imwrite, memmap

        dims, shape = zip(*sizes.items())

        metadata: dict[str, Any] = self._sequence_metadata()
        metadata["axes"] = "".join(dims).upper()

        # append the position key to the filename if there are multiple positions
        if (seq := self.current_sequence) and seq.sizes.get("p", 1) > 1:
            ext = ".ome.tif" if self._is_ome else ".tif"
            fname = self._filename.replace(ext, f"_{position_key}{ext}")
        else:
            fname = self._filename

        # create parent directories if they don't exist
        # Path(fname).parent.mkdir(parents=True, exist_ok=True)
        # write empty file to disk
        imwrite(
            fname,
            shape=shape,
            bigtiff=True, #jgronemeyer24
            dtype=dtype,
            metadata=metadata,
            imagej=not self._is_ome,
            ome=self._is_ome,
        )

        # memory-mapped NumPy array of image data stored in TIFF file.
        mmap = memmap(fname, dtype=dtype)
        # This line is important, as tifffile.memmap appears to lose singleton dims
        mmap.shape = shape

        return mmap  # type: ignore

    def finalize_metadata(self) -> None:
        """Called during sequenceFinished before clearing sequence metadata.

        Custom Override to save the frame metadata to a JSON file.
        jgronemeyer24
        """
        
        # Convert defaultdict to a regular dictionary
        regular_dict = dict(self.frame_metadatas)

        # Serialize to JSON using CustomJSONEncoder
        json_str = json.dumps(regular_dict, indent=4, cls=CustomJSONEncoder)
        # Save to a file
        with open(self._frame_metadata_filename, "w") as file:
            file.write(json_str)
        
        
        #self.plot() #TODO plot metadata in dev mode
        
    def plot(self):
        import json
        import pandas as pd
        import matplotlib.pyplot as plt

        with open('c:/dev/frame_metadatas.json', 'r') as file:
            data = json.load(file)

        # Normalize the nested JSON data into a flat table
        normalized_data = pd.json_normalize(data, record_path=None)

        # Convert the normalized data into a pandas DataFrame
        df = pd.DataFrame(normalized_data)

        # Extract the 'TimeReceivedByCore' field and convert it to datetime format
        df['TimeReceivedByCore'] = pd.to_datetime(df['camera_metadata.TimeReceivedByCore'])

        # Plot the 'TimeReceivedByCore' field
        plt.figure(figsize=(10, 6))
        plt.plot(df['TimeReceivedByCore'], marker='o')
        plt.xlabel('Index')
        plt.ylabel('TimeReceivedByCore')
        plt.title('TimeReceivedByCore Plot')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def _sequence_metadata(self) -> dict:
        """Create metadata for the sequence, when creating a new file."""
        if not self._is_ome:
            return {}

        metadata: dict = {}
        # see tifffile.tifffile for more metadata options
        if seq := self.current_sequence:
            if seq.time_plan and hasattr(seq.time_plan, "interval"):
                interval = seq.time_plan.interval
                if isinstance(interval, timedelta):
                    interval = interval.total_seconds()
                metadata["TimeIncrement"] = interval
                metadata["TimeIncrementUnit"] = "s"
            if seq.z_plan and hasattr(seq.z_plan, "step"):
                metadata["PhysicalSizeZ"] = seq.z_plan.step
                metadata["PhysicalSizeZUnit"] = "µm"
            if seq.channels:
                metadata["Channel"] = {"Name": [c.config for c in seq.channels]}

        return metadata
                
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, object: Any) -> Any:
        if isinstance(object, MDAEvent):
            return None #ignore the MDAEvents for now
        return super().default(object)


class CV2Writer(_5DWriterBase[Any]):
    """Write incoming MDA frames directly to an mp4/avi video using OpenCV."""

    def __init__(self, filename: Path | str, fps: int = 30, fourcc: str = "mp4v") -> None:
        try:
            import cv2  # noqa: F401
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "opencv-python is required to use this handler. "
                "Please `pip install opencv-python`."
            ) from e

        self._filename = str(filename)
        if not self._filename.endswith((".mp4", ".avi")):
            raise ValueError("filename must end with '.mp4' or '.avi'")
        self._fps = fps
        self._fourcc = fourcc
        self._frame_metadata_filename = self._filename + FRAME_MD_FILENAME

        super().__init__()

    def new_array(self, position_key: str, dtype: np.dtype, sizes: dict[str, int]):
        import cv2

        width = sizes["x"]
        height = sizes["y"]
        is_color = sizes.get("c", 1) > 1

        if (seq := self.current_sequence) and seq.sizes.get("p", 1) > 1:
            fname = self._filename.replace(".mp4", f"_{position_key}.mp4")
            fname = fname.replace(".avi", f"_{position_key}.avi")
        else:
            fname = self._filename

        fourcc = cv2.VideoWriter.fourcc(*self._fourcc)
        writer = cv2.VideoWriter(fname, fourcc, self._fps, (width, height), isColor=is_color)
        return writer

    def write_frame(self, ary: Any, index: tuple[int, ...], frame: np.ndarray) -> None:
        import cv2

        # allocate a destination array for normalization
        dst = np.empty_like(frame)
        frame_8u = cv2.normalize(frame, dst, 0, 255, cv2.NORM_MINMAX)
        frame_8u = cv2.convertScaleAbs(frame)  # More robust conversion
        ary.write(frame_8u)

    def finalize_metadata(self) -> None:
        for writer in self.position_arrays.values():
            try:
                writer.release()
            except Exception:
                pass

        regular_dict = dict(self.frame_metadatas)
        json_str = json.dumps(regular_dict, indent=4, cls=CustomJSONEncoder)
        with open(self._frame_metadata_filename, "w") as file:
            file.write(json_str)
 