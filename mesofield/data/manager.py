from __future__ import annotations
"""
DataManager & DataSaver Workflow Overview
1. Initialization
    ├─ dm = DataManager()
    └─ dm.setup(config, path, devices)
2. Saving experiment outputs via DataSaver
    ├─ dm.saver.configuration()       # writes experiment config to disk
    ├─ dm.saver.all_hardware()        # writes hardware outputs
    ├─ dm.saver.all_notes()           # writes notes.txt if any
    └─ dm.saver.writer_for(camera)    # returns writer & updates camera output paths
3. Final database update: dm.update_database()
    update_database() records the :class:`DataPaths` for the current configuration
    into the HDF5 database using a MultiIndex of (Subject, Session, Task).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict, Iterable
from logging import Logger

import os
import queue
import csv
import threading
import time
from datetime import datetime

import pandas as pd

from mesofield.config import ExperimentConfig
from mesofield.data.writer import CustomWriter, CV2Writer
from mesofield.io.h5db import H5Database
from mesofield.utils._logger import get_logger, log_this_fr
from typing import Dict


@dataclass
class DataPacket:
    """Entry in :class:`DataQueue`."""

    device_id: str
    timestamp: datetime
    payload: Any
    device_ts: float | None = None
    meta: Dict[str, Any] = field(default_factory=dict)


class DataQueue:
    """Thread-safe queue for data streaming between devices and consumers.
    
    Registered DataProducer (the :class:`DataManager`) devices can push data packets to this queue,
    which can then be consumed by other parts of the system.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: queue.Queue[DataPacket] = queue.Queue(maxsize)

    def push(
        self,
        device_id: str,
        payload: Any,
        *,
        timestamp: datetime | None = None,
        device_ts: float | None = None,
        **meta: Any,
    ) -> None:
        """Add a new data packet to the queue."""
        if timestamp is None:
            timestamp = datetime.now()
        self._queue.put(DataPacket(device_id, timestamp, payload, device_ts, meta))

    def pop(self, block: bool = True, timeout: float | None = None) -> DataPacket:
        """Return the next :class:`DataPacket` from the queue."""
        return self._queue.get(block=block, timeout=timeout)

    def empty(self) -> bool:
        return self._queue.empty()


@dataclass
class DataPaths:
    """Structured storage for all output paths used by the :class:`DataSaver`.
    
    Relies on the :class:`ExperimentConfig` to generate paths based on the
    experiment's BIDS directory using the `make_path` method, which relies
    on HardwareDevices implementing a file_type, bids_type.
    
    These paths can then be passed to HardwareDevice.save_data(path)  
    """
    configuration: str
    notes: str
    timestamps: str
    hardware: Dict[str, str] = field(default_factory=dict)
    writers: Dict[str, str] = field(default_factory=dict)
    queue: str = ""

    @classmethod
    def build(cls, cfg: ExperimentConfig) -> DataPaths:
        cfg_paths = {
            "configuration": cfg.make_path("configuration", "csv"),
            "notes": cfg.make_path("notes", "txt"),
            "timestamps": cfg.make_path("timestamps", "csv"),
        }
        hw_paths: Dict[str, str] = {}
        for dev_id, device in cfg.hardware.devices.items():
            args = getattr(device, "path_args", {})
            suffix = args.get("suffix", dev_id)
            ext = args.get("extension", getattr(device, "file_type", "dat"))
            bids_type = args.get("bids_type", getattr(device, "bids_type"))
            hw_paths[dev_id] = cfg.make_path(suffix, ext, bids_type)
        return cls(
            configuration=cfg_paths["configuration"],
            notes=cfg_paths["notes"],
            timestamps=cfg_paths["timestamps"],
            hardware=hw_paths,
            writers={},
            queue=cfg.make_path("dataqueue", "csv", "beh"),
        )

@dataclass
class DataSaver:
    """Helper class for saving experiment data to disk.
    
    This class handles saving configuration, hardware data, notes, timestamps,
    and camera data to disk.
    
    Takes paths generated from :class:`DataPaths` instance and calls `Device.save_data(path)`
    on all hardware devices that implement this method.
    
    (NOTE: If a device does not implement `save_data`, it will be skipped.)
    
    Takes an :class:`ExperimentConfig` to save configuration.
    """
    
    cfg: ExperimentConfig
    paths: DataPaths = field(init=False)
    logger: Logger = field(default_factory=lambda: get_logger("DataSaver"))

    def __post_init__(self) -> None:
        self.paths = DataPaths.build(self.cfg)
        self.logger.info(f"Prepared output paths: {self.paths}")

    def configuration(self) -> None:
        path = self.paths.configuration
        try:
            params = self.cfg.items()
            df = pd.DataFrame(params.items(), columns=["Parameter", "Value"])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            self.logger.info(f"Configuration saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")

    def all_hardware(self) -> None:
        for dev_id, path in self.paths.hardware.items():
            device = self.cfg.hardware.devices.get(dev_id)
            if not device:
                continue
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                device.output_path = path
                if hasattr(device, "save_data"):
                    device.save_data(path)
                self.logger.info(f"Device {dev_id} data saved to {path}")
            except Exception as e:
                self.logger.error(f"Error saving device {dev_id}: {e}")

    def all_notes(self) -> None:
        if not self.cfg.notes:
            return
        path = self.paths.notes
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write("\n".join(self.cfg.notes))
            self.logger.info(f"Notes saved to {path}")
            self.cfg.notes.clear()  # Clear notes after saving
        except Exception as e:
            self.logger.error(f"Error saving notes: {e}")

    def writer_for(self, camera: Any) -> Any:
        """Prepare and return a writer for the provided camera device."""
        if not hasattr(camera, "set_writer"):
            raise AttributeError("camera does not implement set_writer")

        camera.set_writer(self.cfg.make_path)
        writer = getattr(camera, "writer", None)
        output_path = getattr(camera, "output_path", None)

        name = getattr(camera, "name", getattr(camera, "device_id", "camera"))
        if output_path:
            self.paths.writers[name] = output_path
        return writer

    def save_timestamps(self, id, start_time, stop_time) -> None:
        path = self.paths.timestamps
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["device_id", "started", "stopped"])
                writer.writerow([id, start_time, stop_time])
                for dev_id, device in self.cfg.hardware.devices.items():
                    started = getattr(device, "_started", "")
                    stopped = getattr(device, "_stopped", "")
                    writer.writerow([dev_id, started, stopped])
            self.logger.info(f"Timestamps saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving timestamps: {e}")

    def save_queue(self, rows: list[list[Any]], path: str | None = None) -> None:
        """Save queued data rows to CSV file specified in DataPaths or override path."""
        if path is None:
            path = self.paths.queue
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    "queue_elapsed",
                    "packet_ts",
                    "device_ts",
                    "device_id",
                    "payload",
                ])
                writer.writerows(rows)
            self.logger.info(f"Queue log saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving queue log: {e}")


class DataManager:
    """Very small wrapper providing optional :class:`DataSaver`."""

    def __init__(self, h5_path: str) -> None:
        self.save: DataSaver
        self.base = H5Database(h5_path)
        self.queue = DataQueue()

        self.devices: List[Any] = []
        self._registered_ids: set[str] = set()
        
        self.queue_log_path: Optional[str] = None
        self.queue_packets: list[list[Any]] = []
        
        self._queue_thread: Optional[threading.Thread] = None
        self._stop_queue: bool = False

    def setup(
        self, config: ExperimentConfig, devices: Iterable[Any] | None = None
    ) -> None:
        """Attach configuration and optionally register devices."""
        self.save = DataSaver(config)
        if devices is not None:
            self.register_devices(devices)

    #@log_this_fr
    def register_devices(self, devices: Iterable[Any]) -> None:
        """Register a list of hardware devices with the manager."""
        for dev in devices:
            if hasattr(dev, "device_type") and hasattr(dev, "get_data"):
                self.register_hardware_device(dev)

    def append_to_database(self, df: pd.DataFrame, key: str = "data") -> None:
        """Append ``df`` to the database if configured."""
        if self.base:
            self.base.update(df, key)

    # ------------------------------------------------------------------
    def start_queue_logger(self, path: str | None = None) -> None:
        """Begin capturing :class:`DataQueue` contents."""
        # determine log path, default to DataPaths.queue
        if path is None and getattr(self, "save", None):
            path = self.save.paths.queue
        if path is None:
            return

        self.queue_log_path = path
        # in-memory storage for log rows
        self.queue_packets = []
        self._stop_queue = False

        # start background thread to record queue packets
        self._queue_thread = threading.Thread(
            target=self._queue_writer_loop,
            daemon=True,
        )
        self._queue_thread.start()

    def stop_queue_logger(self) -> None:
        """Stop the queue logging thread and flush data to disk."""
        self._stop_queue = True
        if self._queue_thread:
            self._queue_thread.join(timeout=1)

        # save recorded packets via DataSaver
        if self.save and self.queue_log_path:
            # use the configured log path for writing
            self.save.save_queue(self.queue_packets, self.queue_log_path)

    def _queue_writer_loop(self) -> None:

        while not self._stop_queue or not self.queue.empty():
            try:
                pkt = self.queue.pop(timeout=0.1)
            except queue.Empty:
                continue

            now = time.perf_counter()  # Use monotonic time for consistency
            row = [now, pkt.timestamp, pkt.device_ts, pkt.device_id, pkt.payload]
            self.queue_packets.append(row)

            # if self._stream and self._writer:
            #     self._writer.writerow(row)
            #     assert self._file is not None
            #     self._file.flush()

    def register_hardware_device(self, device: Any) -> None:  # pragma: no cover - convenience
        """Track a hardware device and connect its data stream to the queue."""
        dev_id = getattr(device, "device_id", getattr(device, "id", "unknown"))
        if device in self.devices or dev_id in self._registered_ids:
            return

        self.devices.append(device)
        self._registered_ids.add(dev_id)

        # Try to connect various callback styles to push data to our queue
        def _push(payload: Any, device_ts: Any = None, *, dev=device) -> None:
            dev_id = getattr(dev, "device_id", getattr(dev, "id", "unknown"))
            #print(f"DataManager: Pushing data from {dev_id} to queue: {payload}")
            self.queue.push(dev_id, payload, device_ts=device_ts)

        # Connect using a standard data_event if present
        evt = getattr(device, "data_event", None)
        if evt is not None and hasattr(evt, "connect"):
            try:
                evt.connect(_push)
            except Exception:
                pass
            
        sig = getattr(device, "serialDataReceived", None)
        if sig is not None and hasattr(sig, "connect"):
            try:
                sig.connect(_push)
            except Exception:
                pass

        if sig is None and hasattr(device, "core"):
            # connect metadata-only from core.mda.events.frameReady
            sig = getattr(device.core.mda.events, "frameReady", None)
            if sig is not None and hasattr(sig, "connect"):
                try:
                    # frameReady callback signature: (image, metadata) You can find these in the tiff_frame_metadata.json files for reference
                    sig.connect(lambda _img, event, metadata: _push(payload=metadata['camera_metadata']['ImageNumber'],
                                                                    device_ts=metadata['camera_metadata']['TimeReceivedByCore']))
                except Exception:
                    pass
                
    #TODO: move this logic to DataSaver
    def get_device_outputs(self, subject: str, session: str) -> pd.DataFrame:
        """Return a DataFrame of output file paths for registered devices."""
        records = {}
        for dev in self.devices:
            dev_id = getattr(dev, "device_id", getattr(dev, "id", "unknown"))
            out = getattr(dev, "output_path", None)
            if out:
                key = (dev.device_type, dev_id, "file")
                records[key] = out

                # derive metadata path if attribute not set
                meta = getattr(dev, "metadata_path", None)
                if not meta:
                    if out.endswith("ome.tiff"):
                        meta = out.replace("ome.tiff", "ome.tiff_frame_metadata.json")
                    elif out.endswith("ome.tif"):
                        meta = out.replace("ome.tif", "ome.tif_frame_metadata.json")
                if meta and os.path.exists(meta):
                    key = (dev.device_type, dev_id, "metadata")
                    records[key] = meta
        if not records:
            return pd.DataFrame()
        idx = pd.MultiIndex.from_arrays([[subject], [session]], names=["Subject", "Session"])
        return pd.DataFrame(records, index=idx)

    # ------------------------------------------------------------------
    def update_database(self) -> None:
        """Record the current :class:`DataPaths` into the HDF5 database."""
        if not (self.base and self.save):
            return

        cfg = self.save.cfg
        subject, session, task = cfg.subject, cfg.session, getattr(cfg, "task", "")

        # build single row dataframe with output paths
        records: dict[str, str] = {
            "configuration": self.save.paths.configuration,
            "notes": self.save.paths.notes,
            "timestamps": self.save.paths.timestamps,
            "queue": self.queue_log_path or self.save.paths.queue,
        }

        for dev_id, path in self.save.paths.hardware.items():
            records[dev_id] = path

        for name, path in self.save.paths.writers.items():
            records[name] = path

        idx = pd.MultiIndex.from_arrays(
            [[subject], [session], [task]],
            names=["Subject", "Session", "Task"],
        )

        df = pd.DataFrame([records], index=idx)
        self.append_to_database(df, key="datapaths")

    def read_database(self, key: str = "datapaths") -> Optional[pd.DataFrame]:
        """Read a DataFrame from the underlying :class:`H5Database`."""
        if self.base:
            result = self.base.read(key)
            if isinstance(result, pd.Series):
                return result.to_frame().T
            return result
        return None
