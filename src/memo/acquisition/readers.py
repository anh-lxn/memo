from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import re
import threading

import numpy as np
import pandas as pd

from memo.types import SensorFrame

try:
    import serial
    from serial import SerialException
except ImportError:  # pragma: no cover - depends on local environment
    serial = None

    class SerialException(Exception):
        pass


SENSOR_COLUMNS = [f"Sensor R{i}" for i in range(1, 9)]


class SensorReader(ABC):
    """Abstract interface for membrane sensor acquisition backends."""

    @abstractmethod
    def read(self) -> SensorFrame:
        """Return the next sensor frame."""


@dataclass
class SerialForceReader:
    """Reads live force values from a serial connection."""

    port: str = "COM3"
    baudrate: int = 57600
    timeout: float = 0.2

    def __post_init__(self):
        self._serial = None
        self._number_pattern = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self.last_raw_text = ""
        self.last_error = ""
        self.latest_force_value: float | None = None
        self.latest_force_timestamp: datetime | None = None

    def _ensure_connection(self):
        if self._serial is not None:
            return
        if serial is None:
            raise RuntimeError("pyserial is not installed.")
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        try:
            self._serial.reset_input_buffer()
        except AttributeError:
            pass

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._ensure_connection()
        self._thread = threading.Thread(target=self._read_loop, name="serial-force-reader", daemon=True)
        self._thread.start()

    def _parse_force_text(self, text: str) -> float | None:
        if not text:
            return None

        if text.upper().endswith("N"):
            text = text[:-1].strip()
            if not text:
                return None

        text = text.replace(",", ".")
        match = self._number_pattern.search(text)
        if match is None:
            return None
        return float(match.group(0))

    def _read_loop(self):
        while not self._stop_event.is_set():
            try:
                line = self._serial.readline()
                if not line:
                    continue

                text = line.decode("ascii", errors="ignore").strip()
                value = self._parse_force_text(text)
                with self._lock:
                    self.last_raw_text = text
                    if value is not None:
                        self.latest_force_value = value
                        self.latest_force_timestamp = datetime.utcnow()
                    self.last_error = ""
            except Exception as exc:
                with self._lock:
                    self.last_error = str(exc)
                break

    def get_latest_force(self) -> float | None:
        with self._lock:
            return self.latest_force_value

    def get_last_raw_text(self) -> str:
        with self._lock:
            return self.last_raw_text

    def get_last_error(self) -> str:
        with self._lock:
            return self.last_error

    def get_latest_force_timestamp(self) -> datetime | None:
        with self._lock:
            return self.latest_force_timestamp

    def close(self):
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def connection_info(self) -> str:
        if self._serial is None:
            return f"{self.port} geschlossen"
        return f"{self.port} offen @ {self.baudrate}"


@dataclass
class MockReader(SensorReader):
    """Generates synthetic sensor values for local development."""

    sensor_count: int = 8
    seed: int | None = None
    baseline: float = 0.0
    noise_std: float = 1.0
    source: str = "mock"

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def read(self) -> SensorFrame:
        sensors = self._rng.normal(
            loc=self.baseline,
            scale=self.noise_std,
            size=self.sensor_count,
        ).astype(float)
        return SensorFrame(
            sensors=sensors,
            timestamp=datetime.utcnow(),
            source=self.source,
            metadata={"reader": type(self).__name__},
        )


@dataclass
class CsvReplayReader(SensorReader):
    """Replays frames from an existing dataset without changing CSV schema."""

    csv_path: str
    loop: bool = False
    source: str = "csv_replay"

    def __post_init__(self):
        self._data = pd.read_csv(self.csv_path)
        missing_columns = [column for column in SENSOR_COLUMNS if column not in self._data.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"CSV is missing required sensor columns: {missing}")
        self._index = 0

    def read(self) -> SensorFrame:
        if self._index >= len(self._data):
            if not self.loop:
                raise StopIteration("Reached end of CSV replay dataset.")
            self._index = 0

        row = self._data.iloc[self._index]
        self._index += 1

        sensors = row[SENSOR_COLUMNS].to_numpy(dtype=float)
        metadata = {
            "reader": type(self).__name__,
            "row_index": self._index - 1,
        }
        for label in ("X", "Y", "F"):
            if label in row.index:
                metadata[label.lower()] = float(row[label])

        return SensorFrame(
            sensors=sensors,
            timestamp=datetime.utcnow(),
            source=self.source,
            metadata=metadata,
        )
