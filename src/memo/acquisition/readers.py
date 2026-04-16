from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import os
import re
import sys
import threading

import numpy as np
import pandas as pd

from memo.types import SensorFrame

try:
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
except (ImportError, NotImplementedError):  # pragma: no cover - depends on Raspberry Pi environment
    board = None
    busio = None
    ADS = None
    AnalogIn = None

try:
    import serial
    from serial import SerialException
except ImportError:  # pragma: no cover - depends on local environment
    serial = None

    class SerialException(Exception):
        pass


SENSOR_COLUMNS = [f"Sensor R{i}" for i in range(1, 9)]

FORCE_SERIAL_WINDOWS_FALLBACKS = ("COM6", "COM5", "COM4", "COM3")
FORCE_SERIAL_LINUX_FALLBACKS = (
    "/dev/ttyUSB0",
    "/dev/ttyUSB1",
    "/dev/ttyACM0",
    "/dev/ttyACM1",
    "/dev/serial0",
    "/dev/ttyAMA0",
    "/dev/ttyS0",
)


def _looks_like_force_port(description: str, hwid: str, device: str) -> bool:
    text = " ".join(part for part in (description, hwid, device) if part).lower()
    preferred_tokens = (
        "usb",
        "serial",
        "uart",
        "ch340",
        "cp210",
        "acm",
        "ttyusb",
        "arduino",
        "ttyama",
        "serial0",
        "tty",
    )
    return any(token in text for token in preferred_tokens)


def _list_detected_serial_ports() -> list[str]:
    if serial is None:
        return []

    try:
        from serial.tools import list_ports
    except Exception:
        return []

    preferred_ports: list[str] = []
    fallback_ports: list[str] = []
    seen_devices: set[str] = set()

    for port_info in list_ports.comports():
        device = getattr(port_info, "device", "") or ""
        description = getattr(port_info, "description", "") or ""
        hwid = getattr(port_info, "hwid", "") or ""
        manufacturer = getattr(port_info, "manufacturer", "") or ""
        product = getattr(port_info, "product", "") or ""
        interface = getattr(port_info, "interface", "") or ""
        if not device or device in seen_devices:
            continue
        seen_devices.add(device)

        port_text = " ".join((description, hwid, manufacturer, product, interface, device))
        if _looks_like_force_port(port_text, "", ""):
            preferred_ports.append(device)
        else:
            fallback_ports.append(device)

    return preferred_ports + fallback_ports


def resolve_force_serial_port_candidates(port: str | None = None) -> list[str]:
    env_port = os.environ.get("MEMO_FORCE_PORT", "").strip()
    candidates: list[str] = []

    for candidate in (port, env_port):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in _list_detected_serial_ports():
        if candidate not in candidates:
            candidates.append(candidate)

    fallback_ports = FORCE_SERIAL_WINDOWS_FALLBACKS if sys.platform.startswith("win") else FORCE_SERIAL_LINUX_FALLBACKS
    for candidate in fallback_ports:
        if candidate not in candidates:
            candidates.append(candidate)

    return candidates


def resolve_force_serial_port(port: str | None = None) -> str:
    """Return the requested force port or a platform-appropriate default."""

    return resolve_force_serial_port_candidates(port)[0]


class SensorReader(ABC):
    """Abstract interface for membrane sensor acquisition backends."""

    @abstractmethod
    def read(self) -> SensorFrame:
        """Return the next sensor frame."""


@dataclass
class UnavailableSensorReader(SensorReader):
    """Allows the UI to start even when live hardware is unavailable."""

    reason: str
    source: str = "unavailable"

    def read(self) -> SensorFrame:
        raise RuntimeError(self.reason)


@dataclass
class Ads1115Reader(SensorReader):
    """Reads eight membrane sensor voltages from two ADS1115 converters."""

    address_ads0: int = 0x48
    address_ads1: int = 0x49
    source: str = "ads1115"

    def __post_init__(self):
        if board is None or busio is None or ADS is None or AnalogIn is None:
            raise RuntimeError(
                "Adafruit ADS1115 dependencies are not installed."
            )

        self._i2c_bus = busio.I2C(board.SCL, board.SDA)
        self._ads0 = ADS.ADS1115(self._i2c_bus, address=self.address_ads0)
        self._ads1 = ADS.ADS1115(self._i2c_bus, address=self.address_ads1)

        # Use integer channel indices as requested instead of ADS.P0 ... ADS.P3.
        self._channels = [
            AnalogIn(self._ads0, 0),
            AnalogIn(self._ads0, 1),
            AnalogIn(self._ads0, 2),
            AnalogIn(self._ads0, 3),
            AnalogIn(self._ads1, 0),
            AnalogIn(self._ads1, 1),
            AnalogIn(self._ads1, 2),
            AnalogIn(self._ads1, 3),
        ]

    def read(self) -> SensorFrame:
        sensor_r1 = self._channels[0].voltage
        sensor_r2 = self._channels[1].voltage
        sensor_r3 = self._channels[2].voltage
        sensor_r4 = self._channels[3].voltage
        sensor_r5 = self._channels[4].voltage
        sensor_r6 = self._channels[5].voltage
        sensor_r7 = self._channels[6].voltage
        sensor_r8 = self._channels[7].voltage

        sensors = np.array(
            [
                sensor_r1,
                sensor_r2,
                sensor_r3,
                sensor_r4,
                sensor_r5,
                sensor_r6,
                sensor_r7,
                sensor_r8,
            ],
            dtype=float,
        )
        return SensorFrame(
            sensors=sensors,
            timestamp=datetime.utcnow(),
            source=self.source,
            metadata={
                "reader": type(self).__name__,
                "ads0_address": hex(self.address_ads0),
                "ads1_address": hex(self.address_ads1),
            },
        )


@dataclass
class SerialForceReader:
    """Reads live force values from a serial connection."""

    port: str | None = None
    baudrate: int = 57600
    timeout: float = 0.2

    def __post_init__(self):
        self.port = resolve_force_serial_port(self.port)
        self._port_candidates = resolve_force_serial_port_candidates(self.port)
        self._serial = None
        self._number_pattern = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._sample_id = 0
        self._sample_history = deque(maxlen=2000)
        self.last_raw_text = ""
        self.last_error = ""
        self.latest_force_value: float | None = None
        self.latest_force_timestamp: datetime | None = None

    def _ensure_connection(self):
        if self._serial is not None:
            return
        if serial is None:
            raise RuntimeError("pyserial is not installed.")
        last_error: Exception | None = None
        for candidate_port in resolve_force_serial_port_candidates(self.port):
            try:
                trial_serial = serial.Serial(candidate_port, self.baudrate, timeout=self.timeout)
                try:
                    trial_serial.reset_input_buffer()
                except AttributeError:
                    pass
                self._serial = trial_serial
                self.port = candidate_port
                self._port_candidates = resolve_force_serial_port_candidates(candidate_port)
                return
            except Exception as exc:
                last_error = exc

        if last_error is None:
            raise RuntimeError("No serial ports found for force sensor auto-detection.")
        raise RuntimeError(
            f"Could not open any detected force sensor port. Tried: {', '.join(resolve_force_serial_port_candidates(self.port))}. "
            f"Last error: {last_error}"
        )

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

        text = text.replace(",", ".")
        if text.upper().endswith("N"):
            text = text[:-1].strip()
            if not text:
                return None

        matches = self._number_pattern.findall(text)
        if not matches:
            return None
        return float(matches[-1])

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
                        timestamp = datetime.utcnow()
                        self.latest_force_value = value
                        self.latest_force_timestamp = timestamp
                        self._sample_id += 1
                        self._sample_history.append((self._sample_id, timestamp, value))
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

    def get_samples_since(self, last_sample_id: int) -> list[tuple[int, datetime, float]]:
        with self._lock:
            return [sample for sample in self._sample_history if sample[0] > last_sample_id]

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
class SerialSensorReader(SensorReader):
    """Reads membrane sensor values from a serial stream."""

    port: str = "COM3"
    baudrate: int = 57600
    timeout: float = 0.2
    sensor_count: int = 8
    source: str = "serial"

    def __post_init__(self):
        if serial is None:
            raise RuntimeError("pyserial is not installed.")
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        self._number_pattern = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
        try:
            self._serial.reset_input_buffer()
        except AttributeError:
            pass

    def _parse_sensor_text(self, text: str) -> np.ndarray:
        matches = self._number_pattern.findall(text.replace(",", "."))
        if len(matches) < self.sensor_count:
            raise ValueError(
                f"Expected at least {self.sensor_count} sensor values, received {len(matches)} from: {text!r}"
            )
        values = np.asarray(matches[: self.sensor_count], dtype=float)
        return values

    def read(self) -> SensorFrame:
        while True:
            line = self._serial.readline()
            if not line:
                continue

            text = line.decode("ascii", errors="ignore").strip()
            if not text:
                continue

            sensors = self._parse_sensor_text(text)
            return SensorFrame(
                sensors=sensors,
                timestamp=datetime.utcnow(),
                source=self.source,
                metadata={
                    "reader": type(self).__name__,
                    "port": self.port,
                    "baudrate": self.baudrate,
                    "raw_text": text,
                },
            )

    def close(self):
        if self._serial is not None:
            self._serial.close()
            self._serial = None


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
