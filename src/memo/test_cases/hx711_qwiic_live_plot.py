from __future__ import annotations

import re
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

try:
    import pyqtgraph as pg
except ImportError:
    pg = None

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # pragma: no cover - depends on local environment
    serial = None
    list_ports = None

from memo.test_cases.hx711_qwiic_minimal import SMBus, read_raw, set_gain, wake


DEFAULT_ADDRESSES = tuple(range(0x30, 0x38))
EMPIRICAL_MV_PER_COUNT = 2.647355617e-6
EMPIRICAL_MV_OFFSET = -0.146334057
SERIAL_PAIR_PATTERN = re.compile(r"0x([0-7][0-9a-fA-F])\s*[=:]\s*(-?\d+)")
SERIAL_ERROR_PATTERN = re.compile(r"0x([0-7][0-9a-fA-F])\s*[=:]\s*ERR(\d*)", re.IGNORECASE)
SCANNER_LINE_MARKERS = (
    "arduino i2c scanner ready",
    "scanning i2c bus",
    "i2c device found",
    "scan done",
    "found ",
)

# ---------------------------------------------------------------------------
# Live-Plot-Konfiguration
# ---------------------------------------------------------------------------
# Hier umstellen, statt beim Start lange Kommandozeilenargumente zu tippen.
#
# "serial": Arduino liest die HX711 per I2C und sendet die Werte per USB an den Pi.
# "i2c":    Raspberry Pi liest die HX711 direkt ueber seinen I2C-Bus.
CONFIG_SOURCE = "serial"

# Fuer Raspberry Pi mit Arduino per USB ist meistens "/dev/ttyACM0" oder
# "/dev/ttyUSB0" richtig. None versucht eine automatische Erkennung.
CONFIG_SERIAL_PORT = "/dev/ttyACM0"
CONFIG_SERIAL_BAUDRATE = 230400
CONFIG_SERIAL_TIMEOUT = 0.02
CONFIG_SERIAL_DEBUG = False

# Fuer direkte Raspberry-Pi-I2C-Nutzung.
CONFIG_I2C_BUS = 1
CONFIG_HX711_GAIN = 64

# Erstmal nur eine Adresse plotten. Mehrere Adressen und Saturation-Werte wie
# 8388607 machen die Y-Skalierung schnell unlesbar.
CONFIG_ADDRESSES = [0x31]

# Anzeige.
CONFIG_DISPLAY_GAIN = 128.0
CONFIG_REFRESH_MS = 100
CONFIG_HISTORY_SECONDS = 5.0


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value, 0)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


def env_port(name: str, default: str | None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    if value.lower() in {"", "none", "auto"}:
        return None
    return value


def env_addresses(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return list(default)

    addresses = []
    for token in value.replace(";", ",").split(","):
        token = token.strip()
        if token:
            addresses.append(int(token, 0))
    return addresses or list(default)


def counts_to_bridge_voltage_mv(raw_value: int) -> float:
    """Convert raw counts to measured bridge voltage in mV using empirical calibration."""

    return (float(raw_value) * EMPIRICAL_MV_PER_COUNT) + EMPIRICAL_MV_OFFSET


def auto_detect_serial_port() -> str | None:
    if list_ports is None:
        return None

    preferred_terms = ("arduino", "ch340", "usb serial", "usb-serial", "ttyacm", "ttyusb")
    candidates = []
    for port in list_ports.comports():
        text = f"{port.device} {port.description} {port.manufacturer}".lower()
        if any(term in text for term in preferred_terms):
            candidates.append(port.device)

    if candidates:
        return candidates[0]
    return None


class Hx711Source:
    def open(self):
        raise NotImplementedError

    def read_raw_values(self) -> dict[int, int]:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def connection_info(self) -> str:
        raise NotImplementedError


class I2CHx711Source(Hx711Source):
    def __init__(self, bus_number: int, addresses: list[int], gain: int):
        self.bus_number = int(bus_number)
        self.addresses = list(addresses)
        self.gain = int(gain)
        self.bus = None

    def open(self):
        self.bus = SMBus(self.bus_number)
        for address in self.addresses:
            wake(self.bus, address)
            time.sleep(0.02)
            set_gain(self.bus, address, self.gain)
            time.sleep(0.02)

    def read_raw_values(self) -> dict[int, int]:
        if self.bus is None:
            raise RuntimeError("I2C bus ist nicht geoeffnet.")

        values: dict[int, int] = {}
        for address in self.addresses:
            values[address] = read_raw(self.bus, address)
        return values

    def close(self):
        if self.bus is not None:
            self.bus.close()
            self.bus = None

    def connection_info(self) -> str:
        address_text = ", ".join(f"0x{address:02X}" for address in self.addresses)
        return f"I2C bus={self.bus_number}, addr={address_text}, gain={self.gain}"


class SerialHx711Source(Hx711Source):
    def __init__(self, port: str | None, baudrate: int, timeout: float, addresses: list[int], debug: bool = False):
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.addresses = list(addresses)
        self.debug = bool(debug)
        self.connection = None
        self.last_line = ""
        self.last_parse_error = ""
        self.latest_values: dict[int, int] = {}
        self._line_buffer = b""

    def open(self):
        if serial is None:
            raise RuntimeError("pyserial ist nicht installiert.")

        port = self.port or auto_detect_serial_port()
        if not port:
            raise RuntimeError("Kein Arduino-Serial-Port gefunden. Bitte --port angeben.")

        self.connection = serial.Serial(port=port, baudrate=self.baudrate, timeout=0)
        self.port = port
        time.sleep(2.0)
        try:
            self.connection.reset_input_buffer()
        except AttributeError:
            pass

    def read_raw_values(self) -> dict[int, int]:
        if self.connection is None:
            raise RuntimeError("Serial-Verbindung ist nicht geoeffnet.")

        waiting = getattr(self.connection, "in_waiting", 0)
        if waiting:
            self._line_buffer += self.connection.read(waiting)

        complete_lines = []
        while b"\n" in self._line_buffer:
            raw_line, self._line_buffer = self._line_buffer.split(b"\n", 1)
            complete_lines.append(raw_line)

        read_any_line = bool(complete_lines)
        max_lines_per_refresh = 25
        for raw_line in complete_lines[-max_lines_per_refresh:]:
            text = raw_line.decode("ascii", errors="ignore").strip()
            self.last_line = text
            if self.debug:
                print(f"Arduino serial: {text}")

            if not text.startswith("HX711"):
                lowered_text = text.lower()
                if any(marker in lowered_text for marker in SCANNER_LINE_MARKERS):
                    self.last_parse_error = (
                        "Arduino sendet I2C-Scanner-Ausgabe statt HX711-Rohwerten. "
                        "Bitte den Arduino-Bridge-Sketch arduino_hx711_qwiic_serial_bridge.ino flashen"
                    )
                else:
                    self.last_parse_error = "Zeile startet nicht mit HX711"
                continue

            values: dict[int, int] = {}
            for match in SERIAL_PAIR_PATTERN.finditer(text):
                address = int(match.group(1), 16)
                if address in self.addresses:
                    values[address] = int(match.group(2))

            if values:
                self.last_parse_error = ""
                self.latest_values = values
                continue

            errors = []
            for match in SERIAL_ERROR_PATTERN.finditer(text):
                address = int(match.group(1), 16)
                if address in self.addresses:
                    errors.append(f"0x{address:02X}=ERR{match.group(2)}")
            if errors:
                self.last_parse_error = "Arduino meldet I2C/HX711-Fehler: " + ", ".join(errors)
            else:
                self.last_parse_error = "HX711-Zeile ohne parsebare Zahlenwerte"

        if self.latest_values:
            return dict(self.latest_values)

        if read_any_line:
            detail = self.last_parse_error or "keine passende HX711-Zeile empfangen"
            if self.last_line:
                raise TimeoutError(f"{detail}. Letzte Arduino-Zeile: {self.last_line!r}")
        raise TimeoutError("Warte auf erste HX711-Zeile vom Arduino.")

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def connection_info(self) -> str:
        return f"Serial {self.port} @ {self.baudrate}"


class MultiVoltagePlotWidget(QWidget):
    def __init__(self, addresses: list[int], history_seconds: float = 5.0, parent=None):
        super().__init__(parent)
        if pg is None:
            raise RuntimeError("pyqtgraph is not installed.")

        self.addresses = list(addresses)
        self.history_seconds = float(history_seconds)
        self.timestamps: deque[datetime] = deque()
        self.values_by_address = {address: deque() for address in self.addresses}

        self.plot_widget = pg.PlotWidget(background="w")
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setMouseEnabled(x=False, y=True)
        self.plot_widget.hideButtons()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.18)
        self.plot_widget.setLabel("bottom", "Zeit [s]")
        self.plot_widget.setLabel("left", "HX711 Raw Counts")
        address_text = ", ".join(f"0x{address:02X}" for address in self.addresses)
        self.plot_widget.setTitle(f"HX711 Raw Counts {address_text}")

        colors = ("#0f8b6d", "#bc4749", "#3a86ff", "#ffbe0b", "#8338ec", "#fb5607", "#2a9d8f", "#6c757d")
        self.curves = {}
        for index, address in enumerate(self.addresses):
            color = colors[index % len(colors)]
            self.curves[address] = self.plot_widget.plot(
                [],
                [],
                pen=pg.mkPen(color, width=2),
                name=f"0x{address:02X}",
            )

        self.zero_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen("#6c757d", width=1, style=Qt.DashLine),
        )
        self.plot_widget.addItem(self.zero_line)
        self.plot_widget.addLegend(offset=(10, 10))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)

    def append_values(self, timestamp: datetime, values: dict[int, float]):
        self.timestamps.append(timestamp)
        for address in self.addresses:
            last_value = self.values_by_address[address][-1] if self.values_by_address[address] else np.nan
            self.values_by_address[address].append(float(values.get(address, last_value)))

        cutoff = timestamp.timestamp() - self.history_seconds
        while self.timestamps and self.timestamps[0].timestamp() < cutoff:
            self.timestamps.popleft()
            for address in self.addresses:
                if self.values_by_address[address]:
                    self.values_by_address[address].popleft()

        self._update_plot()

    def _update_plot(self):
        if not self.timestamps:
            self.plot_widget.setXRange(-self.history_seconds, 0.0, padding=0.0)
            self.plot_widget.setYRange(-1.0, 1.0, padding=0.1)
            return

        latest_ts = self.timestamps[-1].timestamp()
        x_values = np.array([ts.timestamp() - latest_ts for ts in self.timestamps], dtype=float)
        all_values = []

        for address in self.addresses:
            y_values = np.array(self.values_by_address[address], dtype=float)
            self.curves[address].setData(x_values, y_values)
            finite_values = y_values[np.isfinite(y_values)]
            if finite_values.size:
                all_values.append(finite_values)

        self.plot_widget.setXRange(-self.history_seconds, 0.0, padding=0.0)
        if not all_values:
            return

        combined = np.concatenate(all_values)
        min_y = float(np.min(combined))
        max_y = float(np.max(combined))
        if np.isclose(min_y, max_y):
            span = max(abs(min_y), 1e-6) * 0.2 + 1e-6
            min_y -= span
            max_y += span
        self.plot_widget.setYRange(min_y, max_y, padding=0.1)


class Hx711LivePlotWindow(QMainWindow):
    def __init__(
        self,
        source: Hx711Source,
        addresses: list[int],
        refresh_ms: int,
        display_gain: float,
        history_seconds: float,
    ):
        super().__init__()
        if pg is None:
            raise RuntimeError("pyqtgraph is not installed.")

        self.source = source
        self.addresses = list(addresses)
        self.refresh_ms = int(refresh_ms)
        self.display_gain = float(display_gain)

        self.setWindowTitle("MeMo HX711 Qwiic Live Plot")
        self.resize(1200, 760)

        self.status_label = QLabel("Status: Verbinde...")
        self.raw_label = QLabel("Raw: -")
        self.bridge_voltage_label = QLabel("mV-Schaetzung: -")
        self.amplified_voltage_label = QLabel("Verstaerkte mV-Schaetzung: -")
        self.plot_widget = MultiVoltagePlotWidget(addresses=self.addresses, history_seconds=history_seconds)
        self.exit_button = QPushButton("Exit")

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._refresh)
        self.exit_button.clicked.connect(self.close)

        self._build_ui()
        self._apply_styles()
        self._start_reader()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        layout.addWidget(self.status_label)
        layout.addWidget(self.raw_label)
        layout.addWidget(self.bridge_voltage_label)
        layout.addWidget(self.amplified_voltage_label)
        layout.addWidget(self.plot_widget, stretch=1)
        layout.addWidget(self.exit_button)

    def _apply_styles(self):
        self.status_label.setStyleSheet("font-size: 16px; color: #444444;")
        self.raw_label.setStyleSheet("font-size: 18px; font-weight: 700;")
        self.bridge_voltage_label.setStyleSheet("font-size: 18px; font-weight: 700;")
        self.amplified_voltage_label.setStyleSheet("font-size: 18px; font-weight: 700;")
        self.exit_button.setFixedHeight(36)

    def _start_reader(self):
        try:
            self.source.open()
        except Exception as exc:
            self.status_label.setText(f"Status: Verbindung fehlgeschlagen ({exc})")
            return

        self.status_label.setText(
            f"Status: Verbunden ueber {self.source.connection_info()}, Anzeige x{self.display_gain:g}"
        )
        self.timer.start(self.refresh_ms)

    def _refresh(self):
        try:
            raw_values = self.source.read_raw_values()
        except Exception as exc:
            self.status_label.setText(f"Status: Lesefehler ({exc})")
            return

        timestamp = datetime.now()
        estimated_bridge_values = {
            address: counts_to_bridge_voltage_mv(raw_value)
            for address, raw_value in raw_values.items()
        }
        estimated_amplified_values = {
            address: bridge_value * self.display_gain
            for address, bridge_value in estimated_bridge_values.items()
        }

        raw_text = " | ".join(
            f"0x{address:02X}: {raw_values[address]}"
            for address in self.addresses
            if address in raw_values
        )
        bridge_text = " | ".join(
            f"0x{address:02X}: {estimated_bridge_values[address]:.2f} mV"
            for address in self.addresses
            if address in estimated_bridge_values
        )
        amplified_text = " | ".join(
            f"0x{address:02X}: {estimated_amplified_values[address]:.2f} mV"
            for address in self.addresses
            if address in estimated_amplified_values
        )

        self.raw_label.setText(f"Raw: {raw_text or '-'}")
        self.bridge_voltage_label.setText(f"mV-Schaetzung aus alter Kalibrierung: {bridge_text or '-'}")
        self.amplified_voltage_label.setText(f"Verstaerkte mV-Schaetzung x{self.display_gain:g}: {amplified_text or '-'}")
        self.plot_widget.append_values(timestamp, raw_values)

    def closeEvent(self, event: QCloseEvent):
        if self.timer.isActive():
            self.timer.stop()
        self.source.close()
        event.accept()


def main(argv=None) -> int:
    if pg is None:
        print("pyqtgraph is not installed. Please install it first.")
        return 1

    source_mode = os.environ.get("MEMO_HX711_SOURCE", CONFIG_SOURCE).lower().strip()
    addresses = env_addresses("MEMO_HX711_ADDRESSES", CONFIG_ADDRESSES)
    if source_mode == "serial":
        source = SerialHx711Source(
            port=env_port("MEMO_HX711_SERIAL_PORT", CONFIG_SERIAL_PORT),
            baudrate=env_int("MEMO_HX711_SERIAL_BAUDRATE", CONFIG_SERIAL_BAUDRATE),
            timeout=env_float("MEMO_HX711_SERIAL_TIMEOUT", CONFIG_SERIAL_TIMEOUT),
            addresses=addresses,
            debug=env_bool("MEMO_HX711_SERIAL_DEBUG", CONFIG_SERIAL_DEBUG),
        )
    elif source_mode == "i2c":
        source = I2CHx711Source(
            bus_number=env_int("MEMO_HX711_I2C_BUS", CONFIG_I2C_BUS),
            addresses=addresses,
            gain=env_int("MEMO_HX711_GAIN", CONFIG_HX711_GAIN),
        )
    else:
        print(f"Ungueltige CONFIG_SOURCE: {CONFIG_SOURCE!r}. Erlaubt: 'serial' oder 'i2c'.")
        return 1

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = Hx711LivePlotWindow(
        source=source,
        addresses=addresses,
        refresh_ms=CONFIG_REFRESH_MS,
        display_gain=CONFIG_DISPLAY_GAIN,
        history_seconds=CONFIG_HISTORY_SECONDS,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
