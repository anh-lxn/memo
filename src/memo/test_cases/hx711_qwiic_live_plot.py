from __future__ import annotations

import argparse
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

from memo.test_cases.hx711_qwiic_minimal import SMBus, read_raw, set_gain, wake


EMPIRICAL_MV_PER_COUNT = 2.647355617e-6
EMPIRICAL_MV_OFFSET = -0.146334057


def counts_to_bridge_voltage_mv(raw_value: int) -> float:
    """Convert raw counts to measured bridge voltage in mV using empirical calibration."""

    return (float(raw_value) * EMPIRICAL_MV_PER_COUNT) + EMPIRICAL_MV_OFFSET


class VoltagePlotWidget(QWidget):
    def __init__(self, history_seconds: float = 5.0, display_gain: float = 128.0, parent=None):
        super().__init__(parent)
        if pg is None:
            raise RuntimeError("pyqtgraph is not installed.")

        self.history_seconds = float(history_seconds)
        self.display_gain = float(display_gain)
        self.timestamps: deque[datetime] = deque()
        self.bridge_values: deque[float] = deque()
        self.amplified_values: deque[float] = deque()

        self.plot_widget = pg.PlotWidget(background="w")
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setMouseEnabled(x=False, y=True)
        self.plot_widget.hideButtons()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.18)
        self.plot_widget.setLabel("bottom", "Zeit [s]")
        self.plot_widget.setLabel("left", "Spannung [mV]")
        self.plot_widget.setTitle("HX711 Brueckenspannung")

        self.bridge_curve = self.plot_widget.plot([], [], pen=pg.mkPen("#0f8b6d", width=2), name="Bruecke")
        self.amplified_curve = self.plot_widget.plot([], [], pen=pg.mkPen("#bc4749", width=2), name="Verstaerkt")
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

    def append_value(self, timestamp: datetime, bridge_value: float):
        self.timestamps.append(timestamp)
        self.bridge_values.append(float(bridge_value))
        self.amplified_values.append(float(bridge_value) * self.display_gain)

        cutoff = timestamp.timestamp() - self.history_seconds
        while self.timestamps and self.timestamps[0].timestamp() < cutoff:
            self.timestamps.popleft()
            self.bridge_values.popleft()
            self.amplified_values.popleft()

        self._update_plot()

    def _update_plot(self):
        if not self.timestamps:
            self.bridge_curve.setData([], [])
            self.amplified_curve.setData([], [])
            self.plot_widget.setXRange(-self.history_seconds, 0.0, padding=0.0)
            self.plot_widget.setYRange(-0.01, 0.01, padding=0.1)
            return

        latest_ts = self.timestamps[-1].timestamp()
        x_values = np.array([ts.timestamp() - latest_ts for ts in self.timestamps], dtype=float)
        bridge_values = np.array(self.bridge_values, dtype=float)
        amplified_values = np.array(self.amplified_values, dtype=float)

        self.bridge_curve.setData(x_values, bridge_values)
        self.amplified_curve.setData(x_values, amplified_values)
        self.plot_widget.setXRange(-self.history_seconds, 0.0, padding=0.0)

        all_values = np.concatenate((bridge_values, amplified_values))
        min_y = float(np.min(all_values))
        max_y = float(np.max(all_values))
        if np.isclose(min_y, max_y):
            span = max(abs(min_y), 1e-6) * 0.2 + 1e-6
            min_y -= span
            max_y += span
        self.plot_widget.setYRange(min_y, max_y, padding=0.1)


class Hx711LivePlotWindow(QMainWindow):
    def __init__(self, bus_number: int, address: int, gain: int, avdd: float, refresh_ms: int, display_gain: float):
        super().__init__()
        if pg is None:
            raise RuntimeError("pyqtgraph is not installed.")

        self.bus_number = int(bus_number)
        self.address = int(address)
        self.gain = int(gain)
        self.avdd = float(avdd)
        self.refresh_ms = int(refresh_ms)
        self.display_gain = float(display_gain)
        self.bus = None

        self.setWindowTitle("MeMo HX711 Qwiic Live Plot")
        self.resize(1100, 700)

        self.status_label = QLabel("Status: Verbinde...")
        self.raw_label = QLabel("Raw: -")
        self.bridge_voltage_label = QLabel("Brueckenspannung: -")
        self.amplified_voltage_label = QLabel("Verstaerkte Spannung: -")
        self.plot_widget = VoltagePlotWidget(history_seconds=5.0, display_gain=self.display_gain)
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
        self.raw_label.setStyleSheet("font-size: 20px; font-weight: 700;")
        self.bridge_voltage_label.setStyleSheet("font-size: 20px; font-weight: 700;")
        self.amplified_voltage_label.setStyleSheet("font-size: 20px; font-weight: 700;")
        self.exit_button.setFixedHeight(36)

    def _start_reader(self):
        try:
            self.bus = SMBus(self.bus_number)
            wake(self.bus, self.address)
            time.sleep(0.05)
            set_gain(self.bus, self.address, self.gain)
            time.sleep(0.2)
        except Exception as exc:
            self.status_label.setText(f"Status: Verbindung fehlgeschlagen ({exc})")
            return

        self.status_label.setText(
            f"Status: Verbunden auf bus={self.bus_number}, addr=0x{self.address:02x}, gain={self.gain}, AVDD={self.avdd:.3f} V, Anzeige x{self.display_gain:g}"
        )
        self.timer.start(self.refresh_ms)

    def _refresh(self):
        if self.bus is None:
            return

        try:
            raw_value = read_raw(self.bus, self.address)
        except Exception as exc:
            self.status_label.setText(f"Status: Lesefehler ({exc})")
            return

        bridge_voltage_mv = counts_to_bridge_voltage_mv(raw_value)
        amplified_voltage_mv = bridge_voltage_mv * self.display_gain
        timestamp = datetime.now()

        self.raw_label.setText(f"Raw: {raw_value}")
        self.bridge_voltage_label.setText(f"Brueckenspannung: {bridge_voltage_mv:.2f} mV")
        self.amplified_voltage_label.setText(
            f"Verstaerkte Spannung x{self.display_gain:g}: {amplified_voltage_mv:.2f} mV"
        )
        self.plot_widget.append_value(timestamp, bridge_voltage_mv)

    def closeEvent(self, event: QCloseEvent):
        if self.timer.isActive():
            self.timer.stop()
        if self.bus is not None:
            self.bus.close()
            self.bus = None
        event.accept()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Live plot fuer Soldered HX711 easyC/Qwiic")
    parser.add_argument("--bus", type=int, default=1, help="I2C bus, Standard 1")
    parser.add_argument("--address", type=lambda value: int(value, 0), default=0x30, help="I2C-Adresse, Standard 0x30")
    parser.add_argument("--gain", type=int, choices=(128, 64, 32), default=64, help="HX711 gain")
    parser.add_argument("--avdd", type=float, default=3.3, help="Versorgungsspannung der Bruecke in Volt")
    parser.add_argument("--display-gain", type=float, default=128.0, help="Zusaetzlicher Anzeigeverstaerkungsfaktor")
    parser.add_argument("--refresh-ms", type=int, default=50, help="Plot-Update in Millisekunden")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    if pg is None:
        print("pyqtgraph is not installed. Please install it first.")
        return 1

    args = parse_args(argv)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = Hx711LivePlotWindow(
        bus_number=args.bus,
        address=args.address,
        gain=args.gain,
        avdd=args.avdd,
        refresh_ms=args.refresh_ms,
        display_gain=args.display_gain,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
