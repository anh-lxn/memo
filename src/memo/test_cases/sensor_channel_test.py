from __future__ import annotations

import argparse
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from memo.acquisition.readers import Ads1115Reader


class SensorHistoryPlot(QWidget):
    def __init__(self, history_seconds: float = 3.0, parent=None):
        super().__init__(parent)
        self.history_seconds = history_seconds
        self.timestamps: deque[datetime] = deque()
        self.sensor_history: list[deque[float]] = [deque() for _ in range(8)]
        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111)
        self._build_layout()
        self._redraw()

    def _build_layout(self):
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def append_frame(self, timestamp: datetime, sensors: np.ndarray):
        values = np.asarray(sensors, dtype=float)
        self.timestamps.append(timestamp)
        for index, value in enumerate(values[:8]):
            self.sensor_history[index].append(float(value))

        cutoff = timestamp.timestamp() - self.history_seconds
        while self.timestamps and self.timestamps[0].timestamp() < cutoff:
            self.timestamps.popleft()
            for history in self.sensor_history:
                if history:
                    history.popleft()

        self._redraw()

    def _redraw(self):
        self.axis.clear()
        self.axis.set_title("Sensorwerte ueber 3 Sekunden")
        self.axis.set_xlabel("Zeit seit letztem 3s-Fensterbeginn [s]")
        self.axis.set_ylabel("Sensorwert")
        self.axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        if not self.timestamps:
            self.axis.set_xlim(0.0, self.history_seconds)
            self.canvas.draw_idle()
            return

        start_time = self.timestamps[0].timestamp()
        x_values = np.array([timestamp.timestamp() - start_time for timestamp in self.timestamps], dtype=float)
        colors = ["#005f73", "#0a9396", "#94d2bd", "#ee9b00", "#ca6702", "#bb3e03", "#ae2012", "#9b2226"]
        all_values: list[float] = []

        for index, history in enumerate(self.sensor_history):
            if not history:
                continue
            y_values = np.array(history, dtype=float)
            all_values.extend(y_values.tolist())
            self.axis.plot(
                x_values[: len(y_values)],
                y_values,
                linewidth=1.6,
                color=colors[index],
                label=f"R{index + 1}",
            )

        self.axis.set_xlim(max(0.0, x_values[-1] - self.history_seconds), max(self.history_seconds, x_values[-1]))
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            padding = max(0.05, (y_max - y_min) * 0.15)
            self.axis.set_ylim(y_min - padding, y_max + padding)
        self.axis.legend(loc="upper left", ncol=4, fontsize=9)
        self.canvas.draw_idle()


class SensorChannelTestWindow(QMainWindow):
    def __init__(self, refresh_ms: int):
        super().__init__()
        self.reader = Ads1115Reader()
        self.refresh_ms = refresh_ms

        self.setWindowTitle("MeMo Sensor Channel Test")
        self.resize(1280, 760)

        self.status_label = QLabel("Status: Bereit")
        self.values_label = QLabel("R1-R8: -")
        self.plot = SensorHistoryPlot(history_seconds=3.0)
        self.exit_button = QPushButton("Exit")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.exit_button.clicked.connect(self.close)

        self._build_ui()
        self._apply_styles()
        self.timer.start(self.refresh_ms)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        layout.addWidget(self.status_label)
        layout.addWidget(self.values_label)
        layout.addWidget(self.plot, stretch=1)
        layout.addWidget(self.exit_button)

    def _apply_styles(self):
        self.status_label.setStyleSheet("font-size: 16px; color: #444444;")
        self.values_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.exit_button.setFixedHeight(36)

    def _refresh(self):
        try:
            frame = self.reader.read()
        except Exception as exc:
            self.status_label.setText(f"Status: Fehler beim Lesen ({exc})")
            return

        values = np.asarray(frame.sensors, dtype=float)
        self.status_label.setText(f"Status: Live-Daten von {frame.source}")
        self.values_label.setText(
            "R1-R8: " + " | ".join(f"R{index + 1}={value:.3f}" for index, value in enumerate(values[:8]))
        )
        self.plot.append_frame(frame.timestamp, values)

    def closeEvent(self, event: QCloseEvent):
        if self.timer.isActive():
            self.timer.stop()
        event.accept()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Live test for ADS1115 sensor channels")
    parser.add_argument("--refresh-ms", type=int, default=50, help="UI refresh interval in milliseconds")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = SensorChannelTestWindow(refresh_ms=args.refresh_ms)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
