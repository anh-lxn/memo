from __future__ import annotations

import argparse
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QPushButton,
    QApplication,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

try:
    import pyqtgraph as pg
except ImportError:
    pg = None

from memo.acquisition.readers import Ads1115Reader, MockReader


SENSOR_LABELS = tuple(f"R{index}" for index in range(1, 9))
SENSOR_COLORS = (
    "#005f73",
    "#0a9396",
    "#94d2bd",
    "#ee9b00",
    "#ca6702",
    "#bb3e03",
    "#ae2012",
    "#9b2226",
)
FIXED_BASELINES = np.array([2.70, 2.22, 2.97, 1.77, 1.98, 2.55, 2.57, 2.58], dtype=float)


class SensorPlotWidget(QWidget):
    def __init__(self, sensor_index: int, history_seconds: float = 4.0, parent=None):
        super().__init__(parent)
        if pg is None:
            raise RuntimeError("pyqtgraph is not installed.")

        self.history_seconds = history_seconds
        self.sensor_index = sensor_index
        self.timestamps: deque[datetime] = deque()
        self.values: deque[float] = deque()
        self.baseline = 0.0

        self.plot_widget = pg.PlotWidget(background="w")
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setMouseEnabled(x=False, y=True)
        self.plot_widget.hideButtons()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.18)
        self.plot_widget.setLabel("bottom", "Zeit [s]")
        self.plot_widget.setLabel("left", "Spannung [V]")

        self.signal_curve = self.plot_widget.plot([], [], pen=pg.mkPen(SENSOR_COLORS[0], width=2))
        self.baseline_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen("#6c757d", width=2, style=Qt.DashLine),
        )
        self.plot_widget.addItem(self.baseline_line)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)
        self._update_plot()

    def set_baseline(self, baseline: float):
        self.baseline = float(baseline)
        self._update_plot()

    def append_value(self, timestamp: datetime, value: float):
        self.timestamps.append(timestamp)
        self.values.append(float(value))

        cutoff = timestamp.timestamp() - self.history_seconds
        while self.timestamps and self.timestamps[0].timestamp() < cutoff:
            self.timestamps.popleft()
            if self.values:
                self.values.popleft()

        self._update_plot()

    def _update_plot(self):
        sensor_label = SENSOR_LABELS[self.sensor_index]
        baseline = self.baseline
        self.plot_widget.setTitle(f"{sensor_label} Verlauf")
        self.signal_curve.setPen(pg.mkPen(SENSOR_COLORS[self.sensor_index], width=2))

        if not self.timestamps or not self.values:
            self.signal_curve.setData([], [])
            self.baseline_line.setValue(baseline)
            self.plot_widget.setXRange(-self.history_seconds, 0.0, padding=0.0)
            self.plot_widget.setYRange(0.0, 4.0, padding=0.0)
            return

        latest_ts = self.timestamps[-1].timestamp()
        x_values = np.array([ts.timestamp() - latest_ts for ts in self.timestamps], dtype=float)
        y_values = np.array(self.values, dtype=float)

        self.signal_curve.setData(x_values[: len(y_values)], y_values)
        self.baseline_line.setValue(baseline)
        self.plot_widget.setXRange(-self.history_seconds, 0.0, padding=0.0)
        self.plot_widget.setYRange(0.0, 4.0, padding=0.0)


class SensorChannelTestWindow(QMainWindow):
    def __init__(self, refresh_ms: int, source: str):
        super().__init__()
        if pg is None:
            raise RuntimeError("pyqtgraph is not installed.")

        self.reader, self._reader_notice = self._build_reader(source)
        self.refresh_ms = refresh_ms
        self.latest_values = np.zeros(8, dtype=float)
        self.baselines = FIXED_BASELINES.copy()

        self.setWindowTitle("MeMo Sensor Channel Test")
        self.resize(1280, 760)

        self.status_label = QLabel(f"Status: Bereit ({self._reader_notice})")
        self.mapping_label = QLabel("Mapping: ADS0 ch0-3 = R1-R4 | ADS1 ch0-3 = R5-R8")
        self.values_label = QLabel("R1-R8: -")
        self.selection_label = QLabel("Angezeigte Sensoren")
        self.baseline_label = QLabel("Baselines: -")
        self.sensor_checkboxes: list[QCheckBox] = []
        self.plot_widgets: dict[int, SensorPlotWidget] = {}
        self.plot_grid = QGridLayout()
        self.plot_grid.setSpacing(12)
        self.exit_button = QPushButton("Exit")

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._refresh)

        self.exit_button.clicked.connect(self.close)

        self._build_ui()
        self._apply_styles()
        for sensor_index in (0, 1):
            self.sensor_checkboxes[sensor_index].setChecked(True)
        self.timer.start(self.refresh_ms)

    def _build_reader(self, source: str):
        if source == "mock":
            return MockReader(), "MockReader aktiv"
        return Ads1115Reader(), "ADS1115 aktiv"

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        controls_layout.addWidget(self.selection_label)
        controls_layout.addWidget(self.baseline_label, stretch=1)

        checkbox_layout = QHBoxLayout()
        checkbox_layout.setSpacing(8)
        for sensor_index, sensor_label in enumerate(SENSOR_LABELS):
            checkbox = QCheckBox(sensor_label)
            checkbox.toggled.connect(
                lambda checked, idx=sensor_index: self._toggle_sensor_plot(idx, checked)
            )
            self.sensor_checkboxes.append(checkbox)
            checkbox_layout.addWidget(checkbox)

        layout.addWidget(self.status_label)
        layout.addWidget(self.mapping_label)
        layout.addWidget(self.values_label)
        layout.addLayout(controls_layout)
        layout.addLayout(checkbox_layout)
        layout.addLayout(self.plot_grid, stretch=1)
        layout.addWidget(self.exit_button)

    def _apply_styles(self):
        self.status_label.setStyleSheet("font-size: 16px; color: #444444;")
        self.mapping_label.setStyleSheet("font-size: 14px; color: #5f6b7a;")
        self.values_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.selection_label.setStyleSheet("font-size: 14px; font-weight: 600;")
        self.exit_button.setFixedHeight(36)

    def _toggle_sensor_plot(self, sensor_index: int, checked: bool):
        if checked:
            plot_widget = SensorPlotWidget(sensor_index=sensor_index, history_seconds=4.0)
            plot_widget.set_baseline(float(self.baselines[sensor_index]))
            self.plot_widgets[sensor_index] = plot_widget
            self._refresh_plot_grid()
            return

        plot_widget = self.plot_widgets.pop(sensor_index, None)
        if plot_widget is not None:
            plot_widget.setParent(None)
            plot_widget.deleteLater()
            self._refresh_plot_grid()

    def _refresh_plot_grid(self):
        while self.plot_grid.count():
            item = self.plot_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        for position, sensor_index in enumerate(sorted(self.plot_widgets)):
            row = position // 2
            column = position % 2
            self.plot_grid.addWidget(self.plot_widgets[sensor_index], row, column)

    def _update_baseline_label(self):
        selected_indices = sorted(self.plot_widgets)
        if not selected_indices:
            self.baseline_label.setText("Baselines: -")
            return
        label_text = " | ".join(
            f"{SENSOR_LABELS[index]}={float(self.baselines[index]):.4f} V"
            for index in selected_indices
        )
        self.baseline_label.setText(f"Baselines: {label_text}")

    def _refresh(self):
        try:
            frame = self.reader.read()
        except Exception as exc:
            self.status_label.setText(f"Status: Fehler beim Lesen ({exc})")
            return

        values = np.asarray(frame.sensors, dtype=float)[:8]
        self.latest_values = values
        self.status_label.setText(f"Status: Live-Daten von {frame.source} @ {self.refresh_ms} ms")
        self.values_label.setText(
            "R1-R8: " + " | ".join(f"{label}={value:.4f}" for label, value in zip(SENSOR_LABELS, values))
        )
        self._update_baseline_label()
        for sensor_index, plot_widget in self.plot_widgets.items():
            plot_widget.append_value(frame.timestamp, float(values[sensor_index]))

    def closeEvent(self, event: QCloseEvent):
        if self.timer.isActive():
            self.timer.stop()
        event.accept()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Live test for ADS1115 sensor channels")
    parser.add_argument("--refresh-ms", type=int, default=15, help="UI refresh interval in milliseconds")
    parser.add_argument(
        "--source",
        choices=("ads1115", "mock"),
        default="ads1115",
        help="Datenquelle fuer den Test. Standard ist echter ADS1115-Readout ueber 8 Kanaele.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    if pg is None:
        print("pyqtgraph is not installed. Please install it first.")
        return 1

    args = parse_args(argv)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = SensorChannelTestWindow(refresh_ms=args.refresh_ms, source=args.source)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
