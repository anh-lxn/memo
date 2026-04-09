from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from memo.visualization.plots import CalibrationStatusPanel, LiveSensorPlot


SENSOR_COUNT = 8
DEFAULT_SENSOR_VALUES = np.array([2.0] * SENSOR_COUNT, dtype=float)
DEFAULT_TOLERANCE = 0.05
SLIDER_MIN = 0
SLIDER_MAX = 4000


class SliderCalibrationTestWindow(QMainWindow):
    def __init__(self, refresh_ms: int, tolerance: float):
        super().__init__()
        self.refresh_ms = int(refresh_ms)
        self.tolerance = float(tolerance)
        self.baseline_values = DEFAULT_SENSOR_VALUES.copy()
        self.sensor_sliders: list[QSlider] = []
        self.sensor_value_labels: list[QLabel] = []

        self.setWindowTitle("MeMo Slider Calibration Test")
        self.resize(1500, 900)

        self.status_label = QLabel("Status: Bereit")
        self.live_values_label = QLabel("-")
        self.live_plot = LiveSensorPlot()
        self.calibration_panel = CalibrationStatusPanel(
            baseline_values=self.baseline_values,
            tolerance=self.tolerance,
        )
        self.set_baseline_button = QPushButton("Aktuelle Slider als Basis setzen")
        self.start_calibration_button = QPushButton("Kalibrierung starten")
        self.reset_button = QPushButton("Slider auf Basis setzen")
        self.exit_button = QPushButton("Beenden")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)

        self._build_ui()
        self._apply_styles()
        self._connect_signals()
        self._set_slider_values(self.baseline_values)
        self.timer.start(self.refresh_ms)
        self._refresh()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        top_card = QFrame()
        top_card.setObjectName("card")
        top_layout = QVBoxLayout(top_card)
        top_layout.setContentsMargins(14, 14, 14, 14)
        top_layout.setSpacing(10)
        top_layout.addWidget(self.status_label)
        top_layout.addWidget(self.live_values_label)
        layout.addWidget(top_card)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(14)

        left_column = QVBoxLayout()
        left_column.setSpacing(14)
        left_column.addWidget(self.live_plot, stretch=3)
        left_column.addWidget(self._build_slider_card(), stretch=4)

        content_layout.addLayout(left_column, stretch=3)
        content_layout.addWidget(self.calibration_panel, stretch=2)
        layout.addLayout(content_layout, stretch=1)

    def _build_slider_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        title = QLabel("Potentiometer Simulation")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        slider_grid = QGridLayout()
        slider_grid.setHorizontalSpacing(12)
        slider_grid.setVerticalSpacing(10)

        for index in range(SENSOR_COUNT):
            name_label = QLabel(f"Sensor R{index + 1}")
            value_label = QLabel("0.000 V")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(SLIDER_MIN, SLIDER_MAX)
            slider.setSingleStep(1)
            slider.setPageStep(20)

            self.sensor_sliders.append(slider)
            self.sensor_value_labels.append(value_label)

            slider_grid.addWidget(name_label, index, 0)
            slider_grid.addWidget(slider, index, 1)
            slider_grid.addWidget(value_label, index, 2)

        layout.addLayout(slider_grid)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        button_row.addWidget(self.set_baseline_button)
        button_row.addWidget(self.start_calibration_button)
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.exit_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        return card

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f3f5f7;
            }
            QFrame#card {
                background: white;
                border: 1px solid #d9e0e6;
                border-radius: 12px;
            }
            QLabel {
                color: #1f2a37;
            }
            QLabel#sectionTitle {
                font-size: 15px;
                font-weight: 700;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #c7d0d9;
                border-radius: 10px;
                padding: 8px 14px;
                font-weight: 600;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #f7fafc;
            }
            QSlider::groove:horizontal {
                border: 1px solid #c7d0d9;
                height: 8px;
                background: #eef2f5;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0f8b6d;
                border: 1px solid #0b6b54;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            """
        )
        self.status_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.live_values_label.setStyleSheet("font-size: 15px;")

    def _connect_signals(self):
        self.set_baseline_button.clicked.connect(self._capture_baseline_from_sliders)
        self.start_calibration_button.clicked.connect(self._start_calibration)
        self.reset_button.clicked.connect(self._reset_sliders_to_baseline)
        self.exit_button.clicked.connect(self.close)
        for slider in self.sensor_sliders:
            slider.valueChanged.connect(self._refresh)

    def _slider_values(self) -> np.ndarray:
        return np.array([slider.value() / 1000.0 for slider in self.sensor_sliders], dtype=float)

    def _set_slider_values(self, values: np.ndarray):
        for slider, value in zip(self.sensor_sliders, np.asarray(values, dtype=float)):
            slider.blockSignals(True)
            slider.setValue(int(round(float(value) * 1000.0)))
            slider.blockSignals(False)
        self._refresh()

    def _capture_baseline_from_sliders(self):
        self.baseline_values = self._slider_values()
        self._replace_calibration_panel()
        self.status_label.setText("Status: Neue Basiswerte aus den Slider-Positionen uebernommen.")

    def _start_calibration(self):
        self.calibration_panel.start_calibration()
        self.status_label.setText(
            f"Status: Kalibrierung aktiv. Ziel ist Basis +/- {self.tolerance:.3f} V pro Sensor."
        )
        self._refresh()

    def _reset_sliders_to_baseline(self):
        self._set_slider_values(self.baseline_values)
        self.status_label.setText("Status: Slider auf die aktuellen Basiswerte gesetzt.")

    def _replace_calibration_panel(self):
        parent_layout = self.centralWidget().layout().itemAt(1).layout()
        old_panel = self.calibration_panel
        self.calibration_panel = CalibrationStatusPanel(
            baseline_values=self.baseline_values,
            tolerance=self.tolerance,
        )
        parent_layout.replaceWidget(old_panel, self.calibration_panel)
        old_panel.deleteLater()
        self._refresh()

    def _refresh(self):
        values = self._slider_values()
        for value_label, value in zip(self.sensor_value_labels, values):
            value_label.setText(f"{value:.3f} V")

        self.live_plot.update_values(values)
        self.calibration_panel.set_live_values(values)
        self.live_values_label.setText(
            " | ".join(f"R{index + 1}={value:.3f} V" for index, value in enumerate(values))
        )

        if self.calibration_panel.calibration_active:
            if self.calibration_panel.all_calibrated():
                self.status_label.setText("Status: Alle Sensoren liegen im Kalibrierbereich.")
            else:
                self.status_label.setText("Status: Kalibrierung laeuft. Noch nicht alle Sensoren im Bereich.")

    def closeEvent(self, event: QCloseEvent):
        if self.timer.isActive():
            self.timer.stop()
        event.accept()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Slider-based calibration test for membrane sensors")
    parser.add_argument("--refresh-ms", type=int, default=50, help="UI refresh interval in milliseconds")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="Calibration tolerance in volts")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = SliderCalibrationTestWindow(
        refresh_ms=args.refresh_ms,
        tolerance=args.tolerance,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
