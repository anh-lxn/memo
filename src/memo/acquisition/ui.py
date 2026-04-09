from __future__ import annotations

import argparse
from collections import deque
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from memo.acquisition.readers import (
    Ads1115Reader,
    CsvReplayReader,
    MockReader,
    SerialException,
    SerialForceReader,
    SerialSensorReader,
    UnavailableSensorReader,
)
from memo.acquisition.recorder import CsvSampleRecorder, RawSampleFileWriter
from memo.types import LabeledSample
from memo.visualization.plots import CalibrationStatusPanel, LiveForcePlot, LiveSensorPlot, XYGridPlot


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'recorded_samples'
RAW_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'raw'
GRID_SPACING = 40
CORNER_MARKER_SIZE = 30
OFFSET = 50
MEMBRANE_SIDE_LENGTH = 450.0
MEMBRANE_DIAGONAL = MEMBRANE_SIDE_LENGTH * np.sqrt(2.0)
X_LIMITS = (-MEMBRANE_DIAGONAL / 2 - OFFSET, MEMBRANE_DIAGONAL / 2 + OFFSET)
Y_LIMITS = (-MEMBRANE_DIAGONAL / 2 - OFFSET, MEMBRANE_DIAGONAL / 2 + OFFSET)
REFRESH_MS = 20
FORCE_REFRESH_MS = 20
SENSOR_UI_REFRESH_MS = 120
CALIBRATION_SENSOR_UI_REFRESH_MS = 350
FORCE_SERIAL_PORT = 'COM3'
FORCE_SERIAL_BAUDRATE = 57600
FORCE_STREAM_TIMEOUT_S = 0.5
FORCE_TARGET_TOLERANCE_N = 0.5
FORCE_TARGET_HOLD_S = 5.0
BASELINE_SENSOR_VALUES = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000], dtype=float)
CALIBRATION_TOLERANCE = 0.050
DEBUG_SENSOR_MODE = True


def _to_utc_timestamp(value: datetime | None) -> float | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC).timestamp()
    return value.astimezone(UTC).timestamp()


class AcquisitionWindow(QMainWindow):
    def __init__(self, reader, output_dir: Path, force_port: str = FORCE_SERIAL_PORT, force_baudrate: int = FORCE_SERIAL_BAUDRATE):
        super().__init__()
        self.reader = reader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_writer = RawSampleFileWriter(RAW_OUTPUT_DIR, summary_output_dir=self.output_dir)
        self.force_reader = SerialForceReader(port=force_port, baudrate=force_baudrate)
        self.latest_frame = None
        self.latest_force_value: float | None = None
        self._force_reader_available = True
        self.is_sampling = False
        self.recorder: CsvSampleRecorder | None = None
        self.auto_capture_enabled = False
        self._auto_capture_rows: deque[dict[str, float | str]] = deque()
        self._auto_capture_in_progress = False
        self._last_force_sample_id = 0
        self._last_sensor_ui_update_ts = 0.0

        self.setWindowTitle('MeMo Acquisition')
        self.resize(1920, 1080)

        self.measurement_input = QSpinBox()
        self.measurement_input.setRange(1, 999)
        self.measurement_input.setValue(1)

        self.start_button = QPushButton('Sampling starten')
        self.calibrate_button = QPushButton('Kalibrieren')
        self.capture_button = QPushButton('Starte automatische Punktaufnahme')
        self.point_reset_button = QPushButton('Punkt resetten')
        self.measurement_reset_button = QPushButton('Messung resetten')
        self.exit_button = QPushButton('Beenden')
        self.force_input = QSpinBox()
        self.force_input.setRange(-1000, 1000)
        self.force_input.setSingleStep(1)
        self.force_input.setSuffix(' N')
        self.force_input.setValue(5)

        self.live_plot = LiveSensorPlot()
        self.force_plot = LiveForcePlot(
            history_seconds=5.0,
            target_force=float(self.force_input.value()),
            threshold=FORCE_TARGET_TOLERANCE_N,
            hide_x_tick_labels=True,
            fixed_grid=True,
            center_latest_value=True,
        )
        self.grid_plot = XYGridPlot(
            x_limits=X_LIMITS,
            y_limits=Y_LIMITS,
            grid_spacing=GRID_SPACING,
            corner_marker_size=CORNER_MARKER_SIZE,
            membrane_size=MEMBRANE_DIAGONAL,
            on_point_selected=self._update_point_selection,
        )
        self.calibration_panel = CalibrationStatusPanel(
            baseline_values=BASELINE_SENSOR_VALUES,
            tolerance=CALIBRATION_TOLERANCE,
        )

        self.target_label = QLabel('keiner')
        self.point_state_label = QLabel('offen')
        self.sample_status_label = QLabel('0 insgesamt, 0 offen')
        self.count_label = QLabel('0 Punkte')
        self.file_label = QLabel('-')
        self.status_label = QLabel('Bereit')
        self.force_port_label = QLabel(f'{FORCE_SERIAL_PORT} @ {FORCE_SERIAL_BAUDRATE}')
        self.force_value_label = QLabel('-')
        self.force_raw_label = QLabel('-')
        self.force_reader_status_label = QLabel('Nicht gestartet')
        self.force_hold_label = QLabel('Kraft innerhalb Toleranz halten: 0.0 / 5.0 s')
        self.force_hold_progress = QProgressBar()
        self.force_hold_progress.setRange(0, int(FORCE_TARGET_HOLD_S * 1000))
        self.force_hold_progress.setValue(0)
        self._sensor_connection_confirmed = False
        self._force_in_tolerance_since: datetime | None = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_sensor_data)
        self.force_timer = QTimer(self)
        self.force_timer.timeout.connect(self._refresh_force_data)

        self.start_button.clicked.connect(self.toggle_sampling)
        self.calibrate_button.clicked.connect(self.toggle_calibration)
        self.capture_button.clicked.connect(self.toggle_auto_capture)
        self.point_reset_button.clicked.connect(self.reset_point)
        self.measurement_reset_button.clicked.connect(self.reset_measurement)
        self.exit_button.clicked.connect(self.close)
        self.force_input.valueChanged.connect(self._update_force_target)

        self._build_ui()
        self._apply_styles()
        self._update_point_selection(self.grid_plot.get_active_point(), False)
        self._update_sample_status()
        try:
            self.force_reader.start()
            self.force_reader_status_label.setText(self.force_reader.connection_info())
        except (RuntimeError, SerialException, OSError) as exc:
            self._force_reader_available = False
            self.force_reader_status_label.setText(str(exc))
        self.force_timer.start(FORCE_REFRESH_MS)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(10)

        plots_layout = QHBoxLayout()
        plots_layout.setSpacing(10)
        plots_layout.addWidget(self.live_plot, stretch=3)
        force_layout = QVBoxLayout()
        force_layout.setSpacing(6)
        force_layout.addWidget(self.force_hold_label)
        force_layout.addWidget(self.force_hold_progress)
        force_layout.addWidget(self.force_plot, stretch=1)
        plots_layout.addLayout(force_layout, stretch=3)
        plots_layout.addWidget(self.grid_plot, stretch=4)
        main_layout.addLayout(plots_layout, stretch=6)

        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)

        info_panel = QFrame()
        info_panel.setObjectName('infoPanel')
        info_panel.setMaximumHeight(500)
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(12, 12, 12, 12)
        info_layout.setSpacing(8)

        info_layout.addWidget(self._build_section_card('Messung', self._build_measurement_section()))
        info_layout.addWidget(self._build_section_card('Punkt', self._build_point_section()))

        status_grid = QGridLayout()
        status_grid.setHorizontalSpacing(12)
        status_grid.setVerticalSpacing(10)
        status_grid.addWidget(self._make_label_caption('Sample-Status'), 0, 0)
        status_grid.addWidget(self.sample_status_label, 0, 1)
        status_grid.addWidget(self._make_label_caption('Gespeichert'), 1, 0)
        status_grid.addWidget(self.count_label, 1, 1)
        status_grid.addWidget(self._make_label_caption('Datei'), 2, 0)
        status_grid.addWidget(self.file_label, 2, 1)
        status_grid.addWidget(self._make_label_caption('Meldung'), 3, 0)
        status_grid.addWidget(self.status_label, 3, 1)
        info_layout.addLayout(status_grid)
        info_layout.addStretch(1)

        self.calibration_panel.setMaximumHeight(230)
        bottom_layout.addWidget(info_panel, stretch=2)
        bottom_layout.addWidget(self.calibration_panel, stretch=2)
        main_layout.addLayout(bottom_layout, stretch=0)

    def _build_measurement_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel('Nummer:'))
        row1.addWidget(self.measurement_input)
        row1.addSpacing(16)
        row1.addWidget(QLabel('Kraft:'))
        row1.addWidget(self.force_input)
        row1.addStretch(1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(self.start_button)
        row2.addWidget(self.calibrate_button)
        row2.addWidget(self.measurement_reset_button)
        row2.addWidget(self.exit_button)
        row2.addStretch(1)
        layout.addLayout(row2)
        layout.addStretch(1)
        return section

    def _build_point_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        row1 = QGridLayout()
        row1.setHorizontalSpacing(12)
        row1.setVerticalSpacing(10)
        row1.addWidget(self._make_label_caption('Ausgewaehlter Punkt'), 0, 0)
        row1.addWidget(self.target_label, 0, 1)
        row1.addWidget(self._make_label_caption('Punktstatus'), 1, 0)
        row1.addWidget(self.point_state_label, 1, 1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(self.capture_button)
        row2.addWidget(self.point_reset_button)
        row2.addStretch(1)
        layout.addLayout(row2)
        layout.addStretch(1)
        return section

    def _build_section_card(self, title_text: str, content: QWidget) -> QFrame:
        card = QFrame()
        card.setObjectName('sectionCard')
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel(title_text)
        title.setObjectName('sectionTitle')
        layout.addWidget(title)
        layout.addWidget(content)
        return card

    def _make_label_caption(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName('captionLabel')
        return label

    def _apply_styles(self):
        self.measurement_input.setMinimumWidth(90)
        self.force_input.setMinimumWidth(90)

        for button in [self.start_button, self.calibrate_button, self.point_reset_button, self.measurement_reset_button, self.exit_button]:
            button.setMinimumHeight(40)
        self.capture_button.setMinimumHeight(46)
        self.capture_button.setMinimumWidth(260)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #f3f5f7;
            }
            QFrame#infoPanel {
                background: white;
                border: 1px solid #d9e0e6;
                border-radius: 12px;
            }
            QLabel#captionLabel {
                color: #5b6470;
                font-weight: 600;
            }
            QLabel {
                color: #1f2a37;
            }
            QFrame#sectionCard {
                background: #ffffff;
                border: 1px solid #d9e0e6;
                border-radius: 10px;
            }
            QLabel#sectionTitle {
                color: #1f2a37;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #c7d0d9;
                border-radius: 10px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #f7fafc;
            }
            QPushButton:disabled {
                color: #98a2ad;
                background-color: #eef2f5;
            }
            QPushButton[text="Starte automatische Punktaufnahme"], QPushButton[text="Automatische Punktaufnahme stoppen"] {
                background-color: #0f8b6d;
                color: white;
                border: 1px solid #0b6b54;
                font-size: 15px;
                font-weight: 700;
            }
            QPushButton[text="Starte automatische Punktaufnahme"]:hover, QPushButton[text="Automatische Punktaufnahme stoppen"]:hover {
                background-color: #0c7a60;
            }
            QPushButton[text="Punkt resetten"] {
                background-color: #f59e0b;
                color: #1f2a37;
                border: 1px solid #d48806;
            }
            QPushButton[text="Kalibrieren"], QPushButton[text="Kalibrierung stoppen"] {
                background-color: #f3b13f;
                color: #1f2a37;
                border: 1px solid #d79a2c;
            }
            QPushButton[text="Beenden"] {
                background-color: #b42318;
                color: white;
                border: 1px solid #8f1c13;
            }
            QSpinBox {
                background: white;
                border: 1px solid #c7d0d9;
                border-radius: 8px;
                padding: 6px 8px;
                min-height: 30px;
            }
            QProgressBar {
                background: white;
                border: 1px solid #c7d0d9;
                border-radius: 8px;
                min-height: 18px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0f8b6d;
                border-radius: 7px;
            }
        """)

    def _update_force_target(self, value: int):
        self.force_plot.set_target_force(float(value))
        self._reset_force_hold_progress()

    def _reset_force_hold_progress(self):
        self._force_in_tolerance_since = None
        self._auto_capture_rows.clear()
        self.force_hold_progress.setValue(0)
        self.force_hold_label.setText(
            f'Kraft innerhalb Toleranz halten: 0.0 / {FORCE_TARGET_HOLD_S:.1f} s'
        )

    def _update_force_hold_progress(self, force_value: float | None, timestamp: datetime | None, stream_active: bool):
        if force_value is None or timestamp is None or not stream_active:
            self._reset_force_hold_progress()
            return

        target_force = float(self.force_input.value())
        within_tolerance = abs(force_value - target_force) <= FORCE_TARGET_TOLERANCE_N
        if not within_tolerance:
            self._reset_force_hold_progress()
            return

        if self._force_in_tolerance_since is None:
            self._force_in_tolerance_since = timestamp

        if self.auto_capture_enabled:
            self._append_auto_capture_row(timestamp, force_value)

        held_seconds = max(0.0, (timestamp - self._force_in_tolerance_since).total_seconds())
        clamped_seconds = min(held_seconds, FORCE_TARGET_HOLD_S)
        self.force_hold_progress.setValue(int(clamped_seconds * 1000))
        self.force_hold_label.setText(
            f'Kraft innerhalb Toleranz halten: {clamped_seconds:.1f} / {FORCE_TARGET_HOLD_S:.1f} s'
        )
        if self.auto_capture_enabled and held_seconds >= FORCE_TARGET_HOLD_S:
            self._record_auto_capture()

    def _measurement_path(self, measurement_number: int) -> Path:
        force_value = int(self.force_input.value())
        return self.output_dir / f'3D_Messung_{measurement_number:02d}_{force_value}N.csv'

    def _has_started_measurement(self) -> bool:
        return self.recorder is not None and self.recorder.sample_count > 0

    def _initialize_measurement(self, measurement_number: int, allow_resume: bool, prompt_if_exists: bool):
        csv_path = self._measurement_path(measurement_number)
        overwrite = False

        if csv_path.exists() and prompt_if_exists:
            reply = QMessageBox.question(
                self,
                'Datei existiert bereits',
                f'Die Datei {csv_path.name} existiert bereits. Soll sie ueberschrieben werden?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                overwrite = True
            elif allow_resume:
                overwrite = False
            else:
                self.status_label.setText('Neue Messung abgebrochen.')
                return False

        self.recorder = CsvSampleRecorder(csv_path, overwrite=overwrite)
        if overwrite:
            self.grid_plot.reset_samples()
            self.status_label.setText(f'Neue Messung gestartet: {csv_path.name}')
        else:
            saved_points = self.recorder.read_saved_points()
            if saved_points:
                self.grid_plot.load_saved_points(saved_points)
                self.status_label.setText(f'Bestehende Messung fortgesetzt: {csv_path.name}')
            else:
                self.grid_plot.reset_samples()
                self.status_label.setText(f'Neue Messung gestartet: {csv_path.name}')

        self.count_label.setText(f'{self.recorder.sample_count} Punkte')
        self.file_label.setText(self.recorder.csv_path.name)
        self._update_sample_status()
        return True

    def _ensure_measurement_initialized(self, allow_resume: bool = True, prompt_if_exists: bool = True):
        if self.recorder is not None:
            return True
        measurement_number = self.measurement_input.value()
        return self._initialize_measurement(measurement_number, allow_resume=allow_resume, prompt_if_exists=prompt_if_exists)

    def _update_point_selection(self, point, is_saved: bool = False):
        if point is None:
            self.target_label.setText('keiner')
            self.point_state_label.setText('-')
            self.capture_button.setEnabled(False)
            self.capture_button.setText('Starte automatische Punktaufnahme')
            self.point_reset_button.setEnabled(False)
            return

        self.target_label.setText(f'X={point[0]:.1f}, Y={point[1]:.1f}')
        self.point_state_label.setText('gespeichert' if is_saved else 'offen')
        self.capture_button.setEnabled(not is_saved)
        self.capture_button.setText(
            'Automatische Punktaufnahme stoppen' if self.auto_capture_enabled and not is_saved else 'Starte automatische Punktaufnahme'
        )
        self.point_reset_button.setEnabled(is_saved)

    def toggle_auto_capture(self):
        if self.auto_capture_enabled:
            self.auto_capture_enabled = False
            self._reset_force_hold_progress()
            self.capture_button.setText('Starte automatische Punktaufnahme')
            self.status_label.setText('Automatische Punktaufnahme gestoppt.')
            return

        if not self.is_sampling:
            QMessageBox.warning(self, 'Sampling inaktiv', 'Bitte zuerst Sampling starten.')
            return
        if not self._ensure_measurement_initialized(allow_resume=True, prompt_if_exists=True):
            return

        active_point = self.grid_plot.get_active_point()
        if active_point is None:
            QMessageBox.information(self, 'Fertig', 'Alle Testpunkte wurden bereits gespeichert.')
            self._update_point_selection(None, False)
            self._update_sample_status()
            return

        self.auto_capture_enabled = True
        self._reset_force_hold_progress()
        self.capture_button.setText('Automatische Punktaufnahme stoppen')
        self.status_label.setText('Automatische Punktaufnahme aktiv. Halte die Kraft 5 Sekunden im Toleranzbereich.')

    def _append_auto_capture_row(self, timestamp: datetime, force_value: float):
        if self.latest_frame is None:
            return

        active_point = self.grid_plot.get_active_point()
        if active_point is None:
            return

        row = {
            'date': timestamp.date().isoformat(),
            'time': timestamp.time().isoformat(timespec='milliseconds'),
            'X': float(active_point[0]),
            'Y': float(active_point[1]),
            'F': float(force_value),
        }
        for index, value in enumerate(np.asarray(self.latest_frame.sensors, dtype=float), start=1):
            row[f'Sensor R{index}'] = float(value)

        self._auto_capture_rows.append(row)

    def _record_auto_capture(self):
        if self._auto_capture_in_progress or not self._auto_capture_rows:
            return

        active_point = self.grid_plot.get_active_point()
        if active_point is None:
            return
        if not self._ensure_measurement_initialized(allow_resume=True, prompt_if_exists=True):
            return

        self._auto_capture_in_progress = True
        try:
            rows = list(self._auto_capture_rows)
            measurement_number = self.measurement_input.value()
            sensor_matrix = np.array(
                [[float(row[f'Sensor R{i}']) for i in range(1, 9)] for row in rows],
                dtype=float,
            )
            force_values = np.array([float(row['F']) for row in rows], dtype=float)
            last_timestamp = datetime.fromisoformat(f"{rows[-1]['date']}T{rows[-1]['time']}")
            sample = LabeledSample(
                sensors=np.mean(sensor_matrix, axis=0),
                x=float(active_point[0]),
                y=float(active_point[1]),
                force=float(np.mean(force_values)),
                timestamp=last_timestamp,
                metadata={'source': self.latest_frame.source if self.latest_frame is not None else 'auto_capture'},
            )
            self.recorder.append_sample(sample)
            raw_file_path = self.raw_writer.write_timeseries(rows, measurement_number=measurement_number)
            summary_file_path = self.raw_writer.write_timeseries_summary(rows, measurement_number=measurement_number)
            self.grid_plot.mark_point_saved(active_point)
            self.count_label.setText(f'{self.recorder.sample_count} Punkte')
            self._update_sample_status()

            if self.grid_plot.remaining_sample_count() == 0:
                self.auto_capture_enabled = False
                self.capture_button.setText('Starte automatische Punktaufnahme')
                self.status_label.setText(
                    f'Automatisch aufgenommen: {len(rows)} Samples | Raw: {raw_file_path.name} | Summary: {summary_file_path.name}'
                )
                QMessageBox.information(self, 'Messung abgeschlossen', 'Alle Samples dieser Messung wurden gespeichert.')
            else:
                self.status_label.setText(
                    f'Automatisch aufgenommen: {len(rows)} Samples | Raw: {raw_file_path.name} | Summary: {summary_file_path.name} | Naechster Punkt bereit.'
                )
        finally:
            self._reset_force_hold_progress()
            self._auto_capture_in_progress = False
            self._update_point_selection(self.grid_plot.get_active_point(), False)

    def _update_sample_status(self):
        total_count = self.grid_plot.total_sample_count()
        remaining_count = self.grid_plot.remaining_sample_count()
        self.sample_status_label.setText(f'{total_count} insgesamt, {remaining_count} offen')
        self.measurement_reset_button.setEnabled(self.recorder is not None or total_count > 0)

    def toggle_sampling(self):
        if self.is_sampling:
            self.timer.stop()
            self.is_sampling = False
            self.auto_capture_enabled = False
            self._reset_force_hold_progress()
            self._sensor_connection_confirmed = False
            self.start_button.setText('Sampling starten')
            self.capture_button.setText('Starte automatische Punktaufnahme')
            self.status_label.setText('Sampling angehalten.')
            return

        if not self.refresh_sensor_data(show_connection_feedback=True):
            return

        self.timer.start(REFRESH_MS)
        self.is_sampling = True
        self.start_button.setText('Sampling stoppen')

    def toggle_calibration(self):
        if self.calibration_panel.calibration_active:
            self.calibration_panel.calibration_active = False
            self.calibration_panel._refresh_display()
            self.calibrate_button.setText('Kalibrieren')
            self.status_label.setText('Kalibrierung gestoppt.')
            return

        self.calibration_panel.start_calibration()
        self.calibrate_button.setText('Kalibrierung stoppen')
        if not self.is_sampling:
            self.timer.start(REFRESH_MS)
            self.is_sampling = True
            self.start_button.setText('Sampling stoppen')
        self.status_label.setText('Kalibrierung aktiv.')
        self.refresh_sensor_data()

    def _has_valid_sensor_signal(self, sensor_values: np.ndarray) -> bool:
        values = np.asarray(sensor_values, dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            return False

        if isinstance(self.reader, Ads1115Reader):
            return not np.all(np.isclose(values, 0.0, atol=1e-4))
        return True

    def refresh_sensor_data(self, show_connection_feedback: bool = False):
        try:
            self.latest_frame = self.reader.read()
        except StopIteration:
            self.timer.stop()
            self.is_sampling = False
            self._sensor_connection_confirmed = False
            self.start_button.setText('Sampling starten')
            self.status_label.setText('Replay beendet.')
            return False
        except Exception as exc:
            self.timer.stop()
            self.is_sampling = False
            self._sensor_connection_confirmed = False
            self.start_button.setText('Sampling starten')
            self.status_label.setText(f'Fehler beim Lesen: {exc}')
            if show_connection_feedback:
                QMessageBox.warning(
                    self,
                    'Sensoren nicht verbunden',
                    f'Es konnte kein Sensorsignal gelesen werden.\n\nDetails: {exc}',
                )
            return False

        if not self._has_valid_sensor_signal(self.latest_frame.sensors):
            self.timer.stop()
            self.is_sampling = False
            self._sensor_connection_confirmed = False
            self.start_button.setText('Sampling starten')
            self.status_label.setText('Kein Sensorsignal erkannt.')
            if show_connection_feedback:
                QMessageBox.warning(
                    self,
                    'Sensoren nicht verbunden',
                    'Es wurde kein gueltiges Sensorsignal erkannt. Bitte Sensoren und Verkabelung pruefen.',
                )
            return False

        now_ts = datetime.now(UTC).timestamp()
        sensor_ui_refresh_ms = (
            CALIBRATION_SENSOR_UI_REFRESH_MS
            if self.calibration_panel.calibration_active
            else SENSOR_UI_REFRESH_MS
        )
        should_update_sensor_ui = (
            self._last_sensor_ui_update_ts == 0.0
            or (now_ts - self._last_sensor_ui_update_ts) >= (sensor_ui_refresh_ms / 1000.0)
        )
        if should_update_sensor_ui:
            self.live_plot.update_values(self.latest_frame.sensors)
            self.calibration_panel.set_live_values(self.latest_frame.sensors)
            self._last_sensor_ui_update_ts = now_ts
        if show_connection_feedback and not self._sensor_connection_confirmed:
            QMessageBox.information(
                self,
                'Sensoren verbunden',
                'Sensoren erfolgreich verbunden.',
            )
            self.status_label.setText('Sensoren erfolgreich verbunden. Sampling aktiv.')
            self._sensor_connection_confirmed = True
        elif self.is_sampling:
            self.status_label.setText('Sampling aktiv.')
        return True

    def _refresh_force_data(self):
        now = datetime.now(UTC)
        now_ts = now.timestamp()
        self.force_port_label.setText(f'{self.force_reader.port} @ {self.force_reader.baudrate}')
        if not self._force_reader_available:
            self.force_reader_status_label.setText('Deaktiviert nach Fehler')
            self.force_plot.set_stream_active(False)
            self._reset_force_hold_progress()
            return
        reader_error = self.force_reader.get_last_error()
        if reader_error:
            self._force_reader_available = False
            self.status_label.setText(f'Kraftsensor nicht verfuegbar: {reader_error}')
            self.force_reader_status_label.setText(reader_error)
            self.force_raw_label.setText(self.force_reader.get_last_raw_text() or '-')
            self.force_plot.set_stream_active(False)
            self._reset_force_hold_progress()
            return

        force_value = self.force_reader.get_latest_force()
        force_timestamp = self.force_reader.get_latest_force_timestamp()
        force_timestamp_ts = _to_utc_timestamp(force_timestamp)
        new_force_samples = self.force_reader.get_samples_since(self._last_force_sample_id)
        if new_force_samples:
            self._last_force_sample_id = new_force_samples[-1][0]
        stream_active = (
            force_timestamp_ts is not None
            and (now_ts - force_timestamp_ts) <= FORCE_STREAM_TIMEOUT_S
        )
        self.force_plot.set_stream_active(stream_active)
        if force_value is None:
            self.force_plot.advance_time(now)
            self.force_raw_label.setText(self.force_reader.get_last_raw_text() or '-')
            self.force_reader_status_label.setText(self.force_reader.connection_info())
            self._reset_force_hold_progress()
            return

        self.latest_force_value = force_value
        if new_force_samples:
            plot_samples: list[tuple[datetime, float]] = []
            for _, sample_timestamp, sample_force in new_force_samples:
                plot_samples.append((sample_timestamp, sample_force))
                self._update_force_hold_progress(sample_force, sample_timestamp, stream_active)
            self.force_plot.append_values(plot_samples)
        else:
            timestamp = force_timestamp if force_timestamp is not None else now
            self.force_plot.advance_time(timestamp)
            self.force_plot.append_value(timestamp, force_value)
        self.force_value_label.setText(f'{force_value:.3f} N')
        self.force_raw_label.setText(self.force_reader.get_last_raw_text() or '-')
        self.force_reader_status_label.setText(self.force_reader.connection_info())

    def reset_point(self):
        if self.recorder is None:
            QMessageBox.warning(self, 'Keine Messung', 'Es gibt noch keine gespeicherte Messung.')
            return

        selected_point = self.grid_plot.get_selected_point()
        if selected_point is None or not self.grid_plot.is_point_saved(selected_point):
            QMessageBox.warning(self, 'Kein gruener Punkt', 'Bitte zuerst einen grueneren gespeicherten Punkt im XY-Plot auswaehlen.')
            return

        removed_count = self.recorder.remove_point(selected_point)
        if removed_count <= 0:
            QMessageBox.warning(self, 'Nicht gefunden', 'Der ausgewaehlte Punkt konnte nicht aus der CSV entfernt werden.')
            return

        self.grid_plot.reset_saved_point(selected_point)
        self.count_label.setText(f'{self.recorder.sample_count} Punkte')
        self.status_label.setText(f'Punkt zurueckgesetzt: X={selected_point[0]:.1f}, Y={selected_point[1]:.1f}')
        self._update_sample_status()

    def reset_measurement(self):
        self.auto_capture_enabled = False
        self._reset_force_hold_progress()
        self.capture_button.setText('Starte automatische Punktaufnahme')
        if self.recorder is None:
            self.grid_plot.reset_samples()
            self.measurement_input.setValue(self.measurement_input.value() + 1)
            self.file_label.setText('-')
            self.count_label.setText('0 Punkte')
            self.status_label.setText('Messung zurueckgesetzt.')
            self._update_sample_status()
            self._update_point_selection(self.grid_plot.get_active_point(), False)
            return

        remaining_count = self.grid_plot.remaining_sample_count()
        if self._has_started_measurement() and remaining_count > 0:
            reply = QMessageBox.warning(
                self,
                'Messung unvollstaendig',
                'Es sind noch nicht alle Samples gespeichert. Die aktuelle CSV bleibt erhalten. Neue Messung trotzdem starten?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        elif self._has_started_measurement() and remaining_count == 0:
            reply = QMessageBox.question(
                self,
                'Neue Messung starten',
                'Alle Samples sind gespeichert. Neue Messung starten?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return

        self.recorder = None
        self.grid_plot.reset_samples()
        self.measurement_input.setValue(self.measurement_input.value() + 1)
        self.force_input.setValue(0)
        self.file_label.setText('-')
        self.count_label.setText('0 Punkte')
        next_path = self._measurement_path(self.measurement_input.value())
        self.status_label.setText(f'Bereit fuer neue Messung: {next_path.name}')
        self._update_sample_status()
        self._update_point_selection(self.grid_plot.get_active_point(), False)

        if not self.is_sampling:
            self.timer.start(REFRESH_MS)
            self.is_sampling = True
            self.start_button.setText('Sampling stoppen')
        self.refresh_sensor_data()

    def closeEvent(self, event: QCloseEvent):
        self.auto_capture_enabled = False
        if self._has_started_measurement() and self.grid_plot.remaining_sample_count() > 0:
            reply = QMessageBox.warning(
                self,
                'Messung nicht vollstaendig',
                'Nicht alle Samples wurden gespeichert. Die CSV bleibt erhalten. Wirklich beenden?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

        if self.is_sampling:
            self.timer.stop()
        if self.force_timer.isActive():
            self.force_timer.stop()
        self.force_reader.close()
        event.accept()


def build_reader(replay_csv: str | None, sensor_port: str | None, sensor_baudrate: int, sensor_timeout: float):
    if DEBUG_SENSOR_MODE:
        return MockReader()
    if replay_csv:
        return CsvReplayReader(replay_csv, loop=True)
    if sensor_port:
        return SerialSensorReader(
            port=sensor_port,
            baudrate=sensor_baudrate,
            timeout=sensor_timeout,
        )
    try:
        return Ads1115Reader()
    except RuntimeError as exc:
        return UnavailableSensorReader(
            reason=(
                "ADS1115 ist in dieser Umgebung nicht verfuegbar. "
                "Fuer Live-Sensordaten bitte auf dem Raspberry Pi mit aktivem I2C starten "
                "oder alternativ --replay-csv bzw. --sensor-port verwenden. "
                f"Details: {exc}"
            )
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='MeMo acquisition UI')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR), help='Ausgabeordner fuer Messungsdateien')
    parser.add_argument('--replay-csv', default=None, help='Optionales CSV fuer Replay statt Live-Daten vom ADS1115')
    parser.add_argument('--sensor-port', default=None, help='Optionaler serieller Port fuer 8 Sensorwerte statt ADS1115')
    parser.add_argument('--sensor-baudrate', type=int, default=57600, help='Baudrate fuer den seriellen Sensorreader')
    parser.add_argument('--sensor-timeout', type=float, default=0.2, help='Timeout fuer den seriellen Sensorreader')
    parser.add_argument('--force-port', default=FORCE_SERIAL_PORT, help='Serieller Port fuer Kraftmessung')
    parser.add_argument('--force-baudrate', type=int, default=FORCE_SERIAL_BAUDRATE, help='Baudrate fuer Kraftmessung')
    return parser.parse_args(argv)


def run_app(argv=None):
    args = parse_args(argv)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    reader = build_reader(args.replay_csv, args.sensor_port, args.sensor_baudrate, args.sensor_timeout)
    window = AcquisitionWindow(
        reader=reader,
        output_dir=Path(args.output_dir),
        force_port=args.force_port,
        force_baudrate=args.force_baudrate,
    )
    window.show()
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(run_app())
