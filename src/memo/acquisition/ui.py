from __future__ import annotations

import argparse
import sys
from datetime import datetime
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
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from memo.acquisition.readers import CsvReplayReader, MockReader, SerialException, SerialForceReader
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
REFRESH_MS = 250
FORCE_REFRESH_MS = 40
FORCE_SERIAL_PORT = 'COM3'
FORCE_SERIAL_BAUDRATE = 57600
FORCE_STREAM_TIMEOUT_S = 0.5
BASELINE_SENSOR_VALUES = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000], dtype=float)
CALIBRATION_TOLERANCE = 0.050


class AcquisitionWindow(QMainWindow):
    def __init__(self, reader, output_dir: Path, force_port: str = FORCE_SERIAL_PORT, force_baudrate: int = FORCE_SERIAL_BAUDRATE):
        super().__init__()
        self.reader = reader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_writer = RawSampleFileWriter(RAW_OUTPUT_DIR)
        self.force_reader = SerialForceReader(port=force_port, baudrate=force_baudrate)
        self.latest_frame = None
        self.latest_force_value: float | None = None
        self._force_reader_available = True
        self.is_sampling = False
        self.recorder: CsvSampleRecorder | None = None

        self.setWindowTitle('MeMo Acquisition')
        self.resize(1920, 1080)

        self.live_plot = LiveSensorPlot()
        self.force_plot = LiveForcePlot()
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

        self.measurement_input = QSpinBox()
        self.measurement_input.setRange(1, 999)
        self.measurement_input.setValue(1)

        self.start_button = QPushButton('Sampling starten')
        self.calibrate_button = QPushButton('Kalibrieren')
        self.capture_button = QPushButton('Punkt aufnehmen')
        self.point_reset_button = QPushButton('Punkt resetten')
        self.measurement_reset_button = QPushButton('Messung resetten')
        self.exit_button = QPushButton('Beenden')
        self.force_input = QSpinBox()
        self.force_input.setRange(-1000, 1000)
        self.force_input.setSingleStep(1)
        self.force_input.setSuffix(' N')
        self.force_input.setValue(0)

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

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_sensor_data)
        self.force_timer = QTimer(self)
        self.force_timer.timeout.connect(self._refresh_force_data)

        self.start_button.clicked.connect(self.toggle_sampling)
        self.calibrate_button.clicked.connect(self.toggle_calibration)
        self.capture_button.clicked.connect(self.save_sample)
        self.point_reset_button.clicked.connect(self.reset_point)
        self.measurement_reset_button.clicked.connect(self.reset_measurement)
        self.exit_button.clicked.connect(self.close)

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
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)

        plots_layout = QHBoxLayout()
        plots_layout.setSpacing(14)
        plots_layout.addWidget(self.live_plot, stretch=3)
        plots_layout.addWidget(self.force_plot, stretch=3)
        plots_layout.addWidget(self.grid_plot, stretch=4)
        main_layout.addLayout(plots_layout, stretch=4)

        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(14)

        info_panel = QFrame()
        info_panel.setObjectName('infoPanel')
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(16, 16, 16, 16)
        info_layout.setSpacing(12)

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
        status_grid.addWidget(self._make_label_caption('Kraft COM'), 4, 0)
        status_grid.addWidget(self.force_port_label, 4, 1)
        status_grid.addWidget(self._make_label_caption('Kraft live'), 5, 0)
        status_grid.addWidget(self.force_value_label, 5, 1)
        status_grid.addWidget(self._make_label_caption('Rohdaten'), 6, 0)
        status_grid.addWidget(self.force_raw_label, 6, 1)
        status_grid.addWidget(self._make_label_caption('COM-Status'), 7, 0)
        status_grid.addWidget(self.force_reader_status_label, 7, 1)
        info_layout.addLayout(status_grid)
        info_layout.addStretch(1)

        bottom_layout.addWidget(info_panel, stretch=2)
        bottom_layout.addWidget(self.calibration_panel, stretch=2)
        main_layout.addLayout(bottom_layout, stretch=1)

    def _build_measurement_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

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
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

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
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

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
        self.capture_button.setMinimumWidth(170)

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
            QPushButton[text="Punkt aufnehmen"] {
                background-color: #0f8b6d;
                color: white;
                border: 1px solid #0b6b54;
                font-size: 15px;
                font-weight: 700;
            }
            QPushButton[text="Punkt aufnehmen"]:hover {
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
        """)

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
            self.point_reset_button.setEnabled(False)
            return

        self.target_label.setText(f'X={point[0]:.1f}, Y={point[1]:.1f}')
        self.point_state_label.setText('gespeichert' if is_saved else 'offen')
        self.capture_button.setEnabled(not is_saved)
        self.point_reset_button.setEnabled(is_saved)

    def _update_sample_status(self):
        total_count = self.grid_plot.total_sample_count()
        remaining_count = self.grid_plot.remaining_sample_count()
        self.sample_status_label.setText(f'{total_count} insgesamt, {remaining_count} offen')
        self.measurement_reset_button.setEnabled(self.recorder is not None or total_count > 0)

    def toggle_sampling(self):
        if self.is_sampling:
            self.timer.stop()
            self.is_sampling = False
            self.start_button.setText('Sampling starten')
            self.status_label.setText('Sampling angehalten.')
            return

        self.timer.start(REFRESH_MS)
        self.is_sampling = True
        self.start_button.setText('Sampling stoppen')
        self.status_label.setText('Sampling aktiv.')
        self.refresh_sensor_data()

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

    def refresh_sensor_data(self):
        try:
            self.latest_frame = self.reader.read()
        except StopIteration:
            self.timer.stop()
            self.is_sampling = False
            self.start_button.setText('Sampling starten')
            self.status_label.setText('Replay beendet.')
            return
        except Exception as exc:
            self.timer.stop()
            self.is_sampling = False
            self.start_button.setText('Sampling starten')
            self.status_label.setText(f'Fehler beim Lesen: {exc}')
            return

        self.live_plot.update_values(self.latest_frame.sensors)
        self.calibration_panel.set_live_values(self.latest_frame.sensors)

    def _refresh_force_data(self):
        now = datetime.utcnow()
        self.force_plot.advance_time(now)
        self.force_port_label.setText(f'{self.force_reader.port} @ {self.force_reader.baudrate}')
        if not self._force_reader_available:
            self.force_reader_status_label.setText('Deaktiviert nach Fehler')
            self.force_plot.set_stream_active(False)
            return
        reader_error = self.force_reader.get_last_error()
        if reader_error:
            self._force_reader_available = False
            self.status_label.setText(f'Kraftsensor nicht verfuegbar: {reader_error}')
            self.force_reader_status_label.setText(reader_error)
            self.force_raw_label.setText(self.force_reader.get_last_raw_text() or '-')
            self.force_plot.set_stream_active(False)
            return

        force_value = self.force_reader.get_latest_force()
        force_timestamp = self.force_reader.get_latest_force_timestamp()
        stream_active = (
            force_timestamp is not None
            and (now - force_timestamp).total_seconds() <= FORCE_STREAM_TIMEOUT_S
        )
        self.force_plot.set_stream_active(stream_active)
        if force_value is None:
            self.force_raw_label.setText(self.force_reader.get_last_raw_text() or '-')
            self.force_reader_status_label.setText(self.force_reader.connection_info())
            return

        self.latest_force_value = force_value
        timestamp = self.latest_frame.timestamp if self.latest_frame is not None else now
        self.force_plot.append_value(timestamp, force_value)
        self.force_value_label.setText(f'{force_value:.3f} N')
        self.force_raw_label.setText(self.force_reader.get_last_raw_text() or '-')
        self.force_reader_status_label.setText(self.force_reader.connection_info())

    def save_sample(self):
        if not self._ensure_measurement_initialized(allow_resume=True, prompt_if_exists=True):
            return
        if self.latest_frame is None:
            QMessageBox.warning(self, 'Keine Daten', 'Bitte zuerst Sampling starten.')
            return

        active_point = self.grid_plot.get_active_point()
        if active_point is None:
            QMessageBox.information(self, 'Fertig', 'Alle Testpunkte wurden bereits gespeichert.')
            self._update_point_selection(None, False)
            self._update_sample_status()
            return

        sample = LabeledSample(
            sensors=np.asarray(self.latest_frame.sensors, dtype=float),
            x=float(active_point[0]),
            y=float(active_point[1]),
            force=float(self.force_input.value()),
            timestamp=self.latest_frame.timestamp,
            metadata={'source': self.latest_frame.source},
        )
        row = self.recorder.append_sample(sample)
        raw_file_path = self.raw_writer.write_sample(sample)
        self.grid_plot.mark_point_saved(active_point)
        self.count_label.setText(f'{self.recorder.sample_count} Punkte')
        self.status_label.setText(
            f"Punkt aufgenommen: {row['timestamp']} | X={row['X']} | Y={row['Y']} | F={row['F']} | Raw: {raw_file_path.name}"
        )
        self._update_sample_status()

        if self.grid_plot.remaining_sample_count() == 0:
            QMessageBox.information(self, 'Messung abgeschlossen', 'Alle Samples dieser Messung wurden gespeichert.')

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


def build_reader(replay_csv: str | None):
    if replay_csv:
        return CsvReplayReader(replay_csv, loop=True)
    return MockReader()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='MeMo acquisition UI')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR), help='Ausgabeordner fuer Messungsdateien')
    parser.add_argument('--replay-csv', default=None, help='Optionales CSV fuer Replay statt MockReader')
    parser.add_argument('--force-port', default=FORCE_SERIAL_PORT, help='Serieller Port fuer Kraftmessung')
    parser.add_argument('--force-baudrate', type=int, default=FORCE_SERIAL_BAUDRATE, help='Baudrate fuer Kraftmessung')
    return parser.parse_args(argv)


def run_app(argv=None):
    args = parse_args(argv)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    reader = build_reader(args.replay_csv)
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
