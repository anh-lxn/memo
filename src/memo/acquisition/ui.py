from __future__ import annotations

import argparse
from collections import deque
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QCloseEvent, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
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
    resolve_force_serial_port,
    SerialException,
    SerialForceReader,
    SerialSensorReader,
    UnavailableSensorReader,
)
from memo.acquisition.recorder import CsvSampleRecorder, RawSampleFileWriter
from memo.types import LabeledSample, SensorFrame
from memo.visualization.plots import CalibrationStatusPanel, LiveForcePlot, LiveSensorPlot, XYGridPlot


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'recorded_samples'
RAW_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'raw'
ITM_LOGO_PATH = PROJECT_ROOT / 'doc' / 'ITM_Logo.png'
GRID_SPACING = 40
CORNER_MARKER_SIZE = 30
OFFSET = 50
MEMBRANE_SIDE_LENGTH = 450.0
MEMBRANE_DIAGONAL = MEMBRANE_SIDE_LENGTH * np.sqrt(2.0)
X_LIMITS = (-MEMBRANE_DIAGONAL / 2 - OFFSET, MEMBRANE_DIAGONAL / 2 + OFFSET)
Y_LIMITS = (-MEMBRANE_DIAGONAL / 2 - OFFSET, MEMBRANE_DIAGONAL / 2 + OFFSET)
XY_INNER_MAX_ABS_COORD_MM = 120.0
REFRESH_MS = 20
FORCE_REFRESH_MS = 20
SENSOR_UI_REFRESH_MS = 120
CALIBRATION_SENSOR_UI_REFRESH_MS = 120
FORCE_SERIAL_PORT = resolve_force_serial_port()
FORCE_SERIAL_BAUDRATE = 57600
FORCE_STREAM_TIMEOUT_S = 0.5
FORCE_TARGET_TOLERANCE_N = 1.0
FORCE_TARGET_HOLD_S = 2.0
FORCE_SEQUENCE_N = (5, 10, 15)
FORCE_PLOT_HISTORY_S = 8.0
FORCE_DISPLAY_HISTORY_LENGTH = 9
FORCE_DISPLAY_SMOOTHING_ALPHA = 0.20
FORCE_PLOT_UI_REFRESH_MS = 50
FORCE_PLOT_MAX_BATCH_POINTS = 240
BASELINE_SENSOR_VALUES = np.full(8, 2.5, dtype=float)
SENSOR_BASELINE_THRESHOLD_V = 0.15
CALIBRATION_TOLERANCE = 0.100
DEFAULT_REFERENCE_SENSOR_INDEX = 5  # R6
MODE_ONE_LOCK_SENSOR_FILTER = False
MODE_ONE_FORCE_BASELINE_V = 2.5
MODE_ONE_FORCE_VOLTS_PER_NEWTON = 0.1
MODE_ONE_DISTANCE_DECAY_MM = 160.0
MODE_ONE_NOISE_MAX_V = 0.08
MODE_ONE_NOISE_STEP_STD_V = 0.006
MODE_ONE_NOISE_DECAY = 0.90
MODE_ONE_SENSOR_POSITIONS_MM = np.array(
    [
        (-30.0, 200.0),   # R1
        (45.0, 182.0),    # R2
        (-200.0, 30.0),   # R3
        (-182.0, -45.0),  # R4
        (182.0, 45.0),    # R5
        (200.0, -30.0),   # R6
        (-45.0, -182.0),  # R7
        (30.0, -200.0),   # R8
    ],
    dtype=float,
)
DEBUG_SENSOR_MODE = True
SENSOR_SOURCE_REAL_ALL = 'real_all'
SENSOR_SOURCE_MOCK_ALL = 'mock_all'
SENSOR_SOURCE_FORCE_SENSOR = 'force_sensor'
MODE_SWITCH_PASSWORD = 'memo123'


def _compute_mode_one_sensor_coupling(reference_sensor_index: int) -> np.ndarray:
    reference_position = MODE_ONE_SENSOR_POSITIONS_MM[int(reference_sensor_index)]
    distances = np.linalg.norm(MODE_ONE_SENSOR_POSITIONS_MM - reference_position, axis=1)
    coupling = 1.0 / (1.0 + (distances / MODE_ONE_DISTANCE_DECAY_MM))
    coupling[int(reference_sensor_index)] = 1.0
    return coupling.astype(float)


def _to_utc_timestamp(value: datetime | None) -> float | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC).timestamp()
    return value.astimezone(UTC).timestamp()


class AcquisitionWindow(QMainWindow):
    def __init__(
        self,
        reader,
        output_dir: Path,
        force_port: str = FORCE_SERIAL_PORT,
        force_baudrate: int = FORCE_SERIAL_BAUDRATE,
        real_reader=None,
        mock_reader=None,
    ):
        super().__init__()
        self.reader = reader
        self.real_reader = real_reader if real_reader is not None else reader
        self.mock_reader = mock_reader if mock_reader is not None else MockReader(
            baseline=float(BASELINE_SENSOR_VALUES[0]),
            noise_std=0.05,
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_writer = RawSampleFileWriter(RAW_OUTPUT_DIR, summary_output_dir=self.output_dir)
        self.force_reader = SerialForceReader(port=force_port, baudrate=force_baudrate)
        self.latest_frame = None
        self.latest_force_value: float | None = None
        self._force_reader_available = True
        self.is_sampling = False
        self.recorders: dict[int, CsvSampleRecorder] = {}
        self.auto_capture_enabled = False
        self._auto_capture_rows: deque[dict[str, float | str]] = deque()
        self._auto_capture_in_progress = False
        self._auto_force_index = 0
        self._last_force_sample_id = 0
        self._last_sensor_ui_update_ts = 0.0
        self._force_display_history: deque[float] = deque(maxlen=FORCE_DISPLAY_HISTORY_LENGTH)
        self._force_display_value: float | None = None
        self._last_force_plot_update_ts = 0.0
        self._pending_force_plot_samples: deque[tuple[datetime, float]] = deque()
        self._mode_one_selected = False
        self._mode_two_selected = False
        self._admin_selected = False
        self.sensor_baseline_threshold_v = float(SENSOR_BASELINE_THRESHOLD_V)
        self.force_target_tolerance_n = float(FORCE_TARGET_TOLERANCE_N)
        self.sensor_baseline_values = np.asarray(BASELINE_SENSOR_VALUES, dtype=float).copy()
        self.reference_sensor_index = DEFAULT_REFERENCE_SENSOR_INDEX
        self.mode_one_force_baseline_v = float(self.sensor_baseline_values[self.reference_sensor_index])
        self.sensor_source_mode = SENSOR_SOURCE_REAL_ALL
        self._mock_sensor_rng = np.random.default_rng()
        self._mode_one_noise_state = np.zeros(8, dtype=float)
        self.fullscreen_enabled = True
        self.expo_theme_enabled = False

        self.setWindowTitle('MeMo Acquisition')
        self.resize(1920, 1080)

        self.measurement_input = QSpinBox()
        self.measurement_input.setRange(1, 999)
        self.measurement_input.setValue(1)
        self.measurement_input.setButtonSymbols(QSpinBox.NoButtons)

        self.start_button = QPushButton('Betrieb 1')
        self.mode_two_button = QPushButton('Betrieb 2')
        self.calibrate_button = QPushButton('Kalibriermodus')
        self.admin_button = QPushButton('Admin')
        self.capture_button = QPushButton('Starte automatische Punktaufnahme')
        self.point_reset_button = QPushButton('Punkt resetten')
        self.exit_button = QPushButton('Beenden')
        self.force_input = QSpinBox()
        self.force_input.setRange(-1000, 1000)
        self.force_input.setSingleStep(1)
        self.force_input.setSuffix(' N')
        self.force_input.setValue(FORCE_SEQUENCE_N[0])
        self.force_input.setButtonSymbols(QSpinBox.NoButtons)

        self.live_plot = LiveSensorPlot(
            baseline_value=self.mode_one_force_baseline_v,
            threshold=self.sensor_baseline_threshold_v,
        )
        self.force_plot = LiveForcePlot(
            history_seconds=FORCE_PLOT_HISTORY_S,
            target_force=float(self.force_input.value()),
            threshold=self.force_target_tolerance_n,
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
            max_abs_coordinate=XY_INNER_MAX_ABS_COORD_MM,
            on_point_selected=self._update_point_selection,
        )
        self.calibration_panel = CalibrationStatusPanel(
            baseline_values=self.sensor_baseline_values,
            tolerance=CALIBRATION_TOLERANCE,
        )

        self.target_label = QLabel('keiner')
        self.point_state_label = QLabel('offen')
        self.sample_status_label = QLabel('0 insgesamt, 0 offen')
        self.count_label = QLabel('0 Punkte komplett')
        self.file_label = QLabel('-')
        self.status_label = QLabel('Bereit')
        self.force_port_label = QLabel(f'{FORCE_SERIAL_PORT} @ {FORCE_SERIAL_BAUDRATE}')
        self.force_value_label = QLabel('-')
        self.force_raw_label = QLabel('-')
        self.force_reader_status_label = QLabel('Nicht gestartet')
        self.force_hold_label = QLabel('Kraft 5 N halten (1/3): 0.0 / 2.0 s')
        self.force_hold_progress = QProgressBar()
        self.force_hold_progress.setRange(0, int(FORCE_TARGET_HOLD_S * 1000))
        self.force_hold_progress.setValue(0)
        self.logo_label = QLabel()
        self.logo_label.setObjectName('itmLogo')
        logo_pixmap = QPixmap(str(ITM_LOGO_PATH))
        if not logo_pixmap.isNull():
            self.logo_label.setPixmap(logo_pixmap.scaledToHeight(70))
        self._sensor_connection_confirmed = False
        self._force_in_tolerance_since: datetime | None = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_sensor_data)
        self.force_timer = QTimer(self)
        self.force_timer.timeout.connect(self._refresh_force_data)

        self.start_button.clicked.connect(self.activate_mode_one)
        self.mode_two_button.clicked.connect(self.toggle_sampling)
        self.calibrate_button.clicked.connect(self.toggle_calibration)
        self.admin_button.clicked.connect(self.activate_admin_mode)
        self.capture_button.clicked.connect(self.toggle_auto_capture)
        self.point_reset_button.clicked.connect(self.reset_point)
        self.exit_button.clicked.connect(self.close)
        self.force_input.valueChanged.connect(self._update_force_target)

        self._build_ui()
        self._apply_styles()
        self._configure_mode_buttons()
        self._sync_force_sequence_ui()
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

        header_layout = QHBoxLayout()
        header_layout.addStretch(1)
        header_layout.addWidget(self.logo_label)
        main_layout.addLayout(header_layout, stretch=0)

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
        self.plots_container = QWidget()
        self.plots_container.setLayout(plots_layout)
        self.plot_panel = QFrame()
        self.plot_panel.setObjectName('plotPanel')
        plot_panel_layout = QVBoxLayout(self.plot_panel)
        plot_panel_layout.setContentsMargins(14, 14, 14, 14)
        plot_panel_layout.setSpacing(0)
        plot_panel_layout.addWidget(self.plots_container)
        main_layout.addWidget(self.plot_panel, stretch=8)

        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)

        self.info_panel = QFrame()
        self.info_panel.setObjectName('settingsPanel')
        self.info_panel.setMaximumHeight(480)
        info_layout = QVBoxLayout(self.info_panel)
        info_layout.setContentsMargins(12, 12, 12, 12)
        info_layout.setSpacing(6)

        self.measurement_card = self._build_section_card('Messung', self._build_measurement_section())
        info_layout.addWidget(self.measurement_card)
        self.admin_card = self._build_section_card('Admin', self._build_admin_section())
        self.admin_card.setVisible(False)
        info_layout.addWidget(self.admin_card)
        self.point_card = self._build_section_card('Punkt', self._build_point_section())
        info_layout.addWidget(self.point_card)

        self.status_widget = QWidget()
        status_grid = QGridLayout(self.status_widget)
        status_grid.setContentsMargins(0, 0, 0, 0)
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
        info_layout.addWidget(self.status_widget)
        info_layout.addStretch(1)

        bottom_layout.addWidget(self.info_panel, stretch=2)
        main_layout.addLayout(bottom_layout, stretch=2)

    def _build_measurement_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(12)

        row1 = QHBoxLayout()
        row1.setSpacing(12)
        row1.addWidget(QLabel('Nummer:'))
        row1.addWidget(self._build_stepper_controls(self.measurement_input, compact=True))
        row1.addStretch(1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(self.start_button)
        row2.addWidget(self.mode_two_button)
        row2.addWidget(self.calibrate_button)
        row2.addWidget(self.admin_button)
        row2.addWidget(self.exit_button)
        row2.addStretch(1)
        layout.addLayout(row2)
        return section

    def _build_admin_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(20)

        self.voltage_threshold_input = QDoubleSpinBox()
        self.voltage_threshold_input.setRange(0.0, 2.0)
        self.voltage_threshold_input.setDecimals(2)
        self.voltage_threshold_input.setSingleStep(0.01)
        self.voltage_threshold_input.setSuffix(' V')
        self.voltage_threshold_input.setValue(self.sensor_baseline_threshold_v)
        self.voltage_threshold_input.setButtonSymbols(QDoubleSpinBox.NoButtons)

        self.force_threshold_input = QDoubleSpinBox()
        self.force_threshold_input.setRange(0.0, 20.0)
        self.force_threshold_input.setDecimals(1)
        self.force_threshold_input.setSingleStep(0.1)
        self.force_threshold_input.setSuffix(' N')
        self.force_threshold_input.setValue(self.force_target_tolerance_n)
        self.force_threshold_input.setButtonSymbols(QDoubleSpinBox.NoButtons)

        self.sensor_source_input = QComboBox()
        self.sensor_source_input.addItem('Reale Sensordaten', SENSOR_SOURCE_REAL_ALL)
        self.sensor_source_input.addItem('Mock-Daten', SENSOR_SOURCE_MOCK_ALL)
        self.sensor_source_input.addItem('Kraftsensor steuert Referenzsensor', SENSOR_SOURCE_FORCE_SENSOR)
        self.sensor_source_input.setCurrentIndex(0)

        self.reference_sensor_input = QComboBox()
        for index in range(8):
            self.reference_sensor_input.addItem(f'R{index + 1}', index)
        self.reference_sensor_input.setCurrentIndex(self.reference_sensor_index)

        self.fullscreen_checkbox = QCheckBox('Vollbild')
        self.fullscreen_checkbox.setChecked(True)
        self.expo_theme_checkbox = QCheckBox('Messe-Design')
        self.expo_theme_checkbox.setChecked(False)

        self.voltage_threshold_input.valueChanged.connect(self._update_voltage_threshold)
        self.force_threshold_input.valueChanged.connect(self._update_force_threshold)
        self.sensor_source_input.currentIndexChanged.connect(self._update_sensor_source_mode)
        self.reference_sensor_input.currentIndexChanged.connect(self._update_reference_sensor)
        self.fullscreen_checkbox.toggled.connect(self._set_fullscreen_enabled)
        self.expo_theme_checkbox.toggled.connect(self._set_expo_theme_enabled)

        for label_text, controls_widget in (
            ('Datenquelle', self.sensor_source_input),
            ('Referenzsensor', self.reference_sensor_input),
            ('Kraft-Sollwert', self._build_stepper_controls(self.force_input)),
            ('Ansicht', self.fullscreen_checkbox),
            ('Design', self.expo_theme_checkbox),
            ('Spannungs-Threshold', self._build_stepper_controls(self.voltage_threshold_input)),
            ('Kraft-Threshold', self._build_stepper_controls(self.force_threshold_input)),
        ):
            row = QHBoxLayout()
            row.setSpacing(14)
            caption = self._make_label_caption(label_text)
            caption.setMinimumWidth(180)
            row.addWidget(caption)
            row.addWidget(controls_widget, stretch=1)
            layout.addLayout(row)
        layout.addStretch(1)
        return section

    def _build_stepper_controls(self, spinbox, compact: bool = False) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        if compact:
            container.setObjectName('measurementStepperContainer')
            spinbox.setObjectName('measurementSpinBox')
        spinbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(spinbox, stretch=1)

        minus_button = QPushButton('-')
        plus_button = QPushButton('+')
        for button, action in ((minus_button, spinbox.stepDown), (plus_button, spinbox.stepUp)):
            button.setObjectName('measurementStepper' if compact else 'adminStepper')
            button.setMinimumSize(38, 34) if compact else button.setMinimumSize(48, 44)
            button.clicked.connect(action)
            row.addWidget(button)
        return container

    def _build_point_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

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
        self.measurement_input.setMinimumWidth(72)
        self.force_input.setMinimumWidth(90)

        for button in [self.start_button, self.mode_two_button, self.calibrate_button, self.admin_button, self.point_reset_button, self.exit_button]:
            button.setMinimumHeight(40)
        self.capture_button.setMinimumHeight(40)
        self.sensor_source_input.setMinimumWidth(420)
        self.sensor_source_input.setMinimumHeight(44)
        self.reference_sensor_input.setMinimumWidth(220)
        self.reference_sensor_input.setMinimumHeight(44)
        self.fullscreen_checkbox.setMinimumHeight(44)
        self.expo_theme_checkbox.setMinimumHeight(44)
        self.voltage_threshold_input.setMinimumWidth(320)
        self.force_threshold_input.setMinimumWidth(320)
        for spinbox in (self.voltage_threshold_input, self.force_threshold_input):
            spinbox.setMinimumHeight(44)

        if self.expo_theme_enabled:
            main_bg = "#f7fbff"
            widget_bg = "#f7fbff"
            panel_bg = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #081a2a, stop:1 #0f2f44)"
            panel_border = "#4fd1ff"
            card_bg = "rgba(10, 27, 43, 0.84)"
            card_border = "#2bc2ff"
            label_color = "#e6f7ff"
            caption_color = "#8fdcff"
            title_color = "#e6fbff"
            field_bg = "#f8fcff"
            field_border = "#7dd3fc"
            hover_bg = "#eff8ff"
            admin_stepper_bg = "#ecfeff"
            admin_stepper_border = "#22d3ee"
            mode_idle_bg = "#2b1220"
            mode_idle_fg = "#ffd8e4"
            mode_idle_border = "#fb7185"
            mode_active_bg = "#082f1d"
            mode_active_fg = "#d1fae5"
            mode_active_border = "#34d399"
            capture_idle_bg = "#2b1220"
            capture_idle_fg = "#ffd8e4"
            capture_idle_border = "#fb7185"
            capture_active_bg = "#082f1d"
            capture_active_fg = "#d1fae5"
            capture_active_border = "#34d399"
            reset_bg = "#fff0c2"
            reset_fg = "#d97706"
            reset_border = "#fdba74"
            end_bg = "#7f1d1d"
            end_fg = "#ffffff"
            end_border = "#ef4444"
            progress_bg = "#0b2133"
            progress_border = "#2bc2ff"
            progress_chunk = "#22d3ee"
        else:
            main_bg = "#ffffff"
            widget_bg = "#ffffff"
            panel_bg = "#ffffff"
            panel_border = "#d7dee7"
            card_bg = "#ffffff"
            card_border = "#e6ebf1"
            label_color = "#1f2a37"
            caption_color = "#5b6470"
            title_color = "#1f2a37"
            field_bg = "white"
            field_border = "#c7d0d9"
            hover_bg = "#f7fafc"
            admin_stepper_bg = "#ffffff"
            admin_stepper_border = "#94a3b8"
            mode_idle_bg = "#fee2e2"
            mode_idle_fg = "#7f1d1d"
            mode_idle_border = "#fca5a5"
            mode_active_bg = "#dcfce7"
            mode_active_fg = "#166534"
            mode_active_border = "#86efac"
            capture_idle_bg = "#fee2e2"
            capture_idle_fg = "#7f1d1d"
            capture_idle_border = "#fca5a5"
            capture_active_bg = "#dcfce7"
            capture_active_fg = "#166534"
            capture_active_border = "#86efac"
            reset_bg = "#fde68a"
            reset_fg = "#c2410c"
            reset_border = "#fdba74"
            end_bg = "#b42318"
            end_fg = "white"
            end_border = "#8f1c13"
            progress_bg = "white"
            progress_border = "#c7d0d9"
            progress_chunk = "#0f8b6d"

        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {main_bg};
            }}
            QWidget {{
                background-color: {widget_bg};
            }}
            QFrame#plotPanel, QFrame#settingsPanel {{
                background: {panel_bg};
                border: 1px solid {panel_border};
                border-radius: 16px;
            }}
            QLabel#captionLabel {{
                color: {caption_color};
                font-weight: 600;
            }}
            QLabel {{
                color: {label_color};
            }}
            QFrame#sectionCard {{
                background: {card_bg};
                border: 1px solid {card_border};
                border-radius: 12px;
            }}
            QLabel#sectionTitle {{
                color: {title_color};
                font-size: 14px;
                font-weight: 700;
            }}
            QPushButton {{
                background-color: {field_bg};
                border: 1px solid {field_border};
                border-radius: 10px;
                padding: 8px 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {hover_bg};
            }}
            QPushButton:disabled {{
                color: #98a2ad;
                background-color: #eef2f5;
            }}
            QPushButton[text="Punkt resetten"] {{
                background-color: {reset_bg};
                color: {reset_fg};
                border: 1px solid {reset_border};
            }}
            QPushButton[modeButton="true"] {{
                background-color: {mode_idle_bg};
                color: {mode_idle_fg};
                border: 1px solid {mode_idle_border};
            }}
            QPushButton[modeButton="true"]:hover {{
                background-color: {hover_bg};
            }}
            QPushButton[modeButton="true"][modeActive="true"] {{
                background-color: {mode_active_bg};
                color: {mode_active_fg};
                border: 1px solid {mode_active_border};
            }}
            QPushButton[modeButton="true"][modeActive="true"]:hover {{
                background-color: {hover_bg};
            }}
            QPushButton[captureButton="true"] {{
                background-color: {capture_idle_bg};
                color: {capture_idle_fg};
                border: 1px solid {capture_idle_border};
                font-weight: 600;
            }}
            QPushButton[captureButton="true"]:hover {{
                background-color: {hover_bg};
            }}
            QPushButton[captureButton="true"][captureActive="true"] {{
                background-color: {capture_active_bg};
                color: {capture_active_fg};
                border: 1px solid {capture_active_border};
            }}
            QPushButton[captureButton="true"][captureActive="true"]:hover {{
                background-color: {hover_bg};
            }}
            QPushButton[text="Beenden"] {{
                background-color: {end_bg};
                color: {end_fg};
                border: 1px solid {end_border};
            }}
            QPushButton#adminStepper {{
                background: {admin_stepper_bg};
                color: #111827;
                border: 1px solid {admin_stepper_border};
                border-radius: 8px;
                font-size: 22px;
                font-weight: 700;
                padding: 0px;
            }}
            QPushButton#adminStepper:hover {{
                background: {hover_bg};
            }}
            QPushButton#measurementStepper {{
                background: {admin_stepper_bg};
                color: #111827;
                border: 1px solid {admin_stepper_border};
                border-radius: 8px;
                font-size: 18px;
                font-weight: 700;
                padding: 0px;
            }}
            QPushButton#measurementStepper:hover {{
                background: {hover_bg};
            }}
            QSpinBox, QDoubleSpinBox, QComboBox {{
                background: {field_bg};
                border: 1px solid {field_border};
                border-radius: 8px;
                padding: 8px 12px;
                min-height: 40px;
                font-size: 16px;
            }}
            QSpinBox#measurementSpinBox {{
                padding: 4px 8px;
                min-height: 32px;
                font-size: 14px;
            }}
            QCheckBox {{
                color: {label_color};
                font-size: 16px;
                spacing: 10px;
                padding: 8px 0px;
            }}
            QCheckBox::indicator {{
                width: 24px;
                height: 24px;
                border: 2px solid {field_border};
                border-radius: 6px;
                background: {field_bg};
            }}
            QCheckBox::indicator:checked {{
                background: #2563eb;
                border: 2px solid #2563eb;
            }}
            QComboBox::drop-down {{
                width: 34px;
                border-left: 1px solid {field_border};
                background: {hover_bg};
            }}
            QComboBox::down-arrow {{
                width: 12px;
                height: 12px;
            }}
            QProgressBar {{
                background: {progress_bg};
                border: 1px solid {progress_border};
                border-radius: 8px;
                min-height: 18px;
                text-align: center;
                color: {title_color};
            }}
            QProgressBar::chunk {{
                background-color: {progress_chunk};
                border-radius: 7px;
            }}
        """)

    def _configure_mode_buttons(self):
        for button in (self.start_button, self.mode_two_button, self.calibrate_button, self.admin_button):
            button.setProperty('modeButton', True)
            button.setProperty('modeActive', False)
        self.capture_button.setProperty('captureButton', True)
        self.capture_button.setProperty('captureActive', False)
        self._refresh_mode_button_styles()
        self._refresh_capture_button_style()

    def _set_mode_button_active(self, button: QPushButton, active: bool):
        button.setProperty('modeActive', active)
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()

    def _confirm_mode_switch(self, mode_name: str) -> bool:
        password, accepted = QInputDialog.getText(
            self,
            'Passwort eingeben',
            f'Bitte Passwort fuer {mode_name} eingeben:',
        )
        if not accepted:
            self.status_label.setText(f'{mode_name} nicht aktiviert.')
            return False
        if password != MODE_SWITCH_PASSWORD:
            QMessageBox.warning(self, 'Falsches Passwort', 'Das eingegebene Passwort ist nicht korrekt.')
            self.status_label.setText(f'{mode_name} nicht aktiviert.')
            return False
        return True

    def _refresh_capture_button_style(self):
        self.capture_button.setProperty('captureActive', self.auto_capture_enabled)
        self.capture_button.style().unpolish(self.capture_button)
        self.capture_button.style().polish(self.capture_button)
        self.capture_button.update()

    def _refresh_mode_button_styles(self):
        betrieb1_active = self._mode_one_selected
        betrieb2_active = self.is_sampling and not self.calibration_panel.calibration_active and self._mode_two_selected
        kalibriermodus_active = self.calibration_panel.calibration_active
        admin_active = self._admin_selected
        self._set_mode_button_active(self.start_button, betrieb1_active)
        self._set_mode_button_active(self.mode_two_button, betrieb2_active)
        self._set_mode_button_active(self.calibrate_button, kalibriermodus_active)
        self._set_mode_button_active(self.admin_button, admin_active)

    def _set_admin_visible(self, visible: bool):
        self.admin_card.setVisible(bool(visible))
        self.plots_container.setVisible(not visible)
        self.point_card.setVisible(not visible)
        self.status_widget.setVisible(not visible)
        self.info_panel.setMaximumHeight(760 if visible else 480)

    def _update_voltage_threshold(self, value: float):
        self.sensor_baseline_threshold_v = float(value)
        self.live_plot.set_threshold(self.sensor_baseline_threshold_v)

    def _update_force_threshold(self, value: float):
        self.force_target_tolerance_n = float(value)
        self.force_plot.set_threshold(self.force_target_tolerance_n)

    def _update_sensor_source_mode(self, _: int):
        mode = self.sensor_source_input.currentData()
        self.sensor_source_mode = str(mode) if mode is not None else SENSOR_SOURCE_FORCE_SENSOR
        if self.is_sampling:
            self.refresh_sensor_data()

    def _set_expo_theme_enabled(self, enabled: bool):
        self.expo_theme_enabled = bool(enabled)
        self._apply_styles()

    def _current_reference_sensor_label(self) -> str:
        return f'R{self.reference_sensor_index + 1}'

    def _current_sensor_coupling(self) -> np.ndarray:
        return _compute_mode_one_sensor_coupling(self.reference_sensor_index)

    def _update_reference_sensor(self, _: int):
        index = self.reference_sensor_input.currentData()
        self.reference_sensor_index = int(index) if index is not None else DEFAULT_REFERENCE_SENSOR_INDEX
        self.mode_one_force_baseline_v = float(self.sensor_baseline_values[self.reference_sensor_index])
        self.live_plot.set_baseline_value(self.mode_one_force_baseline_v)
        if self.is_sampling:
            self.refresh_sensor_data()

    def _set_fullscreen_enabled(self, enabled: bool):
        self.fullscreen_enabled = bool(enabled)
        if self.fullscreen_enabled:
            self.showFullScreen()
        else:
            self.showNormal()

    def _build_mock_sensor_frame(self) -> SensorFrame:
        mock_values = self.sensor_baseline_values + self._mock_sensor_rng.uniform(
            low=-self.sensor_baseline_threshold_v,
            high=self.sensor_baseline_threshold_v,
            size=8,
        )
        return SensorFrame(
            sensors=np.asarray(mock_values, dtype=float),
            timestamp=datetime.now(UTC),
            source='mock_admin_mode',
            metadata={'reader': 'AdminMock'},
        )

    def _build_force_control_sensor_frame(self) -> SensorFrame:
        return SensorFrame(
            sensors=self.sensor_baseline_values.copy(),
            timestamp=datetime.now(UTC),
            source='force_r6_admin_mode',
            metadata={'reader': 'AdminForceR6'},
        )

    def _read_base_sensor_frame(self) -> SensorFrame:
        if self.sensor_source_mode == SENSOR_SOURCE_MOCK_ALL:
            return self._build_mock_sensor_frame()
        if self.sensor_source_mode == SENSOR_SOURCE_REAL_ALL:
            return self.real_reader.read()
        if self.sensor_source_mode == SENSOR_SOURCE_FORCE_SENSOR and (self._mode_one_selected or self.calibration_panel.calibration_active):
            return self._build_force_control_sensor_frame()
        return self.real_reader.read()

    def _deactivate_calibration_mode(self):
        if not self.calibration_panel.calibration_active:
            return
        self.calibration_panel.calibration_active = False
        self.calibration_panel._refresh_display()

    def _stop_sampling_mode(self):
        if not self.is_sampling:
            return
        self.timer.stop()
        self.is_sampling = False
        self.auto_capture_enabled = False
        self.force_input.setEnabled(True)
        self._reset_force_hold_progress(sync_sequence=False)
        self._sensor_connection_confirmed = False

    def _reset_mode_one_sensor_view(self):
        self.live_plot.reset_sensor_filter()
        self._mode_one_noise_state = np.zeros(8, dtype=float)

    def _reset_live_sensor_plot(self):
        self.live_plot.update_values(np.zeros(8, dtype=float))

    def _set_mode_one_sensor_view(self):
        if MODE_ONE_LOCK_SENSOR_FILTER:
            self.live_plot.set_visible_sensor_indices((self.reference_sensor_index,), lock_selection=True)
            return
        self.live_plot.reset_sensor_filter()

    def _baseline_values_copy(self) -> np.ndarray:
        baseline_values = self.sensor_baseline_values.copy()
        if baseline_values.size < 8:
            padded = np.full(8, self.mode_one_force_baseline_v, dtype=float)
            padded[:baseline_values.size] = baseline_values
            baseline_values = padded
        else:
            baseline_values = baseline_values[:8].copy()
        return baseline_values

    def _update_mode_one_noise(self) -> np.ndarray:
        random_step = self._mock_sensor_rng.normal(loc=0.0, scale=MODE_ONE_NOISE_STEP_STD_V, size=8)
        self._mode_one_noise_state = (MODE_ONE_NOISE_DECAY * self._mode_one_noise_state) + random_step
        self._mode_one_noise_state = np.clip(self._mode_one_noise_state, -MODE_ONE_NOISE_MAX_V, MODE_ONE_NOISE_MAX_V)
        self._mode_one_noise_state[self.reference_sensor_index] = 0.0
        return self._mode_one_noise_state.copy()

    def _apply_reference_coupling(self, reference_voltage: float, with_noise: bool = False) -> np.ndarray:
        baseline_values = self._baseline_values_copy()
        reference_baseline_v = float(baseline_values[self.reference_sensor_index])
        reference_delta_v = reference_voltage - reference_baseline_v
        coupled_values = baseline_values + (reference_delta_v * self._current_sensor_coupling())
        coupled_values[self.reference_sensor_index] = reference_voltage
        if with_noise:
            coupled_values += self._update_mode_one_noise()
            coupled_values[self.reference_sensor_index] = reference_voltage
        return coupled_values

    def _transform_mode_one_sensor_values(self, sensor_values: np.ndarray) -> np.ndarray:
        values = np.asarray(sensor_values, dtype=float)
        reference_voltage = float(values[min(self.reference_sensor_index, max(0, values.size - 1))]) if values.size else self.mode_one_force_baseline_v
        return self._apply_reference_coupling(reference_voltage, with_noise=True)

    def _transform_mode_one_force_sensor_values(self) -> np.ndarray:
        force_value = 0.0 if self.latest_force_value is None else float(self.latest_force_value)
        reference_voltage = self.mode_one_force_baseline_v + (force_value * MODE_ONE_FORCE_VOLTS_PER_NEWTON)
        return self._apply_reference_coupling(reference_voltage, with_noise=True)

    def _transform_calibration_sensor_values(self, sensor_values: np.ndarray) -> np.ndarray:
        values = np.asarray(sensor_values, dtype=float)
        baseline_values = self._baseline_values_copy()
        reference_voltage = float(values[min(self.reference_sensor_index, max(0, values.size - 1))]) if values.size else float(baseline_values[self.reference_sensor_index])
        baseline_values[self.reference_sensor_index] = reference_voltage
        baseline_values += self._update_mode_one_noise()
        baseline_values[self.reference_sensor_index] = reference_voltage
        return baseline_values

    def _current_calibration_reference_voltage(self) -> float:
        force_value = 0.0 if self.latest_force_value is None else float(self.latest_force_value)
        return self.mode_one_force_baseline_v + (force_value * MODE_ONE_FORCE_VOLTS_PER_NEWTON)

    def _transform_calibration_force_sensor_values(self) -> np.ndarray:
        baseline_values = self._baseline_values_copy()
        reference_voltage = self._current_calibration_reference_voltage()
        baseline_values[self.reference_sensor_index] = reference_voltage
        baseline_values += self._update_mode_one_noise()
        baseline_values[self.reference_sensor_index] = reference_voltage
        return baseline_values

    def activate_mode_one(self):
        if self._mode_one_selected and self.is_sampling:
            self._mode_one_selected = False
            self._admin_selected = False
            self._set_admin_visible(False)
            self._reset_mode_one_sensor_view()
            self._reset_live_sensor_plot()
            self._stop_sampling_mode()
            self._refresh_mode_button_styles()
            self.status_label.setText('Betrieb 1 angehalten.')
            return

        if not self._confirm_mode_switch('Betrieb 1'):
            return
        self._deactivate_calibration_mode()
        self._stop_sampling_mode()
        self._mode_two_selected = False
        self._admin_selected = False
        self._set_admin_visible(False)
        self._mode_one_selected = True
        self._set_mode_one_sensor_view()
        if not self.refresh_sensor_data(show_connection_feedback=True):
            self._mode_one_selected = False
            self._reset_mode_one_sensor_view()
            self._refresh_mode_button_styles()
            return
        self.timer.start(REFRESH_MS)
        self.is_sampling = True
        self._refresh_mode_button_styles()
        self.status_label.setText(
            f'Betrieb 1 ist aktiv. 0 N entspricht aktuell {self.mode_one_force_baseline_v:.2f} V, '
            f'und die Sensoranzeigen folgen {self._current_reference_sensor_label()} mit {MODE_ONE_FORCE_VOLTS_PER_NEWTON:.2f} V pro N.'
        )

    def _update_force_target(self, value: int):
        self.force_plot.set_target_force(float(value))
        self._reset_force_hold_progress(sync_sequence=self.auto_capture_enabled)
        if not self.auto_capture_enabled:
            self._update_manual_force_ui()

    def _current_force_target(self) -> int:
        return int(FORCE_SEQUENCE_N[self._auto_force_index])

    def _current_measurement_path(self, measurement_number: int) -> Path:
        return self._measurement_path(measurement_number, force_value=self._current_force_target())

    def _set_force_target(self, value: int):
        self.force_input.blockSignals(True)
        self.force_input.setValue(int(value))
        self.force_input.blockSignals(False)
        self.force_plot.set_target_force(float(value))

    def _sync_force_sequence_ui(self):
        target_force = self._current_force_target()
        self._set_force_target(target_force)
        step_text = f'{self._auto_force_index + 1}/{len(FORCE_SEQUENCE_N)}'
        self.force_hold_label.setText(
            f'Kraft {target_force} N halten ({step_text}): 0.0 / {FORCE_TARGET_HOLD_S:.1f} s'
        )
        measurement_number = self.measurement_input.value()
        if self.recorders:
            self.file_label.setText(self._measurement_path(measurement_number, force_value=target_force).name)
        else:
            self.file_label.setText(f'Messung {measurement_number:02d} | {len(FORCE_SEQUENCE_N)} Kraftstufen')

    def _update_manual_force_ui(self):
        target_force = int(self.force_input.value())
        self.force_hold_label.setText(
            f'Kraft {target_force} N halten: 0.0 / {FORCE_TARGET_HOLD_S:.1f} s'
        )

    def _reset_force_hold_progress(self, sync_sequence: bool = False):
        self._force_in_tolerance_since = None
        self._auto_capture_rows.clear()
        self.force_hold_progress.setValue(0)
        if sync_sequence:
            self._sync_force_sequence_ui()
        else:
            self._update_manual_force_ui()

    def _filter_display_force(self, value: float) -> float:
        raw_value = float(value)

        if len(self._force_display_history) >= 5:
            history = np.asarray(self._force_display_history, dtype=float)
            median = float(np.median(history))
            mad = float(np.median(np.abs(history - median)))
            deviation_limit = max(0.20, 6.0 * mad)
            if abs(raw_value - median) > deviation_limit:
                raw_value = median + np.sign(raw_value - median) * deviation_limit

        self._force_display_history.append(raw_value)

        if self._force_display_value is None:
            self._force_display_value = raw_value
        else:
            alpha = FORCE_DISPLAY_SMOOTHING_ALPHA
            self._force_display_value = ((1.0 - alpha) * self._force_display_value) + (alpha * raw_value)

        return float(self._force_display_value)

    def _downsample_force_plot_samples(self, samples: list[tuple[datetime, float]]) -> list[tuple[datetime, float]]:
        if len(samples) <= FORCE_PLOT_MAX_BATCH_POINTS:
            return samples

        step = max(1, len(samples) // FORCE_PLOT_MAX_BATCH_POINTS)
        reduced = samples[::step]
        if reduced[-1] != samples[-1]:
            reduced.append(samples[-1])
        return reduced

    def _flush_force_plot(self, now_ts: float, force: bool = False):
        should_update_plot = force or (
            self._last_force_plot_update_ts == 0.0
            or (now_ts - self._last_force_plot_update_ts) >= (FORCE_PLOT_UI_REFRESH_MS / 1000.0)
        )
        if not should_update_plot or not self._pending_force_plot_samples:
            return

        plot_samples = self._downsample_force_plot_samples(list(self._pending_force_plot_samples))
        self._pending_force_plot_samples.clear()
        self.force_plot.append_values(plot_samples)
        self._last_force_plot_update_ts = now_ts

    def _update_force_hold_progress(self, force_value: float | None, timestamp: datetime | None, stream_active: bool):
        if force_value is None or timestamp is None or not stream_active:
            self._reset_force_hold_progress(sync_sequence=self.auto_capture_enabled)
            return

        target_force = float(self.force_input.value())
        within_tolerance = abs(force_value - target_force) <= self.force_target_tolerance_n
        if not within_tolerance:
            self._reset_force_hold_progress(sync_sequence=self.auto_capture_enabled)
            return

        if self._force_in_tolerance_since is None:
            self._force_in_tolerance_since = timestamp

        if self.auto_capture_enabled:
            self._append_auto_capture_row(timestamp, force_value)

        held_seconds = max(0.0, (timestamp - self._force_in_tolerance_since).total_seconds())
        clamped_seconds = min(held_seconds, FORCE_TARGET_HOLD_S)
        self.force_hold_progress.setValue(int(clamped_seconds * 1000))
        target_force = self._current_force_target()
        step_text = f'{self._auto_force_index + 1}/{len(FORCE_SEQUENCE_N)}'
        self.force_hold_label.setText(
            f'Kraft {target_force} N halten ({step_text}): {clamped_seconds:.1f} / {FORCE_TARGET_HOLD_S:.1f} s'
        )
        if self.auto_capture_enabled and held_seconds >= FORCE_TARGET_HOLD_S:
            self._record_auto_capture()

    def _measurement_path(self, measurement_number: int, force_value: int | None = None) -> Path:
        if force_value is None:
            force_value = int(self.force_input.value())
        return self.output_dir / f'3D_Messung_{measurement_number:02d}_{force_value}N.csv'

    def save_screenshot(self):
        desktop_dir = Path.home() / 'Desktop'
        desktop_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = desktop_dir / f'memo_ui_{timestamp}.png'
        pixmap = self.grab()
        if pixmap.isNull():
            QMessageBox.warning(self, 'Screenshot fehlgeschlagen', 'Das Fenster konnte nicht als Bild gespeichert werden.')
            return
        if not pixmap.save(str(screenshot_path), 'PNG'):
            QMessageBox.warning(self, 'Screenshot fehlgeschlagen', f'Die Datei konnte nicht gespeichert werden:\n{screenshot_path}')
            return
        self.status_label.setText(f'Screenshot gespeichert: {screenshot_path.name}')
        QMessageBox.information(self, 'Screenshot gespeichert', f'Der Screenshot wurde gespeichert unter:\n{screenshot_path}')

    def _has_started_measurement(self) -> bool:
        return any(recorder.sample_count > 0 for recorder in self.recorders.values())

    def _initialize_measurement(self, measurement_number: int, allow_resume: bool, prompt_if_exists: bool):
        csv_paths = {
            force_value: self._measurement_path(measurement_number, force_value=force_value)
            for force_value in FORCE_SEQUENCE_N
        }
        overwrite = False

        existing_paths = [path for path in csv_paths.values() if path.exists()]
        if existing_paths and prompt_if_exists:
            reply = QMessageBox.question(
                self,
                'Dateien existieren bereits',
                (
                    f'Es existieren bereits {len(existing_paths)} Dateien fuer die Kraftserie '
                    f'von Messung {measurement_number:02d}. Sollen sie ueberschrieben werden?'
                ),
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

        self.recorders = {
            force_value: CsvSampleRecorder(csv_paths[force_value], overwrite=overwrite)
            for force_value in FORCE_SEQUENCE_N
        }
        if overwrite:
            self.grid_plot.reset_samples()
            self.status_label.setText(
                f'Neue Mehrkraft-Messung gestartet: {measurement_number:02d} '
                f'({len(FORCE_SEQUENCE_N)} Kraftstufen pro Punkt)'
            )
        else:
            saved_point_sets = [set(recorder.read_saved_points()) for recorder in self.recorders.values()]
            completed_points = set.intersection(*saved_point_sets) if saved_point_sets else set()
            if completed_points:
                self.grid_plot.load_saved_points(sorted(completed_points))
                self.status_label.setText(
                    f'Bestehende Mehrkraft-Messung fortgesetzt: {measurement_number:02d} '
                    f'({len(completed_points)} Punkte komplett)'
                )
            else:
                self.grid_plot.reset_samples()
                self.status_label.setText(
                    f'Neue Mehrkraft-Messung gestartet: {measurement_number:02d} '
                    f'({len(FORCE_SEQUENCE_N)} Kraftstufen pro Punkt)'
                )

        completed_points = len(self.grid_plot.saved_points)
        self.count_label.setText(f'{completed_points} Punkte komplett')
        self._auto_force_index = 0
        self._sync_force_sequence_ui()
        self._update_sample_status()
        return True

    def _ensure_measurement_initialized(self, allow_resume: bool = True, prompt_if_exists: bool = True):
        if self.recorders:
            return True
        measurement_number = self.measurement_input.value()
        return self._initialize_measurement(measurement_number, allow_resume=allow_resume, prompt_if_exists=prompt_if_exists)

    def _update_point_selection(self, point, is_saved: bool = False):
        if point is None:
            self.target_label.setText('keiner')
            self.point_state_label.setText('-')
            self.capture_button.setEnabled(False)
            self.capture_button.setText('Starte automatische Punktaufnahme')
            self._refresh_capture_button_style()
            self.point_reset_button.setEnabled(False)
            return

        self.target_label.setText(f'X={point[0]:.1f}, Y={point[1]:.1f}')
        self.point_state_label.setText('gespeichert' if is_saved else 'offen')
        self.capture_button.setEnabled(not is_saved)
        self.capture_button.setText(
            'Automatische Punktaufnahme stoppen' if self.auto_capture_enabled and not is_saved else 'Starte automatische Punktaufnahme'
        )
        self._refresh_capture_button_style()
        self.point_reset_button.setEnabled(is_saved)

    def toggle_auto_capture(self):
        if self.auto_capture_enabled:
            self.auto_capture_enabled = False
            self.force_input.setEnabled(True)
            self._reset_force_hold_progress(sync_sequence=False)
            self.capture_button.setText('Starte automatische Punktaufnahme')
            self._refresh_capture_button_style()
            self.status_label.setText('Automatische Punktaufnahme gestoppt.')
            return

        if not self.is_sampling or self.calibration_panel.calibration_active or not (self._mode_one_selected or self._mode_two_selected):
            QMessageBox.warning(self, 'Betriebsmodus inaktiv', 'Bitte zuerst Betrieb 1 oder Betrieb 2 starten.')
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
        self.force_input.setEnabled(False)
        self._auto_force_index = 0
        self._reset_force_hold_progress(sync_sequence=True)
        self.capture_button.setText('Automatische Punktaufnahme stoppen')
        self._refresh_capture_button_style()
        self.status_label.setText(
            f'Automatische Punktaufnahme aktiv. Punkt halten und Kraftfolge {list(FORCE_SEQUENCE_N)} N je {FORCE_TARGET_HOLD_S:.0f}s aufnehmen.'
        )

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
            target_force = self._current_force_target()
            recorder = self.recorders[target_force]
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
            recorder.append_sample(sample)
            raw_file_path = self.raw_writer.write_timeseries(rows, measurement_number=measurement_number)
            summary_file_path = self.raw_writer.write_timeseries_summary(rows, measurement_number=measurement_number)
            force_step_text = f'{self._auto_force_index + 1}/{len(FORCE_SEQUENCE_N)}'

            if self._auto_force_index < (len(FORCE_SEQUENCE_N) - 1):
                self._auto_force_index += 1
                self.status_label.setText(
                    f'Punkt X={active_point[0]:.1f}, Y={active_point[1]:.1f}: Kraft {target_force} N ({force_step_text}) gespeichert. '
                    f'Naechste Kraft: {self._current_force_target()} N.'
                )
            else:
                self.grid_plot.mark_point_saved(active_point)
                completed_points = len(self.grid_plot.saved_points)
                self.count_label.setText(f'{completed_points} Punkte komplett')
                self._update_sample_status()
                self._auto_force_index = 0

                if self.grid_plot.remaining_sample_count() == 0:
                    self.auto_capture_enabled = False
                    self.force_input.setEnabled(True)
                    self.capture_button.setText('Starte automatische Punktaufnahme')
                    self._refresh_capture_button_style()
                    self.status_label.setText(
                        f'Punkt X={active_point[0]:.1f}, Y={active_point[1]:.1f} komplett gespeichert | Raw: {raw_file_path.name} | Summary: {summary_file_path.name}'
                    )
                    QMessageBox.information(
                        self,
                        'Messung abgeschlossen',
                        'Alle 49 Innenpunkte wurden fuer alle Kraftstufen gespeichert.',
                    )
                else:
                    next_point = self.grid_plot.get_active_point()
                    self.status_label.setText(
                        f'Punkt X={active_point[0]:.1f}, Y={active_point[1]:.1f} komplett gespeichert. '
                        f'Naechster Punkt: X={next_point[0]:.1f}, Y={next_point[1]:.1f}. '
                        f'Starte wieder bei {self._current_force_target()} N.'
                    )
        finally:
            self._reset_force_hold_progress(sync_sequence=True)
            self._auto_capture_in_progress = False
            self._update_point_selection(self.grid_plot.get_active_point(), False)

    def _update_sample_status(self):
        total_count = self.grid_plot.total_sample_count()
        remaining_count = self.grid_plot.remaining_sample_count()
        self.sample_status_label.setText(f'{total_count} insgesamt, {remaining_count} offen')
    def toggle_sampling(self):
        self._mode_one_selected = False
        if self.is_sampling and self._mode_two_selected:
            self._stop_sampling_mode()
            self._mode_two_selected = False
            self._admin_selected = False
            self._set_admin_visible(False)
            self._reset_mode_one_sensor_view()
            self._reset_live_sensor_plot()
            self._refresh_mode_button_styles()
            self.capture_button.setText('Starte automatische Punktaufnahme')
            self._refresh_capture_button_style()
            self.status_label.setText('Betrieb 2 angehalten.')
            return

        if not self._confirm_mode_switch('Betrieb 2'):
            return
        self._deactivate_calibration_mode()
        self._admin_selected = False
        self._set_admin_visible(False)
        self._reset_mode_one_sensor_view()
        self._mode_two_selected = True
        if not self.refresh_sensor_data(show_connection_feedback=True):
            self._mode_two_selected = False
            self._refresh_mode_button_styles()
            return

        self.timer.start(REFRESH_MS)
        self.is_sampling = True
        self._refresh_mode_button_styles()

    def toggle_calibration(self):
        self._mode_one_selected = False
        self._mode_two_selected = False
        self._admin_selected = False
        self._set_admin_visible(False)
        self._reset_mode_one_sensor_view()
        if self.calibration_panel.calibration_active:
            self._deactivate_calibration_mode()
            self._stop_sampling_mode()
            self._reset_live_sensor_plot()
            self._refresh_mode_button_styles()
            self.status_label.setText('Kalibrierung gestoppt.')
            return

        if not self._confirm_mode_switch('Kalibriermodus'):
            return
        self.calibration_panel.start_calibration()
        if not self.is_sampling:
            self.timer.start(REFRESH_MS)
            self.is_sampling = True
        self._refresh_mode_button_styles()
        self.status_label.setText('Kalibrierung aktiv.')
        self.refresh_sensor_data()

    def activate_admin_mode(self):
        if self._admin_selected:
            self._admin_selected = False
            self._set_admin_visible(False)
            self._refresh_mode_button_styles()
            self.status_label.setText('Admin-Modus geschlossen.')
            return

        if not self._confirm_mode_switch('Admin-Modus'):
            return
        self._deactivate_calibration_mode()
        self._stop_sampling_mode()
        self._mode_one_selected = False
        self._mode_two_selected = False
        self._admin_selected = True
        self._reset_mode_one_sensor_view()
        self._set_admin_visible(True)
        self._refresh_mode_button_styles()
        self.status_label.setText('Admin-Modus aktiv. Thresholds koennen jetzt angepasst werden.')

    def _has_valid_sensor_signal(self, sensor_values: np.ndarray) -> bool:
        values = np.asarray(sensor_values, dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            return False

        if isinstance(self.reader, Ads1115Reader):
            return not np.all(np.isclose(values, 0.0, atol=1e-4))
        return True

    def refresh_sensor_data(self, show_connection_feedback: bool = False):
        try:
            self.latest_frame = self._read_base_sensor_frame()
        except StopIteration:
            self.timer.stop()
            self.is_sampling = False
            self._sensor_connection_confirmed = False
            self._refresh_mode_button_styles()
            self.status_label.setText('Replay beendet.')
            return False
        except Exception as exc:
            self.timer.stop()
            self.is_sampling = False
            self._sensor_connection_confirmed = False
            self._refresh_mode_button_styles()
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
            self._refresh_mode_button_styles()
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
            display_sensor_values = self.latest_frame.sensors
            if self._mode_one_selected:
                if self.sensor_source_mode == SENSOR_SOURCE_FORCE_SENSOR:
                    display_sensor_values = self._transform_mode_one_force_sensor_values()
                else:
                    display_sensor_values = self._transform_mode_one_sensor_values(self.latest_frame.sensors)
            elif self.calibration_panel.calibration_active:
                if self.sensor_source_mode == SENSOR_SOURCE_FORCE_SENSOR:
                    display_sensor_values = self._transform_calibration_force_sensor_values()
                else:
                    display_sensor_values = self._transform_calibration_sensor_values(self.latest_frame.sensors)
            self.live_plot.update_values(display_sensor_values)
            self.calibration_panel.set_live_values(display_sensor_values)
            self._last_sensor_ui_update_ts = now_ts
        if show_connection_feedback and not self._sensor_connection_confirmed:
            if self.sensor_source_mode == SENSOR_SOURCE_MOCK_ALL:
                self.status_label.setText('Sensoren verbunden. Mock-Daten aktiv.')
            elif self.sensor_source_mode == SENSOR_SOURCE_REAL_ALL:
                self.status_label.setText('Sensoren verbunden. Reale Sensordaten aktiv.')
            elif self._mode_one_selected:
                self.status_label.setText(
                    f'Sensoren erfolgreich verbunden. Betrieb 1 aktiv mit {self._current_reference_sensor_label()} als Referenzsensor.'
                )
            elif self.calibration_panel.calibration_active:
                self.status_label.setText(
                    f'Sensoren erfolgreich verbunden. Kalibriermodus aktiv: nur {self._current_reference_sensor_label()} bleibt beweglich.'
                )
            else:
                self.status_label.setText('Sensoren erfolgreich verbunden. Betrieb 2 aktiv.')
            self._sensor_connection_confirmed = True
        elif self.is_sampling:
            if self.sensor_source_mode == SENSOR_SOURCE_MOCK_ALL:
                self.status_label.setText('Mock-Daten aktiv.')
            elif self.sensor_source_mode == SENSOR_SOURCE_REAL_ALL:
                self.status_label.setText('Reale Sensordaten aktiv.')
            elif self._mode_one_selected:
                self.status_label.setText(
                    f'Betrieb 1 aktiv mit {self._current_reference_sensor_label()} als Referenzsensor.'
                )
            elif self.calibration_panel.calibration_active:
                self.status_label.setText(
                    f'Kalibriermodus aktiv: nur {self._current_reference_sensor_label()} bleibt beweglich.'
                )
            elif self._mode_two_selected:
                self.status_label.setText('Betrieb 2 aktiv.')
        return True

    def _refresh_force_data(self):
        now = datetime.now(UTC)
        now_ts = now.timestamp()
        self.force_port_label.setText(f'{self.force_reader.port} @ {self.force_reader.baudrate}')
        if not self._force_reader_available:
            self.force_reader_status_label.setText('Deaktiviert nach Fehler')
            self.force_plot.set_stream_active(False)
            self._reset_force_hold_progress(sync_sequence=self.auto_capture_enabled)
            return
        reader_error = self.force_reader.get_last_error()
        if reader_error:
            self._force_reader_available = False
            self.status_label.setText(f'Kraftsensor nicht verfuegbar: {reader_error}')
            self.force_reader_status_label.setText(reader_error)
            self.force_raw_label.setText(self.force_reader.get_last_raw_text() or '-')
            self.force_plot.set_stream_active(False)
            self._reset_force_hold_progress(sync_sequence=self.auto_capture_enabled)
            return

        force_value = self.force_reader.get_latest_force()
        force_timestamp = self.force_reader.get_latest_force_timestamp()
        force_timestamp_ts = _to_utc_timestamp(force_timestamp)
        new_force_samples = self.force_reader.get_samples_since(self._last_force_sample_id)
        if new_force_samples:
            self._last_force_sample_id = new_force_samples[-1][0]
        stream_active = (
            force_timestamp_ts is not None
            and (now_ts - force_timestamp_ts) <= max(FORCE_STREAM_TIMEOUT_S, FORCE_REFRESH_MS / 1000.0 * 3.0)
        )
        if force_value is None or not stream_active:
            self.force_plot.set_stream_active(False)
            self.force_raw_label.setText(self.force_reader.get_last_raw_text() or '-')
            self.force_reader_status_label.setText(self.force_reader.connection_info())
            self._reset_force_hold_progress(sync_sequence=self.auto_capture_enabled)
            return
        self.force_plot.set_stream_active(True)

        self.latest_force_value = force_value
        display_value = self._force_display_value if self._force_display_value is not None else force_value
        if new_force_samples:
            for _, sample_timestamp, sample_force in new_force_samples:
                filtered_sample_force = self._filter_display_force(sample_force)
                self._pending_force_plot_samples.append((sample_timestamp, filtered_sample_force))
                self._update_force_hold_progress(filtered_sample_force, sample_timestamp, stream_active)
                display_value = filtered_sample_force
            self._flush_force_plot(now_ts)
        else:
            self._flush_force_plot(now_ts, force=True)
        self.force_value_label.setText(f'{display_value:.3f} N')
        self.force_raw_label.setText(self.force_reader.get_last_raw_text() or '-')
        self.force_reader_status_label.setText(self.force_reader.connection_info())

    def reset_point(self):
        if not self.recorders:
            QMessageBox.warning(self, 'Keine Messung', 'Es gibt noch keine gespeicherte Messung.')
            return

        selected_point = self.grid_plot.get_selected_point()
        if selected_point is None or not self.grid_plot.is_point_saved(selected_point):
            QMessageBox.warning(self, 'Kein gruener Punkt', 'Bitte zuerst einen grueneren gespeicherten Punkt im XY-Plot auswaehlen.')
            return

        removed_count = sum(recorder.remove_point(selected_point) for recorder in self.recorders.values())
        if removed_count <= 0:
            QMessageBox.warning(self, 'Nicht gefunden', 'Der ausgewaehlte Punkt konnte nicht aus der CSV entfernt werden.')
            return

        self.grid_plot.reset_saved_point(selected_point)
        self.count_label.setText(f'{len(self.grid_plot.saved_points)} Punkte komplett')
        self.status_label.setText(f'Punkt zurueckgesetzt: X={selected_point[0]:.1f}, Y={selected_point[1]:.1f}')
        self._update_sample_status()

    def reset_measurement(self):
        self.auto_capture_enabled = False
        self.force_input.setEnabled(True)
        self._reset_force_hold_progress(sync_sequence=False)
        self.capture_button.setText('Starte automatische Punktaufnahme')
        self._refresh_capture_button_style()
        if not self.recorders:
            self.grid_plot.reset_samples()
            self.measurement_input.setValue(self.measurement_input.value() + 1)
            self.recorders = {}
            self._auto_force_index = 0
            self._update_manual_force_ui()
            self.count_label.setText('0 Punkte komplett')
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

        self.recorders = {}
        self.grid_plot.reset_samples()
        self.measurement_input.setValue(self.measurement_input.value() + 1)
        self._auto_force_index = 0
        self._update_manual_force_ui()
        self.count_label.setText('0 Punkte komplett')
        next_path = self._current_measurement_path(self.measurement_input.value())
        self.status_label.setText(f'Bereit fuer neue Mehrkraft-Messung: {next_path.name}')
        self._update_sample_status()
        self._update_point_selection(self.grid_plot.get_active_point(), False)

        if not self.is_sampling:
            self.timer.start(REFRESH_MS)
            self.is_sampling = True
        self._mode_two_selected = False
        self._refresh_mode_button_styles()
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


def build_real_reader(replay_csv: str | None, sensor_port: str | None, sensor_baudrate: int, sensor_timeout: float):
    if replay_csv:
        return CsvReplayReader(replay_csv, loop=True)
    if sensor_port:
        return SerialSensorReader(
            port=sensor_port,
            baudrate=sensor_baudrate,
            timeout=sensor_timeout,
        )
    if sys.platform.startswith("win"):
        return MockReader(baseline=float(BASELINE_SENSOR_VALUES[0]), noise_std=0.05)
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


def build_mock_reader():
    return MockReader(baseline=float(BASELINE_SENSOR_VALUES[0]), noise_std=0.05)


def build_reader(replay_csv: str | None, sensor_port: str | None, sensor_baudrate: int, sensor_timeout: float):
    if DEBUG_SENSOR_MODE:
        return build_mock_reader()
    return build_real_reader(replay_csv, sensor_port, sensor_baudrate, sensor_timeout)


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
    real_reader = build_real_reader(args.replay_csv, args.sensor_port, args.sensor_baudrate, args.sensor_timeout)
    mock_reader = build_mock_reader()
    window = AcquisitionWindow(
        reader=reader,
        output_dir=Path(args.output_dir),
        force_port=args.force_port,
        force_baudrate=args.force_baudrate,
        real_reader=real_reader,
        mock_reader=mock_reader,
    )
    window.showFullScreen()
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(run_app())
